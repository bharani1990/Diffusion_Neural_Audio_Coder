# scripts/eval_codec.py
import sys
from pathlib import Path
import time
import torch
import soundfile as sf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from src.model import DiffusionUNet
from demo.vocoder import griffin_lim_waveform

def load_unet(state_path: Path, in_channels: int = 1):
    model = DiffusionUNet(in_channels=in_channels)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def pt_path_to_flac_path(pt_path: Path) -> Path:
    root = ROOT
    pt_path = Path(pt_path)
    if not pt_path.is_absolute():
        pt_path = root / pt_path

    parts = pt_path.parts
    try:
        idx = parts.index("test-clean")
        rel_parts = parts[idx + 1 :]
    except ValueError:
        raise ValueError(f"'test-clean' not found in {pt_path}")

    flac_rel = Path(*rel_parts)
    flac_path = root / "data" / "raw" / "test-clean" / flac_rel
    return flac_path.with_suffix(".flac")

def main():
    root = ROOT
    test_manifest = root / "data" / "processed" / "test_manifest.jsonl"
    target_frames = 120
    sample_rate = 16000

    test_ds = SpectrogramDataset(str(test_manifest), target_frames=target_frames)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    saved_unet = root / "saved_models" / "diffusion_unet.pt"
    model = load_unet(saved_unet, in_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    stats_path = root / "data" / "processed" / "stats.pt"
    stats = torch.load(stats_path)
    mean = torch.tensor(stats["mean"], device=device)
    std = torch.tensor(stats["std"], device=device)

    import json
    with open(test_manifest, "r", encoding="utf-8") as f:
        paths = [json.loads(line)["path"] for line in f if line.strip()]
    pt_paths = [Path(p) for p in paths]

    out_dir = root / "eval_outputs" / "test-clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    timesteps = 1000

    for i, (x_norm,) in enumerate(zip(test_dl)):
        x_norm = x_norm.to(device)
        b = x_norm.size(0)
        t = torch.randint(0, timesteps, (b,), device=device)

        start = time.perf_counter()
        noise = torch.randn_like(x_norm)
        xt = x_norm + noise
        with torch.no_grad():
            pred_noise = model(xt, t)
        x_denoised_norm = xt - pred_noise
        latency_ms = (time.perf_counter() - start) * 1000.0

        x_denoised = x_denoised_norm * std + mean
        x_denoised = x_denoised.squeeze(0)

        wav = griffin_lim_waveform(x_denoised.unsqueeze(0), sample_rate=sample_rate)
        wav = wav.squeeze(0).cpu().numpy()

        pt_path = pt_paths[i]
        flac_ref = pt_path_to_flac_path(pt_path)

        ref_wav, ref_sr = sf.read(flac_ref)
        if ref_sr != sample_rate:
            raise ValueError("reference sample rate mismatch")

        stem = flac_ref.stem
        out_path = out_dir / f"{stem}_codec.wav"
        ref_out_path = out_dir / f"{stem}_ref.wav"

        sf.write(out_path, wav, samplerate=sample_rate)
        if not ref_out_path.exists():
            sf.write(ref_out_path, ref_wav, samplerate=ref_sr)

        print(f"{i}: saved {out_path.name}, latency {latency_ms:.2f} ms")

        if i >= 9:
            break

if __name__ == "__main__":
    main()
