import sys
import time
import json
import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from src.model import DiffusionUNet
from demo.exp_1.vocoder import mel_to_linear, griffin_lim_waveform

def load_unet(state_path: Path, in_channels: int = 1) -> DiffusionUNet:
    model = DiffusionUNet(in_channels=in_channels)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def pt_path_to_flac_path(pt_path: Path, root: Path) -> Path:
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
    hop_length = 256
    timesteps = 1000

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

    pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, "wb").to(device)
    stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate).to(device)

    metrics = {"pesq": [], "stoi": [], "latency": []}

    with open(test_manifest, "r", encoding="utf-8") as f:
        paths = [json.loads(line)["path"] for line in f if line.strip()]
    pt_paths = [Path(p) for p in paths]

    out_dir = root / "eval_outputs" / "test-clean"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        metrics["latency"].append(latency_ms)

        x_denoised = x_denoised_norm * std + mean
        x_denoised = x_denoised.squeeze(0)

        wav = griffin_lim_waveform(x_denoised.unsqueeze(0), sample_rate=sample_rate)
        wav = wav.squeeze(0).cpu()

        pt_path = pt_paths[i]
        flac_ref = pt_path_to_flac_path(pt_path, root)

        if not flac_ref.exists():
            print(f"Warning! Missing reference: {flac_ref}")
            continue
        ref_wav, ref_sr = sf.read(str(flac_ref))
        if ref_sr != sample_rate:
            ref_wav = torchaudio.functional.resample(torch.tensor(ref_wav), ref_sr, sample_rate).numpy()

        min_len = min(len(ref_wav), len(wav))
        ref_aligned = torch.tensor(ref_wav[:min_len], device=device).unsqueeze(0)
        deg_aligned = wav[:min_len].unsqueeze(0).to(device)

        with torch.no_grad():
            pesq_val = pesq_metric(deg_aligned, ref_aligned).item()
            stoi_val = stoi_metric(deg_aligned, ref_aligned).item()

        metrics["pesq"].append(pesq_val)
        metrics["stoi"].append(stoi_val)

        stem = flac_ref.stem
        out_codec = out_dir / f"{stem}_codec.wav"
        out_ref = out_dir / f"{stem}_ref.wav"
        sf.write(out_codec, wav.cpu().numpy(), sample_rate)
        if not out_ref.exists():
            sf.write(out_ref, ref_wav, sample_rate)

        print(f"{i}: {stem}, PESQ={pesq_val:.3f}, STOI={stoi_val:.3f}, "
              f"latency={latency_ms:.2f} ms")

        if i >= 9:
            break

    if metrics["pesq"]:
        pesq_mean = np.mean(metrics["pesq"])
        stoi_mean = np.mean(metrics["stoi"])
        lat_mean = np.mean(metrics["latency"])
        print("-----------------Aggregate Results----------------")
        print(f"Mean PESQ : {pesq_mean:.3f}")
        print(f"Mean STOI : {stoi_mean:.3f}")
        print(f"Mean latency: {lat_mean:.2f} ms")
        print("--------------------------------------------------")

    mel_bins = 80
    frame_rate = sample_rate / hop_length
    bit_depth = 8
    bitrate_kbps = mel_bins * bit_depth * frame_rate / 1000
    print(f"Estimated effective mel bitrate: {bitrate_kbps:.2f} kbps")

if __name__ == "__main__":
    main()
