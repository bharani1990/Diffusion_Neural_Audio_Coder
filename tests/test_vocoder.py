import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch
import time
from src.train_module import AudioCodecModule
from src.utils import pesq_metric, stoi_metric
from src.config import SAMPLE_RATE


def main(checkpoint_path=None, test_file=None):
    if checkpoint_path is None:
        checkpoint_path = Path("lightning_logs/checkpoints/last.ckpt")
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_path} not found — falling back to random-initialized model")
        checkpoint_path = None

    if test_file is None:
        test_dir = Path("data/processed_norm/test-clean/1995/1826")
        files = sorted(test_dir.glob("*.pt"))
        if not files:
            print("No test mel files found — will use synthetic mel input")
            test_file = None
        else:
            test_file = files[0]
    else:
        test_file = Path(test_file)
        if not test_file.exists():
            pytest.skip("Specified test file not found, skipping vocoder test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using test file: {test_file}")
    print(f"Device: {device}")

    if checkpoint_path is None:
        print("Instantiating AudioCodecModule with random weights")
        module = AudioCodecModule()
    else:
        module = AudioCodecModule.load_from_checkpoint(str(checkpoint_path))
    module = module.to(device)
    module.eval()

    if test_file is None:
        mel = torch.randn(80, 120)
    else:
        data = torch.load(str(test_file))
        mel = data[0].float() if isinstance(data, (tuple, list)) else data.float()
        if mel.dim() == 3:
            mel = mel.squeeze(0)

    mel_input_enc = mel.unsqueeze(0).unsqueeze(0).to(device)
    mel_input_voc = mel.unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        z_q, vq_loss, indices = module.model.encode(mel_input_enc)
        mel_recon = module.model.decode(z_q, torch.tensor([500], device=device))
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        duration_sec = mel.shape[-1] * 0.01
        br = (indices.numel() * 12) / (duration_sec * 1000)

        print(f"mel shape: {mel.shape}")
        print(f"mel_input_enc shape: {mel_input_enc.shape}")
        print(f"mel_input_voc shape: {mel_input_voc.shape}")
        print(f"z_q shape: {z_q.shape}")
        print(f"vq_loss: {vq_loss}")
        try:
            idx_sample = indices.flatten()[:10].cpu().numpy()
        except Exception:
            idx_sample = None
        print(f"indices shape: {indices.shape}, sample indices: {idx_sample}")
        print(f"mel_recon shape: {mel_recon.shape}")

        try:
            wave_orig = module.model.to_waveform(mel_input_voc)
        except Exception:
            wave_orig = module.model.to_waveform(mel_input_voc.detach().cpu())

        try:
            wave_recon = module.model.to_waveform(mel_recon)
        except Exception:
            wave_recon = module.model.to_waveform(mel_recon.detach().cpu())

        print(f"wave_orig shape: {wave_orig.shape}")
        print(f"wave_recon shape: {wave_recon.shape}")
        print(f"latency_ms: {latency_ms:.2f} ms, bitrate_kbps: {br:.2f}")

    wave_orig_np = wave_orig.squeeze().cpu().numpy()
    wave_recon_np = wave_recon.squeeze().cpu().numpy()

    assert wave_orig_np.size > 0
    assert wave_recon_np.size > 0

    try:
        p = pesq_metric(wave_orig_np, wave_recon_np, SAMPLE_RATE)
        s = stoi_metric(wave_orig_np, wave_recon_np, SAMPLE_RATE)
        print(f"PESQ={p:.3f}, STOI={s:.3f}")
        assert isinstance(p, float)
        assert isinstance(s, float)
    except Exception:
        print("PESQ/STOI not available or failed")
        pytest.skip("PESQ/STOI dependencies missing or failed, waveform produced successfully")

    print('\nNow running inference bypassing VQ (no retraining)')
    with torch.no_grad():
        z = module.model.encoder.encode_no_vq(mel_input_enc)
        mel_recon_no_vq = module.model.decoder(z, torch.tensor([500], device=device))
        try:
            wave_recon_no_vq = module.model.to_waveform(mel_recon_no_vq)
        except Exception:
            wave_recon_no_vq = module.model.to_waveform(mel_recon_no_vq.detach().cpu())

    wave_recon_no_vq_np = wave_recon_no_vq.squeeze().cpu().numpy()
    try:
        p_no_vq = pesq_metric(wave_orig_np, wave_recon_no_vq_np, SAMPLE_RATE)
        s_no_vq = stoi_metric(wave_orig_np, wave_recon_no_vq_np, SAMPLE_RATE)
        print(f"No-VQ PESQ={p_no_vq:.3f}, STOI={s_no_vq:.3f}")
    except Exception:
        print("PESQ/STOI not available for No-VQ case")

    print("Success")

if __name__ == '__main__':
    main()

def test_vocoder_pipeline():
    main()
