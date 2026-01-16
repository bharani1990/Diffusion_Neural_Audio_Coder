import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.model import AudioCodec
def main():
    print("Testing AudioCodec model...")

    x = torch.load("data/processed_norm/train-clean-5/19/198/19-198-0000.pt").float()
    if x.dim() == 3:
        x = x.unsqueeze(0)

    print(f"Input shape: {x.shape}")
    B, C, F, T = x.shape

    model = AudioCodec(latent_dim=16, hidden_dim=256)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    x = x.to(device)

    with torch.no_grad():
        print("\n[STAGE 1: COMPRESSION (Encoding)]")
        z_q, vq_loss, idx = model.encode(x)
        z_q_cpu = z_q.detach().cpu()
        vq_val = float(vq_loss.detach().cpu()) if isinstance(vq_loss, torch.Tensor) else float(vq_loss)
        print(f"  z_q: {z_q_cpu.shape}, vq_loss: {vq_val:.4f}, indices: {idx.shape}")
        
        print("\n[STAGE 2: DIFFUSION (Decoding)]")
        t = torch.randint(0, 1000, (B,), device=device)
        mel_recon = model.decode(z_q, t)
        print(f"  reconstructed mel: {mel_recon.detach().cpu().shape}")
        
        mel, vq_loss = model(x, t)
        print(f"  forward output - mel: {mel.detach().cpu().shape}, vq_loss: {float(vq_loss.detach().cpu()):.4f}")
        
        print("\n[STAGE 3: VOCODER (Mel to Sound)]")
        try:
            wave = model.to_waveform(mel)
            wave_shape = wave.detach().cpu().shape
        except Exception:
            wave_shape = model.to_waveform(mel.detach().cpu()).shape
        print(f"  waveform: {wave_shape}")
        
        print("\nAll stages complete!")


if __name__ == '__main__':
    main()

def test_model_forward():
    main()
