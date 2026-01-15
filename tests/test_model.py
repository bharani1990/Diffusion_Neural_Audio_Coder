import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.model import AudioCodec

print("Testing AudioCodec model...")

x = torch.load("data/processed_norm/train-clean-5/19/198/19-198-0000.pt").float()
if x.dim() == 3:
    x = x.unsqueeze(0)

print(f"Input shape: {x.shape}")
B, C, F, T = x.shape

model = AudioCodec(latent_dim=16, hidden_dim=256)
model.eval()

with torch.no_grad():
    print("\n[STAGE 1: COMPRESSION (Encoding)]")
    z_q, vq_loss, idx = model.encode(x)
    print(f"  z_q: {z_q.shape}, vq_loss: {vq_loss:.4f}, indices: {idx.shape}")
    
    print("\n[STAGE 2: DIFFUSION (Decoding)]")
    t = torch.randint(0, 1000, (B,))
    mel_recon = model.decode(z_q, t)
    print(f"  reconstructed mel: {mel_recon.shape}")
    
    mel, vq_loss = model(x, t)
    print(f"  forward output - mel: {mel.shape}, vq_loss: {vq_loss:.4f}")
    
    print("\n[STAGE 3: VOCODER (Mel to Sound)]")
    wave = model.to_waveform(mel)
    print(f"  waveform: {wave.shape}")
    
    print("\nAll stages complete!")
