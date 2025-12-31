import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.model import DiffusionUNet
from demo.exp_2.model import Experiment2Codec

x = torch.load("data/processed_norm/train-clean-5/19/198/19-198-0000.pt").float()
if x.dim() == 3:
    x = x.unsqueeze(0)

B, C, F, T = x.shape
t = torch.randint(0, 1000, (B,))

encoder = DiffusionUNet(in_channels=C)
model = Experiment2Codec(encoder=encoder)

print("Input shape:", x.shape)
latent = model.encode(x, t)
print("Latent shape:", latent.shape)

recon = model.decode(latent)
print("Recon shape:", recon.shape)
