import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.model import DiffusionUNet

x = torch.load("data/processed_norm/train-clean-5/19/198/19-198-0000.pt").float()
if x.dim() == 3:
    x = x.unsqueeze(0)

B, C, F, T = x.shape
t = torch.randint(0, 1000, (B,))

model = DiffusionUNet(in_channels=C)
y = model(x, t)

print("input:", x.shape, "output:", y.shape)
