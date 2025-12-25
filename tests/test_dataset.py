import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader
from src.dataset import SpectrogramDataset

ds = SpectrogramDataset("data/processed/dev_manifest.jsonl", target_frames=120)
dl = DataLoader(ds, batch_size=4, shuffle=True)

for batch in dl:
    print(batch.shape)
    break
