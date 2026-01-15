import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from src.dataset import SpectrogramDataset
from src.utils import collate_fn

ds_train = SpectrogramDataset("data/processed/train_manifest.jsonl", target_frames=120)
ds_val = SpectrogramDataset("data/processed/dev_manifest.jsonl", target_frames=120)
ds_test = SpectrogramDataset("data/processed/test_manifest.jsonl", target_frames=120)

print(f"Train samples: {len(ds_train)}")
print(f"Val samples: {len(ds_val)}")
print(f"Test samples: {len(ds_test)}")

dl_train = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, collate_fn=collate_fn)
dl_test = DataLoader(ds_test, batch_size=4, shuffle=False, collate_fn=collate_fn)

print("\nTrain batch:")
for batch in dl_train:
    print(f"  Shape: {batch.shape}")
    print(f"  Min: {batch.min():.4f}, Max: {batch.max():.4f}, Mean: {batch.mean():.4f}")
    break

print("\nVal batch:")
for batch in dl_val:
    print(f"  Shape: {batch.shape}")
    print(f"  Min: {batch.min():.4f}, Max: {batch.max():.4f}, Mean: {batch.mean():.4f}")
    break

print("\nTest batch:")
for batch in dl_test:
    print(f"  Shape: {batch.shape}")
    print(f"  Min: {batch.min():.4f}, Max: {batch.max():.4f}, Mean: {batch.mean():.4f}")
    break

print("Train & val & test datasets loading works!")
