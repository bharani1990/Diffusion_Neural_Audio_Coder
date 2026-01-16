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

def main():
    print(f"Train samples: {len(ds_train)}")
    print(f"Val samples: {len(ds_val)}")
    print(f"Test samples: {len(ds_test)}")

    dl_train = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, collate_fn=collate_fn)
    dl_test = DataLoader(ds_test, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print("\nTrain batch:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for batch in dl_train:
        batch = batch.to(device)
        b_cpu = batch.detach().cpu()
        print(f"  Shape: {b_cpu.shape}")
        print(f"  Min: {b_cpu.min():.4f}, Max: {b_cpu.max():.4f}, Mean: {b_cpu.mean():.4f}")
        break

    print("\nVal batch:")
    for batch in dl_val:
        batch = batch.to(device)
        b_cpu = batch.detach().cpu()
        print(f"  Shape: {b_cpu.shape}")
        print(f"  Min: {b_cpu.min():.4f}, Max: {b_cpu.max():.4f}, Mean: {b_cpu.mean():.4f}")
        break

    print("\nTest batch:")
    for batch in dl_test:
        batch = batch.to(device)
        b_cpu = batch.detach().cpu()
        print(f"  Shape: {b_cpu.shape}")
        print(f"  Min: {b_cpu.min():.4f}, Max: {b_cpu.max():.4f}, Mean: {b_cpu.mean():.4f}")
        break

    print("Train & val & test datasets loading works!")


if __name__ == '__main__':
    main()

def test_dataset_loading():
    main()
