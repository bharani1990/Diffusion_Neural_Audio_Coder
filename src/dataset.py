from pathlib import Path
import json
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, manifest_path, target_frames=120):
        self.manifest_path = Path(manifest_path)
        self.target_frames = target_frames
        self.entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def _crop_or_pad(self, x):
        c, f, t = x.shape
        if t == self.target_frames:
            return x
        if t > self.target_frames:
            start = (t - self.target_frames) // 2
            end = start + self.target_frames
            return x[:, :, start:end]
        pad_total = self.target_frames - t
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad = (pad_left, pad_right)
        return torch.nn.functional.pad(x, pad=(pad[0], pad[1]))

    def __getitem__(self, idx):
        e = self.entries[idx]
        x = torch.load(e["path"]).float()
        if x.dim() == 3:
            pass
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"unexpected shape {x.shape}")
        x = self._crop_or_pad(x)
        return x
