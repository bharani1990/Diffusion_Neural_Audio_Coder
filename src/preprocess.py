from pathlib import Path
import torch
import json


def list_specs(root):
    root = Path(root)
    paths = []
    for p in root.rglob("*.pt"):
        if "stats" in p.name:
            continue
        paths.append(p)
    return sorted(paths)


def compute_stats(root_in, stats_path):
    paths = list_specs(root_in)
    mean = 0.0
    sq_mean = 0.0
    count = 0
    for p in paths:
        x = torch.load(p)
        x = x.float()
        c, f, t = x.shape
        n = c * f * t
        x_flat = x.reshape(-1)
        mean += x_flat.sum()
        sq_mean += (x_flat * x_flat).sum()
        count += n
    mean = mean / count
    variance = sq_mean / count - mean * mean
    std = torch.sqrt(torch.tensor(variance))
    stats = {"mean": float(mean), "std": std.item()}
    torch.save(stats, stats_path)
    return stats


def normalize_and_manifest(root_in, root_out, stats, manifest_path):
    root_in, root_out = Path(root_in), Path(root_out)
    paths = list_specs(root_in)
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])
    entries = []
    for p in paths:
        x = torch.load(p).float()
        x = (x - mean) / std
        rel = p.relative_to(root_in)
        out_path = root_out / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(x, out_path)
        c, f, t = x.shape
        entry = {
            "path": str(out_path.as_posix()),
            "frames": int(t),
            "channels": int(c),
            "n_mels": int(f),
        }
        entries.append(entry)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
