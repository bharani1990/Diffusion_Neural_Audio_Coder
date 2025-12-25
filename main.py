import argparse
from src.utils_audio import process_root
from src.preprocess import compute_stats, normalize_and_manifest
from pathlib import Path
import torch

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["extract", "preprocess"], required=True)
    p.add_argument("--root_in", required=True)
    p.add_argument("--root_out", required=True)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--stats_path", default="data/processed/stats.pt")
    p.add_argument("--manifest_path", default=None)
    args = p.parse_args()

    if args.stage == "extract":
        process_root(args.root_in, args.root_out, sample_rate=args.sample_rate)
    elif args.stage == "preprocess":
        stats_path = Path(args.stats_path)
        if stats_path.exists():
            stats = torch.load(stats_path)
        else:
            stats = compute_stats(args.root_in, stats_path)
        manifest_path = args.manifest_path
        if manifest_path is None:
            manifest_path = str(Path(args.root_out) / "manifest.jsonl")
        normalize_and_manifest(args.root_in, args.root_out, stats, manifest_path)
