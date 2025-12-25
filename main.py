import argparse
from src.utils_audio import process_root

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root_in", required=True)
    p.add_argument("--root_out", required=True)
    p.add_argument("--sample_rate", type=int, default=16000)
    args = p.parse_args()
    process_root(args.root_in, args.root_out, sample_rate=args.sample_rate)
