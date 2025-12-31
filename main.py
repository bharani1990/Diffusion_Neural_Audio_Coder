import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from src.utils_audio import process_root
from src.preprocess import compute_stats, normalize_and_manifest
from src.dataset import SpectrogramDataset
from src.train_module import DiffusionLightningModule
from demo.exp_2.model import Experiment2Codec
from src.model import DiffusionUNet
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    verbose=True,
)

checkpoint_cb = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="best-{epoch:02d}-{val_loss:.4f}",
)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["extract", "preprocess", "train", "eval"], required=True)
    p.add_argument("--model", choices=["old", "new"], default="old")
    p.add_argument("--root_in", required=False)
    p.add_argument("--root_out", required=False)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--stats_path", default=None)
    p.add_argument("--manifest_path", default=None)
    p.add_argument("--train_manifest", default="data/processed/train_manifest.jsonl")
    p.add_argument("--dev_manifest", default="data/processed/dev_manifest.jsonl")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint_path", default="lightning_logs/version_0/checkpoints/last.ckpt")
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

    elif args.stage == "train":
        train_ds = SpectrogramDataset(args.train_manifest, target_frames=args.frames)
        dev_ds = SpectrogramDataset(args.dev_manifest, target_frames=args.frames)
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )
        dev_dl = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
        model = DiffusionLightningModule(
            in_channels=1, n_mels=80, frames=args.frames, lr=args.lr
        )
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=args.max_epochs,
            callbacks=[early_stop, checkpoint_cb],
        )
        trainer.fit(model, train_dl, dev_dl)

    elif args.stage == "eval":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.model == "new":
            encoder = DiffusionUNet(in_channels=1)
            if Path(args.checkpoint_path).exists():
                checkpoint = torch.load(args.checkpoint_path, map_location=device)
                encoder.load_state_dict(checkpoint['state_dict'])
            model = Experiment2Codec(encoder=encoder)
            model.load_vocoder()
            print("Loaded NEW model (Diffusion + HiFi-GAN)")
        else:
            model = DiffusionLightningModule.load_from_checkpoint(args.checkpoint_path)
            print("Loaded OLD model (Diffusion + Griffin-Lim)")
        
        model = model.to(device).eval()
        print(f"Model: {args.model}, Checkpoint: {args.checkpoint_path}")
        print("Ready for evaluation!")
