import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CSVLogger
from typing import List
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils_audio import process_root
from src.preprocess import compute_stats, normalize_and_manifest
from src.dataset import SpectrogramDataset
from src.train_module import AudioCodecModule
from src.utils import collate_fn
from src import config as cfg

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["extract", "preprocess", "train", "eval"], required=True)
    p.add_argument("--root_in", required=False)
    p.add_argument("--root_out", required=False)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--stats_path", default=None)
    p.add_argument("--manifest_path", default=None)
    p.add_argument("--train_manifest", default="data/processed/train_manifest.jsonl")
    p.add_argument("--dev_manifest", default="data/processed/dev_manifest.jsonl")
    p.add_argument("--batch_size", type=int, default=cfg.TRAIN_BATCH_SIZE)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--max_epochs", type=int, default=cfg.TRAIN_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--latent_dim", type=int, default=cfg.LATENT_DIM)
    p.add_argument("--hidden_dim", type=int, default=cfg.HIDDEN_DIM)
    p.add_argument("--checkpoint_path", default="lightning_logs/checkpoints/last.ckpt")
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
        device = torch.device('cuda' if torch.cuda.is_available() and cfg.USE_GPU else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available() and cfg.USE_GPU:
            torch.set_float32_matmul_precision('medium')
        
        train_ds = SpectrogramDataset(args.train_manifest, target_frames=args.frames)
        dev_ds = SpectrogramDataset(args.dev_manifest, target_frames=args.frames)
        
        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(dev_ds)}")
        
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.USE_GPU,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
        dev_dl = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.USE_GPU,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
        
        model = AudioCodecModule(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            vq_weight=cfg.VQ_WEIGHT,
            perc_weight=cfg.PERCEPTUAL_WEIGHT,
        )
        
        checkpoint_dir = cfg.CHECKPOINT_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir / "checkpoints",
            filename="codec-{epoch:02d}-{val_loss:.4f}",
            monitor=cfg.MONITOR_METRIC,
            mode=cfg.MONITOR_MODE,
            save_top_k=cfg.SAVE_TOP_K,
            save_last=True,
        )
        
        callbacks: List[Callback] = [checkpoint_callback]
        if cfg.ENABLE_EARLY_STOPPING:
            early_cb = EarlyStopping(
                monitor=cfg.EARLY_STOPPING_MONITOR,
                mode=cfg.EARLY_STOPPING_MODE,
                patience=cfg.EARLY_STOPPING_PATIENCE,
                min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
            )
            callbacks.append(early_cb)
        
        accelerator = 'gpu' if torch.cuda.is_available() and cfg.USE_GPU else 'cpu'
        devices = 1 if accelerator == 'gpu' else 'auto'
        
        trainer = L.Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=CSVLogger(save_dir=str(cfg.CHECKPOINT_DIR), name=""),
            precision=cfg.PRECISION,
            gradient_clip_val=cfg.GRADIENT_CLIP_VAL,
            gradient_clip_algorithm=cfg.GRADIENT_CLIP_ALGORITHM,
            enable_progress_bar=True,
            log_every_n_steps=cfg.LOG_INTERVAL,
        )
        
        print("\n" + "="*70)
        print("Training: AudioCodec")
        print("="*70)
        trainer.fit(model, train_dl, dev_dl)
        
        print(f"\nCheckpoints saved to {checkpoint_dir}")

    elif args.stage == "eval":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
        
        model = AudioCodecModule.load_from_checkpoint(
            checkpoint_path,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            vq_weight=cfg.VQ_WEIGHT,
            perc_weight=cfg.PERCEPTUAL_WEIGHT,
        )
        model = model.to(device).eval()
        
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        print(f"Using device: {device}")
        print("Ready for evaluation!")
