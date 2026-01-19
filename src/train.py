import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import CSVLogger
from typing import List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from src.train_module import AudioCodecModule
from src.utils import collate_fn
from src import config as cfg


def train(epochs=cfg.TRAIN_EPOCHS, batch_size=cfg.TRAIN_BATCH_SIZE, lr=cfg.LR, latent_dim=cfg.LATENT_DIM, hidden_dim=cfg.HIDDEN_DIM):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.USE_GPU else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available() and cfg.USE_GPU:
        torch.set_float32_matmul_precision('medium')
    
    train_dataset = SpectrogramDataset("data/processed/train_manifest.jsonl", target_frames=120)
    val_dataset = SpectrogramDataset("data/processed/dev_manifest.jsonl", target_frames=120)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.USE_GPU,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.USE_GPU,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    
    model = AudioCodecModule(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
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
    if getattr(cfg, 'ENABLE_EARLY_STOPPING', False):
        early_cb = EarlyStopping(
            monitor=getattr(cfg, 'EARLY_STOPPING_MONITOR', cfg.MONITOR_METRIC),
            mode=getattr(cfg, 'EARLY_STOPPING_MODE', cfg.MONITOR_MODE),
            patience=getattr(cfg, 'EARLY_STOPPING_PATIENCE', 5),
            min_delta=getattr(cfg, 'EARLY_STOPPING_MIN_DELTA', 1e-4),
        )
        callbacks.append(early_cb)

    accelerator = 'gpu' if torch.cuda.is_available() and cfg.USE_GPU else 'cpu'
    devices = 1 if accelerator == 'gpu' else 'auto'

    trainer = L.Trainer(
        max_epochs=epochs,
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
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nCheckpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train audio codec")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
