import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from src.train_module import AudioCodecModule
from src.utils import collate_fn


def train(epochs=50, batch_size=4, lr=1e-3, latent_dim=16, hidden_dim=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = SpectrogramDataset("data/processed/train_manifest.jsonl", target_frames=120)
    val_dataset = SpectrogramDataset("data/processed/dev_manifest.jsonl", target_frames=120)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    model = AudioCodecModule(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        vq_weight=0.25,
        perc_weight=0.5
    )
    
    checkpoint_dir = Path("lightning_logs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir / "checkpoints",
        filename="codec-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=device.type,
        devices=1,
        callbacks=[checkpoint_callback],
        logger=CSVLogger(save_dir="lightning_logs", name=""),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    print("\n" + "="*70)
    print("Training: AudioCodec")
    print("="*70)
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nCheckpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train audio codec")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
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
