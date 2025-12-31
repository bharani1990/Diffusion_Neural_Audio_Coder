import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from demo.exp_4.train_module import CompressionLightningModule


def train(epochs=100, batch_size=16, lr=1e-4, latent_dim=4, hidden_dim=64):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = SpectrogramDataset("data/processed/train_manifest.jsonl", target_frames=120)
    val_dataset = SpectrogramDataset("data/processed/dev_manifest.jsonl", target_frames=120)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=11,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=11,
                            persistent_workers=True)
    
    model = CompressionLightningModule(latent_dim=latent_dim, hidden_dim=hidden_dim, lr=lr)
    
    checkpoint_dir = Path("lightning_logs/exp_4")
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
        logger=CSVLogger(save_dir="lightning_logs/exp_4", name=""),
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, 
          latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
