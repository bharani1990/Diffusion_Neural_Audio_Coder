import sys
import argparse
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from demo.exp_6.config import Config
from demo.exp_6.train_module import CompressionDiffusionModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default='balanced',
                       choices=['quality', 'speed', 'memory', 'balanced'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_manifest', type=str, 
                       default='data/processed/train_manifest.jsonl')
    parser.add_argument('--val_manifest', type=str, 
                       default='data/processed/dev_manifest.jsonl')
    args = parser.parse_args()
    
    config = Config(preset=args.preset)
    
    train_dataset = SpectrogramDataset(args.train_manifest, target_frames=config.frames)
    val_dataset = SpectrogramDataset(args.val_manifest, target_frames=config.frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=config.num_workers,
                            persistent_workers=True,
                            # pin_memory=True,
                            )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=config.num_workers,
                           persistent_workers=True,
                           # pin_memory=True,
                           )
    
    model = CompressionDiffusionModule(
        in_channels=1,
        timesteps=config.timesteps,
        lr=args.lr,
        latent_dim=config.latent_dim,
        compression_weight=0.5,
        diffusion_weight=0.5
    )
    
    precision_map = {
        'fp32': 32,
        'fp16': '16-mixed',
        'bf16': 'bf16',
    }
    precision = precision_map.get(config.precision, '32')
    
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=precision,
        accumulate_grad_batches=config.grad_accum,
        log_every_n_steps=10,
        callbacks=[
            EarlyStopping(
                monitor='val_loss', patience=10, mode='min'
            ),
            ModelCheckpoint(
                dirpath='lightning_logs/exp_6',
                monitor='val_loss',
                mode='min',
                save_top_k=3
            ),
        ],
        default_root_dir='lightning_logs/exp_6',
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()