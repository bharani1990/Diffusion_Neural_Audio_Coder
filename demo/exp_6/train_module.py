import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pathlib import Path
import json
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.exp_6.model import DiffusionModel, CompressionModel


class CompressionDiffusionModule(L.LightningModule):
    def __init__(self, in_channels=1, timesteps=1000, lr=1e-4, latent_dim=16,
                 compression_weight=0.5, diffusion_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.diffusion_model = DiffusionModel(in_channels=in_channels)
        self.compression_model = CompressionModel(in_channels=in_channels, latent_dim=latent_dim)
        
        self.timesteps = timesteps
        self.lr = lr
        self.compression_weight = compression_weight
        self.diffusion_weight = diffusion_weight
        
        self.train_losses = []
        self.val_losses = []

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def _sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def _add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        t_norm = t.float().view(-1, 1, 1, 1) / (self.timesteps - 1)
        beta_start, beta_end = 1e-4, 0.02
        beta = beta_start + (beta_end - beta_start) * t_norm
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar = alpha_bar[-1]
        noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
        return noisy, noise

    def training_step(self, batch, batch_idx):
        x0 = batch
        b = x0.size(0)
        
        x_recon = self.compression_model(x0)
        comp_loss = 0.7 * F.mse_loss(x_recon, x0) + 0.3 * F.l1_loss(x_recon, x0)
        
        t = self._sample_timesteps(b, x0.device)
        noise = torch.randn_like(x0)
        xt, noise_target = self._add_noise(x0, t, noise)
        noise_pred = self.diffusion_model(xt, t)
        diff_loss = F.mse_loss(noise_pred, noise_target)
        
        loss = self.compression_weight * comp_loss + self.diffusion_weight * diff_loss
        
        self.train_losses.append(loss.detach().cpu().item())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch
        b = x0.size(0)
        
        x_recon = self.compression_model(x0)
        comp_loss = 0.7 * F.mse_loss(x_recon, x0) + 0.3 * F.l1_loss(x_recon, x0)
        
        t = self._sample_timesteps(b, x0.device)
        noise = torch.randn_like(x0)
        xt, noise_target = self._add_noise(x0, t, noise)
        noise_pred = self.diffusion_model(xt, t)
        diff_loss = F.mse_loss(noise_pred, noise_target)
        
        loss = self.compression_weight * comp_loss + self.diffusion_weight * diff_loss
        
        self.val_losses.append(loss.detach().cpu().item())
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_end(self):
        root = Path.cwd()
        log_dir = root / "lightning_logs" / "exp_6"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "loss_curves.json", "w") as f:
            json.dump({"train": self.train_losses, "val": self.val_losses}, f)
