import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
import json
from src.model import DiffusionUNet


class DiffusionLightningModule(L.LightningModule):
    def __init__(self, in_channels=1, n_mels=80, frames=120, timesteps=1000, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = DiffusionUNet(in_channels=in_channels)
        self.timesteps = timesteps
        self.lr = lr
        self.train_losses = []
        self.val_losses = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def _sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def _add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        t = t.float().view(-1, 1, 1, 1)
        beta_start, beta_end = 1e-4, 0.02
        beta = beta_start + (beta_end - beta_start) * t / (self.timesteps - 1)
        alpha = 1.0 - beta
        alpha_bar = alpha.cumprod(dim=0)
        alpha_bar = alpha_bar[-1]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise, noise

    def training_step(self, batch, batch_idx):
        x0 = batch
        b = x0.size(0)
        t = self._sample_timesteps(b, x0.device)
        noise = torch.randn_like(x0)
        xt, noise = self._add_noise(x0, t, noise)
        pred = self.model(xt, t)
        loss = nn.functional.mse_loss(pred, noise)
        self.train_losses.append(loss.detach().cpu().item())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0 = batch
        b = x0.size(0)
        t = self._sample_timesteps(b, x0.device)
        noise = torch.randn_like(x0)
        xt, noise = self._add_noise(x0, t, noise)
        pred = self.model(xt, t)
        loss = nn.functional.mse_loss(pred, noise)
        self.val_losses.append(loss.detach().cpu().item())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_end(self):
        root = Path.cwd()
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        with open(plots_dir / "loss_curves.json", "w", encoding="utf-8") as f:
            json.dump(out, f)
