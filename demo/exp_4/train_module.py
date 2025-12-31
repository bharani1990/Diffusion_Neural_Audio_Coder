import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
import json
from demo.exp_4.model import CompressionCodec


class CompressionLightningModule(L.LightningModule):
    def __init__(self, latent_dim=4, hidden_dim=64, lr=1e-4, vq_weight=0.25):
        super().__init__()
        self.save_hyperparameters()
        
        self.codec = CompressionCodec(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.lr = lr
        self.vq_weight = vq_weight
        self.train_losses = []
        self.val_losses = []
        self.recon_loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch
        mel_recon, vq_loss = self.codec(x)
        
        recon_loss = self.recon_loss_fn(mel_recon, x)
        total_loss = recon_loss + self.vq_weight * vq_loss
        
        self.train_losses.append(total_loss.detach().cpu().item())
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_vq_loss", vq_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        mel_recon, vq_loss = self.codec(x)
        
        recon_loss = self.recon_loss_fn(mel_recon, x)
        total_loss = recon_loss + self.vq_weight * vq_loss
        
        self.val_losses.append(total_loss.detach().cpu().item())
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_vq_loss", vq_loss, prog_bar=True)
        
        return total_loss

    def on_train_end(self):
        root = Path.cwd()
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        with open(plots_dir / "loss_curves_exp4.json", "w", encoding="utf-8") as f:
            json.dump(out, f)