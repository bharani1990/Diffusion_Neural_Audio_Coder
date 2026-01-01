import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pathlib import Path
import json
from demo.exp_5.model import CompressionCodec


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_recon):
        loss = F.l1_loss(x, x_recon)
        if x.size(1) > 1:
            for scale in [1, 2, 4]:
                if scale > 1:
                    x_down = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                    x_recon_down = F.avg_pool2d(x_recon, kernel_size=scale, stride=scale)
                else:
                    x_down = x
                    x_recon_down = x_recon
                loss = loss + 0.1 * F.l1_loss(x_down, x_recon_down)
        
        return loss


class CompressionLightningModule(L.LightningModule):
    def __init__(self, latent_dim=16, hidden_dim=256, lr=1e-3, vq_weight=0.25, perceptual_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        self.codec = CompressionCodec(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.lr = lr
        self.vq_weight = vq_weight
        self.perceptual_weight = perceptual_weight
        self.train_losses = []
        self.val_losses = []
        self.recon_loss_fn = nn.L1Loss()
        self.perceptual_loss_fn = PerceptualLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x = batch
        mel_recon, vq_loss = self.codec(x)
        
        recon_loss = self.recon_loss_fn(mel_recon, x)
        perceptual_loss = self.perceptual_loss_fn(mel_recon, x)
        total_loss = recon_loss + self.perceptual_weight * perceptual_loss + self.vq_weight * vq_loss
        
        self.train_losses.append(total_loss.detach().cpu().item())
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_recon_loss", recon_loss, prog_bar=True)
        self.log("train_perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("train_vq_loss", vq_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        mel_recon, vq_loss = self.codec(x)
        
        recon_loss = self.recon_loss_fn(mel_recon, x)
        perceptual_loss = self.perceptual_loss_fn(mel_recon, x)
        total_loss = recon_loss + self.perceptual_weight * perceptual_loss + self.vq_weight * vq_loss
        
        self.val_losses.append(total_loss.detach().cpu().item())
        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("val_vq_loss", vq_loss, prog_bar=True)
        
        return total_loss

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        print(f"epoch {epoch}")

    def on_train_end(self):
        root = Path.cwd()
        plots_dir = root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        with open(plots_dir / "loss_curves_exp5.json", "w", encoding="utf-8") as f:
            json.dump(out, f)