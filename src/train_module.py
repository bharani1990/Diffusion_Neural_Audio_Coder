import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pathlib import Path
import json
from src.model import AudioCodec


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


class AudioCodecModule(L.LightningModule):
    def __init__(self, latent_dim=16, hidden_dim=256, lr=1e-3, vq_weight=0.25, perc_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AudioCodec(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.lr = lr
        self.vq_weight = vq_weight
        self.perc_weight = perc_weight
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.recon_fn = nn.L1Loss()
        self.perc_fn = PerceptualLoss()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        return [opt], [sched]

    def training_step(self, batch, batch_idx):
        x = batch
        t = torch.randint(0, 1000, (x.size(0),), device=x.device)
        mel, vq_loss = self.model(x, t)
        
        recon_loss = self.recon_fn(mel, x)
        perc_loss = self.perc_fn(mel, x)
        total_loss = recon_loss + self.perc_weight * perc_loss + self.vq_weight * vq_loss
        
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        t = torch.randint(0, 1000, (x.size(0),), device=x.device)
        mel, vq_loss = self.model(x, t)
        
        recon_loss = self.recon_fn(mel, x)
        perc_loss = self.perc_fn(mel, x)
        total_loss = recon_loss + self.perc_weight * perc_loss + self.vq_weight * vq_loss
        
        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss", 0)
        if train_loss:
            self.train_epoch_losses.append(float(train_loss))

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss", 0)
        if val_loss:
            self.val_epoch_losses.append(float(val_loss))

    def on_train_end(self):
        Path("plots").mkdir(exist_ok=True)
        with open("plots/loss_curves.json", "w") as f:
            json.dump({"train": self.train_epoch_losses, "val": self.val_epoch_losses}, f)
            print(f"Loss curves saved: {len(self.train_epoch_losses)} epochs")
