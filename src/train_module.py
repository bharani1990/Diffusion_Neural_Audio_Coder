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

    def forward(self, target, recon):
        if target.dim() == 4 and recon.dim() == 4 and target.size(1) != recon.size(1):
            if target.size(1) > recon.size(1):
                target = target.mean(dim=1, keepdim=True)
            else:
                recon = recon.mean(dim=1, keepdim=True)
        loss = F.l1_loss(target, recon)
        if target.size(1) > 1:
            for scale in [1, 2, 4]:
                if scale > 1:
                    t_down = F.avg_pool2d(target, kernel_size=scale, stride=scale)
                    r_down = F.avg_pool2d(recon, kernel_size=scale, stride=scale)
                else:
                    t_down = target
                    r_down = recon
                loss = loss + 0.1 * F.l1_loss(t_down, r_down)
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

        if mel.dim() == 4 and x.dim() == 4 and mel.size(1) != x.size(1):
            mel_for_loss = mel.mean(dim=1, keepdim=True)
        else:
            mel_for_loss = mel

        recon_loss = self.recon_fn(mel_for_loss, x)
        perc_loss = self.perc_fn(x, mel_for_loss)
        total_loss = recon_loss + self.perc_weight * perc_loss + self.vq_weight * vq_loss

        with torch.no_grad():
            mel_cpu = mel_for_loss.detach().cpu()
            x_cpu = x.detach().cpu()
            vq_cpu = vq_loss.detach().cpu() if isinstance(vq_loss, torch.Tensor) else torch.tensor(vq_loss)
            recon_cpu = self.recon_fn(mel_cpu, x_cpu)
            perc_cpu = self.perc_fn(mel_cpu, x_cpu)
            total_cpu = recon_cpu + self.perc_weight * perc_cpu + self.vq_weight * vq_cpu

        self.log("train_loss", float(total_cpu), prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        t = torch.randint(0, 1000, (x.size(0),), device=x.device)

        mel, vq_loss = self.model(x, t)

        if mel.dim() == 4 and x.dim() == 4 and mel.size(1) != x.size(1):
            mel_for_loss = mel.mean(dim=1, keepdim=True)
        else:
            mel_for_loss = mel

        recon_loss = self.recon_fn(mel_for_loss, x)
        perc_loss = self.perc_fn(x, mel_for_loss)
        total_loss = recon_loss + self.perc_weight * perc_loss + self.vq_weight * vq_loss

        with torch.no_grad():
            mel_cpu = mel_for_loss.detach().cpu()
            x_cpu = x.detach().cpu()
            vq_cpu = vq_loss.detach().cpu() if isinstance(vq_loss, torch.Tensor) else torch.tensor(vq_loss)
            recon_cpu = self.recon_fn(mel_cpu, x_cpu)
            perc_cpu = self.perc_fn(mel_cpu, x_cpu)
            total_cpu = recon_cpu + self.perc_weight * perc_cpu + self.vq_weight * vq_cpu

        self.log("val_loss", float(total_cpu), prog_bar=True)
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
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            if len(self.train_epoch_losses) > 0:
                plt.plot(self.train_epoch_losses, label='train')
            if len(self.val_epoch_losses) > 0:
                plt.plot(self.val_epoch_losses, label='val')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path('plots') / 'loss_curve.png')
            plt.close()
            print('Loss curve image saved: plots/loss_curve.png')
        except Exception:
            pass
