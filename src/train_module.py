import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pathlib import Path
import json
from src.model import AudioCodec
from src import config as cfg
from src.utils import pesq_metric, stoi_metric


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
    def __init__(self, latent_dim=64, hidden_dim=256, lr=1e-3, vq_weight=0.25, perc_weight=0.5, num_embeddings=8192, num_res_blocks=4):
        super().__init__()
        self.save_hyperparameters()
        self.model = AudioCodec(
            latent_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            num_embeddings=num_embeddings, 
            num_res_blocks=num_res_blocks
        )
        self.lr = lr
        self.vq_weight = vq_weight
        self.perc_weight = perc_weight
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.recon_fn = nn.L1Loss()
        self.perc_fn = PerceptualLoss()

    def configure_optimizers(self):
        if getattr(cfg, 'OPTIMIZER', 'adam') == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))

        sched_type = getattr(cfg, 'SCHEDULER', 'step')
        if sched_type == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.TRAIN_EPOCHS))
        else:
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=getattr(cfg, 'SCHEDULER_STEP_SIZE', 10), gamma=getattr(cfg, 'SCHEDULER_GAMMA', 0.5))
        return [opt], [sched]

    def training_step(self, batch, batch_idx):
        x = batch
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(f"Input contains NaN/Inf at batch {batch_idx}")
        t = torch.randint(0, 1000, (x.size(0),), device=x.device)

        mel, vq_loss = self.model(x, t)

        if mel.dim() == 4 and x.dim() == 4 and mel.size(1) != x.size(1):
            mel_for_loss = mel.mean(dim=1, keepdim=True)
        else:
            mel_for_loss = mel

        recon_loss = self.recon_fn(mel_for_loss, x)
        perc_loss = self.perc_fn(x, mel_for_loss)
        total_loss = recon_loss + self.perc_weight * perc_loss + self.vq_weight * vq_loss

        try:
            self.log('recon_loss_step', recon_loss, on_step=True, on_epoch=False, prog_bar=False)
            self.log('perc_loss_step', perc_loss, on_step=True, on_epoch=False, prog_bar=False)
            if isinstance(vq_loss, torch.Tensor):
                self.log('vq_loss_step', vq_loss, on_step=True, on_epoch=False, prog_bar=False)
            else:
                self.log('vq_loss_step', float(vq_loss), on_step=True, on_epoch=False, prog_bar=False)
        except Exception:
            pass

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            msg = {
                'batch_idx': batch_idx,
                'recon_loss': float(recon_loss.detach().cpu()) if torch.isfinite(recon_loss).all() else None,
                'perc_loss': float(perc_loss.detach().cpu()) if torch.isfinite(perc_loss).all() else None,
                'vq_loss': float(vq_loss.detach().cpu()) if isinstance(vq_loss, torch.Tensor) and torch.isfinite(vq_loss).all() else None,
            }
            raise RuntimeError(f"NaN/Inf detected in loss at batch {batch_idx}: {msg}")

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

        if getattr(cfg, 'VALIDATION_WAVEFORM_METRICS', False) and batch_idx == 0:
            try:
                if mel_for_loss.dim() == 4:
                    mel_voc = mel_for_loss.squeeze(1).detach().cpu()
                else:
                    mel_voc = mel_for_loss.detach().cpu()

                if x.dim() == 4:
                    x_voc = x.squeeze(1).detach().cpu()
                else:
                    x_voc = x.detach().cpu()

                wave_ref = self.model.to_waveform(x_voc)
                wave_rec = self.model.to_waveform(mel_voc)

                wave_ref_np = wave_ref.squeeze().cpu().numpy()
                wave_rec_np = wave_rec.squeeze().cpu().numpy()

                p = pesq_metric(wave_ref_np, wave_rec_np, cfg.SAMPLE_RATE)
                s = stoi_metric(wave_ref_np, wave_rec_np, cfg.SAMPLE_RATE)
                self.log('val_pesq', float(p))
                self.log('val_stoi', float(s))
            except Exception:
                pass

        return total_loss

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get("train_loss", 0)
        if train_loss:
            self.train_epoch_losses.append(float(train_loss))

    def on_train_epoch_start(self):
        try:
            total = int(self.trainer.max_epochs) if self.trainer.max_epochs is not None else None
        except Exception:
            total = None
        one_based = int(self.current_epoch) + 1
        if total:
            print(f"Starting epoch {one_based}/{total}")
        else:
            print(f"Starting epoch {one_based}")
        try:
            self.log('epoch', one_based)
        except Exception:
            pass

    def on_after_backward(self):
        try:
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += (param_norm.item() ** 2)
            total_norm = total_norm ** 0.5
            self.log('grad_norm', total_norm, on_step=True, on_epoch=False)
            if total_norm > 1e3:
                print(f"Large grad norm detected: {total_norm:.3f}")
        except Exception:
            pass

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