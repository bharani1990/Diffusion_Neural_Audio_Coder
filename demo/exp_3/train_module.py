import torch
import torch.nn as nn
import lightning as L
from pathlib import Path
import json
from src.model import DiffusionUNet
from demo.exp_3.model import Experiment3Codec, DecoderNetwork


class DecoderLightningModule(L.LightningModule):
    def __init__(self, encoder_checkpoint, latent_channels=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = DiffusionUNet(in_channels=1)
        checkpoint = torch.load(encoder_checkpoint, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.encoder.load_state_dict(state_dict, strict=False)
        self.encoder = self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.latent_proj = nn.Conv2d(1, latent_channels, kernel_size=1)
        
        self.codec = Experiment3Codec(encoder=self.encoder, decoder=DecoderNetwork(latent_channels=latent_channels))
        self.codec.load_vocoder()
        
        self.quantizer = self.codec.quantizer
        self.decoder = self.codec.decoder
        
        self.lr = lr
        self.train_losses = []
        self.val_losses = []
        self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.latent_proj.parameters()) + list(self.decoder.parameters()), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            latent = self.encoder(x, t)
            quantized = self.quantizer(latent)
        
        latent_proj = self.latent_proj(quantized)
        mel_recon = self.decoder(latent_proj)
        loss = self.loss_fn(mel_recon, x)
        
        self.train_losses.append(loss.detach().cpu().item())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            latent = self.encoder(x, t)
            quantized = self.quantizer(latent)
            latent_proj = self.latent_proj(quantized)
            mel_recon = self.decoder(latent_proj)
            loss = self.loss_fn(mel_recon, x)
        
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
        with open(plots_dir / "loss_curves_exp3.json", "w", encoding="utf-8") as f:
            json.dump(out, f)