import torch
import torch.nn as nn
from src.model import DiffusionUNet
from demo.exp_2.vocoder import HiFiGANGenerator


class ScalarQuantizer(nn.Module):
    def __init__(self, num_bits=12):
        super().__init__()
        self.num_bits = num_bits
        self.levels = 2 ** num_bits
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        scaled = x * self.scale
        quantized = torch.round(scaled * (self.levels - 1)) / (self.levels - 1)
        quantized = torch.clamp(quantized, -1.0, 1.0)
        return quantized


class DecoderNetwork(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_channels, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x


class Experiment3Codec(nn.Module):
    def __init__(self, encoder, decoder=None, latent_proj=None):
        super().__init__()
        self.encoder = encoder
        self.quantizer = ScalarQuantizer(num_bits=12)
        self.latent_proj = latent_proj if latent_proj is not None else nn.Conv2d(1, 64, kernel_size=1)
        self.decoder = decoder if decoder is not None else DecoderNetwork(latent_channels=64)
        self.vocoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_vocoder(self):
        self.vocoder = HiFiGANGenerator(num_mels=80).to(self.device).eval()

    def encode(self, mel, t):
        latent = self.encoder(mel, t)
        quantized = self.quantizer(latent)
        return quantized
    
    def decode(self, quantized):
        device = quantized.device
        if self.vocoder is None:
            self.vocoder = HiFiGANGenerator(num_mels=80).to(device).eval()
        latent_proj = self.latent_proj(quantized)
        mel_recon = self.decoder(latent_proj)
        mel_recon = torch.log(torch.clamp(mel_recon, min=1e-5))
        mel_recon = mel_recon.squeeze(1)
        return self.vocoder(mel_recon)
