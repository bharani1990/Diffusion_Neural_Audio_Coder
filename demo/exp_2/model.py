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

class Experiment2Codec(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = ScalarQuantizer(num_bits=12)
        self.vocoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_vocoder(self):
        self.vocoder = HiFiGANGenerator(num_mels=80).to(self.device).eval()

    def encode(self, audio, t):
        latent = self.encoder(audio, t)
        quantized = self.quantizer(latent)
        return quantized
    
    def decode(self, quantized):
        device = quantized.device
        if self.vocoder is None:
            self.vocoder = HiFiGANGenerator(num_mels=80).to(device).eval()
        mel = torch.log(torch.clamp(quantized, min=1e-5))
        mel = mel.squeeze(1)
        return self.vocoder(mel)
# class Experiment2Codec(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         self.quantizer = ScalarQuantizer(num_bits=12)
#         self.mel_proj = nn.Conv2d(1, 80, 1)
#         self.vocoder = None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     def load_vocoder(self):
#         self.mel_proj = self.mel_proj.to(self.device)
#         self.vocoder = HiFiGANGenerator(num_mels=80).to(self.device).eval()

#     def encode(self, audio, t):
#         latent = self.encoder(audio, t)
#         quantized = self.quantizer(latent)
#         return quantized
    
#     def decode(self, quantized):
#         device = quantized.device
#         if self.vocoder is None:
#             self.mel_proj = self.mel_proj.to(device)
#             self.vocoder = HiFiGANGenerator(num_mels=80).to(device).eval()
#         mel = self.mel_proj(quantized)
#         mel = torch.log(torch.clamp(mel, min=1e-5))
#         return self.vocoder(mel)
