import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.dropout1 = nn.Dropout2d(dropout)
        self.time_proj = nn.Linear(time_emb_dim, in_channels)
        self.norm2 = nn.GroupNorm(8, in_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.dropout2 = nn.Dropout2d(dropout)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        h = self.dropout1(h)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        h = self.dropout2(h)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.permute(0, 2, 3, 1).reshape(b * h * w, c)
        qkv = self.to_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b * h * w, self.num_heads, self.head_dim)
        k = k.reshape(b * h * w, self.num_heads, self.head_dim)
        v = v.reshape(b * h * w, self.num_heads, self.head_dim)
        attn = torch.einsum("bhi,bhj->hij", q, k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("hij,bhj->bhi", attn, v)
        out = out.reshape(b * h * w, c)
        out = self.to_out(out)
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + out


class DiffusionModel(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, channel_mults=(1, 2, 4), time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        ch = base_channels
        skip_channels = []
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder_blocks.append(ResidualBlock(ch, time_emb_dim))
            self.downsample_blocks.append(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1))
            skip_channels.append(ch)
            ch = out_ch
        
        self.bottleneck = nn.Sequential(
            ResidualBlock(ch, time_emb_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, time_emb_dim),
        )
        
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for mult, skip_ch in zip(reversed(channel_mults), reversed(skip_channels)):
            out_ch = base_channels * mult
            self.upsample_blocks.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
            self.decoder_blocks.append(ResidualBlock(out_ch + skip_ch, time_emb_dim))
            ch = out_ch + skip_ch
        
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = []
        for encoder_block, downsample_block in zip(self.encoder_blocks, self.downsample_blocks):
            h = encoder_block(h, t_emb)
            skips.append(h)
            h = downsample_block(h)
        h = self.bottleneck[0](h, t_emb)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, t_emb)
        for upsample_block, decoder_block, skip in zip(self.upsample_blocks, self.decoder_blocks, reversed(skips)):
            h = upsample_block(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder_block(h, t_emb)
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h

class CompressionModel(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, latent_dim, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
