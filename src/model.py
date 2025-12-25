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
    def __init__(self, in_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, in_channels)
        self.norm2 = nn.GroupNorm(1, in_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return x + h

class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4), time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        downs = []
        down_samples = []
        ch = base_channels
        skip_channels = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            downs.append(ResidualBlock(ch, time_emb_dim))
            down_samples.append(nn.Conv2d(ch, out_ch, 4, stride=2, padding=1))
            skip_channels.append(ch)
            ch = out_ch
        self.downs = nn.ModuleList(downs)
        self.down_samples = nn.ModuleList(down_samples)

        self.mid1 = ResidualBlock(ch, time_emb_dim)
        self.mid2 = ResidualBlock(ch, time_emb_dim)

        ups = []
        up_samples = []
        for mult, skip_ch in zip(reversed(channel_mults), reversed(skip_channels)):
            out_ch = base_channels * mult
            up_samples.append(nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1))
            ups.append(ResidualBlock(out_ch + skip_ch, time_emb_dim))
            ch = out_ch + skip_ch
        self.up_samples = nn.ModuleList(up_samples)
        self.ups = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(1, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.in_conv(x)
        skips = []
        for rb, ds in zip(self.downs, self.down_samples):
            x = rb(x, t_emb)
            skips.append(x)
            x = ds(x)
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)
        for us, rb in zip(self.up_samples, self.ups):
            x = us(x)
            skip = skips.pop()
            if skip.shape[2] != x.shape[2] or skip.shape[3] != x.shape[3]:
                skip = skip[:, :, : x.shape[2], : x.shape[3]]
            x = torch.cat([x, skip], dim=1)
            x = rb(x, t_emb)
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x
