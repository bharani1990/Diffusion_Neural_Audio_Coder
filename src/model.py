import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
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


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, t_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_c) if t_dim else None
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if t is not None and self.t_proj is not None:
            h = h + self.t_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return self.skip(x) + h


class VQ(nn.Module):
    def __init__(self, num_embed=4096, embed_dim=16, decay=0.99):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.decay = decay
        self.cluster_size = torch.zeros(num_embed)
        w = torch.randn(num_embed, embed_dim)
        nn.init.uniform_(w, -1.0 / num_embed, 1.0 / num_embed)
        self.w = w

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.embed_dim)
        
        w = self.w.to(z.device)
        cluster_size = self.cluster_size.to(z.device)
        
        dist = (
            (z_flat ** 2).sum(1, keepdim=True)
            - 2 * z_flat @ w.t()
            + (w ** 2).sum(1, keepdim=True).t()
        )
        idx = dist.argmin(dim=1)
        z_q_flat = w[idx]
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        loss = torch.mean((z.detach() - z_q) ** 2) + 0.25 * torch.mean((z - z_q.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        
        if self.training:
            with torch.no_grad():
                mask = torch.zeros(self.num_embed, device=z.device)
                mask.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
                cluster_size = cluster_size * self.decay + mask * (1 - self.decay)
                self.cluster_size = cluster_size.to(self.cluster_size.device)
        
        return z_q, loss, idx.view(B, H, W)


class CompressionEncoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim // 2, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim)
        )
        self.conv4 = nn.Conv2d(hidden_dim, latent_dim, 1)
        self.vq = VQ(num_embed=4096, embed_dim=latent_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.res(x)
        z = self.conv4(x)
        z_q, vq_loss, idx = self.vq(z)
        return z_q, vq_loss, idx

    def encode_no_vq(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.res(x)
        z = self.conv4(x)
        return z


class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, t_dim=128):
        super().__init__()
        self.t_emb = TimeEmbedding(t_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim)
        )
        
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.res1 = ResBlock(hidden_dim, hidden_dim, t_dim)
        self.res2 = ResBlock(hidden_dim, hidden_dim, t_dim)
        
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.res3 = ResBlock(hidden_dim, hidden_dim, t_dim)
        
        self.up2 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1)
        self.res4 = ResBlock(hidden_dim // 2, hidden_dim // 2, t_dim)
        
        self.up3 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, stride=2, padding=1)
        self.res5 = ResBlock(hidden_dim // 4, hidden_dim // 4, t_dim)
        
        self.norm = nn.GroupNorm(8, hidden_dim // 4)
        self.out = nn.Conv2d(hidden_dim // 4, 80, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, z_q, t):
        t_emb = self.t_mlp(self.t_emb(t))
        
        x = self.proj(z_q)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        
        x = self.up1(x)
        x = self.res3(x, t_emb)
        
        x = self.up2(x)
        x = self.res4(x, t_emb)
        
        x = self.up3(x)
        x = self.res5(x, t_emb)
        
        x = self.act(self.norm(x))
        x = self.out(x)
        
        return x


class HiFiGAN(nn.Module):
    def __init__(self, num_mels=80):
        super().__init__()
        self.conv_pre = nn.Conv1d(num_mels, 256, 7, 1, padding=3)
        
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 256, 16, 8, padding=4),
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1),
        ])
        
        self.conv_post = nn.Conv1d(32, 1, 7, 1, padding=3)

    def forward(self, mel):
        x = self.conv_pre(mel)
        
        for up in self.ups:
            x = up(x)
            x = nn.functional.leaky_relu(x, 0.1)
        
        x = nn.functional.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class AudioCodec(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, t_dim=128):
        super().__init__()
        self.encoder = CompressionEncoder(latent_dim, hidden_dim)
        self.decoder = DiffusionDecoder(latent_dim, hidden_dim, t_dim)
        self.vocoder = HiFiGAN(num_mels=80)
        self.latent_dim = latent_dim

    def encode(self, x):
        z_q, vq_loss, idx = self.encoder(x)
        return z_q, vq_loss, idx

    def decode(self, z_q, t):
        mel = self.decoder(z_q, t)
        return mel

    def to_waveform(self, mel):
        if mel.dim() == 3:
            mel = mel.unsqueeze(-1)
        B, C, H, W = mel.shape
        mel_flat = mel.reshape(B, C, H * W)
        wave = self.vocoder(mel_flat)
        return wave

    def forward(self, x, t):
        z_q, vq_loss, idx = self.encode(x)
        mel = self.decode(z_q, t)
        return mel, vq_loss
