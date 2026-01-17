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
    def __init__(self, in_c, out_c, t_dim=None, gn_groups=8):
        super().__init__()
        def _safe_groups(channels, target_groups=gn_groups):
            g = min(target_groups, channels)
            while g > 1 and (channels % g) != 0:
                g -= 1
            return max(1, g)

        self.norm1 = nn.GroupNorm(_safe_groups(in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(_safe_groups(out_c), out_c)
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


class VQEMA(nn.Module):
    def __init__(self, num_embed=4096, embed_dim=16, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embed = int(num_embed)
        self.embed_dim = int(embed_dim)
        self.decay = float(decay)
        self.eps = float(eps)

        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embed, 1.0 / self.num_embed)

        self.register_buffer('cluster_size', torch.zeros(self.num_embed))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())

    def forward(self, z):
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_perm.view(-1, self.embed_dim)


        embed_weight = self.embedding.weight
        dist = (
            (z_flat ** 2).sum(dim=1, keepdim=True)
            - 2.0 * z_flat @ embed_weight.t()
            + (embed_weight ** 2).sum(dim=1, keepdim=True).t()
        )

        encoding_indices = dist.argmin(dim=1)
        z_q_flat = embed_weight[encoding_indices]

        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        loss_recon = torch.mean((z.detach() - z_q) ** 2)
        loss_commit = 0.25 * torch.mean((z - z_q.detach()) ** 2)
        loss = loss_recon + loss_commit

        z_q_st = z + (z_q - z).detach()

        if self.training:
            with torch.no_grad():
                encodings = torch.nn.functional.one_hot(encoding_indices, num_classes=self.num_embed).type(z_flat.dtype)
                n = encodings.sum(dim=0)
                embed_sum = encodings.t() @ z_flat  
                self.cluster_size = self.cluster_size.to(z.device) * self.decay + (1 - self.decay) * n.to(self.cluster_size.dtype)
                self.embed_avg = self.embed_avg.to(z.device) * self.decay + (1 - self.decay) * embed_sum.to(self.embed_avg.dtype)
                n_total = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n_total + self.num_embed * self.eps) * n_total
                new_weight = self.embed_avg / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(new_weight)

        return z_q_st, loss, encoding_indices.view(B, H, W)


class CompressionEncoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, num_embeddings=4096, num_res_blocks=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim // 2, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(hidden_dim, hidden_dim))
        self.res = nn.Sequential(*res_blocks)
        
        self.conv4 = nn.Conv2d(hidden_dim, latent_dim, 1)
        self.vq = VQEMA(num_embed=num_embeddings, embed_dim=latent_dim)
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
        
        def _safe_groups(channels, target_groups=8):
            g = min(target_groups, channels)
            while g > 1 and (channels % g) != 0:
                g -= 1
            return max(1, g)

        self.norm = nn.GroupNorm(_safe_groups(hidden_dim // 4), hidden_dim // 4)
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
    def __init__(self, latent_dim=16, hidden_dim=256, t_dim=128, num_embeddings=4096, num_res_blocks=2):
        super().__init__()
        self.encoder = CompressionEncoder(latent_dim, hidden_dim, num_embeddings, num_res_blocks)
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
        if mel.dim() == 4:
            mel = mel.mean(dim=1)
        
        if mel.dim() == 3:
            device = next(self.vocoder.parameters()).device
            mel = mel.to(device)
            wave = self.vocoder(mel)
            return wave
        else:
            raise ValueError(f"Could not resolve mel input to 3 dimensions (B, 80, T). Final shape: {mel.shape}")

    def forward(self, x, t):
        z_q, vq_loss, idx = self.encode(x)
        mel = self.decode(z_q, t)
        return mel, vq_loss
