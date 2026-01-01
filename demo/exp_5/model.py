import torch
import torch.nn as nn


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=4096, embedding_dim=16, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        w = torch.randn(num_embeddings, embedding_dim)
        nn.init.uniform_(w, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        self.register_buffer('w', w)
        
        self.cluster_size: torch.Tensor
        self.w: torch.Tensor

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        
        dist = (
            (z_flat ** 2).sum(1, keepdim=True)
            - 2 * z_flat @ self.w.t()
            + (self.w ** 2).sum(1, keepdim=True).t()
        )
        
        idx = dist.argmin(dim=1)
        z_q_flat = self.w[idx]
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        loss = torch.mean((z.detach() - z_q) ** 2) + 0.25 * torch.mean((z.detach() - z_q.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        
        if self.training:
            with torch.no_grad():
                mask = torch.zeros(self.num_embeddings, device=z.device)
                mask.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
                
                self.cluster_size.mul_(self.decay)
                self.cluster_size.add_(mask, alpha=1 - self.decay)
                
                n = z_flat.size(0)
                updated_cluster_size = (
                    (self.cluster_size + self.epsilon) /
                    (self.cluster_size.sum() + self.num_embeddings * self.epsilon) *
                    self.cluster_size.sum()
                )
        
        return z_q, loss, idx.view(B, H, W)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.relu(self.norm1(x))
        h = self.conv1(h)
        h = self.relu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class CompressionEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256, latent_dim=16, num_res_blocks=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 2, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.conv4 = nn.Conv2d(hidden_dim, latent_dim, 1)
        self.relu = nn.ReLU()
        self.quantizer = VectorQuantizerEMA(num_embeddings=4096, embedding_dim=latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.res_blocks(x)
        z = self.conv4(x)
        z_q, vq_loss, indices = self.quantizer(z)
        return z_q, vq_loss, indices


class CompressionDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256, out_channels=1, num_res_blocks=2):
        super().__init__()
        self.deconv1 = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim // 2, out_channels, 4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, z_q):
        x = self.relu(self.deconv1(z_q))
        x = self.res_blocks(x)
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class CompressionCodec(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.encoder = CompressionEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = CompressionDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.vocoder = None

    def load_vocoder(self):
        from demo.exp_2.vocoder import HiFiGANGenerator
        device = next(self.parameters()).device
        self.vocoder = HiFiGANGenerator(num_mels=80).to(device).eval()

    def encode(self, x):
        z_q, vq_loss, indices = self.encoder(x)
        return z_q, vq_loss, indices

    def decode(self, z_q, use_vocoder=True):
        mel_recon = self.decoder(z_q)
        if use_vocoder and self.vocoder is not None:
            mel_recon_log = torch.log(torch.clamp(mel_recon, min=1e-5))
            mel_recon_log = mel_recon_log.squeeze(1)
            waveform = self.vocoder(mel_recon_log)
            return waveform
        return mel_recon

    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x)
        mel_recon = self.decoder(z_q)
        return mel_recon, vq_loss