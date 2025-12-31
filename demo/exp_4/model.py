import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        z_flattened = z.reshape(-1, self.embedding_dim)
        distances = (z_flattened.pow(2).sum(1, keepdim=True)
                    - 2 * z_flattened @ self.embeddings.weight.t()
                    + self.embeddings.weight.pow(2).sum(1))
        min_encoding_indices = distances.argmin(1)
        z_q = self.embeddings(min_encoding_indices).reshape(z.shape)
        loss = (z_q.detach() - z).pow(2).mean() + 0.25 * (z_q - z.detach()).pow(2).mean()
        z_q = z + (z_q - z).detach()
        return z_q, loss, min_encoding_indices


class CompressionEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, latent_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim * 2, latent_dim, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        z = self.conv4(x)
        B, C, H, W = z.shape
        z_reshaped = z.permute(0, 2, 3, 1).reshape(-1, C)
        z_q, vq_loss, indices = self.quantizer(z_reshaped.view(B, H, W, C).permute(0, 3, 1, 2))
        return z_q, vq_loss, indices


class CompressionDecoder(nn.Module):
    def __init__(self, latent_dim=4, hidden_dim=64, out_channels=1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, hidden_dim * 2, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, z_q):
        x = self.relu(self.deconv1(z_q))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)
        return x


class CompressionCodec(nn.Module):
    def __init__(self, latent_dim=4, hidden_dim=64):
        super().__init__()
        self.encoder = CompressionEncoder(in_channels=1, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = CompressionDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, out_channels=1)
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
        z_q, vq_loss, _ = self.encode(x)
        mel_recon = self.decoder(z_q)
        return mel_recon, vq_loss
    