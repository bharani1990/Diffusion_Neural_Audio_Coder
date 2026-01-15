import torch
import torch.nn.functional as F
from pathlib import Path


def collate_fn(batch):
    max_time = max(mel.shape[-1] for mel in batch)
    padded = []
    for mel in batch:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[-1] < max_time:
            pad = max_time - mel.shape[-1]
            mel = F.pad(mel, (0, pad), mode='constant', value=0)
        padded.append(mel)
    return torch.stack(padded, dim=0)


def pesq_metric(ref, recon, sr=16000):
    import pesq
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
    if isinstance(recon, torch.Tensor):
        recon = recon.cpu().numpy()
    min_len = min(len(ref), len(recon))
    try:
        return float(pesq.pesq(sr, ref[:min_len], recon[:min_len], 'wb'))
    except:
        return 0.0


def stoi_metric(ref, recon, sr=16000):
    import pystoi
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().numpy()
    if isinstance(recon, torch.Tensor):
        recon = recon.cpu().numpy()
    min_len = min(len(ref), len(recon))
    try:
        return float(pystoi.stoi(ref[:min_len], recon[:min_len], sr, extended=False))
    except:
        return 0.0


def bitrate(latent, duration_sec):
    return (latent.numel() * 12) / (duration_sec * 1000)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
