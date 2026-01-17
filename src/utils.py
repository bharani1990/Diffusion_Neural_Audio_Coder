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


def compute_metrics(wave_ref, wave_rec, sr=16000, latent_indices=None, duration_sec=None):
    metric_dict = {}  
    try:
        p = pesq_metric(wave_ref, wave_rec, sr)
    except Exception:
        p = 0.0
    try:
        s = stoi_metric(wave_ref, wave_rec, sr)
    except Exception:
        s = 0.0
    br = None
    if latent_indices is not None and duration_sec is not None:
        try:
            br = bitrate(latent_indices, duration_sec)
        except Exception:
            br = None
            
    metric_dict.update({
        "pesq": p,
        "stoi": s,
        "bitrate_kbps": br
    })
    return metric_dict
