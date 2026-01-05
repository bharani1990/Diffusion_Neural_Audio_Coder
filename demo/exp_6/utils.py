import torch
import torchaudio
import json
from pathlib import Path


def load_audio(audio_path, sr=22050):
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)
    return waveform


def save_audio(waveform, path, sr=22050):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), waveform, sr)


def get_compression_ratio(original_size, compressed_size):
    if compressed_size == 0:
        return 0
    return original_size / compressed_size


def compute_bitrate(num_samples, duration_sec, sr=22050):
    if duration_sec == 0:
        return 0
    return (num_samples / sr) / duration_sec


def save_metrics(metrics_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)


def load_checkpoint(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint
