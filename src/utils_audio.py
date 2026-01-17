from pathlib import Path
import torch
import soundfile as sf
import torchaudio
import librosa
import numpy as np
from src import config as cfg
    

def process_root(
    root_in, root_out, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256
):
    root_in, root_out = Path(root_in), Path(root_out)
    for path in root_in.rglob("*.flac"):
        rel = path.relative_to(root_in)
        out_path = root_out / rel.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav, sr = sf.read(path, always_2d=False)
        if wav.ndim == 1:
            wav = wav[None, :]
        wav = torch.from_numpy(wav).float()
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        spec = spec_transform(wav)
        spec = torch.log(spec + 1e-9)
        torch.save(spec, out_path)


def mel_spectrogram(
    y, n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False
):

    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()
    if y.ndim == 1:
        y = y.unsqueeze(0)

    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_size,
        hop_length=hop_size,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
        center=center
    ).to(y.device)
    
    spec = spec_transform(y)
    spec = torch.log(spec + 1e-9)
    if spec.ndim == 3:
        spec = spec.unsqueeze(0)
    return spec


def griffin_lim(mel, n_fft=1024, hop_length=256, n_iter=32):

    if isinstance(mel, torch.Tensor):
        mel = mel.cpu().numpy()
    
    if mel.ndim == 4:
        mel = mel[0, 0]
    elif mel.ndim == 3:
        mel = mel[0]
        
    S = np.exp(mel)
    wav = librosa.feature.inverse.mel_to_audio(
        S, 
        sr=cfg.SAMPLE_RATE, 
        n_fft=cfg.N_FFT, 
        hop_length=cfg.HOP_LENGTH, 
        win_length=cfg.N_FFT,
        n_iter=n_iter
    )
    return wav
