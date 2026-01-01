from pathlib import Path
import torch
import torchaudio
import librosa

def mel_to_linear(mel_spec, sample_rate=16000, n_fft=1024, n_mels=80):
    if mel_spec.dim() == 4:
        mel_spec = mel_spec.squeeze(0)
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0)

    c, f, t = mel_spec.shape

    mel_filter = torch.from_numpy(librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        htk=True,
    )).to(mel_spec.device, mel_spec.dtype)

    mel_pinv = torch.pinverse(mel_filter)

    mag = torch.matmul(mel_pinv, mel_spec.view(f, t))
    mag = torch.clamp(mag, min=0.0)
    mag = mag.view(1, n_fft // 2 + 1, t)
    return mag


def griffin_lim_waveform(log_mel, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, n_iter=32):
    mel = torch.exp(log_mel.to(torch.float32))
    mag = mel_to_linear(mel, sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)

    wav = torchaudio.functional.griffinlim(
        mag.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=n_iter,
        window=torch.hann_window(n_fft),
        momentum=0.99,
        length=None,
        rand_init=True,
    )
    return wav.unsqueeze(0)
