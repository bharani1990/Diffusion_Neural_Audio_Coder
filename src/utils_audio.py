from pathlib import Path
import torch
import soundfile as sf
import torchaudio

def process_root(root_in, root_out, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=256):
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
