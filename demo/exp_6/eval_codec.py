import sys
import argparse
import torch
import time
import numpy as np
from pathlib import Path
import pesq
import pystoi

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from demo.exp_6.train_module import CompressionDiffusionModule

def mel_to_waveform(mel_spec, sample_rate=22050):
    import torchaudio.transforms as T
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    if mel_spec.dim() == 4:
        mel_spec = mel_spec.squeeze(0).squeeze(0)
    elif mel_spec.dim() == 3:
        mel_spec = mel_spec.squeeze(0)
    mel_spec = torch.exp(mel_spec)
    mel_inv = T.InverseMelScale(n_stft=n_fft//2 + 1, n_mels=n_mels, sample_rate=sample_rate)
    mag_spec = mel_inv(mel_spec)
    griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    waveform = griffin_lim(torch.abs(mag_spec))
    return waveform

def calculate_bitrate(latent, duration_sec):
    compressed_size_bytes = latent.numel() * 4
    if duration_sec > 0:
        bitrate_bps = (compressed_size_bytes * 8) / duration_sec
    else:
        bitrate_bps = 0
    return bitrate_bps / 1000


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='lightning_logs/exp_6/epoch=49-step=2400.ckpt')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for PESQ (must be 8000 or 16000)')
    args = parser.parse_args()

    if args.sr not in [8000, 16000]:
        print('PESQ only supports sample rates 8000 or 16000. Please set --sr accordingly.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_files = []
    root = Path("data/processed")
    for i in range(args.num_samples):
        file_path = root / f"train-clean-5/19/198/19-198-00{i:02d}.pt"
        if file_path.exists():
            test_files.append(str(file_path))
    print(f"Found {len(test_files)} test files")

    norm_files = [f.replace('data/processed', 'data/processed_norm') for f in test_files]

    def load_tensor(file_path):
        x = torch.load(file_path).float()
        if x.dim() == 3:
            pass
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"unexpected shape {x.shape}")
        return x

    model = CompressionDiffusionModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()

    results = []
    for i, (test_file, norm_file) in enumerate(zip(test_files, norm_files)):
        mel = load_tensor(test_file)
        mel_norm = load_tensor(norm_file)
        mel_input = mel.unsqueeze(0).to(device)
        ref_wave = mel_to_waveform(mel_norm.cpu(), sample_rate=args.sr)
        ref_np = ref_wave.numpy()
        start = time.time()
        latent = model.compression_model.encode(mel_input)
        duration_sec = ref_wave.shape[-1] / args.sr
        bitrate_kbps = calculate_bitrate(latent, duration_sec)
        compressed = model.compression_model(mel_input)
        recon_wave = mel_to_waveform(compressed.squeeze(0).cpu(), sample_rate=args.sr)
        latency = (time.time() - start) * 1000
        recon_np = recon_wave.squeeze().cpu().detach().numpy()
        min_len = min(len(ref_np), len(recon_np))
        ref_np = ref_np[:min_len]
        recon_np = recon_np[:min_len]
        try:
            pesq_mode = 'nb' if args.sr == 8000 else 'wb'
            pesq_score = pesq.pesq(args.sr, ref_np, recon_np, pesq_mode)
            stoi_score = pystoi.stoi(ref_np, recon_np, args.sr, extended=False)
        except Exception as e:
            print(f"Metric error {i}: {e}")
            pesq_score, stoi_score = 0.0, 0.0
        results.append({'id': i, 'pesq': pesq_score, 'stoi': stoi_score, 'latency': latency, 'bitrate_kbps': bitrate_kbps})
        print(f"{i}: PESQ={pesq_score:.3f}, STOI={stoi_score:.3f}, latency={latency:.2f}ms, bitrate={bitrate_kbps:.2f} kbps")

    if not results:
        print("No results to aggregate. No test files were found or processed.")
        return
    mean_pesq = np.mean([r['pesq'] for r in results])
    mean_stoi = np.mean([r['stoi'] for r in results])
    mean_latency = np.mean([r['latency'] for r in results])
    mean_bitrate = np.mean([r['bitrate_kbps'] for r in results])
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print(f"Mean PESQ : {mean_pesq:.3f}")
    print(f"Mean STOI : {mean_stoi:.3f}")
    print(f"Mean latency: {mean_latency:.2f}ms")
    print(f"Mean bitrate: {mean_bitrate:.2f} kbps")
    print("="*60)

if __name__ == '__main__':
    main()
