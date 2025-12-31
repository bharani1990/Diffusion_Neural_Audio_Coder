import sys
from pathlib import Path
import torch
import torchaudio
import time
import numpy as np
import pesq
import pystoi
from torchaudio.transforms import GriffinLim
import torchaudio.transforms as T


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import DiffusionUNet
from demo.exp_2.model import Experiment2Codec

def load_test_files():
    test_files = []
    root = Path("data/processed_norm")
    for i in range(10):
        file_path = root / f"train-clean-5/19/198/19-198-00{i:02d}.pt"
        if file_path.exists():
            test_files.append(str(file_path))
    return test_files


def mel_to_waveform(mel_spec, sample_rate=16000):
    n_fft = 1024
    hop_length = 256
    n_mels = 80
    
    if mel_spec.dim() == 4:
        mel_spec = mel_spec.squeeze(0).squeeze(0)
    elif mel_spec.dim() == 3:
        mel_spec = mel_spec.squeeze(0)
    elif mel_spec.dim() == 2:
        pass
    
    mel_spec = torch.exp(mel_spec)
    
    mel_inv = T.InverseMelScale(n_stft=n_fft//2 + 1, n_mels=n_mels, sample_rate=sample_rate)
    mag_spec = mel_inv(mel_spec)
    
    griffin_lim = T.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    waveform = griffin_lim(torch.abs(mag_spec))
    
    return waveform


def evaluate_model(model, test_files):
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    with torch.no_grad():
        for i, test_file in enumerate(test_files):
            mel = torch.load(test_file).float()
            if mel.dim() == 3:
                mel = mel.squeeze(0)
            mel_input = mel.unsqueeze(0).unsqueeze(0).to(device)
            ref_wave = mel_to_waveform(mel.cpu())
            ref_np = ref_wave.numpy()
            B = mel_input.shape[0]
            t = torch.randint(0, 1000, (B,), device=device)            
            start = time.time()
            latent = model.encode(mel_input, t)
            recon_wave = model.decode(latent)
            latency = (time.time() - start) * 1000
            recon_np = recon_wave.squeeze().cpu().detach().numpy()
            min_len = min(len(ref_np), len(recon_np))
            ref_np = ref_np[:min_len]
            recon_np = recon_np[:min_len]
            
            try:
                pesq_score = pesq.pesq(16000, ref_np, recon_np, 'wb')
                stoi_score = pystoi.stoi(ref_np, recon_np, 16000, extended=False)
            except Exception as e:
                print(f"Metric error {i}: {e}")
                pesq_score, stoi_score = 0.0, 0.0
                
            results.append({'id': i, 'pesq': pesq_score, 'stoi': stoi_score, 'latency': latency})
            print(f"{i}: PESQ={pesq_score:.3f}, STOI={stoi_score:.3f}, latency={latency:.2f}ms")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lightning_logs/version_0/checkpoints/last.ckpt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    
    test_files = load_test_files()
    print(f"Found {len(test_files)} test files")
    
    encoder = DiffusionUNet(in_channels=1)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded encoder from {args.checkpoint}")
    
    model = Experiment2Codec(encoder=encoder)
    model.load_vocoder()
    print("NEW MODEL: DiffusionUNet + ScalarQuantizer + HiFi-GAN")
    
    model = model.to(device)
    results = evaluate_model(model, test_files)
    
    mean_pesq = np.mean([r['pesq'] for r in results])
    mean_stoi = np.mean([r['stoi'] for r in results])
    mean_latency = np.mean([r['latency'] for r in results])
    
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print(f"Mean PESQ : {mean_pesq:.3f}  (Target >=3.5)")
    print(f"Mean STOI : {mean_stoi:.3f}  (Target >=0.9)")
    print(f"Mean latency: {mean_latency:.2f}ms (Target <20ms)")
    print(f"Estimated bitrate: ~12kbps")
    print("="*60)
    
    pesq_ok = mean_pesq >= 3.5
    stoi_ok = mean_stoi >= 0.9
    latency_ok = mean_latency < 20
    
    print(f"PESQ: {'PASS' if pesq_ok else 'FAIL'} {mean_pesq:.3f}")
    print(f"STOI: {'PASS' if stoi_ok else 'FAIL'} {mean_stoi:.3f}")
    print(f"Latency: {'PASS' if latency_ok else 'FAIL'} {mean_latency:.2f}ms")
    
    if pesq_ok and stoi_ok and latency_ok:
        print("ALL THRESHOLDS PASSED")
    else:
        print("Some thresholds not met")
