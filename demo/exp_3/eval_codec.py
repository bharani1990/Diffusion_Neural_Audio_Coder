import sys
from pathlib import Path
import torch
import time
import numpy as np
import pesq
import pystoi
import torchaudio.transforms as T

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import DiffusionUNet
from demo.exp_3.model import Experiment3Codec, DecoderNetwork
from demo.exp_3.train_module import DecoderLightningModule


def load_test_files():
    test_files = []
    root = Path("data/processed")
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
            t = torch.zeros(B, dtype=torch.long, device=device)
            
            start = time.time()
            latent = model.encode(mel_input, t)
            recon_wave = model.decode(latent)
            latency = (time.time() - start) * 1000
            
            recon_np = recon_wave.squeeze().cpu().detach().numpy()
            
            latent_elements = latent.numel()
            duration_sec = len(ref_np) / 16000.0
            bitrate = (latent_elements * 12) / (duration_sec * 1000)
            
            min_len = min(len(ref_np), len(recon_np))
            ref_np = ref_np[:min_len]
            recon_np = recon_np[:min_len]
            
            try:
                pesq_score = pesq.pesq(16000, ref_np, recon_np, 'wb')
                stoi_score = pystoi.stoi(ref_np, recon_np, 16000, extended=False)
            except Exception as e:
                print(f"Metric error {i}: {e}")
                pesq_score, stoi_score = 0.0, 0.0
            
            results.append({'id': i, 'pesq': pesq_score, 'stoi': stoi_score, 'latency': latency, 'bitrate': bitrate})
            print(f"{i}: PESQ={pesq_score:.3f}, STOI={stoi_score:.3f}, latency={latency:.2f}ms, bitrate={bitrate:.2f}kbps")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lightning_logs/exp_3/checkpoints/last.ckpt")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_files = load_test_files()
    print(f"Found {len(test_files)} test files")
    
    lightning_module = DecoderLightningModule.load_from_checkpoint(args.checkpoint)
    
    encoder = lightning_module.encoder
    latent_proj = lightning_module.latent_proj
    decoder = lightning_module.decoder
    
    model = Experiment3Codec(encoder=encoder, decoder=decoder, latent_proj=latent_proj)
    model.load_vocoder()
    print("Experiment 3 Codec: DiffusionUNet + ScalarQuantizer + TrainedDecoder + HiFi-GAN")
    
    model = model.to(device)
    results = evaluate_model(model, test_files)
    
    mean_pesq = np.mean([r['pesq'] for r in results])
    mean_stoi = np.mean([r['stoi'] for r in results])
    mean_latency = np.mean([r['latency'] for r in results])
    mean_bitrate = np.mean([r['bitrate'] for r in results])
    
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print(f"Mean PESQ : {mean_pesq:.3f}  (Target >=3.5)")
    print(f"Mean STOI : {mean_stoi:.3f}  (Target >=0.9)")
    print(f"Mean latency: {mean_latency:.2f}ms (Target <20ms)")
    print(f"Mean bitrate: {mean_bitrate:.2f}kbps (Target 8-16kbps)")
    print("="*60)
    
    pesq_ok = mean_pesq >= 3.5
    stoi_ok = mean_stoi >= 0.9
    latency_ok = mean_latency < 20
    bitrate_ok = 8 <= mean_bitrate <= 16
    
    print(f"PESQ: {'PASS' if pesq_ok else 'FAIL'} {mean_pesq:.3f}")
    print(f"STOI: {'PASS' if stoi_ok else 'FAIL'} {mean_stoi:.3f}")
    print(f"Latency: {'PASS' if latency_ok else 'FAIL'} {mean_latency:.2f}ms")
    print(f"Bitrate: {'PASS' if bitrate_ok else 'FAIL'} {mean_bitrate:.2f}kbps")
    
    if pesq_ok and stoi_ok and latency_ok and bitrate_ok:
        print("ALL THRESHOLDS PASSED")
    else:
        print("Some thresholds not met")