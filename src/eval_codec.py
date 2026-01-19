import sys
import os
from pathlib import Path
import torch
import time
import numpy as np

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SAMPLE_RATE
from src.train_module import AudioCodecModule
from src.utils import pesq_metric, stoi_metric, bitrate, compute_metrics, ensure_dir

def load_test_files(num_samples=None):
    mel_dir = Path("data/processed_norm/test-clean/1995/1826")
    
    if not mel_dir.exists():
        print(f"Error: Mel directory not found at {mel_dir}")
        return []
    
    mel_files = sorted(mel_dir.glob("*.pt"))
    if num_samples:
        mel_files = mel_files[:num_samples]
    
    return [str(f) for f in mel_files]


def evaluate_codec(model, test_files):
    device = next(model.parameters()).device
    model.eval()
    results = []
    
    with torch.no_grad():
        for idx, mel_file in enumerate(test_files):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                data = torch.load(mel_file)
                if isinstance(data, (tuple, list)):
                    mel = data[0].float()
                else:
                    mel = data.float()
                if mel.dim() == 3:
                    mel = mel.squeeze(0)
                
                mel_input_enc = mel.unsqueeze(0).unsqueeze(0).to(device)
                mel_input_voc = mel.unsqueeze(0).to(device)
                start_time = time.time()
                z_q, vq_loss, indices = model.model.encode(mel_input_enc)
                mel_recon = model.model.decode(z_q, torch.tensor([500], device=device))
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                duration_sec = mel.shape[-1] * 0.01
                br = bitrate(indices, duration_sec)
                
                wave_orig = model.model.to_waveform(mel_input_voc)
                wave_recon = model.model.to_waveform(mel_recon)

                wave_orig_np = wave_orig.squeeze().cpu().numpy()
                wave_recon_np = wave_recon.squeeze().cpu().numpy()

                min_wav_len = min(wave_orig_np.shape[-1], wave_recon_np.shape[-1])
                wave_orig_np = wave_orig_np[..., :min_wav_len]
                wave_recon_np = wave_recon_np[..., :min_wav_len]

                metrics = compute_metrics(wave_orig_np, wave_recon_np, SAMPLE_RATE, indices, duration_sec)
                result = {
                    'id': idx,
                    'pesq': metrics.get('pesq', 0.0),
                    'stoi': metrics.get('stoi', 0.0),
                    'latency_ms': latency_ms,
                    'bitrate_kbps': metrics.get('bitrate_kbps', br)
                }
                results.append(result)
                print(f"Sample {idx:2d}: PESQ={result['pesq']:.3f} STOI={result['stoi']:.3f} Latency={latency_ms:.2f}ms Bitrate={result['bitrate_kbps']:.2f}kbps")
                
                del mel_input_enc, mel_input_voc, z_q, mel_recon, wave_orig, wave_recon, wave_orig_np, wave_recon_np
                
            except Exception as e:
                import traceback
                print(f"Error processing sample {idx}: {e}")
                traceback.print_exc()
                continue
    
    return results


def print_summary(results, model_name):
    if not results:
        print(f"No results for {model_name}")
        return
    
    print("\n" + "="*70)
    print(f"EVALUATION: {model_name}")
    print("="*70)
    
    pesq_values = [r['pesq'] for r in results if r['pesq'] > 0]
    stoi_values = [r['stoi'] for r in results if r['stoi'] > 0]
    latency_values = [r['latency_ms'] for r in results]
    bitrate_values = [r['bitrate_kbps'] for r in results if r['bitrate_kbps'] is not None]

    
    mean_pesq = np.mean(pesq_values)
    mean_stoi = np.mean(stoi_values)
    mean_latency = np.mean(latency_values)
    mean_bitrate = np.mean(bitrate_values) if bitrate_values else 0.0
    
    print(f"PESQ:    {mean_pesq:.3f}")
    print(f"STOI:    {mean_stoi:.3f}")
    print(f"Latency: {mean_latency:.2f}ms")
    print(f"Bitrate: {mean_bitrate:.2f}kbps")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate audio codec")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    test_files = load_test_files(args.num_samples)
    print(f"Loaded {len(test_files)} test files")
    
    if not test_files:
        print("No test files found")
        sys.exit(1)
    
    if args.checkpoint is None:
        args.checkpoint = "lightning_logs/checkpoints/last.ckpt"
    
    try:
        module = AudioCodecModule.load_from_checkpoint(args.checkpoint, map_location=device)
        model = module.to(device)
        results = evaluate_codec(model, test_files)
        print_summary(results, "AudioCodec")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
