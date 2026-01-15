import sys
from pathlib import Path
import torch
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SAMPLE_RATE
from src.train_module import AudioCodecModule
from src.utils import pesq_metric, stoi_metric, bitrate

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
                data = torch.load(mel_file)
                if isinstance(data, (tuple, list)):
                    mel = data[0].float()
                else:
                    mel = data.float()
                if mel.dim() == 3:
                    mel = mel.squeeze(0)
                
                mel_input = mel.unsqueeze(0).unsqueeze(0).to(device)
                start_time = time.time()
                z_q, vq_loss, indices = model.model.encode(mel_input)
                mel_recon = model.model.decode(z_q, torch.tensor([500], device=device))
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                duration_sec = mel.shape[-1] * 0.01
                br = bitrate(indices, duration_sec)
                
                mel_orig_np = mel.squeeze().cpu().numpy()
                mel_recon_np = mel_recon.squeeze().cpu().numpy()
                
                min_mel_len = min(mel_orig_np.shape[-1], mel_recon_np.shape[-1])
                mel_orig_np = mel_orig_np[..., :min_mel_len]
                mel_recon_np = mel_recon_np[..., :min_mel_len]
                
                mel_orig_flat = mel_orig_np.reshape(-1)
                mel_recon_flat = mel_recon_np.reshape(-1)
                
                p = pesq_metric(mel_orig_flat, mel_recon_flat, SAMPLE_RATE)
                s = stoi_metric(mel_orig_flat, mel_recon_flat, SAMPLE_RATE)
                
                result = {
                    'id': idx,
                    'pesq': p,
                    'stoi': s,
                    'latency_ms': latency_ms,
                    'bitrate_kbps': br
                }
                results.append(result)
                
                print(f"Sample {idx:2d}: PESQ={p:.3f} STOI={s:.3f} Latency={latency_ms:.2f}ms Bitrate={br:.2f}kbps")
                
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
    
    mean_pesq = np.mean([r['pesq'] for r in results])
    mean_stoi = np.mean([r['stoi'] for r in results])
    mean_latency = np.mean([r['latency_ms'] for r in results])
    mean_bitrate = np.mean([r['bitrate_kbps'] for r in results])
    
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
    args = parser.parse_args()
    
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
        module = AudioCodecModule.load_from_checkpoint(args.checkpoint)
        model = module.to(device)
        results = evaluate_codec(model, test_files)
        print_summary(results, "AudioCodec")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
