import sys
import argparse
import json
import torch
import torchaudio
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.exp_6.train_module import CompressionDiffusionModule
from demo.exp_6.utils import load_audio, save_audio, save_metrics


class EvalDataset(Dataset):
    def __init__(self, manifest_path, data_dir, frames=120):
        self.frames = frames
        self.data_dir = Path(data_dir)
        self.samples = []
        
        with open(manifest_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = self.data_dir / sample['audio_filepath']
        
        waveform, sr = torchaudio.load(str(audio_path))
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        target_samples = self.frames * (sr // 100)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        elif waveform.shape[1] < target_samples:
            pad_len = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        
        return waveform.float(), sample.get('audio_filepath', f'sample_{idx}.wav')


def compute_pesq(ref, deg, sr=22050):
    try:
        import pesq
        ref_np = ref.squeeze().cpu().numpy()
        deg_np = deg.squeeze().cpu().numpy()
        score = pesq.pesq(sr, ref_np, deg_np, mode='wb')
        return score
    except:
        return 0.0


def compute_stoi(ref, deg, sr=22050):
    try:
        from pystoi import stoi
        ref_np = ref.squeeze().cpu().numpy()
        deg_np = deg.squeeze().cpu().numpy()
        score = stoi(ref_np, deg_np, sr, extended=False)
        return score
    except:
        return 0.0


def compute_snr(ref, deg):
    ref = ref.squeeze().cpu().numpy()
    deg = deg.squeeze().cpu().numpy()
    
    noise = ref - deg
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='lightning_logs/exp_6/lightning_model.ckpt')
    parser.add_argument('--val_manifest', type=str,
                       default='data/processed/test_manifest.jsonl')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='eval_outputs/exp_6')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--sr', type=int, default=22050)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    model = CompressionDiffusionModule.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    
    eval_dataset = EvalDataset(args.val_manifest, args.data_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_list = []
    
    with torch.no_grad():
        for idx, (batch, filenames) in enumerate(eval_loader):
            if idx >= args.num_samples:
                break
            
            batch = batch.to(device)
            compressed = model.compression_model(batch)
            
            pesq_score = compute_pesq(batch, compressed, sr=args.sr)
            stoi_score = compute_stoi(batch, compressed, sr=args.sr)
            snr_score = compute_snr(batch, compressed)
            
            ref_filename = str(filenames[0]).replace('/', '_')
            comp_filename = f'compressed_{ref_filename}'
            ref_filepath = output_dir / f'ref_{ref_filename}'
            comp_filepath = output_dir / f'comp_{ref_filename}'
            
            save_audio(batch.cpu(), ref_filepath, sr=args.sr)
            save_audio(compressed.detach().cpu(), comp_filepath, sr=args.sr)
            
            metrics = {
                'filename': ref_filename,
                'pesq': float(pesq_score),
                'stoi': float(stoi_score),
                'snr': float(snr_score)
            }
            metrics_list.append(metrics)
            
            print(f"Sample {idx+1}: PESQ={pesq_score:.3f}, STOI={stoi_score:.3f}, SNR={snr_score:.2f}dB")
    
    avg_metrics = {
        'avg_pesq': float(np.mean([m['pesq'] for m in metrics_list])),
        'avg_stoi': float(np.mean([m['stoi'] for m in metrics_list])),
        'avg_snr': float(np.mean([m['snr'] for m in metrics_list])),
        'samples': metrics_list
    }
    
    save_metrics(avg_metrics, output_dir / 'metrics.json')
    print(f"\nAverage PESQ: {avg_metrics['avg_pesq']:.3f}")
    print(f"Average STOI: {avg_metrics['avg_stoi']:.3f}")
    print(f"Average SNR: {avg_metrics['avg_snr']:.2f}dB")


if __name__ == '__main__':
    main()
