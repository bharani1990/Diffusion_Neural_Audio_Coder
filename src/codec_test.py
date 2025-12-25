import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.dataset import SpectrogramDataset
from src.model import DiffusionUNet


def load_unet(state_path: Path, in_channels: int = 1) -> DiffusionUNet:
    model = DiffusionUNet(in_channels=in_channels)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    root = ROOT
    test_manifest = root / "data" / "processed" / "test_manifest.jsonl"
    target_frames = 120

    test_ds = SpectrogramDataset(str(test_manifest), target_frames=target_frames)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    saved_unet = root / "saved_models" / "diffusion_unet.pt"
    model = load_unet(saved_unet, in_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    timesteps = 1000
    for i, x0 in enumerate(test_dl):
        x0 = x0.to(device)
        b = x0.size(0)
        t = torch.randint(0, timesteps, (b,), device=device)
        with torch.no_grad():
            noise = torch.randn_like(x0)
            xt = x0 + noise
            pred = model(xt, t)
        print(f"sample {i}: x0 {x0.shape}, pred {pred.shape}")
        if i >= 9:
            break


if __name__ == "__main__":
    main()
