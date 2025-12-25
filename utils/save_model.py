import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_module import DiffusionLightningModule


def get_latest_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under {log_dir}")
    return candidates[-1]


def main():
    root = ROOT
    log_dir = root / "lightning_logs"
    ckpt_path = get_latest_checkpoint(log_dir)

    model = DiffusionLightningModule.load_from_checkpoint(ckpt_path)
    model.eval()

    save_dir = root / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    unet_path = save_dir / "diffusion_unet.pt"
    torch.save(model.model.state_dict(), unet_path)

    full_path = save_dir / "diffusion_lightning_module.ckpt"
    model.save_hyperparameters()
    torch.save({"state_dict": model.state_dict()}, full_path)


if __name__ == "__main__":
    main()
