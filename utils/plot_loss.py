import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    plots_dir = ROOT / "plots"
    with open(plots_dir / "loss_curves.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("step (or logged point)")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    (plots_dir / "loss_curves").mkdir(exist_ok=True)
    plt.savefig(plots_dir / "loss_curves" / "train_val_loss.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
