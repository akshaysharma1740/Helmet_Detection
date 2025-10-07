import os
import matplotlib.pyplot as plt
import pandas as pd

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_training_curves(csv_path, output_dir):
    """Plot loss & validation metrics from YOLO results.csv"""
    if not os.path.exists(csv_path):
        print(f"[Warning] {csv_path} not found. Skipping plots.")
        return

    df = pd.read_csv(csv_path)
    ensure_dir(output_dir)

    # --- Loss metrics ---
    plt.figure()
    for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Metrics")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_metrics.png"))
    plt.close()

    # --- Validation metrics ---
    plt.figure()
    for col in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if col in df.columns:
            plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "val_metrics.png"))
    plt.close()

    print(f"[Info] Training plots saved to {output_dir}")
