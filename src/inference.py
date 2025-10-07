import os
import random
import cv2
from ultralytics import YOLO
from src import config, utils

def run_inference(weights=config.BEST_PT_PATH, source=config.SOURCE, n_samples=10):
    """Run YOLOv8 inference on a folder of images and save annotated outputs."""
    utils.ensure_dir(config.ANNOTATED_DIR)

    print(f"[INFO] Loading model from: {weights}")
    model = YOLO(weights)

    print(f"[INFO] Running inference on: {source}")
    preds = model.predict(
        source=source,
        imgsz=config.IMG_SIZE,
        conf=config.CONF,
        save=True,
        project=config.ANNOTATED_DIR,
        name="inference_results",
        exist_ok=True
    )

    # Save random annotated overlays (optional)
    sampled_preds = random.sample(preds, min(n_samples, len(preds)))
    for pred in sampled_preds:
        img_overlay = pred.plot()
        out_name = os.path.splitext(os.path.basename(pred.path))[0] + "_overlay.jpg"
        cv2.imwrite(os.path.join(config.ANNOTATED_DIR, out_name), img_overlay)

    print(f"[INFO] ✅ Inference completed! Annotated images saved in: {config.ANNOTATED_DIR}")


if __name__ == "__main__":
    run_inference()
