import os
from ultralytics import YOLO
from src import config

# Load YOLO model
model = YOLO("yolov8s.pt")

# Train
results = model.train(
    data=config.DATA_YAML,
    epochs=config.EPOCHS,
    imgsz=config.IMG_SIZE,
    batch=config.BATCH_SIZE,
    optimizer="AdamW",
    lr0=config.LR,
    patience=10,
    project=config.OUTPUT_DIR,
    name="project",
    exist_ok=True,
    save_period=5
)

print("\n✅ Training completed successfully!")
print(f"Results saved in: {os.path.join(config.OUTPUT_DIR)}")
