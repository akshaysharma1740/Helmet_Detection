import os

BASE_DIR = r"C:\Users\sharm\Downloads\Helmet_Detection"

DATASET_ROOT = r"C:\Users\sharm\Downloads\Helmet_Detection\notebook\construction-helmet-detection-2"
SOURCE = os.path.join(DATASET_ROOT, "test", "images")

DATA_YAML = os.path.join(DATASET_ROOT, "data.yaml")
CLASSES = ["with_helmet", "without_helmet"]
NC = len(CLASSES)

IMG_SIZE = 416
BATCH_SIZE = 10
EPOCHS = 20
LR = 1e-3
CONF = 0.3

# Unified output folder
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "project")
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

BEST_PT_PATH = os.path.join(WEIGHTS_DIR, "best.pt")
