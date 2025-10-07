import os
import uuid
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from src import config, utils
import cv2

app = FastAPI(title="Helmet Detection API")

# Load YOLO model
model = YOLO(config.BEST_PT_PATH)

# Directories
UPLOAD_DIR = os.path.join(config.OUTPUT_DIR, "uploads")
ANNOTATED_DIR = os.path.join(config.OUTPUT_DIR, "annotated")
utils.ensure_dir(UPLOAD_DIR)
utils.ensure_dir(ANNOTATED_DIR)

# Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Serve outputs as static files
app.mount("/static", StaticFiles(directory=ANNOTATED_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """HTML page for image upload"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict helmets on uploaded image"""
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPG/PNG images are supported"}

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    img = cv2.imread(file_path)
    if img is None:
        return {"error": "Failed to read image"}

    results = model.predict(img)
    annotated = results[0].plot()
    out_path = os.path.join(ANNOTATED_DIR, f"{file_id}_annotated.jpg")
    cv2.imwrite(out_path, annotated)

    return FileResponse(out_path, media_type="image/jpeg", filename=f"{file_id}_annotated.jpg")
