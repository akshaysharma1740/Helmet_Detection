import os
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from src import config

# Load YOLO model
model = YOLO(config.BEST_PT_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # default camera
if not cap.isOpened():
    print("Could not open camera. Check index.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
print("Camera FPS:", fps)

# Output directory
os.makedirs(config.ANNOTATED_DIR, exist_ok=True)

# Optional video save
save_output = True
if save_output:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(config.ANNOTATED_DIR, f"webcam_output_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

print("Recording started...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated = results[0].plot()
    cv2.imshow("YOLO Helmet Detection", annotated)

    if save_output:
        out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
