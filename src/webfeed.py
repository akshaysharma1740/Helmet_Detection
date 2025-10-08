import os
import cv2
from datetime import datetime
from ultralytics import YOLO
from src import config

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = config.BEST_PT_PATH
OUTPUT_DIR = config.ANNOTATED_DIR
SAVE_OUTPUT = True          # Set False to disable saving
CAMERA_INDEX = 0            # Default webcam

# -------------------------
# LOAD YOLO MODEL
# -------------------------
print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------
# INITIALIZE CAMERA
# -------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Could not open camera. Check index or permissions.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 Camera initialized (FPS: {fps}, Size: {frame_width}x{frame_height})")

# -------------------------
# OUTPUT VIDEO SETUP
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_OUTPUT:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"webcam_output_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    print(f"💾 Saving annotated output to: {out_path}")

print("✅ Press 'q' to quit the live feed.")

# -------------------------
# LIVE DETECTION LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received — camera may be disconnected.")
        break

    # YOLO prediction
    results = model.predict(frame, conf=config.CONF, verbose=False)

    # Annotate detections
    annotated = results[0].plot()

    # Display annotated live feed
    cv2.imshow("🪖 YOLO Helmet Detection - Live", annotated)

    # Print detected classes in console
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        print("Detected:", label)

    # Save video frame if enabled
    if SAVE_OUTPUT:
        out.write(annotated)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting live stream...")
        break

# -------------------------
# CLEANUP
# -------------------------
cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
print("✅ Resources released. Program ended.")
