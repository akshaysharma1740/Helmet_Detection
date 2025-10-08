# 🪖 Helmet Detection Project

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralyics-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green.svg)

---

## **Project Overview**

This project implements a **helmet detection system** using **YOLOv8**, allowing detection of workers wearing or not wearing helmets. It supports:

- Training on custom helmet datasets
- Inference on test images
- Real-time webcam detection
- Web interface for image upload and prediction

---

## **Problem Statement**

1. Manual monitoring of helmet usage in construction zones is time-consuming, error-prone, and unscalable.
2. Existing automated solutions often fail due to low accuracy and poor handling of live video.

---

## **Solution**

The solution uses **YOLOv8 object detection**, which provides:

- **Accurate detection** of helmets with bounding boxes
- **Fast inference** on images and video streams
- **Web interface** for non-technical users to upload images
- **Real-time webcam detection** for live monitoring

**Key Improvements in This Project**:

1. **Custom-trained YOLOv8 model**: optimized for helmet detection.
2. **Inference module**: saves annotated images in a single output folder, avoiding unnecessary subfolders.
3. **Web upload & API**: users can upload images to get predictions immediately.
4. **Webcam detection**: outputs video with bounding boxes and warnings for missing helmets.
5. **Efficient outputs**: only necessary folders (`weights` and `annotated`) are created.

---

## **Setup**

1. Clone the repository.
2. Install dependencies (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

# Windows

env\Scripts\activate

python -m src.train

python -m src.webfeed(live monitoring)

python -m app.app(for non-technical users to upload images)
