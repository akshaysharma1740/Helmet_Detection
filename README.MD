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

Workplace safety is critical in construction zones. Manually monitoring whether workers wear helmets is time-consuming and error-prone. Previous automated solutions using simple image classification were **inaccurate** because:

- Low precision or recall for "without helmet" images
- High false positives/negatives
- Inefficient handling of live video streams

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
pip install ultralytics fastapi uvicorn opencv-python matplotlib pandas jinja2

```

# Windows

env\Scripts\activate
python -m src.train
python -m src.inference
python -m src.webfeed
python -m app.app
