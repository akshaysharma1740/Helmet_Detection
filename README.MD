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

1. Workplace safety in construction zones requires continuous monitoring of helmet usage.
2. Manual monitoring is time-consuming, error-prone, and cannot scale to large sites.
3. Existing automated solutions using simple image classification have:
   - Low precision and recall for workers without helmets
   - High false positives and false negatives
   - Inefficient handling of live video streams
4. There is a need for a robust, real-time helmet detection system that accurately detects helmet usage in dynamic and complex environments.

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
