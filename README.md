# Pothole Detection System (BSc Engineering Thesis)

[Image of road pothole detection using YOLO bounding boxes]

This repository contains the source code for my Bachelor's Engineering Thesis. The project focuses on developing an automated pothole detection system using deep learning, specifically the state-of-the-art **YOLOv10** object detection model.

## 📌 About the Project
Road maintenance is a critical aspect of urban infrastructure. This project aims to automate the detection of road surface damage (potholes) from images/video using computer vision. 

The system utilizes the **YOLOv10m (Medium)** architecture, fine-tuned on a custom dataset. The training pipeline has been heavily optimized with specific data augmentation strategies to improve the model's robustness in various lighting and weather conditions.

### Key Features
* **Model:** YOLOv10m (Medium)
* **High Resolution:** Trained on 960x960 pixel images for better detection of small road defects.
* **Advanced Augmentation:** Utilizes Mosaic, Mixup, spatial transformations, and HSV color adjustments to prevent overfitting.
* **Hardware Acceleration:** Configured for CUDA (GPU) to drastically reduce training time.

---

## ⚙️ Requirements

To run this project, you need a machine with a CUDA-enabled GPU (highly recommended) and Python installed.

**Dependencies:**
* `torch` (PyTorch with CUDA support)
* `ultralytics` (YOLO framework)

Install the required packages using pip:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install ultralytics
