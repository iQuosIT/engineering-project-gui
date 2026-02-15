# Pothole Detection System (BSc Engineering Thesis)

![Pothole detection bounding boxes example on a road surface]

This repository contains the source code for my Bachelor's Engineering Thesis. The project focuses on developing an automated pothole detection system using deep learning, specifically the state-of-the-art **YOLOv10** object detection model.

## 📌 About the Project
Road maintenance is a critical aspect of urban infrastructure. This project aims to automate the detection of road surface damage (potholes) from images/video using computer vision. 

The system utilizes the **YOLOv10m (Medium)** architecture. The training pipeline has been heavily optimized with specific data augmentation strategies to improve the model's robustness in various lighting and weather conditions.

### Key Features
* **Model:** YOLOv10m (Medium)
* **High Resolution:** Trained on 960x960 pixel images for better detection of small road defects.
* **Advanced Augmentation:** Utilizes Mosaic, Mixup, spatial transformations, and HSV color adjustments to prevent overfitting.
* **Hardware Acceleration:** Configured for CUDA (GPU) to drastically reduce training time.

---

## 📊 Dataset
The model was trained using the **Road Damage Dataset 2022 (RDD2022)**. 

🔗 **Dataset Link:** [RDD-2022 on Kaggle](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022/data)

**Setup Instructions:**
1. Download the dataset from the Kaggle link above.
2. Locate the downloaded archive containing the `RDD_SPLIT` folder.
3. Extract the `RDD_SPLIT` folder directly into the root directory of this project's source code.

---

## ⚙️ Requirements & Hardware

To run this project, you need Python installed and a CUDA-enabled GPU. 

### ⚠️ VRAM Requirements
Because this configuration uses a Medium model (`YOLOv10m`) paired with a high image resolution (`960x960`) and a `batch_size` of 20, it is highly memory-intensive:
* **~24 GB VRAM** (e.g., RTX 3090 / 4090) is recommended to run the default script without modifications.
* **8 GB - 12 GB VRAM:** If you are using a standard GPU, you **must** decrease the `batch_size` in the script to `8` or `4` to avoid `CUDA OutOfMemoryError`.

### Dependencies
* `torch` (PyTorch with CUDA support)
* `ultralytics` (YOLO framework)

Install the required packages using pip:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install ultralytics
