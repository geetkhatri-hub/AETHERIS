# AETHERIS — Infrared Image Enhancement, Colorization & Object Detection

AETHERIS is a web-based image-processing application for working with infrared (IR) / thermal images. It provides an interactive dashboard where users can upload an image or select a sample, compare the original monochrome IR image with a colorized result, and inspect object-detection overlays and image-quality metrics.

The project combines a React frontend with a Python FastAPI backend. The backend includes image enhancement utilities, thermal-to-color processing, object detection, and a conditional GAN training implementation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Frontend Setup](#1-frontend-setup)
  - [Backend Setup](#2-backend-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Model Training](#model-training)
- [Dataset Format](#dataset-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Demo Mode and Important Implementation Notes](#demo-mode-and-important-implementation-notes)

## Overview

Infrared images can reveal thermal patterns but are often difficult to interpret visually. AETHERIS provides a dashboard for enhancing and colorizing IR images and comparing the resulting visual output with the original thermal image.

The application also runs object detection on the IR and colorized images. Detection results are displayed as bounding boxes with class labels and confidence values. When a ground-truth image is supplied, the backend can calculate image-quality metrics against that reference.

## Key Features

### 1. Infrared Image Upload

- Upload an image through the dashboard.
- Send the selected image to the backend for processing when the API is available.
- Use the included demo data when the backend is offline.

### 2. IR Image Enhancement

The backend includes an image-enhancement pipeline with:

- Bilateral filtering for edge-preserving denoising.
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement.
- Unsharp masking to improve edge sharpness.

### 3. Thermal Image Colorization

The backend includes a LAB color-mapping pipeline that uses the enhanced IR image as its luminance foundation and maps intensity ranges to color information. A separate pseudo-colorization function is also implemented.

The repository additionally contains an Attention U-Net generator and PatchGAN discriminator for conditional GAN-based IR-to-RGB training.

### 4. Object Detection

The object-detection module includes:

- YOLOv8n-based vehicle detection for cars, motorcycles, buses, and trucks when the Ultralytics model is available.
- Heuristic building detection based on image contours and shapes.
- Heuristic road detection based on edges and line detection.
- Non-maximum suppression to reduce overlapping detections.
- Bounding-box visualization with class labels and confidence scores.

### 5. Before-and-After Image Comparison

The React dashboard includes an interactive comparison slider for viewing:

- Original monochrome infrared image.
- Colorized RGB image.
- Detection boxes on both image layers.

The slider supports mouse and touch interaction.

### 6. Dashboard and Metrics

The dashboard displays:

- System/API connection status.
- Sample satellite/thermal image feeds.
- Object-detection visibility control.
- Detection counts for IR and colorized images.
- PSNR (Peak Signal-to-Noise Ratio).
- SSIM (Structural Similarity Index).

## How It Works

The general processing flow is:

```text
Upload IR Image / Select Sample
              |
              v
       React Frontend
              |
              v
      FastAPI Backend
              |
              v
     Read and Decode Image
              |
              v
      IR Image Enhancement
   (Denoising + CLAHE + Sharpening)
              |
              v
       Color Processing
       (LAB Color Mapping)
              |
              v
       Object Detection
    (YOLOv8 + Image Heuristics)
              |
              v
       Quality Evaluation
       (PSNR + SSIM)
              |
              v
       JSON API Response
              |
              v
     Interactive Dashboard
   (Image Slider + Detection Boxes
          + Metrics)
```

**Note:** The current `/api/colorize` inference route calls the LAB-based `run_model_inference()` function. The Attention U-Net and PatchGAN are implemented in the repository for GAN training, but the current API route does not call the generator for its main colorization result.

## Technology Stack

### Frontend

| Technology | Purpose |
|---|---|
| React 19 | User interface |
| JavaScript (JSX) | Frontend application logic |
| Vite 8 | Development server and build tool |
| CSS | Dashboard styling and responsive layout |

### Backend and Image Processing

| Technology | Purpose |
|---|---|
| Python | Backend and image-processing logic |
| FastAPI | REST API |
| Uvicorn | ASGI server |
| OpenCV | Image processing, filtering, and computer vision |
| NumPy | Numerical and image-array operations |
| Pillow | Image loading and handling |
| PyTorch | Deep-learning model implementation and training |
| Torchvision | PyTorch vision dependencies |
| Ultralytics YOLOv8 | Vehicle detection |
| scikit-image | SSIM metric calculation |
| Matplotlib | Training/visualization dependency |

## Project Structure

```text
AETHERIS-main/
│
├── backend/
│   ├── app.py                       # FastAPI application and inference endpoints
│   ├── dataset.py                   # FLIR paired-image dataset and synthetic fallback
│   ├── dummy.jpg
│   ├── requirements.txt             # Python dependencies
│   │
│   ├── models/
│   │   ├── generator.py             # Attention U-Net generator
│   │   ├── discriminator.py         # PatchGAN discriminator
│   │   └── yolov8_helper.py         # YOLOv8 and heuristic object detection
│   │
│   ├── utils/
│   │   ├── enhancement.py           # IR enhancement utilities
│   │   └── metrics.py               # PSNR and SSIM calculations
│   │
│   └── samples/                     # Sample IR and ground-truth images
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   └── ImageSlider.jsx      # Before/after comparison slider
│   │   ├── assets/
│   │   ├── App.jsx                  # Main dashboard
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
│
├── notebooks/
│   └── colab_training.ipynb         # Google Colab training notebook
│
├── backend.zip                      # Additional backend archive
└── README.md
```

## Getting Started

### Prerequisites

Install the following software:

- Node.js and npm
- Python 3.10 or a compatible Python version
- pip
- Git (optional)

For GPU-based model training or inference, install a PyTorch build compatible with your CUDA environment. CPU execution is also supported by the backend code.

### 1. Frontend Setup

Open a terminal in the project directory and navigate to the frontend:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in the terminal. Vite commonly uses:

```text
http://localhost:5173
```

To create a production build:

```bash
npm run build
```

To preview the production build:

```bash
npm run preview
```

### 2. Backend Setup

Open another terminal from the project root:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it.

**Windows PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Keep the backend terminal running while using the frontend.

## Configuration

The frontend reads the backend URL from the Vite environment variable `VITE_API_URL`.

In `frontend/`, create a `.env` file if you want to override the configured default:

```env
VITE_API_URL=http://localhost:8000
```

Restart the Vite server after changing environment variables.

If `VITE_API_URL` is not set, the frontend code uses this configured backend URL:

```text
https://geet-2406-backend.hf.space
```

The backend attempts to download generator weights from the configured Hugging Face model URL if `backend/models/generator.pth` does not exist. If the weights cannot be loaded, the backend continues with its fallback image-processing path.

## API Reference

The FastAPI backend currently defines the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Not defined as a custom route in the current backend file |
| GET | `/api/samples` | Returns available sample IR and ground-truth images as Base64 data |
| POST | `/api/colorize` | Processes an uploaded IR image and returns enhanced/colorized images, detections, and metrics |

### POST `/api/colorize`

This endpoint accepts multipart form data.

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | Image file | Yes | Input infrared/thermal image |
| `gt_file` | Image file | No | Optional ground-truth/reference image for metric comparison |

The response includes:

- `ir` — original grayscale IR image in Base64 PNG format.
- `enhanced` — enhanced IR image in Base64 PNG format.
- `colorized` — colorized output image in Base64 PNG format.
- `annotated_ir` — IR image with detection boxes.
- `annotated_rgb` — colorized image with detection boxes.
- `detections_ir` — detections from the IR image.
- `detections_rgb` — detections from the colorized image.
- `metrics` — PSNR, SSIM, validation availability, and detection counts.

Example request using cURL:

```bash
curl -X POST "http://localhost:8000/api/colorize" \
  -F "file=@sample_ir.png"
```

Example request with a ground-truth image:

```bash
curl -X POST "http://localhost:8000/api/colorize" \
  -F "file=@sample_ir.png" \
  -F "gt_file=@sample_rgb.png"
```

## Model Training

The backend includes a conditional GAN training implementation.

### Model Architecture

**Generator — Attention U-Net**

- Accepts a single-channel infrared image.
- Produces a three-channel RGB image.
- Uses encoder and decoder blocks with skip connections.
- Uses attention gates to filter skip-connection features.
- Uses a final `Tanh` activation.

**Discriminator — PatchGAN**

- Receives the concatenation of the one-channel IR image and three-channel RGB image.
- Evaluates local image patches as real or generated.
- Uses convolutional layers, Batch Normalization, and LeakyReLU activations.

### Training Method

The training script uses:

- Adversarial loss: `BCEWithLogitsLoss`
- Reconstruction loss: `L1Loss`
- Adam optimizer
- Default learning rate: `0.0002`
- Default batch size: `4`
- Default epochs: `50`
- Default L1 loss weight: `100`

Run training from the `backend` directory:

```bash
python train.py --dataset_path dataset --checkpoint_dir models --epochs 50 --batch_size 4
```

The script saves generator and discriminator checkpoints at the configured intervals and writes final weights as:

```text
backend/models/generator.pth
backend/models/discriminator.pth
```

Training can use a CUDA device when available; otherwise it uses the CPU. GAN training can require substantial memory and processing time.

## Dataset Format

The dataset loader supports paired FLIR-style images. Place the dataset inside the backend directory in this structure:

```text
backend/
└── dataset/
    ├── train/
    │   ├── thermal/
    │   │   ├── image_001.png
    │   │   └── image_002.png
    │   └── rgb/
    │       ├── image_001.png
    │       └── image_002.png
    │
    └── test/
        ├── thermal/
        └── rgb/
```

Thermal and RGB images must have matching filenames so that the loader can pair them.

If the dataset directory is missing or contains no matching pairs, `FLIRDataset` uses its procedural synthetic image generator. This fallback is useful for testing the training pipeline, but synthetic examples should not be treated as a replacement for a real paired training dataset.

## Evaluation Metrics

### PSNR — Peak Signal-to-Noise Ratio

PSNR measures the pixel-level reconstruction difference between a reference image and a processed image. It is expressed in decibels (dB). Higher values generally indicate lower pixel-level error.

### SSIM — Structural Similarity Index

SSIM compares structural similarity between a reference image and a processed image. Values closer to 1 generally indicate greater structural similarity.

For meaningful reference-based evaluation, provide a corresponding ground-truth image using `gt_file`.

## Demo Mode and Important Implementation Notes

- The React frontend contains two built-in local demo samples. It can display these when the backend is unavailable.
- The frontend attempts to contact the configured backend for sample data and image processing.
- The backend's current main colorization route uses LAB-based image processing; it does not invoke the Attention U-Net generator for that route.
- The repository contains GAN architecture and training code, but a trained generator checkpoint is needed to load generator weights successfully.
- YOLOv8n is loaded when the Ultralytics package and model weights are available. If it cannot be loaded, the detector uses its heuristic fallback paths.
- Building and road detections use image-processing heuristics, not a dedicated trained building/road detection model.
- When no ground-truth image is uploaded, the backend returns estimated/randomized PSNR and SSIM values. These should not be interpreted as validated image-quality measurements.
- The backend enables CORS for all origins in its current configuration. Restrict allowed origins before using it in a production deployment.
- The current repository does not include a complete production authentication or user-management system.



## Project Information

**Project Name:** AETHERIS  
**Project Type:** Infrared Image Processing / Computer Vision / Deep Learning  
**Frontend:** React + Vite  
**Backend:** Python + FastAPI  
**Core Areas:** IR Enhancement, Image Colorization, Object Detection, Image Quality Evaluation

---

