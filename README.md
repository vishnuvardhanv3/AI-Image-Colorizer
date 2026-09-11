# AI Image Colorizer

An AI-powered image colorization application developed as a deep-learning project for restoring natural-looking color to black-and-white photographs.

## Overview

This project combines an image-colorization model with a lightweight FastAPI backend and browser interface. A grayscale photograph is processed by the colorization pipeline and returned as a full-color image.

### Pipeline

```text
Black-and-white image
        ↓
Image preprocessing
        ↓
Deep-learning colorization model
        ↓
Color reconstruction
        ↓
Colorized image
```

## Model Development

I developed and evaluated the image-colorization pipeline as part of this project. The development process included preparing photographic images as grayscale inputs, preserving their original RGB versions as color targets, training and validating colorization approaches, and integrating the resulting inference pipeline into a usable application.

A dedicated training experiment used a 20,000-image subset of a large photographic dataset:

- 19,000 images for training
- 1,000 images for validation
- RGB photographs converted to grayscale inputs
- Original RGB images used as color targets
- Image captions were not required for the colorization task

The training experiments were used to study how a neural network learns to reconstruct plausible color information from luminance-only images.

## Application

The trained/developed colorization workflow is integrated into a local web application so that the model can be used without requiring an external image-processing service.

### Features

- Upload JPG, JPEG, and PNG photographs
- AI-based automatic colorization
- Local FastAPI inference server
- Browser-based interface
- Automatic model preparation on first use
- Cached model for subsequent runs
- CUDA acceleration when available
- CPU fallback when CUDA is unavailable
- Generated PNG results

## Automatic Model Preparation

The model file is not stored directly in the Git repository. The application prepares the required ONNX model automatically on first use and stores it locally as:

```text
models/colorize.onnx
```

After the first successful preparation, subsequent launches reuse the local cached model instead of downloading it again.

The application validates the ONNX file before using it, preventing incomplete or invalid model files from being loaded.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the application:

```powershell
python run.py
```

Then open the local URL printed by the application.

## Project Structure

```text
AI-Image-Colorizer/
├── app/
│   ├── colorizer.py       # Model loading and inference
│   ├── main.py            # FastAPI application
│   └── __init__.py
├── frontend/
│   ├── index.html         # Web interface
│   ├── app.js             # Frontend logic
│   └── styles.css         # UI styling
├── models/
│   └── colorize.onnx      # Local model cache
├── input/
├── output/
├── requirements.txt
├── run.py
└── run.bat
```

## API

### `GET /health`

Returns the application and model status, including whether the model is available and which execution provider is being used.

### `POST /api/colorize`

Upload a JPG, JPEG, or PNG image and receive a result identifier and result URL.

### `GET /api/result/{job_id}`

Returns the generated colorized PNG image.

## Project Goal

The goal of this project is to turn monochrome photographs into visually convincing color images through deep-learning-based color prediction while keeping the workflow simple enough for local use.

The project focuses on the complete pipeline: dataset preparation, model experimentation, validation, inference, backend integration, and a practical user interface.

## Notes

AI colorization predicts plausible colors; it cannot guarantee the historically correct colors of an original photograph. Results therefore represent learned color reconstruction rather than exact recovery of lost color information.
