# AI Image Colorizer

A local web application for restoring color to black-and-white photographs using a pretrained DeOldify ONNX colorization model.

## Overview

This project provides a simple browser interface for uploading a JPG, JPEG, or PNG image and generating an AI-colorized result locally through a FastAPI backend.

### Pipeline

```text
Input image
    ↓
Image preprocessing
    ↓
Pretrained DeOldify ONNX model
    ↓
Colorized image
    ↓
Result / download
```

## Model

The production inference pipeline uses the **pretrained DeOldify quantized ONNX model**.

- Model family: DeOldify
- Format: ONNX
- Runtime: ONNX Runtime
- Local filename: `models/colorize.onnx`
- Remote source: `https://huggingface.co/xrds/deoldify/resolve/main/onnx/model_quantized.onnx`
- Execution: CUDA when an ONNX Runtime CUDA provider is available; otherwise CPU

The model binary is intentionally not committed to this repository. On first use, the application downloads it automatically and caches it as `models/colorize.onnx`. Later launches reuse the cached file.

You can override the model source with the `DEOLDIFY_MODEL_URL` environment variable.

## Dataset & Model Development

During development, this project included dataset-based image colorization experiments using the **COCO2017 Image Caption Train** dataset from Kaggle:

https://www.kaggle.com/datasets/seungjunleeofficial/coco2017-image-caption-train

The dataset provides approximately 118K training images. For the colorization experiments, a 20,000-image subset was used:

- Training: 19,000 images
- Validation: 1,000 images
- Captions: not required for the image colorization task

The experiments used RGB photographs to construct grayscale inputs and corresponding color targets for evaluating image colorization approaches.

**Important:** the production model in this repository is the pretrained DeOldify model. The COCO2017 dataset was used for the project's separate dataset-based colorization experiments and evaluation; the DeOldify pretrained weights are not claimed as weights trained by this project.

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

## Automatic Model Download

You do not need to manually place the ONNX model in the repository.

On first startup/use:

1. The application checks `models/colorize.onnx`.
2. If it is missing, the verified DeOldify ONNX model is downloaded.
3. The download is written to a temporary file first.
4. The downloaded ONNX file is validated with ONNX Runtime.
5. It is then stored as `models/colorize.onnx`.
6. Subsequent launches load the cached model without downloading it again.

If the download or validation fails, the application reports the error instead of silently using a partial model.

## Project Structure

- `app/` — FastAPI backend and DeOldify inference logic
- `frontend/` — browser UI
- `models/` — local cached ONNX model location (ignored by Git)
- `input/` — input images (ignored by Git)
- `output/` — generated images (ignored by Git)
- `run.py` — application entry point
- `run.bat` — Windows launcher

## API

### `GET /health`

Returns model availability, load status, model type, execution provider, and whether the model was loaded from cache or downloaded during the current process.

### `POST /api/colorize`

Upload a JPG, JPEG, or PNG image and receive a result ID and result URL.

### `GET /api/result/{job_id}`

Returns the generated colorized PNG.

## Notes

AI-generated colors are predictions and may not be historically accurate. The model is intended for image colorization/restoration rather than forensic reconstruction of original colors.
