# AI Image Colorizer

A web UI for colorizing images using a quantized DeOldify ONNX model.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Then open the local URL printed by the application.

## Model

The ONNX model is intentionally not included in this repository. Place `deoldify-quant.onnx` in `models/` before running the colorizer.

## Project structure

- `app/` — FastAPI backend and colorization logic
- `frontend/` — browser UI
- `models/` — local ONNX model location (ignored by Git)
- `input/` — input images (ignored by Git)
- `output/` — generated images (ignored by Git)
- `run.py` — application entry point
- `run.bat` — Windows launcher
