from pathlib import Path
from io import BytesIO
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from .colorizer import Colorizer, MODEL_PATH

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Image Colorizer", version="1.2.0")
colorizer = Colorizer()

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_present": MODEL_PATH.exists(),
        "model_loaded": colorizer.session is not None,
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "model_type": "DeOldify ONNX",
        "execution_provider": colorizer.providers,
        "loaded_from_cache": colorizer.loaded_from_cache,
        "downloaded_this_process": colorizer.downloaded_this_process,
    }


@app.post("/api/colorize")
async def colorize(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(400, "Please upload a JPG, JPEG, or PNG image.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Maximum image size is 15 MB.")

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Invalid image file.")

    job_id = uuid.uuid4().hex
    path = OUTPUT_DIR / f"{job_id}.png"

    try:
        result = colorizer.colorize(image)
        result.save(path, "PNG")
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Colorization failed: {e}")

    return {"id": job_id, "result_url": f"/api/result/{job_id}"}


@app.get("/api/result/{job_id}")
def result(job_id: str):
    path = OUTPUT_DIR / f"{job_id}.png"
    if not path.exists():
        raise HTTPException(404, "Result not found.")
    return FileResponse(path, media_type="image/png", filename="colorized.png")
