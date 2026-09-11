from pathlib import Path
from urllib.request import Request, urlopen
import os
import tempfile

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "colorize.onnx"
MODEL_URL = os.getenv(
    "DEOLDIFY_MODEL_URL",
    "https://huggingface.co/xrds/deoldify/resolve/main/onnx/model_quantized.onnx",
)


class Colorizer:
    def __init__(self):
        self.session = None
        self.loaded_from_cache = False
        self.downloaded_this_process = False
        self.providers = []

    def _validate_model(self, path: Path) -> None:
        if not path.exists() or path.stat().st_size < 1024:
            raise RuntimeError(f"Downloaded model is missing or empty: {path}")
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            if not session.get_inputs() or not session.get_outputs():
                raise RuntimeError("ONNX model has no usable inputs or outputs")
        except Exception as exc:
            raise RuntimeError(f"Invalid DeOldify ONNX model at {path}: {exc}") from exc

    def _download_model(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="colorize-", suffix=".onnx.download", dir=str(MODEL_DIR)
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            print(f"Downloading DeOldify model to {MODEL_PATH}...")
            request = Request(MODEL_URL, headers={"User-Agent": "AI-Image-Colorizer/1.0"})
            with urlopen(request, timeout=60) as response, temp_path.open("wb") as output:
                total = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\rModel download: {downloaded / total * 100:.1f}%",
                            end="",
                            flush=True,
                        )
            print()
            self._validate_model(temp_path)
            temp_path.replace(MODEL_PATH)
            self.downloaded_this_process = True
            print(f"DeOldify model ready: {MODEL_PATH}")
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def ensure_model(self) -> Path:
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= 1024:
            self._validate_model(MODEL_PATH)
            self.loaded_from_cache = True
            return MODEL_PATH
        self._download_model()
        return MODEL_PATH

    def load(self):
        if self.session is not None:
            return

        model_path = self.ensure_model()
        available = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            raise RuntimeError("ONNX Runtime has no supported execution provider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.providers = self.session.get_providers()

        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        print(f"DeOldify model loaded: {model_path}")
        print(f"Input: {input_meta.name} {input_meta.shape} {input_meta.type}")
        print(f"Output: {output_meta.name} {output_meta.shape} {output_meta.type}")
        print(f"Providers: {self.providers}")

    def colorize(self, image: Image.Image) -> Image.Image:
        self.load()
        original = image.convert("RGB")
        w, h = original.size

        # DeOldify ONNX expects a 256x256, 3-channel float32 image.
        square = ImageOps.fit(
            original,
            (256, 256),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        gray = ImageOps.grayscale(square)
        gray_arr = np.asarray(gray, dtype=np.float32) / 255.0
        tensor = np.stack([gray_arr, gray_arr, gray_arr], axis=0)[None].astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        output = np.asarray(self.session.run(None, {input_name: tensor})[0])
        if output.ndim == 4:
            output = output[0]
        if output.ndim != 3:
            raise RuntimeError(f"Unexpected model output shape: {output.shape}")

        if output.shape[0] == 3:
            out = np.transpose(output, (1, 2, 0))
        elif output.shape[-1] == 3:
            out = output
        else:
            raise RuntimeError(f"Unexpected model output shape: {output.shape}")

        out_min, out_max = float(out.min()), float(out.max())
        if out_min < -0.1 or out_max > 1.1:
            if out_min >= -1.2 and out_max <= 1.2:
                out = (out + 1.0) / 2.0
            else:
                raise RuntimeError(
                    f"Unexpected DeOldify output range: min={out_min:.4f}, max={out_max:.4f}"
                )

        out = np.clip(out, 0.0, 1.0)
        result = Image.fromarray((out * 255).astype(np.uint8), "RGB")
        result = ImageOps.fit(
            result,
            (w, h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return result
