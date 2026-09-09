from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "deoldify-quant.onnx"

class Colorizer:
    def __init__(self):
        self.session = None

    def load(self):
        if self.session is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"]
        )

    def colorize(self, image: Image.Image) -> Image.Image:
        self.load()
        original = image.convert("RGB")
        w, h = original.size
        square = ImageOps.fit(original, (256, 256), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
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
                out = (out - out_min) / max(out_max - out_min, 1e-6)
        out = np.clip(out, 0.0, 1.0)
        L = gray_arr
        corr = np.corrcoef(L.flatten(), out[..., 0].flatten())[0, 1]
        if np.isfinite(corr) and corr > 0.75:
            lab_L = L
            a = (out[..., 1] - 0.5) * 2.0
            b = (out[..., 2] - 0.5) * 2.0
            fy = (lab_L * 100.0 + 16.0) / 116.0
            fx = fy + a * 0.5
            fz = fy - b * 0.5
            eps = 216 / 24389
            kap = 24389 / 27
            def finv(t):
                return np.where(t**3 > eps, t**3, (116*t - 16) / kap)
            X = 0.95047 * finv(fx)
            Y = 1.00000 * finv(fy)
            Z = 1.08883 * finv(fz)
            rgb = np.stack([
                3.2406*X - 1.5372*Y - 0.4986*Z,
                -0.9689*X + 1.8758*Y + 0.0415*Z,
                0.0557*X - 0.2040*Y + 1.0570*Z
            ], axis=-1)
            rgb = np.clip(rgb, 0, 1)
        else:
            rgb = out
        result = Image.fromarray((rgb * 255).astype(np.uint8), "RGB")
        result = ImageOps.fit(result, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        return ImageEnhance.Contrast(result).enhance(1.03)
