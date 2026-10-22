"""Face boxes from OpenCV YuNet and the soft weight map of the faces loss.

YuNet (face_detection_yunet_2023mar.onnx from opencv_zoo, MIT) runs on the
CPU, on the frame exactly as the batch hands it to the model, after any crop.
The frame is scaled down so its longer side is at most DETECT_SIDE first: a
small face is missed then, which only means that it is weighed like the rest.
Each thread keeps its own detector, so the loader thread and validation on
the main thread never share one.

The model file lives in the user cache, $XDG_CACHE_HOME/whoodeo (default
~/.cache/whoodeo). A missing file is downloaded once from opencv_zoo and its
SHA-256 checked.

The soft mask is 1 inside a box enlarged by `expand` about its center and
falls off as a Gaussian-blurred box at the edges: with sigma = feather times
the shorter side of the enlarged box, the mask is 0.5 on the enlarged edge.
Overlapping faces take the maximum. The weight map is 1 + (weight - 1) x mask,
divided by its own mean per frame, so each frame keeps a mean weight of 1.
"""

import hashlib
import math
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_NAME
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
# YuNet sees the frame with its longer side scaled to at most this many pixels.
DETECT_SIDE = 640
_NMS = 0.3
_TOP_K = 50
# A pixel counts as face for val/psnr_face and the face fraction at mask >= 0.5,
# about the enlarged box.
FACE_LEVEL = 0.5


def model_path():
    cache = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(cache) if cache and Path(cache).is_absolute() else Path.home() / ".cache"
    return base / "whoodeo" / MODEL_NAME


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_model():
    """The YuNet file, downloaded on first use and checked."""
    path = model_path()
    if path.is_file() and _sha256(path) == MODEL_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    print(f"faces: downloading {MODEL_URL}", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            partial.write_bytes(response.read())
    except OSError as exc:
        raise SystemExit(f"faces: cannot download {MODEL_URL}: {exc}") from exc
    found = _sha256(partial)
    if found != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"faces: {MODEL_NAME} has sha256 {found}, expected {MODEL_SHA256}")
    os.replace(partial, path)
    return path


class FaceDetector:
    """YuNet boxes (x, y, w, h, score) in frame pixels, one detector per thread."""

    def __init__(self, score, path=None):
        import cv2  # noqa: F401  (fail here, at startup, when OpenCV is missing)

        self.score = float(score)
        self.path = str(path if path is not None else ensure_model())
        self._local = threading.local()

    def _net(self, width, height):
        import cv2

        local = self._local
        net = getattr(local, "net", None)
        if net is None:
            net = cv2.FaceDetectorYN.create(
                self.path, "", (width, height), self.score, _NMS, _TOP_K,
            )
            local.net = net
            local.size = (width, height)
        elif local.size != (width, height):
            net.setInputSize((width, height))
            local.size = (width, height)
        return net

    def detect(self, rgb):
        """Boxes of one HWC uint8 RGB frame, float32 (N, 5)."""
        import cv2

        height, width = rgb.shape[:2]
        scale = min(1.0, DETECT_SIDE / max(height, width))
        if scale < 1.0:
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        else:
            size = (width, height)
            small = rgb
        bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        _ok, faces = self._net(*size).detect(bgr)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        sx = width / size[0]
        sy = height / size[1]
        boxes = np.stack([
            faces[:, 0] * sx, faces[:, 1] * sy, faces[:, 2] * sx, faces[:, 3] * sy, faces[:, 14],
        ], axis=1).astype(np.float32)
        # Clip to the frame; a box with nothing left inside is dropped.
        x0 = np.clip(boxes[:, 0], 0, width)
        y0 = np.clip(boxes[:, 1], 0, height)
        x1 = np.clip(boxes[:, 0] + boxes[:, 2], 0, width)
        y1 = np.clip(boxes[:, 1] + boxes[:, 3], 0, height)
        boxes = np.stack([x0, y0, x1 - x0, y1 - y0, boxes[:, 4]], axis=1)
        return boxes[(boxes[:, 2] >= 1) & (boxes[:, 3] >= 1)]

    def detect_image(self, image):
        """Boxes of one CHW float frame in 0..1."""
        rgb = (image.detach().clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
        return self.detect(np.ascontiguousarray(rgb.permute(1, 2, 0).cpu().numpy()))


def _soft_span(coords, start, end, sigma):
    """A 0..1 box from start to end, blurred by a Gaussian of `sigma`."""
    scale = 1.0 / (sigma * math.sqrt(2.0))
    return 0.5 * (torch.erf((coords - start) * scale) - torch.erf((coords - end) * scale))


def face_mask(boxes, height, width, expand, feather):
    """Soft face mask, float32 (height, width) in 0..1; zeros without boxes."""
    mask = torch.zeros(height, width)
    if len(boxes) == 0:
        return mask
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    for x, y, w, h, _score in np.asarray(boxes, dtype=np.float64):
        half_w = 0.5 * w * expand
        half_h = 0.5 * h * expand
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        sigma = max(feather * 2.0 * min(half_w, half_h), 0.5)
        span_x = _soft_span(xs, cx - half_w, cx + half_w, sigma)
        span_y = _soft_span(ys, cy - half_h, cy + half_h, sigma)
        torch.maximum(mask, span_y[:, None] * span_x[None, :], out=mask)
    return mask


def detect_batch(detector, images, expand, feather):
    """Boxes and the soft mask (B, 1, H, W) of a CPU batch (B, 3, H, W)."""
    _batch, _channels, height, width = images.shape
    boxes = [detector.detect_image(image) for image in images]
    masks = torch.stack([face_mask(found, height, width, expand, feather) for found in boxes])
    return boxes, masks.unsqueeze(1)


def weight_map(mask, weight):
    """1 + (weight - 1) x mask, divided by its mean per frame (mean weight 1)."""
    raw = 1.0 + (weight - 1.0) * mask
    return raw / raw.mean(dim=(-2, -1), keepdim=True)


def face_fraction(mask):
    """Share of pixels at mask >= FACE_LEVEL, over the whole batch."""
    return (mask >= FACE_LEVEL).float().mean()
