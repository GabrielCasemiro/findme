"""Face detection + embedding, wrapping InsightFace (buffalo_l / ArcFace).

Everything runs locally on CPU. The model auto-downloads to ~/.insightface on first
use. Embeddings are L2-normalized 512-d vectors, so cosine similarity is just a dot
product.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Model is heavy to build and not guaranteed thread-safe; load once, guard with a lock.
_analyzer = None
_lock = threading.Lock()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in original-image pixels
    det_score: float
    embedding: np.ndarray  # (512,) float32, L2-normalized


def get_analyzer():
    """Lazily build the InsightFace analyzer (detection + recognition, CPU)."""
    global _analyzer
    if _analyzer is None:
        with _lock:
            if _analyzer is None:
                from insightface.app import FaceAnalysis

                a = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                )
                a.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 -> CPU
                _analyzer = a
    return _analyzer


def load_image_bgr(path: str | Path) -> np.ndarray | None:
    """Load an image as an OpenCV-style BGR uint8 array, honoring EXIF orientation.

    Returns None if the file can't be decoded.
    """
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # rotate per camera orientation
            im = im.convert("RGB")
            rgb = np.asarray(im)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def detect_faces(image_bgr: np.ndarray, min_score: float = 0.5) -> list[DetectedFace]:
    """Detect all faces in an image and return their boxes + normalized embeddings."""
    analyzer = get_analyzer()
    h, w = image_bgr.shape[:2]
    with _lock:  # serialize model calls; onnxruntime session is shared
        faces = analyzer.get(image_bgr)

    out: list[DetectedFace] = []
    for f in faces:
        if float(f.det_score) < min_score:
            continue
        x1, y1, x2, y2 = f.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        emb = np.asarray(f.normed_embedding, dtype=np.float32)
        out.append(
            DetectedFace(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                det_score=float(f.det_score),
                embedding=emb,
            )
        )
    return out


def embed_primary_face(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return the embedding of the largest (closest) face — used for a selfie query."""
    faces = detect_faces(image_bgr, min_score=0.4)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding


def crop_face(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.35) -> np.ndarray:
    """Crop a face with some margin so the thumbnail shows head + a little context."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
    return image_bgr[cy1:cy2, cx1:cx2]


def save_jpeg(image_bgr: np.ndarray, path: Path, max_side: int = 0, quality: int = 88) -> None:
    """Write a BGR array to JPEG, optionally downscaling so the longest side <= max_side."""
    img = image_bgr
    if max_side:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / longest
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
