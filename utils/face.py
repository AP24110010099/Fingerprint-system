"""Shared face utilities: detection (MTCNN), cropping, and FaceNet embedding.

A single lazy-loaded FaceNet (InceptionResnetV1, vggface2 pretrained) is reused
across all modules so we only download weights once.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image

_mtcnn = None
_facenet = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mtcnn():
    """Lazy singleton MTCNN face detector.

    Thresholds are lowered from the defaults so detection succeeds on smaller
    / lower-contrast face photos common in anti-spoofing datasets.
    """
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(keep_all=True, post_process=True,
                       thresholds=[0.4, 0.4, 0.4], min_face_size=40,
                       device=_DEVICE)
    return _mtcnn


def get_facenet():
    """Lazy singleton FaceNet (InceptionResnetV1) embedding model."""
    global _facenet
    if _facenet is None:
        from facenet_pytorch import InceptionResnetV1
        _facenet = InceptionResnetV1(pretrained="vggface2").eval().to(_DEVICE)
    return _facenet


def detect_faces(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) bounding boxes for detected faces."""
    mtcnn = get_mtcnn()
    boxes, _ = mtcnn.detect(np.array(image.convert("RGB")))
    if boxes is None:
        return []
    return [tuple(int(v) for v in box) for box in boxes]


def crop_face(image: Image.Image, box: tuple[int, int, int, int],
              margin: float = 0.2) -> Image.Image:
    """Crop a face box with an optional margin (fraction of box size)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = int(w * margin), int(h * margin)
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
    x2 = min(image.width, x2 + dx); y2 = min(image.height, y2 + dy)
    return image.crop((x1, y1, x2, y2))


def get_face_embedding(face_image: Image.Image) -> tuple[np.ndarray, float]:
    """Extract a 512-D FaceNet embedding from a face image.

    Returns (embedding, inference_time_seconds).

    If MTCNN detects a face, it is cropped and aligned. If no face is detected,
    a center square crop is used as a fallback (so the demo still works on
    already-cropped face photos where the detector may miss).
    """
    import time
    from torchvision.transforms import functional as TF
    facenet = get_facenet()
    mtcnn = get_mtcnn()

    start = time.perf_counter()
    rgb = np.array(face_image.convert("RGB"))
    face_tensor = mtcnn(rgb)
    if face_tensor is None:
        # fallback: center square crop -> 160x160 -> normalized tensor
        h, w = rgb.shape[:2]
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        crop = Image.fromarray(rgb[y0:y0 + size, x0:x0 + size])
        face_tensor = TF.to_tensor(TF.resize(crop, (160, 160)))
        # match FaceNet's fixed_image_standardization
        face_tensor = (face_tensor - 0.5) / 0.5
    else:
        if face_tensor.ndim == 4:
            boxes, _ = mtcnn.detect(rgb)
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            face_tensor = face_tensor[int(np.argmax(areas))]
    face_tensor = face_tensor.unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        emb = facenet(face_tensor).cpu().numpy().flatten()
    return emb, time.perf_counter() - start


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))
