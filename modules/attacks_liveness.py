"""Student 4 - Biometric Attacks and Anti-Spoofing (Liveness Detection).

Two liveness detectors are compared on the same face image:
  1. MiniFASNet (Silent-Face-Anti-Spoofing) via ONNX - 3-class (live/print/replay)
     ~0.08 GFLOPs, ~0.4M params, pretrained, CPU.
  2. Heuristic CV2 baseline (Laplacian + LBP + FFT) - the kind of hand-rolled
     detector that is easily fooled, shown for contrast.

Plus an attack taxonomy table (print, replay/display, mask, deepfake,
adversarial) mapped to defences.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "minifasnet"
_MINIFASNET_PATH = _MODELS_DIR / "2.7_80x80_MiniFASNetV2.onnx"

_session = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not _MINIFASNET_PATH.exists():
            raise FileNotFoundError(
                f"MiniFASNet ONNX not found at {_MINIFASNET_PATH}. "
                "Run download_minifasnet() or place the file manually.")
        _session = ort.InferenceSession(
            str(_MINIFASNET_PATH),
            providers=["CPUExecutionProvider"])
    return _session


def _get_new_box(src_w, src_h, bbox, scale):
    """Expand a face box [x,y,w,h] by `scale`, clamping to image bounds.

    Mirrors the upstream Silent-Face-Anti-Spoofing box logic so the 2.7x
    margin matches the '2.7_80x80' training prefix.
    """
    x, y, bw, bh = bbox
    scale = min((src_h - 1) / bh, min((src_w - 1) / bw, scale))
    nw, nh = bw * scale, bh * scale
    cx, cy = bw / 2 + x, bh / 2 + y
    x1, y1 = cx - nw / 2, cy - nh / 2
    x2, y2 = cx + nw / 2, cy + nh / 2
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > src_w - 1: x1 -= x2 - src_w + 1; x2 = src_w - 1
    if y2 > src_h - 1: y1 -= y2 - src_h + 1; y2 = src_h - 1
    return int(x1), int(y1), int(x2), int(y2)


def minifasnet_liveness(face_image: Image.Image,
                        crop_box=None) -> dict:
    """Run MiniFASNetV2 on a face crop. Returns live/spoof + 3-class scores.

    Preprocessing (per upstream Silent-Face-Anti-Spoofing):
      - crop the face with a 2.7x margin box
      - resize to 80x80
      - keep BGR, float32 in [0,255] (NO normalisation to [0,1])
      - NCHW
    Class labels: 0=paper photo, 1=real face, 2=screen photo.
    `crop_box` may be an MTCNN (x1,y1,x2,y2) box; it is converted to
    [x,y,w,h]. If None, the center 80% square is used as a fallback.
    """
    sess = _get_session()
    bgr = cv2.cvtColor(np.array(face_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    src_h, src_w = bgr.shape[:2]
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        bbox = [x1, y1, x2 - x1, y2 - y1]
    else:
        # fallback: center square = 80% of min dimension
        size = int(0.8 * min(src_h, src_w))
        x = (src_w - size) // 2
        y = (src_h - size) // 2
        bbox = [x, y, size, size]
    x1, y1, x2, y2 = _get_new_box(src_w, src_h, bbox, 2.7)
    crop = bgr[y1:y2 + 1, x1:x2 + 1]
    crop = cv2.resize(crop, (80, 80))
    tensor = crop.astype(np.float32).transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    probs = np.exp(out - out.max(axis=1, keepdims=True))
    probs = (probs / probs.sum(axis=1, keepdims=True))[0]
    # 0=paper, 1=real, 2=screen
    classes = ["paper-attack", "live", "replay/screen-attack"]
    idx = int(np.argmax(probs))
    return {
        "status": "live" if idx == 1 else "spoof",
        "predicted_class": classes[idx],
        "live_score": round(float(probs[1]), 4),
        "paper_score": round(float(probs[0]), 4),
        "screen_score": round(float(probs[2]), 4),
        "method": "MiniFASNetV2 (ONNX)",
    }


# ---------------------------------------------------------------------------
# Heuristic baseline (the easily-fooled approach)
# ---------------------------------------------------------------------------

def heuristic_liveness(image: Image.Image) -> dict:
    """Laplacian variance + LBP uniformity + FFT high-frequency ratio."""
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # simplified LBP uniformity
    h, w = gray.shape
    if h >= 3 and w >= 3:
        center = gray[1:-1, 1:-1]
        neigh = [gray[:-2, 1:-1], gray[2:, 1:-1],
                 gray[1:-1, :-2], gray[1:-1, 2:]]
        pat = sum((n >= center).astype(np.uint8) for n in neigh)
        lbp = len(np.unique(pat)) / max(pat.size, 1)
    else:
        lbp = 0.0

    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    hh, ww = gray.shape
    mask = np.zeros_like(fft)
    mask[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4] = 1
    hf_ratio = float(fft[mask == 0].mean() / (fft.mean() + 1e-8))

    score = 0.0
    if lap_var > 100: score += 0.35
    if lbp > 0.3: score += 0.35
    if hf_ratio > 0.05: score += 0.30
    return {
        "status": "live" if score >= 0.5 else "spoof",
        "confidence": round(score * 100, 2),
        "laplacian_var": round(lap_var, 2),
        "lbp_uniformity": round(lbp, 4),
        "hf_ratio": round(hf_ratio, 4),
        "method": "Heuristic (Laplacian+LBP+FFT)",
    }


# ---------------------------------------------------------------------------
# Attack taxonomy
# ---------------------------------------------------------------------------

def attack_taxonomy() -> pd.DataFrame:
    return pd.DataFrame([
        {"attack": "Print attack", "category": "presentation",
         "description": "printed photo of the enrolled face",
         "defence": "liveness (texture/depth), MiniFASNet",
         "difficulty": "easy to mount"},
        {"attack": "Replay / display attack", "category": "presentation",
         "description": "face shown on a phone/laptop screen",
         "defence": "Moire detection, liveness, MiniFASNet",
         "difficulty": "easy"},
        {"attack": "Silicone / 3D mask", "category": "presentation",
         "description": "physical mask of the face",
         "defence": "depth/IR sensor, rPPG heartbeat",
         "difficulty": "hard (needs hardware)"},
        {"attack": "Deepfake", "category": "synthesis",
         "description": "AI-generated face swap / replay",
         "defence": "frequency artefacts, blink/head dynamics, deepfake detectors",
         "difficulty": "medium"},
        {"attack": "Adversarial", "category": "model-level",
         "description": "imperceptible perturbation to fool the matcher",
         "defence": "adversarial training, input randomisation",
         "difficulty": "medium (needs model access)"},
        {"attack": "Template leakage / database breach", "category": "data",
         "description": "stolen biometric template from the server",
         "defence": "cancelable biometrics, fuzzy commitment, encryption",
         "difficulty": "depends on server security"},
        {"attack": "Hill-climbing / brute force", "category": "score-level",
         "description": "iteratively synthesise an image to raise the match score",
         "defence": "rate limiting, liveness, cancelable templates",
         "difficulty": "medium"},
    ])


def defence_comparison_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"method": "MiniFASNetV2 (Silent-Face)", "type": "deep learning",
         "params_M": 0.435, "flops_G": 0.081, "input": "80x80 BGR",
         "classes": "live/print/replay", "val_acc": "~98%",
         "note": "lightweight, CPU, pretrained"},
        {"method": "Heuristic (Laplacian+LBP+FFT)", "type": "handcrafted",
         "params_M": 0, "flops_G": "~0", "input": "any grayscale",
         "classes": "live/spoof", "val_acc": "low (easily fooled)",
         "note": "baseline for contrast"},
        {"method": "rPPG (remote photoplethysmography)", "type": "deep learning",
         "params_M": "varies", "flops_G": "varies", "input": "video",
         "classes": "live/spoof", "val_acc": "high",
         "note": "detects heartbeat from skin colour - needs video"},
        {"method": "Depth/IR sensor", "type": "hardware",
         "params_M": "n/a", "flops_G": "n/a", "input": "depth/IR",
         "classes": "live/spoof", "val_acc": "very high",
         "note": "requires specialised camera"},
    ])
