"""Student 1 - Biometric Feature Extraction Techniques.

Handcrafted vs deep feature extraction for fingerprint and face:
  - Gabor filter enhancement (fingerprint)
  - Minutiae detection via binarize -> thin -> crossing number (fingerprint)
  - Local Binary Pattern histogram (fingerprint / face)
  - SIFT keypoints (fingerprint / face)
  - FaceNet 512-D deep embedding (face)

Each function returns numpy arrays / dicts that the Streamlit tab visualises.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.morphology import thin


# ---------------------------------------------------------------------------
# Fingerprint: Gabor enhancement
# ---------------------------------------------------------------------------

def gabor_enhance(gray: np.ndarray,
                  orientations: int = 8,
                  kernel_size: int = 21,
                  sigma: float = 4.0,
                  lamda: float = 10.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply a bank of Gabor filters across `orientations` and combine.

    Returns (enhanced_image, list_of_per_orientation_responses).
    """
    gray = gray.astype(np.float32)
    responses = []
    accum = np.zeros_like(gray)
    for i in range(orientations):
        theta = i * np.pi / orientations
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), sigma, theta, lamda, gamma=0.5,
            psi=0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(resp)
        accum = np.maximum(accum, np.abs(resp))
    enhanced = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced, responses


# ---------------------------------------------------------------------------
# Fingerprint: minutiae detection (crossing-number algorithm)
# ---------------------------------------------------------------------------

def extract_minutiae(gray: np.ndarray,
                     block_size: int = 16) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Detect ridge endings and bifurcations.

    Pipeline: Gabor enhance -> adaptive threshold -> skeletonise -> crossing
    number (CN) on a 3x3 neighbourhood.

    Returns (minutiae_list, skeleton, binary).
    Each minutia: {"type": "ending"|"bifurcation", "x": int, "y": int}.
    """
    enhanced, _ = gabor_enhance(gray)
    # adaptive threshold -> binary ridge image (ridges = 1)
    bin_img = cv2.adaptiveThreshold(
        enhanced, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        block_size * 2 + 1, 2)
    skel = thin(bin_img > 0).astype(np.uint8)

    # crossing number: count 0->1 transitions in the ordered neighbour ring
    pad = np.pad(skel, 1, mode="constant", constant_values=0)
    minutiae: list[dict] = []
    h, w = skel.shape
    for y in range(h):
        for x in range(w):
            if skel[y, x] == 0:
                continue
            p = pad[y:y + 3, x:x + 3]
            # ordered neighbour ring clockwise from top-left
            ring = [p[0, 0], p[0, 1], p[0, 2],
                    p[1, 2], p[2, 2], p[2, 1],
                    p[2, 0], p[1, 0]]
            transitions = sum(1 for i in range(8) if ring[i] == 0 and ring[(i + 1) % 8] == 1)
            if transitions == 1:
                minutiae.append({"type": "ending", "x": int(x), "y": int(y)})
            elif transitions == 3:
                minutiae.append({"type": "bifurcation", "x": int(x), "y": int(y)})
    return minutiae, skel, bin_img


def draw_minutiae(image: np.ndarray, minutiae: list[dict]) -> np.ndarray:
    """Draw minutiae on a colour copy of the image (red=ending, green=bifurcation)."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for m in minutiae:
        color = (0, 0, 255) if m["type"] == "ending" else (0, 255, 0)
        cv2.circle(out, (m["x"], m["y"]), 4, color, -1)
    return out


# ---------------------------------------------------------------------------
# LBP histogram (fingerprint or face)
# ---------------------------------------------------------------------------

def lbp_histogram(gray: np.ndarray, radius: int = 2, n_points: int = 16,
                  n_bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Uniform LBP histogram. Returns (histogram, lbp_image)."""
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist, lbp.astype(np.float32)


# ---------------------------------------------------------------------------
# SIFT keypoints
# ---------------------------------------------------------------------------

def sift_keypoints(gray: np.ndarray, max_keypoints: int = 200) -> tuple[list, np.ndarray]:
    """Detect SIFT keypoints and draw them. Returns (keypoints, drawn_image)."""
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kp, _ = sift.detectAndCompute(gray, None)
    drawn = cv2.drawKeypoints(gray, kp, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, drawn


# ---------------------------------------------------------------------------
# Face: deep embedding via FaceNet
# ---------------------------------------------------------------------------

def face_deep_embedding(face_image: Image.Image) -> tuple[np.ndarray, float]:
    """FaceNet 512-D embedding for a face image."""
    from utils.face import get_face_embedding
    return get_face_embedding(face_image)


# ---------------------------------------------------------------------------
# Comparison table (handcrafted vs deep)
# ---------------------------------------------------------------------------

import pandas as pd

def feature_comparison_table() -> pd.DataFrame:
    """Static comparison of the feature methods demonstrated here."""
    return pd.DataFrame([
        {"method": "Gabor filters", "modality": "fingerprint", "type": "handcrafted",
         "feature_dim": "orientations x filter", "learned": "no",
         "advantage": "ridge enhancement, rotation robust", "limitation": "needs orientation tuning"},
        {"method": "Minutiae (CN)", "modality": "fingerprint", "type": "handcrafted",
         "feature_dim": "#endings + #bifurcations", "learned": "no",
         "advantage": "interpretable, ISO standard", "limitation": "sensitive to noise/quality"},
        {"method": "LBP", "modality": "face/fingerprint", "type": "handcrafted",
         "feature_dim": "histogram bins", "learned": "no",
         "advantage": "fast, texture descriptor", "limitation": "not discriminative alone"},
        {"method": "SIFT", "modality": "face/fingerprint", "type": "handcrafted",
         "feature_dim": "128-D per keypoint", "learned": "no",
         "advantage": "scale/rotation invariant", "limitation": "many keypoints, matching cost"},
        {"method": "FaceNet (InceptionResnetV1)", "modality": "face", "type": "deep",
         "feature_dim": 512, "learned": "yes (vggface2)",
         "advantage": "highly discriminative", "limitation": "needs GPU to train, opaque"},
    ])
