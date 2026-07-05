"""Student 3 - Deep Learning Models for Biometric Authentication.

Live comparison of three real, pretrained, CPU-runnable backbones used as face
embedding extractors:
  - FaceNet (InceptionResnetV1, vggface2)        -> 512-D   (legacy 2015)
  - MobileNetV2 (ImageNet pretrained, repurposed)-> 1280-D  (edge)
  - ResNet18    (ImageNet pretrained, repurposed) -> 512-D   (modern CNN)

For each model we measure: parameters, FLOPs (thop), inference time, embedding
dim, and cosine similarity between two face images of the same person.

The full 2026 landscape (ArcFace, AdaFace, MagFace, EdgeFace, NPTFace) is
provided as a cited comparison table for the literature survey; the live demo
validates the measurement methodology on three runnable models.
"""

from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.face import get_facenet, get_mtcnn, _DEVICE


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class FaceNetWrapper:
    name = "FaceNet (InceptionResnetV1)"
    year = 2015
    dim = 512

    def __init__(self):
        self.model = get_facenet().eval()
        self.mtcnn = get_mtcnn()

    def embed(self, pil: Image.Image) -> tuple[np.ndarray, float]:
        import numpy as _np
        t = self.mtcnn(pil.convert("RGB"))
        if t is None:
            raise ValueError("No face detected")
        if t.ndim == 4:
            t = t[0]
        t = t.unsqueeze(0).to(_DEVICE)
        s = time.perf_counter()
        with torch.no_grad():
            e = self.model(t).cpu().numpy().flatten()
        return e, time.perf_counter() - s

    def params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def flops(self) -> int:
        # thop needs a dummy input matching FaceNet's 160x160 input
        dummy = torch.zeros(1, 3, 160, 160).to(_DEVICE)
        try:
            from thop import profile
            macs, _ = profile(self.model, inputs=(dummy,), verbose=False)
            return int(macs * 2)
        except Exception:
            return 0


class TorchvisionWrapper:
    """Repurpose an ImageNet-pretrained torchvision backbone as an embedding
    extractor (global average pool -> flattened feature vector)."""

    def __init__(self, backbone: str):
        import torchvision.models as tvm
        self.backbone = backbone
        if backbone == "mobilenet_v2":
            net = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            self.dim = 1280
            self.name = "MobileNetV2 (ImageNet)"
            self.year = 2018
            self.input_size = 224
        elif backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(net.children())[:-1],
                                          nn.Flatten())
            self.dim = 512
            self.name = "ResNet18 (ImageNet)"
            self.year = 2016
            self.input_size = 224
        else:
            raise ValueError(backbone)
        self.features = self.features.eval().to(_DEVICE)
        self.mtcnn = get_mtcnn()

    def _prep(self, pil: Image.Image) -> torch.Tensor:
        # use MTCNN to crop+align the face, then resize to backbone input
        face = self.mtcnn(pil.convert("RGB"))
        if face is None:
            # fallback: center square crop -> backbone input size
            import numpy as _np
            from PIL import Image as _Image
            rgb = _np.array(pil.convert("RGB"))
            h, w = rgb.shape[:2]
            size = min(h, w)
            y0 = (h - size) // 2; x0 = (w - size) // 2
            crop = _Image.fromarray(rgb[y0:y0 + size, x0:x0 + size])
            from torchvision.transforms import functional as _F
            face = _F.to_tensor(_F.resize(crop, (self.input_size, self.input_size)))
            face = (face - 0.5) / 0.5
        if face.ndim == 4:
            face = face[0]
        # MTCNN gives 160x160 normalized; resize to backbone size
        face = torch.nn.functional.interpolate(
            face.unsqueeze(0).to(_DEVICE), size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False)
        return face

    def embed(self, pil: Image.Image) -> tuple[np.ndarray, float]:
        t = self._prep(pil)
        s = time.perf_counter()
        with torch.no_grad():
            e = self.features(t).cpu().numpy().flatten()
        return e, time.perf_counter() - s

    def params(self) -> int:
        return sum(p.numel() for p in self.features.parameters())

    def flops(self) -> int:
        dummy = torch.zeros(1, 3, self.input_size, self.input_size).to(_DEVICE)
        try:
            from thop import profile
            macs, _ = profile(self.features, inputs=(dummy,), verbose=False)
            return int(macs * 2)
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(face_a: Image.Image, face_b: Image.Image) -> dict:
    """Run all three models on two face images and collect metrics."""
    results = []
    models = []
    try:
        models.append(FaceNetWrapper())
    except Exception as e:
        results.append({"model": "FaceNet", "error": str(e)})
    for bb in ("mobilenet_v2", "resnet18"):
        try:
            models.append(TorchvisionWrapper(bb))
        except Exception as e:
            results.append({"model": bb, "error": str(e)})

    for m in models:
        row = {"model": m.name, "year": m.year, "params": m.params(),
               "flops": m.flops(), "embedding_dim": m.dim}
        try:
            ea, ta = m.embed(face_a)
            eb, tb = m.embed(face_b)
            sim = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))
            row["inference_time_s"] = round((ta + tb) / 2, 4)
            row["cosine_similarity"] = round(sim, 4)
            row["same_person_hint"] = "high" if sim > 0.5 else "low"
        except Exception as e:
            row["error"] = str(e)
        results.append(row)
    return {"results": results}


# ---------------------------------------------------------------------------
# Cited 2026 landscape table (for the literature survey)
# ---------------------------------------------------------------------------

import pandas as pd

def landscape_table() -> pd.DataFrame:
    """Published numbers from recent papers - cite these in the survey."""
    return pd.DataFrame([
        {"model": "FaceNet (InceptionResnetV1)", "year": 2015, "params_M": 23,
         "flops_G": 1.6, "embedding_dim": 512, "LFW_acc": 99.65, "note": "legacy baseline"},
        {"model": "ArcFace (ResNet50)", "year": 2019, "params_M": 43,
         "flops_G": 5.5, "embedding_dim": 512, "LFW_acc": 99.83, "note": "modern standard"},
        {"model": "ArcFace (ResNet100)", "year": 2019, "params_M": 65,
         "flops_G": 8.3, "embedding_dim": 512, "LFW_acc": 99.83, "note": "InsightFace buffalo_l"},
        {"model": "AdaFace", "year": 2022, "params_M": 65,
         "flops_G": 8.3, "embedding_dim": 512, "LFW_acc": 99.82, "note": "quality-adaptive margin"},
        {"model": "MagFace", "year": 2021, "params_M": 65,
         "flops_G": 8.3, "embedding_dim": 512, "LFW_acc": 99.83, "note": "magnitude-aware"},
        {"model": "EdgeFace", "year": 2023, "params_M": 1.77,
         "flops_G": 0.42, "embedding_dim": 256, "LFW_acc": 99.73, "note": "edge SOTA, <2M params"},
        {"model": "MobileNetV2 (ImageNet)", "year": 2018, "params_M": 3.5,
         "flops_G": 0.3, "embedding_dim": 1280, "LFW_acc": "n/a (repurposed)",
         "note": "edge backbone, not face-trained"},
        {"model": "ResNet18 (ImageNet)", "year": 2016, "params_M": 11.7,
         "flops_G": 1.8, "embedding_dim": 512, "LFW_acc": "n/a (repurposed)",
         "note": "general backbone"},
        {"model": "NPTFace (ViT+PPE 3D)", "year": 2026, "params_M": "~90",
         "flops_G": "~18", "embedding_dim": 512, "LFW_acc": 99.47,
         "note": "CVPR 2026, pose-aligned transformer, SOTA on TinyFace"},
        {"model": "FunFace", "year": 2026, "params_M": 65,
         "flops_G": 8.3, "embedding_dim": 512, "LFW_acc": 99.80,
         "note": "utility-aware margin, strong on low-quality"},
    ])
