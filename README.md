# Biometric Authentication Demo

A single Streamlit app that demonstrates four biometric-authentication topics
as **working implementations**, not mockups. Built for a group project covering:

1. **Feature Extraction** - handcrafted (Gabor, minutiae, LBP, SIFT) vs deep (FaceNet)
2. **Template Protection** - cancelable biometrics (BioHashing, IoM-URP, Bloom),
   fuzzy commitment, chaotic-map scrambling
3. **Deep Models** - live FLOPs/params/inference-time comparison of FaceNet,
   MobileNetV2, ResNet18 + a cited 2026 landscape table (ArcFace, AdaFace,
   EdgeFace, NPTFace...)
4. **Attacks & Liveness** - MiniFASNet (Silent-Face-Anti-Spoofing) vs a
   heuristic baseline, plus an attack taxonomy

The **Home** page performs *real* face verification (FaceNet embedding + cosine
similarity) - this is the part the original full-stack project faked.

---

## Quick start

```bash
cd biometric-demo

# 1. Install torch (CPU is fine) - already installed on the dev machine
pip install torch

# 2. Install torchvision matching your torch version, WITHOUT pulling the
#    multi-GB CUDA stack:
pip install --no-deps torchvision==0.24.1   # match torch 2.9.x

# 3. The rest:
pip install streamlit opencv-python-headless scikit-image onnxruntime \
            pandas numpy scipy scikit-learn matplotlib pillow reedsolo thop
pip install --no-deps facenet-pytorch      # avoid its numpy<2 pin conflict

# 3a. If you hit "cannot import BaseDefaultEventLoopPolicy" on Python 3.14,
#     upgrade uvloop (a streamlit dependency):
pip install --upgrade uvloop

# 4. Download the MiniFASNet ONNX weights (Student 4):
mkdir -p models/minifasnet
curl -L -o models/minifasnet/2.7_80x80_MiniFASNetV2.onnx \
  https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx

# 5. Run
streamlit run app.py
```

Open the printed URL (usually http://localhost:8501). Use the sidebar to switch
between the five pages.

> **First run** downloads FaceNet (vggface2, ~100 MB) and torchvision ImageNet
> weights automatically. Subsequent runs are fast.

---

## Project layout

```
biometric-demo/
├── app.py                       # Streamlit entry - 5 pages
├── requirements.txt
├── modules/
│   ├── feature_extraction.py    # Student 1
│   ├── template_protection.py   # Student 2
│   ├── deep_models.py           # Student 3
│   └── attacks_liveness.py      # Student 4
├── utils/
│   ├── face.py                  # MTCNN + FaceNet singleton, cosine sim
│   └── viz.py                   # embedding bar/heatmap, image grid
├── data/
│   ├── faces/                   # sample real faces
│   ├── fingerprints/            # sample fingerprint
│   └── spoof/                   # sample spoof (print/screen) faces
└── models/minifasnet/           # ONNX liveness weights (gitignored)
```

---

## What each student owns

Each student writes their **literature survey (15-20 papers)** AND can defend
their working module in a viva. The comparison tables in the app are seeded;
expand them with cited papers.

### Student 1 - Feature Extraction (`modules/feature_extraction.py`)
- Gabor filter bank (8 orientations), minutiae via crossing-number on the
  skeleton, LBP histogram, SIFT keypoints, FaceNet 512-D embedding.
- **Survey table columns:** dataset, method, accuracy, advantages, limitations.
- **Seed refs:** Hong et al. 1998 (Gabor fingerprint), NIST NBIS (minutiae),
  Ahonen 2006 (LBP face), Lowe 2004 (SIFT), Schroff 2015 (FaceNet),
  Tang 2017 (FingerNet), Engelsma 2021 (DeepPrint), Cappelli 2025 (pyfing/SNFEN).

### Student 2 - Template Protection (`modules/template_protection.py`)
- BioHashing, IoM-URP, Bloom filter, fuzzy commitment (Reed-Solomon),
  logistic-map chaotic scrambling. Live revocability + unlinkability demos.
- **Survey table columns:** scheme, security, complexity, computational cost,
  revocation capability, unlinkability.
- **Seed refs:** Teoh 2004 (BioHashing), Jin 2018 (IoM), Rathgeb 2013 (Bloom),
  Juels 2002 (fuzzy commitment/vault), Otroshi 2025 (Springer benchmark),
  BioDeepHash (IEEE TDSC 2026), ISO/IEC 24745.
- **Key point to make:** AES encryption (what the old project did) is
  *reversible* and therefore NOT cancelable - it lacks revocability and
  unlinkability. The schemes above are.

### Student 3 - Deep Models (`modules/deep_models.py`)
- Live measurement of params (count), FLOPs (thop), inference time, embedding
  dim, cosine similarity for FaceNet / MobileNetV2 / ResNet18.
- A cited 2026 landscape table (ArcFace, AdaFace, MagFace, EdgeFace, NPTFace,
  FunFace) with published numbers.
- **Survey table columns:** FLOPs, parameters, accuracy, inference time.
- **Seed refs:** Schroff 2015 (FaceNet), Deng 2019 (ArcFace), Kim 2022 (AdaFace),
  Meng 2021 (MagFace), Boutros 2023 (EdgeFace), Howard 2017 (MobileNet),
  Dosovitskiy 2021 (ViT), Ran 2026 (NPTFace, CVPR), FunFace 2026.

### Student 4 - Attacks & Liveness (`modules/attacks_liveness.py`)
- MiniFASNetV2 (ONNX, ~0.08 GFLOPs, ~0.4M params) 3-class liveness
  (paper / real / screen) vs a heuristic CV2 baseline.
- Attack taxonomy (print, replay, mask, deepfake, adversarial, template
  leakage, hill-climbing) mapped to defences.
- **Survey table columns:** attack type, detection method, dataset, TPR, FPR.
- **Seed refs:** Yu 2020 (Silent-Face-Anti-Spoofing/MiniFASNet),
  CASIA-FASD / Replay-Attack (Idiap) datasets, Rossler 2019 (FaceForensics++),
  Dong 2019 (adversarial face attacks), ISO/IEC 30107 (PAD).

---

## Why this replaced the old full-stack app

The original `Fingerprint-system/` (Next.js + FastAPI + MongoDB + PowerBI) had a
critical bug: the "Face Authentication" page waited 2 seconds and showed
hardcoded `98.7%` for *any* image, and `/login` only checked username+password
- the face was never compared. This app computes a real cosine similarity every
time. Upload a different person's face on the Home page and the score drops
below the threshold.

The old app also never implemented cancelable biometrics (Student 2's topic) -
only reversible AES. This app implements five real protection schemes.

---

## Notes / limitations

- Thresholds (face match 0.6) are defaults, not tuned on a dataset. For a real
  system you would compute EER/FAR/FRR on a benchmark (LFW, IJB-C).
- MobileNetV2 and ResNet18 are ImageNet-pretrained (repurposed as embedding
  extractors) - they are NOT face-trained, so their cosine similarities are
  weaker than FaceNet's. They are included to show the FLOPs/params/time
  methodology on multiple architectures; the cited table covers face-trained
  SOTA.
- The synthetic fingerprint sample is for demonstrating the extraction
  pipeline; upload a real fingerprint image for biometric-meaningful minutiae.
- CPU-only is fine for all models here. First inference is slower (warm-up).
