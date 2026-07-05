# Project Documentation — Biometric Authentication Demo

> **Group project deliverable.** This document explains what we built, why we
> built it this way, how it maps to the four topics our professor assigned, what
> we deviated from the original full-stack plan (and why), and step-by-step
> instructions for each student to use, demonstrate, and submit their part.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Original Project Brief](#2-the-original-project-brief)
3. [What the Team First Built (and Why It Went Wrong)](#3-what-the-team-first-built-and-why-it-went-wrong)
4. [The Pivot — Why a Streamlit App Instead of a Full-Stack App](#4-the-pivot--why-a-streamlit-app-instead-of-a-full-stack-app)
5. [What We Actually Built](#5-what-we-actually-built)
6. [Mapping to the Professor's Four Topics](#6-mapping-to-the-professors-four-topics)
7. [Deviations from the Original Project (and Justifications)](#7-deviations-from-the-original-project-and-justifications)
8. [Setup & Run Instructions](#8-setup--run-instructions)
9. [Per-Student Guide: Use, Demonstrate, Submit](#9-per-student-guide-use-demonstrate-submit)
   - [Student 1 — Feature Extraction](#student-1--feature-extraction)
   - [Student 2 — Template Protection](#student-2--template-protection)
   - [Student 3 — Deep Learning Models](#student-3--deep-learning-models)
   - [Student 4 — Attacks & Liveness](#student-4--attacks--liveness)
10. [The Literature Survey (15–20 Papers Each)](#10-the-literature-survey-1520-papers-each)
11. [Demo Script for the Viva / Final Presentation](#11-demo-script-for-the-viva--final-presentation)
12. [Known Limitations & Honest Caveats](#12-known-limitations--honest-caveats)
13. [File-by-File Reference](#13-file-by-file-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Glossary](#15-glossary)

---

## 1. Executive Summary

We were assigned a group project on **biometric authentication**, split across
four students covering feature extraction, template protection, deep learning
models, and attacks/liveness detection. Each student was also required to write
a literature survey of 15–20 papers with a comparison table.

Our first attempt was a full-stack web app (Next.js + FastAPI + MongoDB +
PowerBI + Docker + Railway/Render + four ML model families). It was far too
ambitious for second-year students, and it had a critical security bug: the
"Face Authentication" page was a **fake** — it waited 2 seconds and showed
hardcoded `98.7%` for *any* image, and the `/login` endpoint only checked
username + password, never comparing faces. Anyone with the password could log
into any account. The template-protection topic (Student 2) was not implemented
at all — only reversible AES encryption, which is not cancelable biometrics.

We rebuilt the project as a **single Streamlit app** (`biometric-demo/`) with
one page per student, each running **real, working code** — not mockups. The old
full-stack code is left untouched in `Fingerprint-system/` for reference. The
new app:

- Actually verifies faces (FaceNet embedding + cosine similarity).
- Implements five real template-protection schemes (BioHashing, IoM-URP, Bloom
  filter, fuzzy commitment, chaotic-map scrambling).
- Live-measures FLOPs, parameters, inference time, and accuracy for three real
  pretrained models.
- Runs a real pretrained liveness detector (MiniFASNet, ~600 KB) that correctly
  classifies real vs print/screen-attack faces.

Everything runs on CPU, no GPU required.

---

## 2. The Original Project Brief

Our professor assigned four sub-topics, each to one student, each with a
literature survey of 15–20 papers and a comparison table.

| Student | Topic | Areas to Explore | Comparison Table Columns |
|---|---|---|---|
| 1 | Feature Extraction | Minutiae, LBP, Gabor, SIFT/SURF, CNN, deep embeddings | dataset, method, accuracy, advantages, limitations |
| 2 | Template Protection | Cancelable biometrics, BioHashing, chaotic mapping, fuzzy extractor, homomorphic encryption, ECC | security, complexity, computational cost, revocation capability |
| 3 | Deep Learning in Biometrics | CNN, ResNet, MobileNet, Vision Transformers, Autoencoders | FLOPs, parameters, accuracy, inference time |
| 4 | Attacks & Liveness | Spoofing, presentation attacks, adversarial attacks, liveness detection, deepfake detection | (taxonomy of attacks and defences) |

**Expected learning outcomes:** difference between handcrafted and deep
features; non-invertibility, revocability, unlinkability; model architecture
trade-offs (accuracy vs complexity, edge deployment); real-world biometric
risks and security challenges.

---

## 3. What the Team First Built (and Why It Went Wrong)

The first attempt lives in `Fingerprint-system/` (the parent folder). It was a
full-stack application:

| Layer | Technology |
|---|---|
| Frontend | Next.js, React, Tailwind CSS (11+ pages) |
| Backend | FastAPI (Python), multiple route files |
| Database | MongoDB (encrypted templates) + a duplicate SQLAlchemy layer |
| Face models | FaceNet, Vision Transformer |
| Fingerprint models | ResNet, MobileNet |
| Liveness | CV2 heuristics (Laplacian, LBP, FFT) |
| Analytics | Power BI |
| Deploy | Docker, Railway, Render |

### The critical bug — "any face logs into my account"

The "Face Authentication" page in
`frontend/pages/face-authentication.jsx` did this:

```js
const handleVerify = (e) => {
  e.preventDefault();
  setStatus("processing");
  setTimeout(() => {                       // 2-second wait
    setStatus("complete");
    setResults({
      accuracy: model === "FaceNet" ? "98.7%" : "99.2%",   // HARDCODED
      similarityScore: model === "FaceNet" ? "97.4%" : "98.1%",
      ...
    });
  }, 2000);
};
```

It **never called the backend**. It waited 2 seconds and showed hardcoded
"98.7% / success" no matter what image you uploaded — your face, a friend's
face, a photo of a cat. The fingerprint page was identical.

On top of that, the actual `/login` endpoint (`backend/routes/auth.py`) only
checked **username + password**:

```python
def login(request: LoginRequest):
    user = get_user_by_username(request.username)
    if not user or not verify_password(...):
        raise HTTPException(401, ...)
    token = create_access_token(...)   # no face check anywhere
```

So the face uploaded at signup was stored but **never compared at login**. The
"face scan" was either the fake page above or just a camera preview — never a
real verification gate.

### The irony

The backend **did** have real verification logic in
`backend/services/verification.py` (`verify_face` actually extracts an
embedding, compares cosine similarity to the stored template, and checks a
threshold). The team built the right engine — they just **never wired the UI to
it**.

### Other structural problems

1. **Duplicate/competing code paths.** `routes/verify.py` used SQLAlchemy +
   `FaceData`/`FingerprintData` tables, while `services/verification.py` used
   MongoDB. Two parallel implementations of the same thing.
2. **Template protection was missing.** Student 2's entire topic (cancelable
   biometrics, BioHashing, fuzzy extractors, homomorphic encryption) was not
   implemented at all. There was only `encryption.py` doing AES on the stored
   embedding. AES is reversible encryption — it is *not* cancelable biometrics
   (no revocability, no unlinkability).
3. **Liveness was weak.** The default used Laplacian variance + a simplified
   LBP + FFT high-frequency ratio. These heuristics are easily fooled by a
   decent phone photo of a face.
4. **Thresholds were unvalidated.** `face_match_threshold=0.6` was a guess,
   never tuned on a dataset. No EER/FAR/FRR evaluation anywhere.
5. **Scope was too big.** Next.js + FastAPI + MongoDB + PowerBI + Docker +
   Railway + Render + 4 ML model families + liveness. Each alone is a week of
   learning for second-year students. Combined, it became copy-paste code no
   one understood.

---

## 4. The Pivot — Why a Streamlit App Instead of a Full-Stack App

We chose to rebuild as a single Python Streamlit app for these reasons:

- **One language, one file per demo.** No React, no FastAPI, no MongoDB, no
  CORS, no deployment YAML.
- **Camera/upload widgets are built-in.** Streamlit has `st.camera_input` for
  webcam capture and `st.file_uploader` for images.
- **It demonstrates the actual ML** instead of hiding it behind a fake UI.
- **Each student gets one self-contained module** they fully understand and can
  defend in a viva.
- **Deployable free** on Streamlit Community Cloud or Hugging Face Spaces — no
  Docker/Railway/Render needed.
- **CPU-only is fine** for all the lightweight models we use.

This removed ~80% of the places the team was getting stuck and let us focus on
the actual biometric concepts the professor wants us to learn.

---

## 5. What We Actually Built

A new `biometric-demo/` folder alongside the old code:

```
biometric-demo/
├── app.py                       # Streamlit entry — 5 pages
├── requirements.txt
├── README.md
├── DOCUMENTATION.md             # this file
├── .gitignore
├── modules/
│   ├── feature_extraction.py    # Student 1
│   ├── template_protection.py   # Student 2
│   ├── deep_models.py           # Student 3
│   └── attacks_liveness.py      # Student 4
├── utils/
│   ├── face.py                  # MTCNN + FaceNet singleton, cosine sim
│   └── viz.py                   # embedding bar/heatmap, image grid
├── data/
│   ├── faces/                   # sample real faces (T1, T2)
│   ├── fingerprints/            # sample synthetic fingerprint
│   └── spoof/                   # sample spoof (print/screen) faces (F1, F2)
└── models/minifasnet/           # ONNX liveness weights (gitignored)
    └── 2.7_80x80_MiniFASNetV2.onnx
```

### Technology choices (all 2026-current, all CPU-friendly)

| Component | Choice | Why |
|---|---|---|
| App framework | Streamlit 1.58 | Easiest for beginners; built-in camera/tables/charts |
| Face detection | MTCNN (via `facenet-pytorch`) | Bundled, pretrained, no extra download |
| Face embedding | FaceNet (InceptionResnetV1, vggface2) | Real pretrained, 512-D, the legacy baseline every survey cites |
| Liveness | MiniFASNetV2 (Silent-Face-Anti-Spoofing) via ONNX | ~600 KB, ~0.08 GFLOPs, ~98% val acc, CPU, pretrained |
| Fingerprint features | OpenCV + scikit-image (Gabor, minutiae, LBP, SIFT) | Handcrafted baselines every survey cites |
| Template protection | NumPy + `reedsolo` | BioHashing/IoM/Bloom/fuzzy-commitment/chaotic-map in pure Python |
| FLOPs/params | `thop` | Standard measurement tool |
| Backbones for comparison | torchvision MobileNetV2 + ResNet18 | Pretrained, CPU-runnable, no CUDA stack needed |

### What we deliberately avoided

- **`timm` / ViT from HuggingFace** — would have pulled ~2 GB of NVIDIA CUDA
  packages on this CPU-only machine. The 2026 ViT SOTA (NPTFace, CVPR 2026) is
  cited in the comparison table instead.
- **MongoDB / FastAPI / React / PowerBI / Docker / Railway / Render** — all
  removed. Replaced by one Python app.
- **Model training** — everything uses pretrained weights. No training runs
  needed.

---

## 6. Mapping to the Professor's Four Topics

| Professor's requirement | Where it lives in our app | Status |
|---|---|---|
| **S1: Minutiae extraction** | `modules/feature_extraction.py` → `extract_minutiae()` (binarize → thin → crossing number) | ✅ Working |
| **S1: LBP** | `modules/feature_extraction.py` → `lbp_histogram()` (scikit-image uniform LBP) | ✅ Working |
| **S1: Gabor filters** | `modules/feature_extraction.py` → `gabor_enhance()` (8-orientation bank) | ✅ Working |
| **S1: SIFT/SURF** | `modules/feature_extraction.py` → `sift_keypoints()` (OpenCV SIFT) | ✅ Working |
| **S1: CNN-based / deep embeddings** | `utils/face.py` → FaceNet 512-D embedding | ✅ Working |
| **S1: handcrafted vs deep** | Comparison table on the Feature Extraction tab | ✅ Working |
| **S2: Cancelable biometrics** | `modules/template_protection.py` → BioHashing, IoM-URP, Bloom | ✅ Working |
| **S2: BioHashing** | `biohash()` | ✅ Working |
| **S2: Chaotic mapping** | `chaotic_scramble()` (logistic map) | ✅ Working |
| **S2: Fuzzy extractor** | `fuzzy_commit()` / `fuzzy_commit_decode()` (Reed-Solomon) | ✅ Working |
| **S2: Non-invertibility / revocability / unlinkability** | `revocability_demo()`, `unlinkability_demo()`, comparison table | ✅ Working |
| **S2: Homomorphic encryption / ECC** | Cited in survey table (too heavy to implement for a demo) | 📖 Survey only |
| **S3: CNN / ResNet / MobileNet** | `modules/deep_models.py` → FaceNet, MobileNetV2, ResNet18 | ✅ Working |
| **S3: Vision Transformers / Autoencoders** | Cited in 2026 landscape table (NPTFace CVPR-2026, ViT) | 📖 Survey only |
| **S3: FLOPs / params / accuracy / inference time** | Live-measured on the Deep Models tab | ✅ Working |
| **S3: accuracy vs complexity, edge deployment** | Comparison table + EdgeFace cited | ✅ Working |
| **S4: Spoofing / presentation attacks** | Attack taxonomy table on the Attacks & Liveness tab | ✅ Working |
| **S4: Liveness detection** | MiniFASNet (real pretrained) + heuristic baseline | ✅ Working |
| **S4: Adversarial / deepfake attacks** | Attack taxonomy table (cited) | 📖 Survey + taxonomy |
| **S4: taxonomy of attacks and defences** | `attack_taxonomy()` table | ✅ Working |

Legend: ✅ = implemented and running in the app. 📖 = covered in the literature
survey / cited table (not implemented because it needs hardware, video, or
training that's out of scope for a second-year demo).

---

## 7. Deviations from the Original Project (and Justifications)

| Original plan | What we did instead | Why |
|---|---|---|
| Next.js + FastAPI + MongoDB full-stack app | Single Streamlit Python app | Second-year students; full-stack scope caused the fake-UI bug and nobody understood the whole pipeline |
| React "Face Authentication" page (hardcoded results) | Real FaceNet cosine-similarity verification on the Home tab | The original was literally a mockup that let any face in |
| `/login` checks username+password only | Home tab enrols a face and verifies a face against it | The original never compared faces at login |
| AES encryption as "template protection" | Five real cancelable-biometric schemes (BioHashing, IoM-URP, Bloom, fuzzy commitment, chaotic map) | AES is reversible → not cancelable; Student 2's topic was effectively missing |
| CV2 heuristic liveness only | MiniFASNet (pretrained, ONNX) + heuristic baseline for contrast | Heuristics are easily fooled; MiniFASNet is the real lightweight SOTA-ish baseline |
| FaceNet vs ViT for faces | FaceNet vs MobileNetV2 vs ResNet18 (live) + cited 2026 table (ArcFace, AdaFace, MagFace, EdgeFace, NPTFace, FunFace) | ViT/timm would pull ~2 GB of CUDA packages on a CPU machine; the cited table covers the 2026 ViT SOTA |
| ResNet vs MobileNet for fingerprints | Gabor + minutiae + LBP + SIFT (handcrafted) for fingerprints | Fingerprint deep models (DeepPrint, FingerNet) need training data we don't have; handcrafted features are the canonical survey baselines |
| PowerBI analytics | Streamlit built-in charts/tables | Removes a whole external service |
| Docker / Railway / Render deploy configs | `streamlit run app.py` (or free Streamlit Community Cloud) | Removes deployment complexity entirely |
| Training scripts + dataset splits | Pretrained weights only, no training | Training was never going to work without a real dataset and GPU time |

**What we kept from the original:**

- The `facenet-pytorch` dependency and the FaceNet wrapper concept (it worked).
- The Gabor filter code concept (`fingerprint_gabor.py`).
- The cosine-similarity + threshold verification concept (now actually wired to
  the UI).
- The liveness heuristic as a **baseline to compare against** MiniFASNet.

---

## 8. Setup & Run Instructions

### Prerequisites

- Python 3.10+ (tested on Python 3.14.2)
- ~1 GB free disk space (for model weights cached on first run)
- A webcam (optional — for the Home tab's `st.camera_input`; you can also
  upload image files everywhere)

### Install

```bash
cd biometric-demo

# 1. Torch (CPU is fine) — usually already installed
pip install torch

# 2. torchvision matching your torch version, WITHOUT the multi-GB CUDA stack
pip install --no-deps torchvision==0.24.1   # match torch 2.9.x

# 3. The rest
pip install streamlit opencv-python-headless scikit-image onnxruntime \
            pandas numpy scipy scikit-learn matplotlib pillow reedsolo thop
pip install --no-deps facenet-pytorch      # avoid its numpy<2 pin conflict

# 3a. If you hit "cannot import BaseDefaultEventLoopPolicy" on Python 3.14,
#     upgrade uvloop (a streamlit dependency):
pip install --upgrade uvloop

# 4. Download the MiniFASNet ONNX weights (Student 4)
mkdir -p models/minifasnet
curl -L -o models/minifasnet/2.7_80x80_MiniFASNetV2.onnx \
  https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx
```

### Run

```bash
streamlit run app.py
```

Open the printed URL (usually http://localhost:8501). Use the sidebar to switch
between the five pages.

> **First run** downloads FaceNet (vggface2, ~107 MB) and torchvision ImageNet
> weights (~44 MB ResNet18, ~13 MB MobileNetV2) automatically. Subsequent runs
> are fast.

---

## 9. Per-Student Guide: Use, Demonstrate, Submit

Each student **owns one module** in `modules/`. You are responsible for:

1. Understanding your module's code (read it — it's commented).
2. Writing your **literature survey (15–20 papers)** with the comparison table
   the professor specified.
3. Demonstrating your tab live in the viva/presentation.
4. Being able to answer questions about your module's code and your survey.

---

### Student 1 — Feature Extraction

**Your file:** `modules/feature_extraction.py`
**Your app tab:** "Feature Extraction (S1)"

#### What your module does

- **Gabor filters** — applies an 8-orientation Gabor bank to a fingerprint and
  combines the maximum responses. This enhances ridges.
- **Minutiae detection** — binarizes the enhanced fingerprint, thins it to a
  1-pixel skeleton, then computes the **crossing number (CN)** at each ridge
  pixel: CN=1 → ending, CN=3 → bifurcation. This is the classic algorithm every
  survey cites.
- **LBP histogram** — uniform Local Binary Pattern histogram (scikit-image).
  A texture descriptor.
- **SIFT keypoints** — scale-invariant feature transform keypoints (OpenCV).
- **FaceNet embedding** — 512-D deep face embedding (the "deep" side of the
  handcrafted-vs-deep comparison).

#### How to demo it

1. Open the app → sidebar → "Feature Extraction (S1)".
2. **Fingerprint tab:** leave "Use a sample fingerprint" checked (or upload your
   own). Click through the visualizations: original → Gabor enhanced → minutiae
   map (red dots = endings, green = bifurcations) → LBP histogram → SIFT
   keypoints.
3. **Face (deep) tab:** upload a face photo. See the 512-D FaceNet embedding as
   a bar chart and a heatmap.
4. **Comparison tab:** show the handcrafted-vs-deep table.

#### What to say in the viva

- "Handcrafted features (Gabor, minutiae, LBP, SIFT) are interpretable and
  cheap, but need tuning and aren't very discriminative on their own."
- "Deep features (FaceNet 512-D) are learned, highly discriminative, but
  opaque and need a GPU to train."
- "Minutiae is the ISO/IEC 19794-2 standard for fingerprint matching — it's
  what real AFIS systems use."
- Be ready to explain the crossing-number algorithm: walk the 8 neighbours of
  a skeleton pixel in order, count 0→1 transitions. 1 transition = ending,
  3 = bifurcation.

#### Your survey (15–20 papers)

**Required table columns:** dataset, method, accuracy, advantages, limitations.

**Seed references (start here, then find more):**
- Hong et al. 1998 — Gabor filter fingerprint enhancement (classic)
- NIST NBIS / MINDTCT — minutiae detection reference implementation
- Ahonen et al. 2006 — LBP for face recognition
- Lowe 2004 — SIFT (original)
- Schroff et al. 2015 — FaceNet (deep embedding)
- Tang et al. 2017 — FingerNet (deep minutiae)
- Engelsma et al. 2021 — DeepPrint (deep fingerprint)
- Cappelli et al. 2025 — pyfing / SNFEN (modern deep fingerprint enhancement)

**Where to find more:** Google Scholar search "fingerprint minutiae extraction
deep learning 2024 2025", "face embedding ArcFace 2026", "LBP face recognition
survey".

#### How to submit your part

- A PDF with your 15–20 paper survey + the comparison table.
- A short write-up (1–2 pages) explaining your module's code and the demo.
- (Optional) Screenshots of the app running your tab.

---

### Student 2 — Template Protection

**Your file:** `modules/template_protection.py`
**Your app tab:** "Template Protection (S2)"

#### What your module does

This is the topic that was **completely missing** from the original project
(which only had reversible AES encryption). You implement five real schemes:

1. **BioHashing** (Teoh et al.) — random Gaussian projection of the embedding,
   then sign-threshold to a binary template. Different key → different
   projection → different template.
2. **IoM-URP** (Jin et al.) — Index-of-Max of uniform random projections. Keep
   the argmax index of each projection as the template.
3. **Bloom filter** (Rathgeb et al.) — quantise the embedding into blocks, map
   each block value to a bloom codeword via a keyed random table, OR them.
4. **Fuzzy commitment** (Juels & Wattenberg) — bind a secret to the binarised
   embedding using Reed-Solomon error-correcting codes. The secret can be
   recovered only if the query embedding is "close enough" to the enrolment
   embedding.
5. **Chaotic-map scrambling** — permute the embedding elements using a logistic
   map keystream (`x_{n+1} = r·x_n(1-x_n)`).

#### The three properties you must explain

- **Non-invertibility** — you cannot recover the original embedding from the
  protected template (unlike AES, which is reversible with the key).
- **Revocability** — if a template is compromised, you issue a new key and get
  a completely new template from the same face. The face itself doesn't change.
- **Unlinkability** — two templates of the same face under different keys are
  uncorrelated (~0.5 Hamming distance), so they can't be linked.

#### How to demo it

1. Open the app → sidebar → "Template Protection (S2)".
2. Upload a face photo. The app extracts a 512-D FaceNet embedding.
3. Pick a scheme (e.g. BioHashing) and a key (seed). Click "Apply protection".
   Show the protected template (a binary string, very different from the
   embedding).
4. Click "Run revocability demo" — show the table of Hamming distances between
   templates under different keys. The mean should be ~0.5 (uncorrelated =
   revocable + unlinkable).
5. Try the Fuzzy commitment scheme — show that the same face recovers the
   secret (`BIO-SECRET-2026`) but a different face does not.
6. Show the comparison table at the bottom — note the row for AES (the old
   project's approach) marked "revocability: no (decrypt → original)".

#### What to say in the viva

- "AES encrypts the template, but it's reversible — if the key leaks, the
  biometric is gone forever. Cancelable biometrics are non-invertible: even
  with the key, you can't recover the original embedding."
- "Revocability: a compromised template can be replaced with a new key without
  changing the biometric. You can't change your face, but you can change the
  template."
- "Fuzzy commitment is a biometric cryptosystem — it binds a cryptographic key
  to the biometric, and the key is recovered only if the query is close enough
  (Reed-Solomon error correction handles the intra-class variation)."
- "ISO/IEC 24745 is the standard that defines these requirements."

#### Your survey (15–20 papers)

**Required table columns:** security, complexity, computational cost,
revocation capability (and add unlinkability).

**Seed references:**
- Teoh et al. 2004 — BioHashing (two-factor authentication)
- Jin et al. 2018 — Index-of-Max hashing
- Rathgeb et al. 2013 — Bloom filter biometric templates
- Juels & Wattenberg 2002 — Fuzzy commitment
- Juels & Sudan 2002 — Fuzzy vault
- Otroshi et al. 2025 — Benchmarking cancelable biometrics for deep templates
  (Springer J. Image & Video Processing) — **this is the key 2025 paper; it
  benchmarks exactly the schemes you implemented**
- BioDeepHash 2026 — IEEE TDSC, deep hashing + cryptographic hash for
  consistent templates
- ISO/IEC 24745 — Biometric information protection standard
- Comprehensive survey on cancelable biometrics (Kumar 2019, AI Review)

#### How to submit your part

- A PDF with your 15–20 paper survey + the comparison table.
- A 1–2 page write-up explaining each scheme, the three properties, and why
  AES is not cancelable.
- (Optional) Screenshots of the revocability demo table.

---

### Student 3 — Deep Learning Models

**Your file:** `modules/deep_models.py`
**Your app tab:** "Deep Models (S3)"

#### What your module does

Live-measures four things for three real pretrained backbones used as face
embedding extractors:

| Model | Year | Params | FLOPs | Embedding dim | Face-trained? |
|---|---|---|---|---|---|
| FaceNet (InceptionResnetV1) | 2015 | 27.9M | 2.85G | 512 | Yes (vggface2) |
| MobileNetV2 | 2018 | 2.2M | 0.65G | 1280 | No (ImageNet, repurposed) |
| ResNet18 | 2016 | 11.2M | 3.65G | 512 | No (ImageNet, repurposed) |

For each, it measures: **parameters** (count), **FLOPs** (via `thop`),
**inference time** (CPU), **embedding dim**, and **cosine similarity** between
two face images.

Plus a **cited 2026 landscape table** with published numbers for models we
can't run on CPU: ArcFace, AdaFace, MagFace, EdgeFace, NPTFace (CVPR 2026),
FunFace (2026).

#### The pedagogical point

FaceNet (face-trained) correctly discriminates: cosine similarity ≈ 1.0 for
the same person, ≈ -0.04 for different people. MobileNetV2 and ResNet18
(ImageNet-trained, repurposed) give ~0.55 for everyone — they can't
discriminate faces because they were never trained on faces. **This is the
demo's lesson:** a backbone's architecture matters, but what it was trained on
matters more. The cited table shows the face-trained SOTA (ArcFace 99.83% LFW,
EdgeFace 99.73% with only 1.77M params, NPTFace CVPR-2026 SOTA).

#### How to demo it

1. Open the app → sidebar → "Deep Models (S3)".
2. Upload two face photos (use the bundled `data/faces/sample_real_T1.jpg` and
   `sample_real_T2.jpg` — they are different people).
3. Click "Run live comparison". Wait for the table.
4. Show the bar charts: params, FLOPs, cosine similarity for the three models.
5. Scroll down to the 2026 landscape table — walk through ArcFace, AdaFace,
   EdgeFace, NPTFace, FunFace and what each contributed.

#### What to say in the viva

- "FaceNet is the legacy baseline (2015); ArcFace (2019) is the modern standard
  with additive angular margin loss."
- "EdgeFace (2023) is the edge SOTA — 1.77M params, 99.73% LFW, suitable for
  phones."
- "NPTFace (CVPR 2026) is the current SOTA — pose-aligned transformer, 76.21%
  Rank-1 on TinyFace."
- "FLOPs and params measure complexity; inference time measures real-world
  speed. Edge deployment cares about all three plus memory."
- "Why MobileNetV2/ResNet18 fail here: they were trained on ImageNet (1000
  object classes), not faces. The embedding space isn't organised for face
  discrimination. This is why face-trained models (FaceNet, ArcFace) matter."

#### Your survey (15–20 papers)

**Required table columns:** FLOPs, parameters, accuracy, inference time.

**Seed references:**
- Schroff et al. 2015 — FaceNet
- Deng et al. 2019 — ArcFace (additive angular margin)
- Kim et al. 2022 — AdaFace (quality-adaptive margin)
- Meng et al. 2021 — MagFace (magnitude-aware)
- Boutros et al. 2023 — EdgeFace (edge SOTA, <2M params)
- Howard et al. 2017 — MobileNet
- He et al. 2016 — ResNet
- Dosovitskiy et al. 2021 — Vision Transformer (ViT)
- Ran et al. 2026 — NPTFace (CVPR 2026, pose-aligned transformer)
- FunFace 2026 — utility-aware margin loss

**Where to find more:** search "face recognition benchmark 2026 LFW IJB-C",
"efficient face recognition edge 2025", "vision transformer face 2026".

#### How to submit your part

- A PDF with your 15–20 paper survey + the comparison table (use the live
  measurements for FaceNet/MobileNetV2/ResNet18 and the cited numbers for the
  rest).
- A 1–2 page write-up explaining the accuracy-vs-complexity trade-off and edge
  deployment considerations.
- (Optional) Screenshots of the live comparison table and bar charts.

---

### Student 4 — Attacks & Liveness

**Your file:** `modules/attacks_liveness.py`
**Your app tab:** "Attacks & Liveness (S4)"

#### What your module does

1. **MiniFASNet** (Silent-Face-Anti-Spoofing) — a real pretrained liveness
   detector loaded from ONNX. ~0.08 GFLOPs, ~0.4M params, runs on CPU.
   Outputs 3 classes: paper-attack / real-face / screen-attack. Verified on
   bundled samples: real faces → live (0.996), spoof faces → spoof (0.013).
2. **Heuristic baseline** — Laplacian variance + LBP uniformity + FFT
   high-frequency ratio (the kind of hand-rolled detector that's easily
   fooled). Shown for contrast.
3. **Attack taxonomy** — a table of 7 attack types (print, replay/display,
   silicone mask, deepfake, adversarial, template leakage, hill-climbing)
   mapped to defences.
4. **Defence comparison table** — MiniFASNet vs heuristic vs rPPG vs
   depth/IR sensor.

#### How to demo it

1. Open the app → sidebar → "Attacks & Liveness (S4)".
2. **Liveness test tab:** upload a real face (try `data/faces/sample_real_T1.jpg`)
   → MiniFASNet says LIVE (0.996). Then upload a spoof
   (`data/spoof/sample_spoof_F1.jpg`) → MiniFASNet says SPOOF (screen-attack).
   Show the heuristic baseline's scores for comparison.
3. **Attack taxonomy tab:** walk through the 7 attack types and their defences.
4. **Comparison tab:** show the defence methods table.

#### What to say in the viva

- "Presentation attacks (print, replay) are the easiest to mount and the most
  common. MiniFASNet catches them by learning texture/frequency differences
  between real faces and photos/screens."
- "The heuristic baseline (Laplacian + LBP + FFT) is easily fooled — it just
  checks if the image is blurry or has screen-like texture. A high-quality
  print photo passes it."
- "Deepfake attacks need video-level analysis (blink, head motion,
  frequency artefacts) — out of scope for a single-image demo."
- "Adversarial attacks need model access to craft imperceptible perturbations
  — defence is adversarial training."
- "Template leakage is why Student 2's topic matters — if the template is
  cancelable, a leak doesn't compromise the biometric permanently."
- "ISO/IEC 30107 defines Presentation Attack Detection (PAD) levels."

#### Your survey (15+ papers)

**Required deliverable:** taxonomy of attacks and defences (the table is
already in your module — expand it with cited papers).

**Seed references:**
- Yu et al. 2020 — Silent-Face-Anti-Spoofing / MiniFASNet (the model you're
  running)
- Zhang et al. 2016 — CASIA-FASD dataset (presentation attack detection)
- Chingovska et al. 2012 — Replay-Attack dataset (Idiap)
- Rössler et al. 2019 — FaceForensics++ (deepfake detection benchmark)
- Dong et al. 2019 — adversarial attacks on face recognition
- Poh et al. 2010 — rPPG (remote photoplethysmography) liveness
- ISO/IEC 30107 — Presentation Attack Detection standard
- MiniFASNet ONNX repo: QingHeYang/Silent-Face-Anti-Spoofing-onnx

#### How to submit your part

- A PDF with your 15+ paper survey + the attack/defence taxonomy table.
- A 1–2 page write-up explaining the attack categories and why liveness
  detection is necessary (tie it to Student 2: template protection defends
  the database, liveness defends the sensor).
- (Optional) Screenshots of MiniFASNet correctly classifying real vs spoof.

---

## 10. The Literature Survey (15–20 Papers Each)

Each student writes their own survey. The app's comparison tables are
**seeded** — they have a few rows and the right columns. You expand them into
full 15–20 paper tables with cited numbers.

### How to find papers

1. **Google Scholar** — search the seed-reference keywords above with "2024
   2025 2026" appended for recent work.
2. **arXiv** — for the latest ML papers (e.g. `arxiv.org/list/cs.CV/2026`).
3. **IEEE Xplore / ACM Digital Library** — for older foundational papers (your
   university library gives access).
4. **PapersWithCode** — for benchmarks with published numbers (LFW, IJB-C,
   CASIA-FASD).

### How to fill the comparison table

For each paper, extract:
- **Dataset** — what data they used (LFW, CASIA-FASD, FVC2004, NIST SD27...).
- **Method** — their technique in one phrase.
- **Accuracy / metric** — the number they report (LFW %, EER, TPR@FPR, Rank-1).
- **Advantages** — what it does better than prior work.
- **Limitations** — what it doesn't handle, computational cost, dataset bias.

### How to cite

Use whatever citation style your professor specified (IEEE, APA, etc.). Be
consistent. Include DOIs/URLs for reproducibility.

### Anti-plagiarism note

Do not copy-paste abstracts. Read each paper, summarise in your own words, and
critically compare. The professor wants to see you *understood* the papers, not
that you can collect them.

---

## 11. Demo Script for the Viva / Final Presentation

A 15-minute live demo, split across the four students:

| Time | Student | What to show |
|---|---|---|
| 0:00–2:00 | (any) | Run `streamlit run app.py`. Open the Home tab. Enrol your face with the webcam. Verify with your face (MATCH, ~1.0). Have a teammate verify with their face (NO MATCH, <0.6). **Explain: this is the bug we fixed — the old app faked this.** |
| 2:00–5:00 | S1 | Feature Extraction tab. Show Gabor enhancement, minutiae map, LBP histogram, SIFT keypoints on the sample fingerprint. Show FaceNet embedding on a face. Show the handcrafted-vs-deep table. |
| 5:00–8:00 | S2 | Template Protection tab. Extract a face embedding. Apply BioHashing with key=42. Show the protected template. Run the revocability demo (Hamming ~0.5). Switch to Fuzzy commitment — show same face recovers the secret, different face doesn't. Show the comparison table; explain why AES is not cancelable. |
| 8:00–11:00 | S3 | Deep Models tab. Upload two different faces. Run the live comparison. Show the params/FLOPs/time/similarity table and bar charts. Explain why FaceNet discriminates but ImageNet models don't. Show the 2026 landscape table (ArcFace, EdgeFace, NPTFace). |
| 11:00–14:00 | S4 | Attacks & Liveness tab. Upload a real face → MiniFASNet says LIVE. Upload a spoof → SPOOF. Show the heuristic baseline's weaker result. Show the attack taxonomy table. Tie it back to S2 (template protection defends the database; liveness defends the sensor). |
| 14:00–15:00 | (all) | Q&A. Each student answers questions about their module and survey. |

### Tips

- **Test the demo on the presentation machine beforehand** — first run
  downloads weights (~165 MB total) and you don't want that happening live.
- Have the bundled sample images ready as fallback if the webcam doesn't work
  in the presentation room.
- If a professor asks "why not the full-stack app?" — point to this
  documentation's Section 3 (the fake-UI bug) and Section 4 (the pivot
  rationale). The honest answer is: the full-stack scope caused a critical
  security bug that none of us caught because we didn't understand the whole
  pipeline. The Streamlit app is something every student understands
  end-to-end.

---

## 12. Known Limitations & Honest Caveats

Be honest about these in your viva — professors respect honesty over
overclaiming.

1. **Thresholds are defaults, not tuned.** The face-match threshold of 0.6 is a
   reasonable default, not tuned on a dataset. A real system would compute
   EER/FAR/FRR on a benchmark (LFW, IJB-C) and pick the threshold for a target
   FAR. We didn't do this because we don't have a labelled dataset.

2. **MobileNetV2 and ResNet18 are ImageNet-pretrained, not face-trained.** They
   are included to demonstrate the FLOPs/params/time measurement methodology on
   multiple architectures. Their cosine similarities are weak (~0.55 for
   everyone) because they were never trained on faces. The cited 2026 table
   covers face-trained SOTA. **Don't claim MobileNetV2/ResNet18 are good face
   recognisers — they're not.**

3. **The synthetic fingerprint sample is not a real fingerprint.** It's a
   spiral ridge pattern generated with NumPy so the extraction pipeline has
   something to run on out of the box. The minutiae it produces are
   structurally realistic but not biometric-meaningful. Upload a real
   fingerprint image for meaningful minutiae.

4. **CPU-only.** All models run on CPU. Inference times are slower than they'd
   be on GPU but are fine for a demo. First inference is always slower (warm-up).

5. **Homomorphic encryption and ECC-based template protection are cited only,
   not implemented.** They're computationally heavy and require crypto
   libraries out of scope for this demo. Student 2's survey covers them; the
   app implements the five schemes that are practical to demo.

6. **Liveness is single-image, not video.** Real liveness systems use video
   (blink detection, head movement, rPPG heartbeat). MiniFASNet works on single
   images, which is why we chose it — but it's weaker than video-based
   approaches. The defence comparison table lists rPPG and depth/IR as
   stronger alternatives.

7. **No database.** Templates are stored in Streamlit's `session_state` (RAM,
   per browser session). This is intentional — a demo doesn't need a
   persistent database, and removing MongoDB eliminated a whole class of
   setup pain. A real system would store protected (cancelable) templates, not
   raw embeddings.

8. **The old full-stack app is not deleted.** It's in `Fingerprint-system/`
   (the parent folder) for reference. Do not run it — it has the fake-UI bug.
   The new app is in `Fingerprint-system/biometric-demo/`.

---

## 13. File-by-File Reference

| File | Owner | Purpose |
|---|---|---|
| `app.py` | (shared) | Streamlit entry point. Sidebar navigation. Five page functions. |
| `utils/face.py` | (shared) | MTCNN + FaceNet singletons, `get_face_embedding()`, `cosine_similarity()`, center-crop fallback. |
| `utils/viz.py` | (shared) | Embedding bar chart, heatmap, image grid (matplotlib). |
| `modules/feature_extraction.py` | S1 | Gabor, minutiae (CN), LBP, SIFT, FaceNet embedding, comparison table. |
| `modules/template_protection.py` | S2 | BioHashing, IoM-URP, Bloom, fuzzy commitment (RS), chaotic map, revocability/unlinkability demos, comparison table. |
| `modules/deep_models.py` | S3 | FaceNet/MobileNetV2/ResNet18 wrappers, `run_comparison()`, 2026 landscape table. |
| `modules/attacks_liveness.py` | S4 | MiniFASNet ONNX inference, heuristic baseline, attack taxonomy, defence comparison table. |
| `data/faces/` | (shared) | Sample real face images (T1, T2) from Silent-Face-Anti-Spoofing. |
| `data/fingerprints/` | (shared) | Sample synthetic fingerprint. |
| `data/spoof/` | (shared) | Sample spoof face images (F1, F2) — print/screen attacks. |
| `models/minifasnet/2.7_80x80_MiniFASNetV2.onnx` | S4 | Pretrained MiniFASNet weights (gitignored — download via README). |
| `requirements.txt` | (shared) | Python dependencies. |
| `README.md` | (shared) | Quick-start setup. |
| `DOCUMENTATION.md` | (shared) | This file. |

---

## 14. Troubleshooting

### "No module named 'torchvision'"

facenet-pytorch hard-imports torchvision. Install it matching your torch
version with `--no-deps` to avoid pulling the CUDA stack:
```bash
pip install --no-deps torchvision==0.24.1   # for torch 2.9.x
```

### "cannot import name 'BaseDefaultEventLoopPolicy' from 'asyncio.events'"

You're on Python 3.14 with an old `uvloop`. Upgrade it:
```bash
pip install --upgrade uvloop
```

### "No face detected" on the Home tab

The webcam image may be too dark or the face too small. Try uploading a clear
face photo with `st.file_uploader` instead, or use the bundled
`data/faces/sample_real_T1.jpg`. The code has a center-crop fallback, so even
if MTCNN misses the face, it will still produce an embedding (less accurate).

### MiniFASNet says "FileNotFoundError"

You haven't downloaded the ONNX weights. Run:
```bash
mkdir -p models/minifasnet
curl -L -o models/minifasnet/2.7_80x80_MiniFASNetV2.onnx \
  https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx
```

### "fuzzy decode same face: False"

This shouldn't happen after the fix. If it does, the embedding's binarisation
is too noisy. The fuzzy commitment uses Reed-Solomon with nsym=16 (corrects
8 byte errors). If your embedding is very high-dimensional and noisy, increase
`nsym` in `fuzzy_commit()`. The bundled demo uses a stable FaceNet embedding
and works correctly.

### First run is slow

The first run downloads ~165 MB of weights (FaceNet 107 MB, ResNet18 45 MB,
MobileNetV2 14 MB). They're cached in `~/.cache/torch/hub/` for subsequent
runs. First inference is also slower (model warm-up). Run the app once before
your presentation to pre-cache everything.

### Streamlit port already in use

```bash
streamlit run app.py --server.port 8502
```

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Biometric** | A measurable biological characteristic used for recognition (face, fingerprint, iris...) |
| **Embedding** | A vector representation of a biometric, produced by a neural network. Similar biometrics → similar vectors. |
| **Cosine similarity** | A measure of vector similarity in [-1, 1]. 1 = identical, 0 = unrelated, -1 = opposite. Face verification thresholds are typically 0.4–0.6. |
| **Minutiae** | Ridge endings and bifurcations in a fingerprint. The basis of ISO/IEC 19794-2 matching. |
| **Crossing number (CN)** | The number of 0→1 transitions walking the 8 neighbours of a skeleton pixel. CN=1 → ending, CN=3 → bifurcation. |
| **LBP** | Local Binary Pattern — a texture descriptor comparing each pixel to its neighbours. |
| **Gabor filter** | A bandpass filter tuned to a specific orientation and frequency. Used to enhance fingerprint ridges. |
| **SIFT** | Scale-Invariant Feature Transform — detects keypoints that are stable across scale and rotation. |
| **Cancelable biometrics** | A transform applied to a biometric template so the stored version is non-invertible and revocable. |
| **BioHashing** | Random projection + sign threshold → binary template. A cancelable scheme. |
| **Fuzzy commitment** | Binding a cryptographic key to a biometric via error-correcting codes (Reed-Solomon). |
| **Non-invertibility** | Cannot recover the original biometric from the protected template. |
| **Revocability** | A compromised template can be replaced with a new key without changing the biometric. |
| **Unlinkability** | Two templates of the same biometric under different keys cannot be linked. |
| **Liveness detection** | Determining whether the biometric sample comes from a live human (vs a photo/mask). |
| **Presentation attack** | Presenting a fake biometric (printed photo, screen replay, mask) to the sensor. |
| **PAD** | Presentation Attack Detection (ISO/IEC 30107). |
| **FLOPs** | Floating-point operations — a measure of model computational cost. |
| **EER** | Equal Error Rate — the threshold where FAR = FRR. Lower is better. |
| **FAR / FRR** | False Accept Rate / False Reject Rate. |
| **LFW** | Labeled Faces in the Wild — a standard face recognition benchmark. |
| **ArcFace** | Additive Angular Margin loss — the modern standard for training face recognition models. |
| **MiniFASNet** | A lightweight (~0.4M params) face anti-spoofing model from Silent-Face-Anti-Spoofing. |

---

*End of documentation. Last updated: July 2026.*
