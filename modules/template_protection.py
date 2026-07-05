"""Student 2 - Biometric Template Protection.

Implementations of cancelable-biometric and biometric-cryptosystem schemes
applied to a FaceNet deep embedding. All schemes are keyed: changing the key
revokes the template (revocability) and different keys yield uncorrelated
templates (unlinkability).

Schemes (benchmarked in Otroshi et al., 2025, Springer J. Image & Video Proc.):
  1. BioHashing        - random Gaussian projection + sign threshold
  2. IoM-URP           - Index-of-Max of uniform random projections
  3. Bloom filter      - per-block bloom mapping of quantised features
  4. Fuzzy commitment  - Reed-Solomon binding of a binarised embedding to a key
  5. Chaotic map       - logistic-map scrambling of the embedding

Each function takes a real-valued embedding + a seed/key and returns a
protected template (binary string or array) plus diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from reedsolo import RSCodec


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(len(a), len(b))
    return float(np.mean(a[:n] != b[:n]))


# ---------------------------------------------------------------------------
# 1. BioHashing  (Teoh et al.)
# ---------------------------------------------------------------------------

def biohash(embedding: np.ndarray, seed: int, n_bits: int = 256) -> np.ndarray:
    """Random Gaussian projection -> sign threshold -> binary template."""
    g = _rng(seed)
    d = len(embedding)
    proj = g.standard_normal((n_bits, d))
    scores = proj @ embedding
    return (scores >= 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Index-of-Max (IoM) hashing - URP variant  (Jin et al.)
# ---------------------------------------------------------------------------

def iom_urp(embedding: np.ndarray, seed: int, m: int = 256,
            k: int = 32) -> np.ndarray:
    """Apply k random uniform projections; keep the arg-max index of each.

    Returns an integer template of length k with values in [0, m).
    """
    g = _rng(seed)
    d = len(embedding)
    proj = g.uniform(-1, 1, (m, d))
    scores = proj @ embedding           # (m,)
    # repeat with independent projections to get k indices
    idx = np.empty(k, dtype=np.int32)
    for i in range(k):
        proj_i = g.uniform(-1, 1, (m, d))
        idx[i] = int(np.argmax(proj_i @ embedding))
    return idx


# ---------------------------------------------------------------------------
# 3. Bloom filter template  (Rathgeb et al.)
# ---------------------------------------------------------------------------

def bloom_template(embedding: np.ndarray, seed: int, block: int = 4,
                   bloom_size: int = 8) -> np.ndarray:
    """Quantise embedding into blocks, map each block to a bloom code word."""
    g = _rng(seed)
    d = len(embedding)
    n_blocks = d // block
    # random mapping table: block value -> bloom codeword
    table = g.integers(0, 2, size=(16, bloom_size)).astype(np.uint8)
    # quantise each dim to 4 bits (0..15)
    e = embedding - embedding.min()
    e = (e / (e.max() + 1e-8) * 15).astype(np.int32)
    template = np.zeros((n_blocks, bloom_size), dtype=np.uint8)
    for b in range(n_blocks):
        chunk = e[b * block:(b + 1) * block]
        code = np.zeros(bloom_size, dtype=np.uint8)
        for v in chunk:
            code |= table[v]
        template[b] = code
    return template.ravel()


# ---------------------------------------------------------------------------
# 4. Fuzzy commitment  (Juels & Wattenberg) via Reed-Solomon
# ---------------------------------------------------------------------------

def fuzzy_commit(embedding: np.ndarray, secret: bytes,
                 nsym: int = 16) -> dict:
    """Bind `secret` to a binarised embedding using Reed-Solomon.

    Returns {"helper": bytes, "secret": secret, "nsym": nsym}.
    Decoder recovers the secret iff the queried embedding is close enough
    (within RS error-correction capacity) to the enrolment embedding.
    """
    rs = RSCodec(nsym=nsym, nsize=255 - nsym)
    enc = rs.encode(secret)             # codeword bytes
    # binarise embedding to a bit string of same length as codeword (in bits)
    e = embedding - embedding.mean()
    bits = (e >= 0).astype(np.uint8)
    # pad / truncate bits to len(enc)*8
    target = len(enc) * 8
    if len(bits) < target:
        bits = np.concatenate([bits, np.zeros(target - len(bits), dtype=np.uint8)])
    else:
        bits = bits[:target]
    codeword_bits = np.unpackbits(np.frombuffer(enc, dtype=np.uint8))
    helper_bits = (bits ^ codeword_bits).astype(np.uint8)
    return {"helper": helper_bits, "secret": secret, "nsym": nsym}


def fuzzy_commit_decode(query_embedding: np.ndarray, helper: dict) -> tuple[bool, bytes | None]:
    """Try to recover the bound secret from a query embedding."""
    rs = RSCodec(nsym=helper["nsym"], nsize=255 - helper["nsym"])
    e = query_embedding - query_embedding.mean()
    bits = (e >= 0).astype(np.uint8)
    target = len(helper["helper"])
    if len(bits) < target:
        bits = np.concatenate([bits, np.zeros(target - len(bits), dtype=np.uint8)])
    else:
        bits = bits[:target]
    codeword_bits = bits ^ helper["helper"]
    codeword = np.packbits(codeword_bits).tobytes()
    try:
        # reedsolo >=1.7 returns (message, full_message, errata_pos)
        decoded = rs.decode(codeword)
        msg = decoded[0] if isinstance(decoded, tuple) else decoded
        return True, bytes(msg)
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# 5. Chaotic-map scrambling  (logistic map)
# ---------------------------------------------------------------------------

def chaotic_scramble(embedding: np.ndarray, seed: int,
                     r: float = 3.99) -> np.ndarray:
    """Permute embedding elements using a logistic-map keystream."""
    g = _rng(seed)
    n = len(embedding)
    x = g.uniform(0.01, 0.99)
    seq = np.empty(n)
    for i in range(n):
        x = r * x * (1 - x)
        seq[i] = x
    order = np.argsort(seq)
    return embedding[order]


# ---------------------------------------------------------------------------
# Security / property diagnostics
# ---------------------------------------------------------------------------

def revocability_demo(embedding: np.ndarray, scheme: str = "biohash",
                      n_keys: int = 5) -> pd.DataFrame:
    """Show that different keys produce different templates (revocability)."""
    rows = []
    fn = {"biohash": biohash, "iom_urp": iom_urp,
          "bloom": bloom_template, "chaotic": chaotic_scramble}[scheme]
    templates = [fn(embedding, seed=s) for s in range(n_keys)]
    for i in range(n_keys):
        for j in range(i + 1, n_keys):
            rows.append({"key_a": i, "key_b": j,
                         "hamming_distance": hamming(templates[i], templates[j])})
    return pd.DataFrame(rows)


def unlinkability_demo(embedding_a: np.ndarray, embedding_b: np.ndarray,
                       scheme: str = "biohash",
                       n_keys: int = 10) -> dict:
    """Compare same-key vs cross-key distances for two different faces."""
    fn = {"biohash": biohash, "iom_urp": iom_urp,
          "bloom": bloom_template, "chaotic": chaotic_scramble}[scheme]
    same_key = []      # same face, different keys -> should be ~0.5 (uncorrelated)
    cross_face = []    # different faces, same key -> genuine impostor distance
    for s in range(n_keys):
        for t in range(s + 1, n_keys):
            same_key.append(hamming(fn(embedding_a, s), fn(embedding_a, t)))
    for s in range(n_keys):
        cross_face.append(hamming(fn(embedding_a, s), fn(embedding_b, s)))
    return {
        "same_face_diff_key_mean": float(np.mean(same_key)),
        "diff_face_same_key_mean": float(np.mean(cross_face)),
        "same_face_diff_key": same_key,
        "diff_face_same_key": cross_face,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def protection_comparison_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"scheme": "BioHashing", "category": "cancelable",
         "security": "medium (random-projection irreversibility)",
         "complexity": "O(n_bits x d)", "computational_cost": "low",
         "revocability": "yes (new key)", "unlinkability": "high",
         "accuracy_impact": "low"},
        {"scheme": "IoM-URP", "category": "cancelable",
         "security": "medium", "complexity": "O(m x k x d)", "computational_cost": "low",
         "revocability": "yes", "unlinkability": "high", "accuracy_impact": "low"},
        {"scheme": "Bloom filter", "category": "cancelable",
         "security": "medium-high", "complexity": "O(d/block x bloom_size)",
         "computational_cost": "low", "revocability": "yes",
         "unlinkability": "high", "accuracy_impact": "medium (perf drop reported)"},
        {"scheme": "Fuzzy commitment (RS)", "category": "biometric cryptosystem",
         "security": "high (bounded by code distance)", "complexity": "RS encode/decode",
         "computational_cost": "medium", "revocability": "yes (new secret)",
         "unlinkability": "medium", "accuracy_impact": "low (error-tolerant)"},
        {"scheme": "Chaotic-map scrambling", "category": "cancelable",
         "security": "low-medium (permutation only)", "complexity": "O(d)",
         "computational_cost": "very low", "revocability": "yes",
         "unlinkability": "medium", "accuracy_impact": "none (lossless)"},
        {"scheme": "AES (baseline, NOT cancelable)", "category": "encryption",
         "security": "high (reversible)", "complexity": "AES",
         "computational_cost": "low", "revocability": "no (decrypt -> original)",
         "unlinkability": "n/a (invertible)", "accuracy_impact": "none"},
    ])
