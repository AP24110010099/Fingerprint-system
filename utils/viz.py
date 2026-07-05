"""Visualization helpers for embeddings and feature maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def embedding_bar_chart(embedding: np.ndarray, title: str = "Embedding",
                        max_bars: int = 64) -> plt.Figure:
    """Bar chart of the first `max_bars` embedding dimensions."""
    fig, ax = plt.subplots(figsize=(8, 3))
    vals = embedding[:max_bars]
    colors = ["#4C3FE4" if v >= 0 else "#E4534C" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_title(f"{title} (first {len(vals)} of {len(embedding)} dims)")
    ax.set_xlabel("dimension")
    ax.set_ylabel("value")
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    return fig


def embedding_heatmap(embedding: np.ndarray, title: str = "Embedding heatmap") -> plt.Figure:
    """1-row heatmap of the full embedding vector."""
    fig, ax = plt.subplots(figsize=(8, 1.6))
    grid = embedding.reshape(1, -1)
    im = ax.imshow(grid, aspect="auto", cmap="RdBu",
                   vmin=-np.max(np.abs(embedding)), vmax=np.max(np.abs(embedding)))
    ax.set_yticks([])
    ax.set_xlabel("dimension")
    ax.set_title(f"{title} ({len(embedding)}-D)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    return fig


def image_grid(images: list[np.ndarray], titles: list[str],
               cols: int = 2, cmap: str = "gray") -> plt.Figure:
    """Plot a grid of images with titles."""
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
