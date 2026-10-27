"""Draw instance masks and boxes on images and save them as PNG files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"]


def to_uint8(image: Tensor) -> Tensor:
    """Convert a ``(3, H, W)`` image in ``[0, 1]`` floats or ``uint8`` into ``uint8``."""
    if image.dtype == torch.uint8:
        return image
    return (image.clamp(0, 1) * 255).round().to(torch.uint8)


def overlay(
    image: Tensor,
    masks: Tensor,
    boxes: Tensor | None = None,
    captions: Sequence[str] | None = None,
    alpha: float = 0.45,
) -> NDArray[np.uint8]:
    """Return an ``(H, W, 3)`` array with masks (and optionally boxes) drawn on the image.

    Args:
        image: ``(3, H, W)`` image tensor.
        masks: ``(N, H, W)`` binary masks, or ``(N, 1, H, W)`` probabilities (thresholded at 0.5).
        boxes: Optional ``(N, 4)`` boxes in ``x1, y1, x2, y2``.
        captions: Optional text for each box.
        alpha: Mask opacity.
    """
    canvas = to_uint8(image.cpu())
    if masks.ndim == 4:
        masks = masks[:, 0] >= 0.5
    masks = masks.cpu().bool()
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(masks.shape[0])]
    if masks.shape[0]:
        canvas = draw_segmentation_masks(canvas, masks, alpha=alpha, colors=colors)
    if boxes is not None and boxes.shape[0]:
        canvas = draw_bounding_boxes(
            canvas, boxes.cpu(), labels=list(captions) if captions else None, colors=colors, width=2
        )
    return canvas.permute(1, 2, 0).numpy()


def save_grid(panels: Sequence[NDArray[np.uint8]], titles: Sequence[str], path: str | Path, cols: int = 2) -> Path:
    """Save several images side by side in one PNG file."""
    if len(panels) != len(titles):
        raise ValueError("panels and titles must have the same length")
    if not panels:
        raise ValueError("nothing to draw")
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, panel, title in zip(axes.flat, panels, titles, strict=False):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
