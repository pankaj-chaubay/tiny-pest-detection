"""Post-processing: score filtering, class-wise NMS and Soft-NMS.

Standard NMS deletes any box that overlaps a higher-scoring box by more than a threshold.
For clustered pests that is harmful: two real insects side by side overlap, and one gets
deleted. **Soft-NMS** (Bodla et al., 2017) lowers the overlapping box's score instead of
deleting it, so a genuine neighbour keeps a chance to survive the final score threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_iou

Method = Literal["none", "nms", "soft-nms"]


@dataclass(frozen=True)
class PostprocessConfig:
    """Settings for :func:`postprocess`.

    Attributes:
        method: ``"none"``, ``"nms"`` or ``"soft-nms"``.
        iou_threshold: Overlap above which NMS suppresses (or linear Soft-NMS decays).
        score_threshold: Detections below this score are dropped at the end.
        sigma: Width of the Gaussian decay in Soft-NMS. Smaller sigma = stronger decay.
        soft_kind: ``"gaussian"`` or ``"linear"`` decay.
    """

    method: Method = "nms"
    iou_threshold: float = 0.5
    score_threshold: float = 0.05
    sigma: float = 0.5
    soft_kind: Literal["gaussian", "linear"] = "gaussian"

    def __post_init__(self) -> None:
        if self.method not in ("none", "nms", "soft-nms"):
            raise ValueError(f"unknown method {self.method!r}")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


def soft_nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    kind: Literal["gaussian", "linear"] = "gaussian",
) -> tuple[Tensor, Tensor]:
    """Soft-NMS for one class.

    Repeatedly takes the highest-scoring remaining box and decays the scores of the others
    by their overlap with it: ``s * exp(-iou^2 / sigma)`` (Gaussian) or ``s * (1 - iou)`` when
    ``iou > iou_threshold`` (linear). Boxes that fall below ``score_threshold`` are dropped.

    Args:
        boxes: ``(N, 4)`` boxes in ``x1, y1, x2, y2``.
        scores: ``(N,)`` confidence scores.

    Returns:
        ``(keep, new_scores)``: indices into the input, ordered by decayed score, and the
        decayed scores of those boxes.
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.shape != boxes.shape[:1]:
        raise ValueError("boxes must be (N, 4) and scores (N,)")
    remaining = torch.arange(boxes.shape[0], device=boxes.device)
    current = scores.clone().float()
    keep: list[int] = []
    kept_scores: list[float] = []
    while remaining.numel():
        best_pos = int(torch.argmax(current[remaining]))
        best = int(remaining[best_pos])
        if float(current[best]) < score_threshold:
            break
        keep.append(best)
        kept_scores.append(float(current[best]))
        remaining = torch.cat([remaining[:best_pos], remaining[best_pos + 1 :]])
        if not remaining.numel():
            break
        ious = box_iou(boxes[best].unsqueeze(0), boxes[remaining]).squeeze(0)
        if kind == "gaussian":
            decay = torch.exp(-(ious**2) / sigma)
        else:
            decay = torch.where(ious > iou_threshold, 1.0 - ious, torch.ones_like(ious))
        current[remaining] = current[remaining] * decay
        remaining = remaining[current[remaining] >= score_threshold]
    index = torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    return index, torch.tensor(kept_scores, dtype=scores.dtype, device=boxes.device)


def postprocess(output: Mapping[str, Tensor], config: PostprocessConfig) -> dict[str, Tensor]:
    """Apply the configured suppression class by class, then the score threshold.

    ``output`` is one image's result from a torchvision detector (``boxes``, ``scores``,
    ``labels`` and optionally ``masks``). Extra per-detection tensors are filtered with the
    same indices, so masks stay aligned with boxes.
    """
    boxes, scores, labels = output["boxes"], output["scores"], output["labels"]
    if config.method == "none" or boxes.shape[0] == 0:
        keep = torch.arange(boxes.shape[0], device=boxes.device)
        new_scores = scores
    elif config.method == "nms":
        keep = batched_nms(boxes, scores, labels, config.iou_threshold)
        new_scores = scores[keep]
    else:
        parts_keep, parts_scores = [], []
        for cls in torch.unique(labels):
            idx = torch.nonzero(labels == cls, as_tuple=True)[0]
            k, s = soft_nms(boxes[idx], scores[idx], config.iou_threshold, config.sigma, kind=config.soft_kind)
            parts_keep.append(idx[k])
            parts_scores.append(s)
        keep = torch.cat(parts_keep)
        new_scores = torch.cat(parts_scores)
        order = torch.argsort(new_scores, descending=True)
        keep, new_scores = keep[order], new_scores[order]

    result = {k: v[keep] for k, v in output.items() if k != "scores"}
    result["scores"] = new_scores if config.method != "none" else scores[keep]
    passed = result["scores"] >= config.score_threshold
    return {k: v[passed] for k, v in result.items()}
