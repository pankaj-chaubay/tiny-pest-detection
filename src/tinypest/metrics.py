"""Detection metrics: precision, recall and F1 at one IoU, and COCO mAP by object size.

COCO mAP is computed with ``pycocotools``. ``AP_small`` (objects under 32x32 pixels) is the
number to watch for tiny pests; the overall AP can look fine while small objects fail.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from tinypest.dataset import DetectionDataset, collate_fn
from tinypest.engine import predict
from tinypest.postprocess import PostprocessConfig

COCO_STATS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100")


@dataclass
class MatchCounts:
    """True positives, false positives and false negatives, summed over images."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match_detections(
    pred_boxes: Tensor,
    pred_scores: Tensor,
    pred_labels: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_threshold: float = 0.5,
) -> MatchCounts:
    """Greedy, class-aware matching: each prediction (highest score first) claims the
    best-overlapping unclaimed ground-truth box of the same class at or above the threshold."""
    counts = MatchCounts()
    if pred_boxes.shape[0] == 0:
        counts.fn = int(gt_boxes.shape[0])
        return counts
    if gt_boxes.shape[0] == 0:
        counts.fp = int(pred_boxes.shape[0])
        return counts
    ious = box_iou(pred_boxes, gt_boxes)
    ious[pred_labels[:, None] != gt_labels[None, :]] = -1.0
    claimed = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for i in torch.argsort(pred_scores, descending=True).tolist():
        row = ious[i].clone()
        row[claimed] = -1.0
        j = int(torch.argmax(row))
        if row[j] >= iou_threshold:
            claimed[j] = True
            counts.tp += 1
        else:
            counts.fp += 1
    counts.fn = int((~claimed).sum())
    return counts


def to_coco_results(
    output: Mapping[str, Tensor], image_id: int, label_to_category: Mapping[int, int], mask_threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Turn one image's detections into COCO results records (xywh boxes, RLE masks if present)."""
    records = []
    masks = output.get("masks")
    for k in range(output["boxes"].shape[0]):
        x1, y1, x2, y2 = output["boxes"][k].tolist()
        record: dict[str, Any] = {
            "image_id": image_id,
            "category_id": label_to_category[int(output["labels"][k])],
            "score": float(output["scores"][k]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        }
        if masks is not None:
            binary = (masks[k, 0] >= mask_threshold).numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("ascii")
            record["segmentation"] = rle
        records.append(record)
    return records


def coco_map(gt: Mapping[str, Any], results: Sequence[Mapping[str, Any]], iou_type: str = "bbox") -> dict[str, float]:
    """COCO AP/AR for ``results``. Returns zeros when there are no detections."""
    if iou_type not in ("bbox", "segm"):
        raise ValueError("iou_type must be 'bbox' or 'segm'")
    if not results:
        return dict.fromkeys(COCO_STATS, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {k: gt[k] for k in ("images", "annotations", "categories")}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(results))
        evaluator = COCOeval(coco_gt, coco_dt, iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {name: round(float(v), 4) for name, v in zip(COCO_STATS, evaluator.stats[:9], strict=True)}


@dataclass(frozen=True)
class DetectionReport:
    """Evaluation numbers for one split and one post-processing setting."""

    images: int
    ground_truth: int
    detections: int
    postprocess: str
    precision: float
    recall: float
    f1: float
    box_map: dict[str, float]
    mask_map: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: DetectionDataset,
    device: torch.device,
    postprocess_config: PostprocessConfig | None = None,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    batch_size: int = 2,
) -> DetectionReport:
    """Evaluate on every image in ``dataset`` (use evaluation transforms).

    Precision/recall/F1 count detections at or above ``score_threshold``; COCO mAP uses all
    detections because it ranks them by score itself.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    counts = MatchCounts()
    results: list[dict[str, Any]] = []
    n_gt = n_det = 0
    for images, targets in loader:
        outputs = predict(model, images, device, score_threshold=0.0, postprocess_config=postprocess_config)
        for out, target in zip(outputs, targets, strict=True):
            results.extend(to_coco_results(out, int(target["image_id"]), dataset.label_to_category))
            keep = out["scores"] >= score_threshold
            m = match_detections(
                out["boxes"][keep], out["scores"][keep], out["labels"][keep],
                target["boxes"], target["labels"], iou_threshold,
            )  # fmt: skip
            counts.tp, counts.fp, counts.fn = counts.tp + m.tp, counts.fp + m.fp, counts.fn + m.fn
            n_gt += int(target["boxes"].shape[0])
            n_det += int(keep.sum())

    has_masks = bool(results) and "segmentation" in results[0]
    return DetectionReport(
        images=len(dataset),
        ground_truth=n_gt,
        detections=n_det,
        postprocess="built-in NMS" if postprocess_config is None else postprocess_config.method,
        precision=round(counts.precision, 4),
        recall=round(counts.recall, 4),
        f1=round(counts.f1, 4),
        box_map=coco_map(dataset.coco, results, "bbox"),
        mask_map=coco_map(dataset.coco, results, "segm") if has_masks else None,
    )
