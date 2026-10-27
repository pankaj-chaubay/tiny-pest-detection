"""Exploratory analysis of a detection dataset, with an anchor-size recommendation.

Tiny objects fail in Mask/Faster R-CNN mostly because no anchor is small enough to overlap
them well, so the region proposal network never proposes them. This module measures object
sizes **as the model sees them** (after its internal resize) and proposes anchor sizes for
the five FPN levels that cover the observed range.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FPN_LEVELS = 5


@dataclass(frozen=True)
class AnchorAdvice:
    """Proposed anchor sizes and how much of the data they cover."""

    sizes: tuple[int, ...]
    object_side_p05: float
    object_side_p50: float
    object_side_p95: float
    coverage: float
    default_coverage: float


@dataclass(frozen=True)
class DatasetReport:
    """Everything the EDA step measures."""

    images: int
    objects: int
    images_without_objects: int
    classes_present: int
    classes_total: int
    imbalance_ratio: float
    top_classes: list[tuple[str, int]]
    rarest_classes: list[tuple[str, int]]
    image_width: dict[str, float]
    image_height: dict[str, float]
    objects_per_image: dict[str, float]
    relative_object_size: dict[str, float]
    size_buckets: dict[str, int]
    anchors: AnchorAdvice

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary."""
        return asdict(self)


def _describe(values: NDArray[np.float64]) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": round(float(np.min(values)), 3),
        "median": round(float(np.median(values)), 3),
        "mean": round(float(values.mean()), 3),
        "max": round(float(np.max(values)), 3),
    }


def resized_object_sides(coco: Mapping[str, Any], min_size: int = 800, max_size: int = 1333) -> NDArray[np.float64]:
    """Square-root of each box area after the resize torchvision detection models apply."""
    images = {i["id"]: i for i in coco["images"]}
    sides = []
    for ann in coco["annotations"]:
        info = images[ann["image_id"]]
        h, w = float(info["height"]), float(info["width"])
        scale = min(min_size / min(h, w), max_size / max(h, w))
        _, _, bw, bh = ann["bbox"]
        sides.append(math.sqrt(max(bw * bh, 0.0)) * scale)
    return np.asarray(sides, dtype=np.float64)


def anchor_coverage(sides: NDArray[np.float64], anchor_sizes: Sequence[int]) -> float:
    """Share of objects within a factor of two of their nearest anchor size.

    A rough proxy for whether the RPN can match an object: an object half or double
    the nearest anchor still reaches the usual IoU threshold at a matching aspect ratio.
    """
    if sides.size == 0:
        return 0.0
    anchors = np.asarray(anchor_sizes, dtype=np.float64)
    ratio = sides[:, None] / anchors[None, :]
    nearest = np.min(np.abs(np.log2(ratio)), axis=1)
    return float((nearest <= 1.0).mean())


def suggest_anchor_sizes(
    coco: Mapping[str, Any], min_size: int = 800, max_size: int = 1333, smallest_allowed: int = 4
) -> AnchorAdvice:
    """Propose five anchor sizes (one per FPN level, doubling each level).

    The smallest anchor is the largest power of two at or below the 5th percentile of the
    resized object sides, but never below ``smallest_allowed``. The largest anchor is kept
    within the resized image (``max_size``), so for datasets of big objects the advice
    stays at or near the default instead of growing past the image.
    """
    sides = resized_object_sides(coco, min_size, max_size)
    default = (32, 64, 128, 256, 512)
    if sides.size == 0:
        return AnchorAdvice(default, 0.0, 0.0, 0.0, 0.0, 0.0)
    p05, p50, p95 = (float(np.quantile(sides, q)) for q in (0.05, 0.5, 0.95))
    smallest = max(smallest_allowed, 2 ** math.floor(math.log2(max(p05, 1.0))))
    ceiling = 2 ** math.floor(math.log2(max_size)) // 2 ** (FPN_LEVELS - 1)
    smallest = max(smallest_allowed, min(smallest, ceiling))
    sizes = tuple(smallest * 2**i for i in range(FPN_LEVELS))
    return AnchorAdvice(
        sizes=sizes,
        object_side_p05=round(p05, 1),
        object_side_p50=round(p50, 1),
        object_side_p95=round(p95, 1),
        coverage=round(anchor_coverage(sides, sizes), 4),
        default_coverage=round(anchor_coverage(sides, default), 4),
    )


def analyse(coco: Mapping[str, Any], min_size: int = 800, max_size: int = 1333, top_k: int = 5) -> DatasetReport:
    """Measure class balance, image sizes, object sizes and anchor fit."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_class = Counter(names[a["category_id"]] for a in coco["annotations"])
    per_image = Counter(a["image_id"] for a in coco["annotations"])
    images = {i["id"]: i for i in coco["images"]}

    rel_sizes = []
    areas = []
    for ann in coco["annotations"]:
        info = images[ann["image_id"]]
        _, _, bw, bh = ann["bbox"]
        areas.append(bw * bh)
        rel_sizes.append(math.sqrt(bw * bh) / math.sqrt(info["width"] * info["height"]))
    area_arr = np.asarray(areas, dtype=np.float64)
    ranked = per_class.most_common()

    return DatasetReport(
        images=len(coco["images"]),
        objects=len(coco["annotations"]),
        images_without_objects=sum(1 for i in coco["images"] if i["id"] not in per_image),
        classes_present=len(per_class),
        classes_total=len(coco["categories"]),
        imbalance_ratio=round(ranked[0][1] / ranked[-1][1], 2) if ranked else 0.0,
        top_classes=ranked[:top_k],
        rarest_classes=ranked[-top_k:][::-1] if ranked else [],
        image_width=_describe(np.asarray([i["width"] for i in coco["images"]], dtype=np.float64)),
        image_height=_describe(np.asarray([i["height"] for i in coco["images"]], dtype=np.float64)),
        objects_per_image=_describe(np.asarray([per_image.get(i, 0) for i in images], dtype=np.float64)),
        relative_object_size=_describe(np.asarray(rel_sizes, dtype=np.float64)),
        size_buckets={
            "small": int((area_arr < 32**2).sum()),
            "medium": int(((area_arr >= 32**2) & (area_arr < 96**2)).sum()),
            "large": int((area_arr >= 96**2).sum()),
        },
        anchors=suggest_anchor_sizes(coco, min_size, max_size),
    )


def plot_report(coco: Mapping[str, Any], out_dir: str | Path, min_size: int = 800, max_size: int = 1333) -> list[Path]:
    """Save a class-distribution chart and an object-size histogram with anchor sizes marked."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in coco["categories"]}
    counts = Counter(names[a["category_id"]] for a in coco["annotations"]).most_common(25)
    paths = []

    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.3 * len(counts))))
    if counts:
        labels, values = zip(*counts, strict=True)
        ax.barh(list(labels)[::-1], list(values)[::-1], color="#1F3864")
    ax.set_xlabel("instances")
    ax.set_title("Instances per class (top 25)")
    fig.tight_layout()
    paths.append(out / "class_distribution.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    sides = resized_object_sides(coco, min_size, max_size)
    advice = suggest_anchor_sizes(coco, min_size, max_size)
    fig, ax = plt.subplots(figsize=(8, 4))
    if sides.size:
        bins = np.logspace(0, math.log10(max(float(np.max(sides)), 2.0)), 40).tolist()
        ax.hist(sides, bins=bins, color="#8FA9D6")
        ax.set_xscale("log")
    for size in advice.sizes:
        ax.axvline(size, color="#C00000", linestyle="--", linewidth=1)
    ax.set_xlabel("object side after resize (pixels, log scale)")
    ax.set_ylabel("objects")
    ax.set_title(f"Object sizes vs proposed anchors {advice.sizes}")
    fig.tight_layout()
    paths.append(out / "object_sizes.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths
