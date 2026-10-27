"""Load, validate, summarise and split COCO annotation files.

The converter in :mod:`tinypest.voc` writes COCO, and every later step reads it. Polygon,
uncompressed RLE and compressed RLE segmentations are all understood, so the same code also
works with real instance masks (for example a CVAT export) when training Mask R-CNN.

All functions work on the plain ``dict`` that ``json.load`` returns.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils

from tinypest.errors import AnnotationError

CocoDict = dict[str, Any]

_REQUIRED_KEYS = ("images", "annotations", "categories")
_SPLIT_NAMES = ("train", "val", "test")


def load_coco(path: str | Path) -> CocoDict:
    """Read a COCO JSON file and check that it is internally consistent.

    Args:
        path: Location of the COCO annotation file.

    Returns:
        The parsed annotation dictionary.

    Raises:
        AnnotationError: If the file is missing, is not valid JSON, or fails
            :func:`validate_coco`.
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Annotation file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            coco: CocoDict = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{path} is not valid JSON: {exc}") from exc
    validate_coco(coco, source=str(path))
    return coco


def validate_coco(coco: Mapping[str, Any], source: str = "<memory>") -> None:
    """Check the structural rules the rest of the package relies on.

    The checks are: the three top-level lists exist; image, annotation and category ids are
    unique; every annotation points at an existing image and category; every image has a
    positive width and height; and every annotation has a segmentation.

    Args:
        coco: Parsed COCO dictionary.
        source: Name used in error messages, usually the file path.

    Raises:
        AnnotationError: On the first rule that is broken.
    """
    if not isinstance(coco, Mapping):
        raise AnnotationError(f"{source}: top level must be a JSON object")
    for key in _REQUIRED_KEYS:
        if not isinstance(coco.get(key), list):
            raise AnnotationError(f"{source}: missing or invalid list '{key}'")

    image_ids = _unique_ids(coco["images"], "image", source)
    category_ids = _unique_ids(coco["categories"], "category", source)
    _unique_ids(coco["annotations"], "annotation", source)

    for image in coco["images"]:
        if int(image.get("width", 0)) <= 0 or int(image.get("height", 0)) <= 0:
            raise AnnotationError(f"{source}: image {image['id']} has no valid width/height")
        if not image.get("file_name"):
            raise AnnotationError(f"{source}: image {image['id']} has no file_name")

    for ann in coco["annotations"]:
        if ann.get("image_id") not in image_ids:
            raise AnnotationError(f"{source}: annotation {ann['id']} points at unknown image {ann.get('image_id')}")
        if ann.get("category_id") not in category_ids:
            raise AnnotationError(f"{source}: annotation {ann['id']} has unknown category {ann.get('category_id')}")
        if not ann.get("segmentation"):
            raise AnnotationError(f"{source}: annotation {ann['id']} has no segmentation")


def _unique_ids(records: Sequence[Mapping[str, Any]], kind: str, source: str) -> set[int]:
    ids: list[int] = []
    for record in records:
        if "id" not in record:
            raise AnnotationError(f"{source}: a {kind} record has no 'id'")
        ids.append(int(record["id"]))
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise AnnotationError(f"{source}: duplicate {kind} ids {sorted(duplicates)[:5]}")
    return set(ids)


def annotation_to_mask(annotation: Mapping[str, Any], height: int, width: int) -> NDArray[np.uint8]:
    """Rasterise one annotation into a binary ``(height, width)`` mask.

    Args:
        annotation: A COCO annotation with a polygon, uncompressed RLE or compressed RLE
            ``segmentation``.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        A ``uint8`` array with 1 inside the object and 0 elsewhere.

    Raises:
        AnnotationError: If the segmentation has an unknown form, has polygons with fewer
            than three points, or its RLE size does not match the image.
    """
    segmentation = annotation.get("segmentation")
    ann_id = annotation.get("id", "?")
    if isinstance(segmentation, list):
        polygons = [p for p in segmentation if len(p) >= 6]
        if not polygons:
            raise AnnotationError(f"annotation {ann_id}: no polygon with at least 3 points")
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and "counts" in segmentation:
        size = list(segmentation.get("size", []))
        if size != [height, width]:
            raise AnnotationError(f"annotation {ann_id}: RLE size {size} does not match image {[height, width]}")
        counts = segmentation["counts"]
        rle = mask_utils.frPyObjects(segmentation, height, width) if isinstance(counts, list) else segmentation
    else:
        raise AnnotationError(f"annotation {ann_id}: unsupported segmentation format")
    return np.asarray(mask_utils.decode(rle), dtype=np.uint8)


@dataclass(frozen=True)
class CocoSummary:
    """Headline numbers that describe an annotation file."""

    images: int
    annotations: int
    images_without_annotations: int
    instances_per_category: dict[str, int]
    max_instances_per_image: int
    polygon_annotations: int
    rle_annotations: int
    crowd_annotations: int
    area_quantiles: dict[str, float] = field(default_factory=dict)
    size_buckets: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a JSON-serialisable dictionary."""
        return asdict(self)


def summarize(coco: Mapping[str, Any]) -> CocoSummary:
    """Describe an annotation file: counts, formats and object sizes.

    Object sizes use the COCO convention: *small* below 32x32 pixels of area, *medium* up to
    96x96, *large* above.

    Args:
        coco: A validated COCO dictionary.

    Returns:
        A :class:`CocoSummary`.
    """
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_image = Counter(a["image_id"] for a in coco["annotations"])
    per_category = Counter(names[a["category_id"]] for a in coco["annotations"])
    areas = np.array([float(a.get("area", 0.0)) for a in coco["annotations"]], dtype=float)

    quantiles: dict[str, float] = {}
    buckets = {"small": 0, "medium": 0, "large": 0}
    if areas.size:
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            quantiles[f"p{int(q * 100)}"] = round(float(np.quantile(areas, q)), 1)
        buckets["small"] = int((areas < 32**2).sum())
        buckets["medium"] = int(((areas >= 32**2) & (areas < 96**2)).sum())
        buckets["large"] = int((areas >= 96**2).sum())

    return CocoSummary(
        images=len(coco["images"]),
        annotations=len(coco["annotations"]),
        images_without_annotations=sum(1 for i in coco["images"] if i["id"] not in per_image),
        instances_per_category=dict(per_category),
        max_instances_per_image=max(per_image.values(), default=0),
        polygon_annotations=sum(isinstance(a["segmentation"], list) for a in coco["annotations"]),
        rle_annotations=sum(isinstance(a["segmentation"], dict) for a in coco["annotations"]),
        crowd_annotations=sum(int(a.get("iscrowd", 0)) for a in coco["annotations"]),
        area_quantiles=quantiles,
        size_buckets=buckets,
    )


def split_coco(
    coco: Mapping[str, Any],
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, CocoDict]:
    """Split a dataset into train, validation and test sets **by image**.

    Splitting by image (not by annotation) keeps every instance of an image in one subset,
    so no picture is seen during both training and evaluation.

    Args:
        coco: A validated COCO dictionary.
        ratios: Fractions for train, val and test. They must be non-negative and sum to 1.
        seed: Seed for the shuffle, so the same input always gives the same split.

    Returns:
        A mapping ``{"train": ..., "val": ..., "test": ...}`` of complete COCO dictionaries.

    Raises:
        ValueError: If the ratios are invalid or a non-empty subset would get no images.
    """
    if len(ratios) != 3 or any(r < 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must be three non-negative numbers summing to 1, got {ratios}")

    image_ids = sorted(int(i["id"]) for i in coco["images"])
    random.Random(seed).shuffle(image_ids)
    n_total = len(image_ids)
    n_train = round(ratios[0] * n_total)
    n_val = round(ratios[1] * n_total)
    chunks = (image_ids[:n_train], image_ids[n_train : n_train + n_val], image_ids[n_train + n_val :])
    for name, ratio, chunk in zip(_SPLIT_NAMES, ratios, chunks, strict=True):
        if ratio > 0 and not chunk:
            raise ValueError(f"split '{name}' would be empty; the dataset has only {n_total} images")

    return {name: subset_coco(coco, chunk) for name, chunk in zip(_SPLIT_NAMES, chunks, strict=True)}


def subset_coco(coco: Mapping[str, Any], image_ids: Sequence[int]) -> CocoDict:
    """Return a COCO dictionary restricted to ``image_ids`` (categories are kept whole)."""
    keep = set(image_ids)
    return {
        **{k: v for k, v in coco.items() if k not in ("images", "annotations")},
        "images": [i for i in coco["images"] if i["id"] in keep],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
    }


def write_coco(coco: Mapping[str, Any], path: str | Path) -> Path:
    """Write a COCO dictionary to ``path``, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, separators=(",", ":"))
    return path
