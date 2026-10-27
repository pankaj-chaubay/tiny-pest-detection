"""Read Pascal VOC XML annotations and convert them to COCO.

IP102's detection subset ships one VOC XML file per image. Object names are class indices
as strings (``"0"`` is the first line of ``classes.txt``), and coordinates are pixels.

The converter is strict by default: a malformed file stops the run with a message naming
the file. With ``strict=False`` bad files are skipped and listed in the returned report, so
one broken XML file does not block a conversion of 18,000 images.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinypest.errors import AnnotationError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VocObject:
    """One labelled box. Coordinates are pixels, ``x1, y1`` inclusive, ``x2, y2`` exclusive."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    difficult: bool = False

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class VocAnnotation:
    """The content of one VOC XML file."""

    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...]


@dataclass
class ConversionReport:
    """What happened during a conversion."""

    images: int = 0
    objects: int = 0
    skipped_files: list[str] = field(default_factory=list)
    clipped_boxes: int = 0
    dropped_boxes: int = 0


def _text(node: ET.Element, path: str, source: Path) -> str:
    found = node.find(path)
    if found is None or found.text is None or not found.text.strip():
        raise AnnotationError(f"{source.name}: missing <{path}>")
    return found.text.strip()


def _number(node: ET.Element, path: str, source: Path) -> float:
    raw = _text(node, path, source)
    try:
        return float(raw)
    except ValueError as exc:
        raise AnnotationError(f"{source.name}: <{path}> is not a number: {raw!r}") from exc


def parse_voc(path: str | Path) -> VocAnnotation:
    """Parse one VOC XML file.

    Raises:
        AnnotationError: If the file is missing, is not valid XML, or lacks a required tag.
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Annotation file not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise AnnotationError(f"{path.name}: invalid XML ({exc})") from exc

    width = int(_number(root, "size/width", path))
    height = int(_number(root, "size/height", path))
    if width <= 0 or height <= 0:
        raise AnnotationError(f"{path.name}: image size must be positive, got {width}x{height}")

    objects = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        objects.append(
            VocObject(
                name=_text(obj, "name", path),
                x1=_number(obj, "bndbox/xmin", path),
                y1=_number(obj, "bndbox/ymin", path),
                x2=_number(obj, "bndbox/xmax", path),
                y2=_number(obj, "bndbox/ymax", path),
                difficult=difficult is not None and (difficult.text or "0").strip() == "1",
            )
        )
    return VocAnnotation(_text(root, "filename", path), width, height, tuple(objects))


def load_class_names(path: str | Path) -> list[str]:
    """Read one class name per line, ignoring blank lines.

    A leading number (``"1  rice leaf roller"``, as in the original IP102 file) is removed.

    Raises:
        AnnotationError: If the file is missing, empty or has duplicate names.
    """
    path = Path(path)
    if not path.is_file():
        raise AnnotationError(f"Class list not found: {path}")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        names.append(rest.strip() if head.isdigit() and rest.strip() else line)
    if not names:
        raise AnnotationError(f"{path} contains no class names")
    if len(set(names)) != len(names):
        raise AnnotationError(f"{path} contains duplicate class names")
    return names


def resolve_label(name: str, class_names: Sequence[str], index_base: int = 0) -> int:
    """Return the 0-based class index for a VOC object name.

    Numeric names are treated as indices (IP102 style); anything else is looked up by name.

    Raises:
        AnnotationError: If the name is not a known class.
    """
    if name.isdigit():
        index = int(name) - index_base
        if 0 <= index < len(class_names):
            return index
        raise AnnotationError(f"class index {name} is outside 0..{len(class_names) - 1 + index_base}")
    try:
        return list(class_names).index(name)
    except ValueError as exc:
        raise AnnotationError(f"unknown class name {name!r}") from exc


def box_polygon(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Rectangle polygon in COCO order, used as a placeholder mask for box-only data."""
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def voc_to_coco(
    xml_dir: str | Path,
    class_names: Sequence[str],
    index_base: int = 0,
    strict: bool = True,
    min_box_size: float = 1.0,
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert a folder of VOC XML files into one COCO dictionary.

    Image ids are assigned 1, 2, 3 ... in sorted file-name order, so they are stable and
    cannot collide (hash-based ids can). Category ids are ``class index + 1`` for every
    class in ``class_names``, so subsets of the same dataset share one id scheme.
    Boxes are clipped to the image; boxes smaller than ``min_box_size`` after clipping are
    dropped and counted. Each annotation also gets a rectangle polygon as ``segmentation``,
    which lets box-only data run through mask-based tooling; it is not a real outline.

    Args:
        xml_dir: Folder with ``*.xml`` files.
        class_names: Class names in index order.
        index_base: 0 if numeric names start at 0 (IP102), 1 if they start at 1.
        strict: Raise on the first bad file (``True``) or skip it and record it (``False``).
        min_box_size: Minimum width and height in pixels after clipping.

    Returns:
        The COCO dictionary and a :class:`ConversionReport`.

    Raises:
        AnnotationError: If the folder has no XML files, or (strict mode) a file is invalid.
    """
    xml_dir = Path(xml_dir)
    files = sorted(xml_dir.glob("*.xml"))
    if not files:
        raise AnnotationError(f"No .xml files in {xml_dir}")

    coco: dict[str, Any] = {
        "info": {"description": f"Converted from Pascal VOC ({xml_dir.name})"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n, "supercategory": "pest"} for i, n in enumerate(class_names)],
    }
    report = ConversionReport()
    ann_id = 1
    for path in files:
        try:
            voc = parse_voc(path)
            labels = [resolve_label(o.name, class_names, index_base) for o in voc.objects]
        except AnnotationError as exc:
            if strict:
                raise
            LOGGER.warning("Skipping %s: %s", path.name, exc)
            report.skipped_files.append(path.name)
            continue

        image_id = len(coco["images"]) + 1
        coco["images"].append({"id": image_id, "file_name": voc.filename, "width": voc.width, "height": voc.height})
        report.images += 1
        for obj, label in zip(voc.objects, labels, strict=True):
            x1, y1 = max(0.0, obj.x1), max(0.0, obj.y1)
            x2, y2 = min(float(voc.width), obj.x2), min(float(voc.height), obj.y2)
            if (x1, y1, x2, y2) != (obj.x1, obj.y1, obj.x2, obj.y2):
                report.clipped_boxes += 1
            w, h = x2 - x1, y2 - y1
            if w < min_box_size or h < min_box_size:
                report.dropped_boxes += 1
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": label + 1,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "segmentation": [box_polygon(x1, y1, x2, y2)],
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            report.objects += 1
    return coco, report
