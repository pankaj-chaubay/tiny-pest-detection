"""Synthetic IP102-style data: small images with tiny bright 'pests' and VOC XML files.

Object names are numeric class indices, as in IP102. One image has no objects, one box
spills past the image border (to exercise clipping) and the rest are tiny (6-14 px).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tinypest.voc import load_class_names, voc_to_coco

CLASSES = ["rice leaf roller", "rice leaf caterpillar", "paddy stem maggot"]
HEIGHT, WIDTH = 120, 160


def _xml(filename: str, boxes: list[tuple[int, int, int, int, int]]) -> str:
    objects = "".join(
        f"<object><name>{c}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>"
        f"<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
        for c, x1, y1, x2, y2 in boxes
    )
    return (
        f"<annotation><folder>IP102</folder><filename>{filename}</filename>"
        f"<size><width>{WIDTH}</width><height>{HEIGHT}</height><depth>3</depth></size>"
        f"<segmented>0</segmented>{objects}</annotation>"
    )


def make_voc_dataset(root: Path, n_images: int = 8) -> dict[str, Path]:
    """Write images, XML files and a classes file; return their locations."""
    images, xmls = root / "JPEGImages", root / "Annotations"
    images.mkdir(parents=True, exist_ok=True)
    xmls.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n_images):
        name = f"IP{i:09d}"
        pixels = np.full((HEIGHT, WIDTH, 3), (60, 110, 50), dtype=np.uint8)
        pixels = np.clip(pixels + rng.normal(0, 4, pixels.shape), 0, 255).astype(np.uint8)
        boxes: list[tuple[int, int, int, int, int]] = []
        n_objects = 0 if i == n_images - 1 else 1 + i % 3
        for _ in range(n_objects):
            side = int(rng.integers(6, 15))
            x1, y1 = int(rng.integers(5, WIDTH - side - 5)), int(rng.integers(5, HEIGHT - side - 5))
            cls = int(rng.integers(0, len(CLASSES)))
            pixels[y1 : y1 + side, x1 : x1 + side] = (235, 210, 40 + 60 * cls)
            boxes.append((cls, x1, y1, x1 + side, y1 + side))
        if i == 0:
            boxes.append((0, WIDTH - 10, 10, WIDTH + 20, 30))  # spills past the right edge
            pixels[10:30, WIDTH - 10 :] = (235, 210, 40)
        Image.fromarray(pixels).save(images / f"{name}.jpg", quality=95)
        (xmls / f"{name}.xml").write_text(_xml(f"{name}.jpg", boxes), encoding="utf-8")
    classes = root / "classes.txt"
    classes.write_text("".join(f"{i + 1}  {n} \t\t\r\n" for i, n in enumerate(CLASSES)), encoding="utf-8")
    return {"images": images, "xml": xmls, "classes": classes}


@pytest.fixture()
def voc(tmp_path: Path) -> dict[str, Path]:
    return make_voc_dataset(tmp_path)


@pytest.fixture()
def coco_file(voc: dict[str, Path], tmp_path: Path) -> Path:
    import json

    coco, _ = voc_to_coco(voc["xml"], load_class_names(voc["classes"]))
    path = tmp_path / "all.json"
    path.write_text(json.dumps(coco), encoding="utf-8")
    return path
