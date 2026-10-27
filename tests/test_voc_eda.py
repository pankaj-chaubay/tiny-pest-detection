from __future__ import annotations

from pathlib import Path

import pytest

from tinypest.coco import load_coco, validate_coco
from tinypest.eda import analyse, anchor_coverage, plot_report, resized_object_sides, suggest_anchor_sizes
from tinypest.errors import AnnotationError
from tinypest.voc import load_class_names, parse_voc, resolve_label, voc_to_coco

REAL_CLASSES = Path(__file__).resolve().parents[1] / "data" / "ip102_classes.txt"


def test_parse_voc(voc: dict[str, Path]) -> None:
    ann = parse_voc(voc["xml"] / "IP000000001.xml")
    assert ann.filename == "IP000000001.jpg" and (ann.width, ann.height) == (160, 120)
    assert len(ann.objects) == 2 and all(o.width > 0 for o in ann.objects)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("<annotation><size><width>10</width></size></annotation>", "size/height"),
        ("<annotation", "invalid XML"),
        (
            "<annotation><filename>a.jpg</filename><size><width>x</width><height>5</height></size></annotation>",
            "not a number",
        ),
    ],
)
def test_parse_voc_errors(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.xml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AnnotationError, match=message):
        parse_voc(path)


def test_class_names_strip_numbers_and_whitespace(voc: dict[str, Path]) -> None:
    assert load_class_names(voc["classes"]) == ["rice leaf roller", "rice leaf caterpillar", "paddy stem maggot"]


def test_class_names_reject_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "c.txt"
    path.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="duplicate"):
        load_class_names(path)


def test_resolve_label() -> None:
    names = ["a", "b", "c"]
    assert resolve_label("0", names) == 0
    assert resolve_label("3", names, index_base=1) == 2
    assert resolve_label("b", names) == 1
    with pytest.raises(AnnotationError):
        resolve_label("3", names)
    with pytest.raises(AnnotationError):
        resolve_label("zebra", names)


@pytest.mark.skipif(not REAL_CLASSES.is_file(), reason="class list not present")
def test_ip102_class_list() -> None:
    names = load_class_names(REAL_CLASSES)
    assert len(names) == 102 and names[0] == "rice leaf roller"


def test_voc_to_coco(voc: dict[str, Path]) -> None:
    coco, report = voc_to_coco(voc["xml"], load_class_names(voc["classes"]))
    validate_coco(coco)
    assert [i["id"] for i in coco["images"]] == list(range(1, 9))
    assert len(coco["categories"]) == 3 and coco["categories"][0]["id"] == 1
    assert report.clipped_boxes == 1 and report.objects == len(coco["annotations"])
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        assert x >= 0 and y >= 0 and x + w <= 160 and y + h <= 120


def test_voc_to_coco_strict_and_lenient(voc: dict[str, Path]) -> None:
    (voc["xml"] / "broken.xml").write_text("<annotation>", encoding="utf-8")
    names = load_class_names(voc["classes"])
    with pytest.raises(AnnotationError):
        voc_to_coco(voc["xml"], names)
    coco, report = voc_to_coco(voc["xml"], names, strict=False)
    assert report.skipped_files == ["broken.xml"] and len(coco["images"]) == 8


def test_empty_folder(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match=r"No \.xml"):
        voc_to_coco(tmp_path, ["a"])


def test_anchor_advice_prefers_small_anchors(coco_file: Path) -> None:
    coco = load_coco(coco_file)
    advice = suggest_anchor_sizes(coco, min_size=120, max_size=160)
    assert advice.sizes[0] <= 8 and len(advice.sizes) == 5
    assert advice.coverage >= advice.default_coverage
    assert advice.coverage > 0.9


def test_anchor_advice_never_exceeds_image() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 400, "height": 300}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 350, 270]}],
        "categories": [{"id": 1, "name": "big"}],
    }
    advice = suggest_anchor_sizes(coco, min_size=800, max_size=1333)
    assert max(advice.sizes) <= 1333 and advice.sizes == (64, 128, 256, 512, 1024)


def test_anchor_coverage_bounds() -> None:
    import numpy as np

    sides = np.array([8.0, 16.0, 300.0])
    assert anchor_coverage(sides, (8, 16, 32, 64, 128)) == pytest.approx(2 / 3)
    assert anchor_coverage(np.array([]), (8,)) == 0.0


def test_resized_sides_scale(coco_file: Path) -> None:
    coco = load_coco(coco_file)
    native = resized_object_sides(coco, min_size=120, max_size=160)
    doubled = resized_object_sides(coco, min_size=240, max_size=320)
    assert doubled == pytest.approx(native * 2)


def test_analyse_and_plots(coco_file: Path, tmp_path: Path) -> None:
    coco = load_coco(coco_file)
    report = analyse(coco, 120, 160)
    assert report.images == 8 and report.images_without_objects == 1
    assert report.size_buckets["small"] >= report.objects - 1
    paths = plot_report(coco, tmp_path / "eda", 120, 160)
    assert all(p.is_file() and p.stat().st_size > 1000 for p in paths)
