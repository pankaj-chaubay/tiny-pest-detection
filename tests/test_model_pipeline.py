"""Model construction, metrics and an end-to-end CLI run on CPU with random weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tinypest.cli import main
from tinypest.coco import load_coco
from tinypest.dataset import DetectionDataset, collate_fn
from tinypest.engine import TrainConfig, fit
from tinypest.metrics import coco_map, match_detections, to_coco_results
from tinypest.model import (
    build_model,
    is_mask_model,
    layers_for_epoch,
    parse_unfreeze_schedule,
    set_trainable_backbone_layers,
)
from tinypest.transforms import build_transforms

CPU = torch.device("cpu")


def test_small_anchors_are_used() -> None:
    model = build_model(4, pretrained=False, anchor_sizes=(4, 8, 16, 32, 64))
    assert model.rpn.anchor_generator.sizes == ((4,), (8,), (16,), (32,), (64,))
    assert model.roi_heads.box_predictor.cls_score.out_features == 4
    assert not is_mask_model(model)
    assert is_mask_model(build_model(3, arch="maskrcnn", pretrained=False))


def test_build_model_validation() -> None:
    with pytest.raises(ValueError):
        build_model(1, pretrained=False)
    with pytest.raises(ValueError):
        build_model(3, pretrained=False, anchor_sizes=(8, 16, 32))
    with pytest.raises(ValueError):
        build_model(3, arch="yolo", pretrained=False)  # type: ignore[arg-type]


def test_progressive_unfreezing() -> None:
    model = build_model(3, pretrained=False)
    assert set_trainable_backbone_layers(model, 2) == ["layer4", "layer3"]
    trainable = {n.split(".")[0] for n, p in model.backbone.body.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "layer3"}
    set_trainable_backbone_layers(model, 5)
    assert all(p.requires_grad for p in model.backbone.body.parameters())


def test_unfreeze_schedule_parsing() -> None:
    schedule = parse_unfreeze_schedule("1:2, 4:3,7:5")
    assert schedule == {1: 2, 4: 3, 7: 5}
    assert [layers_for_epoch(schedule, e) for e in (1, 3, 4, 9)] == [2, 2, 3, 5]
    assert layers_for_epoch({}, 3) is None and parse_unfreeze_schedule("") == {}
    for bad in ("1-2", "0:3", "2:9"):
        with pytest.raises(ValueError):
            parse_unfreeze_schedule(bad)


def test_match_detections() -> None:
    gt = torch.tensor([[0.0, 0, 10, 10], [20.0, 20, 30, 30]])
    preds = torch.tensor([[0.0, 0, 10, 10], [1.0, 1, 10, 10], [20.0, 20, 30, 30]])
    counts = match_detections(preds, torch.tensor([0.9, 0.8, 0.7]), torch.tensor([1, 1, 2]), gt, torch.tensor([1, 1]))
    assert (counts.tp, counts.fp, counts.fn) == (1, 2, 1)  # wrong class for the third box
    assert counts.precision == pytest.approx(1 / 3) and counts.recall == pytest.approx(0.5)


def test_perfect_detections_give_full_map(coco_file: Path, voc: dict[str, Path]) -> None:
    ds = DetectionDataset(coco_file, voc["images"])
    results = []
    for i in range(len(ds)):
        _, t = ds[i]
        out = {"boxes": t["boxes"], "labels": t["labels"], "scores": torch.ones(len(t["labels"]))}
        results.extend(to_coco_results(out, int(t["image_id"]), ds.label_to_category))
    stats = coco_map(load_coco(coco_file), results)
    assert stats["AP"] == pytest.approx(1.0) and stats["AP_small"] == pytest.approx(1.0)


def test_fit_with_unfreeze_schedule(coco_file: Path, voc: dict[str, Path], tmp_path: Path) -> None:
    torch.manual_seed(0)
    ds = DetectionDataset(coco_file, voc["images"], build_transforms(True), skip_empty=True)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    model = build_model(ds.num_classes, pretrained=False, min_size=120, max_size=160)
    history = fit(model, loader, None, TrainConfig(epochs=2, lr=0.001, unfreeze={1: 1, 2: 5}), tmp_path, CPU)
    assert [h.trainable_layers for h in history] == [1, 5]
    assert (tmp_path / "last.pt").is_file() and not (tmp_path / "best.pt").exists()


def test_cli_end_to_end(voc: dict[str, Path], tmp_path: Path) -> None:
    coco_path = tmp_path / "coco" / "all.json"
    assert main(["convert", str(voc["xml"]), "--classes", str(voc["classes"]), "--out", str(coco_path)]) == 0
    assert (
        main(["eda", str(coco_path), "--out-dir", str(tmp_path / "eda"), "--min-size", "120", "--max-size", "160"]) == 0
    )
    assert (tmp_path / "eda" / "eda_report.json").is_file()
    assert main(["split", str(coco_path), "--out-dir", str(tmp_path / "s"), "--ratios", "0.5", "0.25", "0.25"]) == 0

    run = tmp_path / "run"
    common = ["--images", str(voc["images"]), "--device", "cpu"]
    split = tmp_path / "s"
    train_args = ["train", "--train", str(split / "train.json"), "--val", str(split / "val.json"),
                  "--out", str(run), "--epochs", "1", "--batch-size", "2", "--workers", "0",
                  "--min-size", "120", "--max-size", "160", "--no-pretrained",
                  "--arch", "maskrcnn", *common]  # fmt: skip
    assert main(train_args) == 0
    assert {"best.pt", "last.pt", "history.csv", "model_meta.json"} <= {p.name for p in run.iterdir()}

    report_path = tmp_path / "report.json"
    eval_args = ["evaluate", str(tmp_path / "s" / "test.json"), "--checkpoint", str(run / "best.pt"),
                 "--compare", "--score-threshold", "0.0", "--out", str(report_path), *common]  # fmt: skip
    assert main(eval_args) == 0
    reports = json.loads(report_path.read_text())
    assert [r["postprocess"] for r in reports] == ["built-in NMS", "nms", "soft-nms"]
    assert reports[0]["mask_map"] is not None

    image = next(voc["images"].iterdir())
    predict_args = ["predict", str(image), "--checkpoint", str(run / "best.pt"),
                    "--out-dir", str(tmp_path / "pred"), "--device", "cpu"]  # fmt: skip
    assert main(predict_args) == 0
    assert main(["curves", str(run / "history.csv"), "--out", str(tmp_path / "curves.png")]) == 0
    assert (tmp_path / "curves.png").is_file()


def test_cli_errors(tmp_path: Path) -> None:
    assert main(["convert", str(tmp_path), "--classes", str(tmp_path / "none.txt")]) == 2
    assert main(["evaluate", "x.json", "--images", str(tmp_path), "--checkpoint", str(tmp_path / "m.pt")]) == 2
