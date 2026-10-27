"""Command-line interface: ``tinypest <command> [options]``.

Commands
--------
``convert``   Pascal VOC XML folder -> one COCO JSON file.
``eda``       Class balance, image and object sizes, anchor-size advice, two charts.
``split``     Train/val/test COCO files, split by image.
``train``     Fine-tune Faster or Mask R-CNN with small anchors and staged unfreezing.
``evaluate``  Precision/recall/F1 and COCO mAP; ``--compare`` tries built-in NMS, NMS and Soft-NMS.
``predict``   Draw detections on images.
``curves``    Plot training and validation loss from ``history.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tinypest.coco import load_coco, split_coco, write_coco
from tinypest.dataset import DetectionDataset, collate_fn, read_rgb
from tinypest.eda import analyse, plot_report
from tinypest.engine import TrainConfig, fit, load_checkpoint, predict
from tinypest.errors import TinyPestError
from tinypest.metrics import evaluate
from tinypest.model import TINY_ANCHORS, build_model, parse_unfreeze_schedule
from tinypest.postprocess import PostprocessConfig
from tinypest.transforms import AugmentConfig, build_transforms
from tinypest.visualize import overlay, save_grid
from tinypest.voc import load_class_names, voc_to_coco

LOGGER = logging.getLogger("tinypest")
META_FILE = "model_meta.json"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dump(data: Any, out: str | None) -> None:
    text = json.dumps(data, indent=2)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")


def cmd_convert(args: argparse.Namespace) -> int:
    names = load_class_names(args.classes)
    coco, report = voc_to_coco(args.xml_dir, names, index_base=args.index_base, strict=not args.lenient)
    path = write_coco(coco, args.out)
    LOGGER.info(
        "%d images, %d objects -> %s (clipped %d, dropped %d, skipped files %d)",
        report.images, report.objects, path, report.clipped_boxes, report.dropped_boxes, len(report.skipped_files),
    )  # fmt: skip
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    coco = load_coco(args.annotations)
    report = analyse(coco, args.min_size, args.max_size)
    _dump(report.to_dict(), str(Path(args.out_dir) / "eda_report.json"))
    for path in plot_report(coco, args.out_dir, args.min_size, args.max_size):
        LOGGER.info("Saved %s", path)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    for name, subset in split_coco(load_coco(args.annotations), ratios=args.ratios, seed=args.seed).items():
        path = write_coco(subset, Path(args.out_dir) / f"{name}.json")
        LOGGER.info("%-5s %5d images -> %s", name, len(subset["images"]), path)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    device = _device(args.device)
    with_masks = args.arch == "maskrcnn"
    aug = AugmentConfig(zoom_in_p=args.zoom_in, noise_p=args.noise)
    train_ds = DetectionDataset(args.train, args.images, build_transforms(True, aug), with_masks, skip_empty=True)
    val_ds = DetectionDataset(args.val, args.images, build_transforms(False), with_masks) if args.val else None
    kw: dict[str, Any] = {"batch_size": args.batch_size, "collate_fn": collate_fn, "num_workers": args.workers}
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw) if val_ds is not None else None

    schedule = parse_unfreeze_schedule(args.unfreeze)
    model = build_model(
        train_ds.num_classes, arch=args.arch, anchor_sizes=args.anchors, pretrained=not args.no_pretrained,
        trainable_backbone_layers=args.trainable_layers, min_size=args.min_size, max_size=args.max_size,
    )  # fmt: skip
    meta = {
        "arch": args.arch, "anchors": list(args.anchors), "num_classes": train_ds.num_classes,
        "class_names": train_ds.class_names, "min_size": args.min_size, "max_size": args.max_size,
    }  # fmt: skip
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    config = TrainConfig(epochs=args.epochs, lr=args.lr, lr_step=args.lr_step, seed=args.seed, unfreeze=schedule)
    LOGGER.info("%s on %s, %d train images, anchors %s", args.arch, device, len(train_ds), tuple(args.anchors))
    fit(model, train_loader, val_loader, config, args.out, device)
    return 0


def _load_trained(checkpoint: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    meta_path = Path(checkpoint).with_name(META_FILE)
    if not meta_path.is_file():
        raise FileNotFoundError(f"{META_FILE} not found next to {checkpoint}; it is written by 'tinypest train'")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = build_model(
        meta["num_classes"], arch=meta["arch"], anchor_sizes=meta["anchors"], pretrained=False,
        min_size=meta["min_size"], max_size=meta["max_size"],
    )  # fmt: skip
    return load_checkpoint(model, checkpoint, device), meta


def cmd_evaluate(args: argparse.Namespace) -> int:
    device = _device(args.device)
    model, meta = _load_trained(args.checkpoint, device)
    dataset = DetectionDataset(args.annotations, args.images, build_transforms(False), meta["arch"] == "maskrcnn")
    settings: list[PostprocessConfig | None]
    if args.compare:
        settings = [None, PostprocessConfig("nms"), PostprocessConfig("soft-nms")]
    else:
        settings = [None if args.postprocess == "builtin" else PostprocessConfig(args.postprocess)]
    reports = [evaluate(model, dataset, device, s, args.score_threshold).to_dict() for s in settings]
    _dump(reports if len(reports) > 1 else reports[0], args.out)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    device = _device(args.device)
    model, meta = _load_trained(args.checkpoint, device)
    post = None if args.postprocess == "builtin" else PostprocessConfig(args.postprocess)
    out_dir = Path(args.out_dir)
    for path in args.images:
        image = read_rgb(path).float() / 255
        result = predict(model, [image], device, args.score_threshold, post)[0]
        names = meta["class_names"]
        captions = [f"{names[int(c)]} {s:.2f}" for c, s in zip(result["labels"], result["scores"], strict=True)]
        masks = result.get("masks", torch.zeros((0, *image.shape[-2:]), dtype=torch.bool))
        panel = overlay(image, masks, result["boxes"], captions)
        target = save_grid(
            [panel], [f"{Path(path).name}: {len(captions)} detections"], out_dir / f"{Path(path).stem}_pred.png", cols=1
        )
        LOGGER.info("%s -> %s", path, target)
    return 0


def cmd_curves(args: argparse.Namespace) -> int:
    with Path(args.history).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{args.history} has no rows")
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [float(r["train_loss"]) for r in rows], marker="o", label="train", color="#1F3864")
    if rows[0]["val_loss"]:
        ax.plot(epochs, [float(r["val_loss"]) for r in rows], marker="o", label="validation", color="#C00000")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title("Training convergence")
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved %s", args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinypest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert", help="VOC XML -> COCO JSON")
    p.add_argument("xml_dir")
    p.add_argument("--classes", required=True, help="one class name per line")
    p.add_argument("--out", default="data/coco/all.json")
    p.add_argument("--index-base", type=int, default=0, choices=(0, 1), help="first numeric class id in the XML")
    p.add_argument("--lenient", action="store_true", help="skip bad files instead of stopping")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("eda", help="dataset analysis and anchor advice")
    p.add_argument("annotations")
    p.add_argument("--out-dir", default="outputs/eda")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("split", help="train/val/test split by image")
    p.add_argument("annotations")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("train", help="fine-tune a detector")
    p.add_argument("--train", required=True)
    p.add_argument("--val")
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--arch", choices=("fasterrcnn", "maskrcnn"), default="fasterrcnn")
    p.add_argument("--anchors", type=int, nargs=5, default=TINY_ANCHORS, metavar="PX")
    p.add_argument("--trainable-layers", type=int, default=3)
    p.add_argument("--unfreeze", default="", help="staged unfreezing, e.g. '1:2,4:3,7:5'")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--lr-step", type=int, default=4)
    p.add_argument("--zoom-in", type=float, default=0.5, help="probability of the zoom-in crop")
    p.add_argument("--noise", type=float, default=0.3, help="probability of Gaussian noise")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="metrics for a checkpoint")
    p.add_argument("annotations")
    p.add_argument("--images", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--postprocess", choices=("builtin", "nms", "soft-nms", "none"), default="builtin")
    p.add_argument("--compare", action="store_true", help="evaluate built-in NMS, NMS and Soft-NMS")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--out")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="draw detections")
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--postprocess", choices=("builtin", "nms", "soft-nms", "none"), default="soft-nms")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--out-dir", default="outputs/predictions")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("curves", help="plot loss curves")
    p.add_argument("history")
    p.add_argument("--out", default="outputs/loss_curves.png")
    p.set_defaults(func=cmd_curves)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Expected errors end with a message and exit code 2."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")  # fmt: skip
    try:
        return int(args.func(args))
    except (TinyPestError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
