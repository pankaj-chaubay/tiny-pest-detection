"""PyTorch dataset that serves COCO boxes (and optional masks) to torchvision detectors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors

from tinypest.coco import annotation_to_mask, load_coco, validate_coco
from tinypest.errors import DatasetError

Target = dict[str, Tensor]
Sample = tuple[Tensor, Target]
TransformFn = Callable[[Any, Any], tuple[Any, Any]]


def read_rgb(path: str | Path) -> Tensor:
    """Read an image file as a ``uint8`` tensor of shape ``(3, H, W)`` in RGB order."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


class DetectionDataset(Dataset[Sample]):
    """Images with boxes, labels and (optionally) masks from a COCO file.

    Args:
        annotations: COCO JSON path or an already-loaded dictionary.
        image_dir: Folder holding ``images[*].file_name``.
        transforms: Callable from :func:`tinypest.transforms.build_transforms`.
        with_masks: Also return ``masks``. Needed for Mask R-CNN, not for Faster R-CNN.
        skip_empty: Leave out images without objects (useful for training on big datasets
            where negatives are plentiful). Keep them for evaluation.
    """

    def __init__(
        self,
        annotations: str | Path | Mapping[str, Any],
        image_dir: str | Path,
        transforms: TransformFn | None = None,
        with_masks: bool = False,
        skip_empty: bool = False,
    ) -> None:
        if isinstance(annotations, Mapping):
            validate_coco(annotations)
            self.coco: Mapping[str, Any] = annotations
        else:
            self.coco = load_coco(annotations)
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise DatasetError(f"Image folder not found: {self.image_dir}")
        self.transforms = transforms
        self.with_masks = with_masks

        categories = sorted(self.coco["categories"], key=lambda c: c["id"])
        self.category_to_label = {c["id"]: i + 1 for i, c in enumerate(categories)}
        self.label_to_category = {v: k for k, v in self.category_to_label.items()}
        self.class_names = ["__background__", *(c["name"] for c in categories)]

        self._anns: dict[int, list[Mapping[str, Any]]] = {i["id"]: [] for i in self.coco["images"]}
        for ann in self.coco["annotations"]:
            self._anns[ann["image_id"]].append(ann)
        images = sorted(self.coco["images"], key=lambda i: i["id"])
        self.images = [i for i in images if self._anns[i["id"]] or not skip_empty]

        missing = [i["file_name"] for i in self.images if not (self.image_dir / i["file_name"]).is_file()]
        if missing:
            raise DatasetError(f"{len(missing)} image(s) missing from {self.image_dir}, for example {missing[:3]}")

    @property
    def num_classes(self) -> int:
        """Number of classes including the background."""
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        info = self.images[index]
        path = self.image_dir / info["file_name"]
        try:
            image = read_rgb(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise DatasetError(f"Cannot read image {path}: {exc}") from exc
        height, width = int(info["height"]), int(info["width"])
        if tuple(image.shape[-2:]) != (height, width):
            raise DatasetError(f"{path} is {tuple(image.shape[-2:])} (H, W) but the annotation says {(height, width)}")

        anns = self._anns[info["id"]]
        xywh = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32).reshape(-1, 4)
        xyxy = torch.cat([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], dim=1)
        target: dict[str, Any] = {
            "boxes": tv_tensors.BoundingBoxes(xyxy, format="XYXY", canvas_size=(height, width)),
            "labels": torch.tensor([self.category_to_label[a["category_id"]] for a in anns], dtype=torch.int64),
        }
        if self.with_masks:
            stack = [annotation_to_mask(a, height, width) for a in anns]
            masks = np.stack(stack) if stack else np.zeros((0, height, width), dtype=np.uint8)
            target["masks"] = tv_tensors.Mask(torch.from_numpy(masks))

        img: Any = tv_tensors.Image(image)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        out: Target = {
            "boxes": torch.as_tensor(target["boxes"]).to(torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(target["labels"]),
            "image_id": torch.tensor(int(info["id"]), dtype=torch.int64),
        }
        if self.with_masks:
            out["masks"] = torch.as_tensor(target["masks"]).to(torch.uint8)
        return torch.as_tensor(img), out


def collate_fn(batch: Sequence[Sample]) -> tuple[tuple[Tensor, ...], tuple[Target, ...]]:
    """Detection models take lists of differently sized images, so do not stack."""
    images, targets = zip(*batch, strict=True)
    return tuple(images), tuple(targets)
