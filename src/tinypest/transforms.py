"""Augmentation for small-object detection.

Besides flips and colour changes, two transforms target tiny pests:

* **Zoom-in crop** (``RandomIoUCrop``): crops a region that still contains the objects,
  which makes them larger relative to the image after the model's resize.
* **Gaussian noise**: imitates sensor noise in field photographs, so the model does not
  learn to rely on perfectly clean texture.

Boxes, masks and the image are transformed together through ``torchvision.transforms.v2``;
:class:`~torchvision.transforms.v2.SanitizeBoundingBoxes` then removes objects that the
crop cut away, along with their labels and masks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms import v2


@dataclass(frozen=True)
class AugmentConfig:
    """Strength of each training augmentation."""

    hflip_p: float = 0.5
    vflip_p: float = 0.0
    zoom_in_p: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    noise_p: float = 0.3
    noise_sigma: float = 0.03

    def __post_init__(self) -> None:
        for name in ("hflip_p", "vflip_p", "zoom_in_p", "noise_p"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if min(self.brightness, self.contrast, self.noise_sigma) < 0:
            raise ValueError("brightness, contrast and noise_sigma must be non-negative")


class GaussianNoise(v2.Transform):  # type: ignore[misc]
    """Add zero-mean Gaussian noise to float images in ``[0, 1]`` with probability ``p``."""

    def __init__(self, sigma: float = 0.03, p: float = 0.5) -> None:
        super().__init__()
        self.sigma = sigma
        self.p = p

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {"apply": random.random() < self.p}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not params["apply"]:
            return inpt
        if isinstance(inpt, tv_tensors.Image) or (type(inpt) is Tensor and inpt.is_floating_point()):
            noisy = (inpt + torch.randn_like(inpt) * self.sigma).clamp(0.0, 1.0)
            return tv_tensors.wrap(noisy, like=inpt) if isinstance(inpt, tv_tensors.TVTensor) else noisy
        return inpt


class ZoomIn(v2.Transform):  # type: ignore[misc]
    """``RandomIoUCrop`` applied with probability ``p`` and only when the image has objects."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.crop = v2.RandomIoUCrop()

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        boxes = [x for x in _flatten(sample) if isinstance(x, tv_tensors.BoundingBoxes)]
        if not boxes or boxes[0].shape[0] == 0 or random.random() >= self.p:
            return sample
        return self.crop(*inputs)


def _flatten(obj: Any) -> list[Any]:
    if isinstance(obj, dict):
        return [y for v in obj.values() for y in _flatten(v)]
    if isinstance(obj, (list, tuple)):
        return [y for v in obj for y in _flatten(v)]
    return [obj]


def build_transforms(train: bool, config: AugmentConfig | None = None) -> v2.Compose:
    """Build the pipeline. Inputs: ``tv_tensors.Image`` plus a target dict with
    ``boxes`` (``tv_tensors.BoundingBoxes``), ``labels`` and optionally ``masks``."""
    to_float = v2.ToDtype(torch.float32, scale=True)
    if not train:
        return v2.Compose([to_float])
    cfg = config or AugmentConfig()
    steps: list[v2.Transform] = [ZoomIn(cfg.zoom_in_p), v2.RandomHorizontalFlip(cfg.hflip_p)]
    if cfg.vflip_p > 0:
        steps.append(v2.RandomVerticalFlip(cfg.vflip_p))
    if cfg.brightness > 0 or cfg.contrast > 0:
        steps.append(v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast))
    steps.append(to_float)
    if cfg.noise_p > 0 and cfg.noise_sigma > 0:
        steps.append(GaussianNoise(cfg.noise_sigma, cfg.noise_p))
    steps.append(v2.SanitizeBoundingBoxes(min_size=1.0))
    return v2.Compose(steps)
