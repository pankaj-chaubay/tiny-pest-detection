"""Faster / Mask R-CNN tuned for small objects, with staged backbone unfreezing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from torch import nn
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

Arch = Literal["fasterrcnn", "maskrcnn"]
TINY_ANCHORS: tuple[int, ...] = (8, 16, 32, 64, 128)
DEFAULT_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)

# ResNet stages from the top of the network down; unfreezing goes in this order.
_BACKBONE_STAGES = ("layer4", "layer3", "layer2", "layer1", "conv1")


def build_model(
    num_classes: int,
    *,
    arch: Arch = "fasterrcnn",
    anchor_sizes: Sequence[int] = TINY_ANCHORS,
    aspect_ratios: Sequence[float] = DEFAULT_RATIOS,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
    detections_per_image: int = 100,
) -> FasterRCNN:
    """Build a ResNet-50-FPN detector with anchors sized for small objects.

    The default anchors (8-128 px) are four times smaller than torchvision's (32-512 px).
    Keeping three aspect ratios per location keeps the RPN head's shape, so the COCO-trained
    RPN weights still load; only the anchor geometry changes.

    Args:
        num_classes: Classes **including** the background.
        arch: ``"fasterrcnn"`` for boxes (right for box-only data such as IP102) or
            ``"maskrcnn"`` to add a mask head (needs real masks to be useful).
        anchor_sizes: One size per FPN level (five levels).
        aspect_ratios: Height/width ratios used at every level.
        pretrained: Start from COCO weights (downloaded on first use).
        trainable_backbone_layers: ResNet stages fine-tuned from the start (0-5).
        min_size: Shorter side after the model's resize. Raising it enlarges tiny objects.
        max_size: Cap on the longer side after resizing.
        detections_per_image: Maximum detections returned per image.

    Raises:
        ValueError: On invalid ``num_classes``, anchor settings or layer count.
    """
    if num_classes < 2:
        raise ValueError("num_classes counts the background, so it must be at least 2")
    if len(anchor_sizes) != 5 or any(s <= 0 for s in anchor_sizes):
        raise ValueError("anchor_sizes needs five positive sizes, one per FPN level")
    if pretrained and len(aspect_ratios) != 3:
        raise ValueError("pretrained RPN weights expect exactly three aspect ratios")
    if not 0 <= trainable_backbone_layers <= 5:
        raise ValueError("trainable_backbone_layers must be between 0 and 5")

    anchors = AnchorGenerator(
        sizes=tuple((int(s),) for s in anchor_sizes),
        aspect_ratios=(tuple(float(r) for r in aspect_ratios),) * len(anchor_sizes),
    )
    common = {
        "weights_backbone": None,
        "trainable_backbone_layers": trainable_backbone_layers if pretrained else None,
        "rpn_anchor_generator": anchors,
        "min_size": min_size,
        "max_size": max_size,
        "box_detections_per_img": detections_per_image,
    }
    model: FasterRCNN
    if arch == "maskrcnn":
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None, **common)
        mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
    elif arch == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None, **common
        )
    else:
        raise ValueError(f"unknown arch {arch!r}")

    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, num_classes)
    return model


def is_mask_model(model: nn.Module) -> bool:
    """True if the model predicts masks."""
    return isinstance(model, MaskRCNN)


def set_trainable_backbone_layers(model: nn.Module, layers: int) -> list[str]:
    """Unfreeze the top ``layers`` ResNet stages and freeze the rest.

    Used for progressive unfreezing: train the heads and top stages first, then open up
    lower stages once the new heads have settled.

    Returns:
        Names of the stages that are now trainable.
    """
    if not 0 <= layers <= 5:
        raise ValueError("layers must be between 0 and 5")
    body: nn.Module = model.backbone.body  # type: ignore[union-attr,assignment]
    open_stages = list(_BACKBONE_STAGES[:layers])
    for name, param in body.named_parameters():
        param.requires_grad = any(name.startswith(stage) for stage in open_stages) or (
            "conv1" in open_stages and name.startswith("bn1")
        )
    return open_stages


def parse_unfreeze_schedule(spec: str) -> dict[int, int]:
    """Parse ``"1:3,4:4,7:5"`` into ``{epoch: trainable_layers}``.

    Raises:
        ValueError: If the text is malformed or values are out of range.
    """
    schedule: dict[int, int] = {}
    if not spec.strip():
        return schedule
    for part in spec.split(","):
        try:
            epoch_text, layers_text = part.split(":")
            epoch, layers = int(epoch_text), int(layers_text)
        except ValueError as exc:
            raise ValueError(f"bad unfreeze entry {part!r}; use EPOCH:LAYERS, e.g. 1:3,5:5") from exc
        if epoch < 1 or not 0 <= layers <= 5:
            raise ValueError(f"bad unfreeze entry {part!r}")
        schedule[epoch] = layers
    return schedule


def layers_for_epoch(schedule: Mapping[int, int], epoch: int) -> int | None:
    """Trainable layers that apply at ``epoch`` (the latest entry at or before it)."""
    applicable = [e for e in schedule if e <= epoch]
    return schedule[max(applicable)] if applicable else None
