from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision import tv_tensors

from tinypest.dataset import DetectionDataset
from tinypest.postprocess import PostprocessConfig, postprocess, soft_nms
from tinypest.transforms import AugmentConfig, GaussianNoise, ZoomIn, build_transforms

# Two pests side by side (IoU about 0.43) and one exact duplicate of the first.
BOXES = torch.tensor([[0.0, 0.0, 10.0, 10.0], [4.0, 0.0, 14.0, 10.0], [0.0, 0.0, 10.0, 10.0], [50, 50, 60, 60]])
SCORES = torch.tensor([0.9, 0.8, 0.7, 0.6])
LABELS = torch.tensor([1, 1, 1, 1])


def test_hard_nms_removes_duplicate_and_keeps_far_box() -> None:
    out = postprocess({"boxes": BOXES, "scores": SCORES, "labels": LABELS}, PostprocessConfig("nms", 0.3, 0.0))
    assert out["scores"].tolist() == pytest.approx([0.9, 0.6])  # the neighbour (IoU 0.43) is lost


def test_soft_nms_keeps_neighbour_with_lower_score() -> None:
    keep, scores = soft_nms(BOXES, SCORES, sigma=0.5)
    order = keep.tolist()
    assert order[0] == 0 and 1 in order  # the neighbour survives
    assert 0.3 < float(scores[order.index(1)]) < 0.8  # decayed, not deleted
    duplicate_score = scores[keep.tolist().index(2)]
    assert duplicate_score < 0.7 * 0.2  # exact duplicate (IoU 1) decays strongly


def test_soft_nms_linear_and_isolated_boxes() -> None:
    far = torch.tensor([[0.0, 0, 5, 5], [100.0, 100, 105, 105]])
    keep, scores = soft_nms(far, torch.tensor([0.5, 0.9]), kind="linear")
    assert keep.tolist() == [1, 0] and scores.tolist() == pytest.approx([0.9, 0.5])


def test_postprocess_is_class_wise_and_keeps_masks_aligned() -> None:
    labels = torch.tensor([1, 2, 1, 1])
    masks = torch.arange(4).float().view(4, 1, 1, 1).expand(4, 1, 3, 3).clone()
    out = postprocess(
        {"boxes": BOXES, "scores": SCORES, "labels": labels, "masks": masks}, PostprocessConfig("nms", 0.3, 0.0)
    )
    assert sorted(out["labels"].tolist()) == [1, 1, 2]  # different class survives the overlap
    for score, mask in zip(out["scores"], out["masks"], strict=True):
        assert SCORES.tolist().index(pytest.approx(float(score))) == int(mask[0, 0, 0])


def test_postprocess_score_threshold_and_empty() -> None:
    out = postprocess(
        {"boxes": BOXES, "scores": SCORES, "labels": LABELS}, PostprocessConfig("none", score_threshold=0.75)
    )
    assert len(out["scores"]) == 2
    empty = {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}
    assert postprocess(empty, PostprocessConfig("soft-nms"))["boxes"].shape == (0, 4)


def test_postprocess_config_validation() -> None:
    with pytest.raises(ValueError):
        PostprocessConfig("fancy")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PostprocessConfig(iou_threshold=0.0)
    with pytest.raises(ValueError):
        soft_nms(torch.zeros(3, 4), torch.zeros(2))


def test_noise_and_zoom_edge_cases() -> None:
    image = tv_tensors.Image(torch.full((3, 20, 20), 0.5))
    noisy = GaussianNoise(sigma=0.1, p=1.0)(image)
    assert float(noisy.min()) >= 0.0 and float(noisy.max()) <= 1.0 and not torch.equal(noisy, image)
    assert torch.equal(GaussianNoise(p=0.0)(image), image)
    no_boxes = {"boxes": tv_tensors.BoundingBoxes(torch.zeros(0, 4), format="XYXY", canvas_size=(20, 20))}
    out_image, _ = ZoomIn(p=1.0)(image, no_boxes)
    assert out_image.shape == image.shape


@pytest.mark.parametrize("seed", range(4))
def test_training_pipeline_keeps_targets_consistent(coco_file: Path, voc: dict[str, Path], seed: int) -> None:
    torch.manual_seed(seed)
    cfg = AugmentConfig(zoom_in_p=1.0, noise_p=1.0)
    ds = DetectionDataset(coco_file, voc["images"], build_transforms(True, cfg), with_masks=True)
    image, target = ds[1]
    n = target["boxes"].shape[0]
    assert image.dtype == torch.float32
    assert target["labels"].shape == (n,) and target["masks"].shape[0] == n
    h, w = image.shape[-2:]
    assert target["masks"].shape[-2:] == (h, w)
    if n:
        assert float(target["boxes"][:, 2].max()) <= w and float(target["boxes"][:, 3].max()) <= h
        assert bool((target["boxes"][:, 2:] > target["boxes"][:, :2]).all())


def test_bad_augment_config() -> None:
    with pytest.raises(ValueError):
        AugmentConfig(zoom_in_p=2.0)
