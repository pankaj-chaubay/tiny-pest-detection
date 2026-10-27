# Review of the original coursework code

**Assignment:** *Practical skills assessment: tiny object detection and classification with Mask R-CNN on IP102* (BSBI Berlin, M.Sc. Data Analytics, submitted 5 February 2025).

**Original form:** Google Colab notebooks (dataset exploration, VOC to COCO conversion, Mask R-CNN definition, training loop, fine-tuning and NMS), shown as screenshots in the written report, plus a standalone `Task 3 code.txt` script for the Matterport Mask R-CNN library.

## Summary

| Area | Rating | Main issue |
|---|---|---|
| Correctness | Low | Undefined variable in the dataset class; box rectangles used as masks; hash ids can collide. |
| Brief coverage | Partial | Augmentation, Soft-NMS, staged fine-tuning and evaluation were described but not run. |
| Code structure | Medium-low | Reasonable functions, but notebook state, repeated `main1..main4` blocks and duplicated model code. |
| Robustness | Low | Broad `except Exception: print(...)` hides failures; no validation of XML or labels. |
| Reproducibility | Low | Colab/Drive paths, no seed, no saved split, one epoch logged, weights not kept. |
| Readability | Medium | Docstrings on most functions; names are clear. |

## Findings

1. **Undefined name in the dataset.** `CustomDataset.__getitem__` uses `num_classes`, which is only defined later as a global. It runs in a notebook by accident of cell order and fails anywhere else.
2. **Silent label corruption.** `labels = [l if 1 <= l <= num_classes - 1 else 0 ...]` turns unknown classes into background instead of reporting them.
3. **Rectangles as masks.** `xml_to_coco` writes each box as a 4-point polygon, and the Mask R-CNN mask head is trained on those. The model learns to draw rectangles; for box-only data Faster R-CNN is the right model.
4. **Image id collisions.** `int(md5(filename)) % 10**9` can map two files to one id; COCO tools would then merge their annotations.
5. **Slow and fragile tensor creation.** `torch.as_tensor(list_of_numpy_masks)` triggers PyTorch's "extremely slow" warning (visible in the report output).
6. **"Tiny object" assumption not checked.** Anchors were reduced to 16 px, but the 200-image subset contains no small boxes (see the README); the report's claim that anchor optimisation "gave much better results" has no measurement behind it.
7. **Described, not implemented.** Augmentation was demonstrated on one image but not used in training; Soft-NMS, progressive unfreezing, heatmaps and mAP appear only in prose.
8. **Matterport script (`Task 3 code.txt`).** Uses `imgaug` without importing it, imports `train_test_split` without using it, and targets TensorFlow 1.x, which current Colab no longer supports.
9. **Error handling.** `try: ... except Exception as e: print(...)` in every entry point turns crashes into log lines and lets the notebook continue in a broken state.

## What was good

- The overall plan matched the brief well: explore, convert, adapt the model, train, evaluate, fine-tune, post-process.
- Replacing both predictors and building a custom `AnchorGenerator` is the correct way to adapt torchvision's Mask R-CNN.
- The class-wise IoU function and the NMS helper were correct.

## How each finding is addressed

| Finding | Fix in this repository |
|---|---|
| 1, 2 | Labels are resolved once in `tinypest.voc.resolve_label`; unknown classes raise `AnnotationError` naming the file. |
| 3 | `--arch fasterrcnn` is the default for box data; Mask R-CNN is used only with real masks. |
| 4 | Sequential image ids in sorted file order. |
| 5 | Masks are stacked with NumPy once, then converted. |
| 6 | `tinypest eda` measures object sizes after resize and advises anchors; the README reports the result. |
| 7 | Zoom-in crop, noise, Soft-NMS and staged unfreezing are implemented, configurable and tested; `evaluate --compare` reports mAP for each post-processing choice. |
| 8 | Replaced by the torchvision implementation; no TensorFlow dependency. |
| 9 | Specific exception types with clear messages; the CLI exits with code 2 on expected errors and lets unexpected ones surface. |
