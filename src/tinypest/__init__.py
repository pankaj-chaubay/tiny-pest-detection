"""tinypest: tiny insect pest detection on IP102 with Faster / Mask R-CNN.

* :mod:`tinypest.voc` reads Pascal VOC XML and converts it to COCO.
* :mod:`tinypest.eda` measures class balance and object sizes and proposes anchor sizes.
* :mod:`tinypest.transforms` adds zoom-in crops and sensor noise for small objects.
* :mod:`tinypest.model` builds detectors with small anchors and staged backbone unfreezing.
* :mod:`tinypest.postprocess` implements class-wise NMS and Soft-NMS.
* :mod:`tinypest.metrics` reports precision/recall/F1 and COCO mAP by object size.
"""

from tinypest.errors import AnnotationError, DatasetError, TinyPestError

__all__ = ["AnnotationError", "DatasetError", "TinyPestError", "__version__"]
__version__ = "1.0.0"
