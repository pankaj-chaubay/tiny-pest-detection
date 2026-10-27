# Data

| Path | Content |
|---|---|
| `ip102_classes.txt` | The 102 IP102 class names, one per line, in index order (line 1 = label `0` in the XML files). |
| `coco/`, `splits/` | Created by `tinypest convert` and `tinypest split`. Not committed. |

## Getting IP102

IP102 (Wu et al., CVPR 2019) is published by its authors for research use and is not redistributed here.

1. Download the detection part (VOC-style `Annotations/` with XML files and `JPEGImages/`) from the official IP102 repository, `github.com/xpwu95/IP102`, or a mirror such as the Kaggle dataset `rtlmhjbn/ip02-dataset`.
2. Convert: `tinypest convert path/to/Annotations --classes data/ip102_classes.txt --out data/coco/all.json`
3. Check the result with `tinypest eda data/coco/all.json` before training.

Object names in IP102's XML files are 0-based class indices. If your copy uses 1-based indices, add `--index-base 1`. If it uses class names, the converter matches them against the class list.
