"""
Metric : Intersection over Union (IoU) + L2 Distance (Fan et al. CVPR 2018)
Dataset: VideoCoAttention
Task   : Co-attention localization

GT format  : {"bbox": [x1, y1, x2, y2]}
Pred format: {"bbox": [x1, y1, x2, y2]}

Coordinates in [0, 1000] (Qwen convention). IoU is scale-invariant.
L2 distance is normalized by 1000 (coordinate range), matching the
Fan et al. convention of normalizing by max(W, H) of the image.
"""

import math
from typing import Any


def _parse_bbox(sample: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = sample.get("bbox")
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return x1, y1, x2, y2
    except (TypeError, ValueError):
        return None


def compute_iou(pred: dict[str, Any], gt: dict[str, Any]) -> float | None:
    p = _parse_bbox(pred)
    g = _parse_bbox(gt)
    if p is None or g is None:
        return None
    px1, py1, px2, py2 = p
    gx1, gy1, gx2, gy2 = g

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    gt_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_l2(pred: dict[str, Any], gt: dict[str, Any]) -> float | None:
    """Normalized L2 distance between bbox centers (Fan et al. CVPR 2018)."""
    p = _parse_bbox(pred)
    g = _parse_bbox(gt)
    if p is None or g is None:
        return None
    pcx, pcy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    gcx, gcy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    dist = math.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
    return dist / 1000.0


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    ious, l2s = [], []
    for p, g in zip(preds, gts):
        iou = compute_iou(p, g)
        l2 = compute_l2(p, g)
        if iou is not None:
            ious.append(iou)
        if l2 is not None:
            l2s.append(l2)
    n_used = len(ious)
    return {
        "mean_iou": sum(ious) / n_used if n_used else float("nan"),
        "l2_dist": sum(l2s) / len(l2s) if l2s else float("nan"),
        "n_used": n_used,
    }
