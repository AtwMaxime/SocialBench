"""
Metric : AUC (heatmap-based)
Dataset: GazeFollow
Task   : Gaze target estimation

GT format  (validation): {"gaze_points": [[x1, y1], [x2, y2], ...]}
GT format  (train)     : {"gaze_point": [x, y]}
Pred format            : {"gaze_point": [x, y]}

Approach (standard for point-output models):
  - Predicted point → 2D Gaussian heatmap (σ = SIGMA on a HEATMAP_SIZE grid)
  - GT points       → sum of 2D Gaussians, normalized to [0, 1]
  - AUC computed by treating each pixel as a binary sample:
      positive  = pixels where GT heatmap > 0.5
      score     = pred heatmap value at that pixel
  Uses sklearn.metrics.roc_auc_score over the flattened heatmaps.

Coordinates are in [0, 1000], mapped to [0, HEATMAP_SIZE-1] internally.
"""

import math
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

HEATMAP_SIZE = 64
SIGMA = 3.0


def _gaussian_heatmap(point: tuple[float, float], size: int, sigma: float) -> np.ndarray:
    """2D Gaussian heatmap centered at `point` (in [0, size-1] coords)."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    heatmap = np.exp(
        -((xg - point[0]) ** 2 + (yg - point[1]) ** 2) / (2 * sigma**2)
    )
    return heatmap


def _to_heatmap_coord(v: float, size: int) -> float:
    return (v / 1000.0) * (size - 1)


def _pred_heatmap(pred: dict[str, Any]) -> np.ndarray:
    gp = pred.get("gaze_point", [])
    if not isinstance(gp, list) or len(gp) < 2:
        return np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    px = _to_heatmap_coord(gp[0], HEATMAP_SIZE)
    py = _to_heatmap_coord(gp[1], HEATMAP_SIZE)
    return _gaussian_heatmap((px, py), HEATMAP_SIZE, SIGMA)


def _gt_heatmap(gt: dict[str, Any]) -> np.ndarray:
    if "gaze_points" in gt:
        points = gt["gaze_points"]
    else:
        points = [gt["gaze_point"]]
    heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    for p in points:
        px = _to_heatmap_coord(p[0], HEATMAP_SIZE)
        py = _to_heatmap_coord(p[1], HEATMAP_SIZE)
        heatmap += _gaussian_heatmap((px, py), HEATMAP_SIZE, SIGMA)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap


def compute_auc(pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, float]:
    pred_h = _pred_heatmap(pred).flatten()
    gt_h = _gt_heatmap(gt).flatten()
    gt_binary = (gt_h >= 0.5).astype(np.int32)
    if gt_binary.sum() == 0 or gt_binary.sum() == len(gt_binary):
        return {"auc": float("nan")}
    auc = roc_auc_score(gt_binary, pred_h)
    return {"auc": float(auc)}


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    aucs = [compute_auc(p, g)["auc"] for p, g in zip(preds, gts)]
    valid = [a for a in aucs if not math.isnan(a)]
    return {"auc": sum(valid) / len(valid) if valid else float("nan")}
