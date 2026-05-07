"""
Metric : MAE, Accuracy (exact count)
Dataset: VoxConverse
Task   : Speaker count estimation

GT format  : {"num_speakers": 3}
Pred format: {"num_speakers": 3}
"""

from typing import Any

import numpy as np


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    pairs = []
    for p, g in zip(preds, gts):
        try:
            pairs.append((int(g["num_speakers"]), int(p["num_speakers"])))
        except (KeyError, TypeError, ValueError):
            continue
    n_used = len(pairs)
    if not pairs:
        return {"mae": float("nan"), "exact_accuracy": float("nan"), "n_used": 0}
    gt_counts = np.array([g for g, _ in pairs])
    pred_counts = np.array([p for _, p in pairs])
    return {
        "mae": float(np.abs(gt_counts - pred_counts).mean()),
        "exact_accuracy": float((gt_counts == pred_counts).mean()),
        "n_used": n_used,
    }
