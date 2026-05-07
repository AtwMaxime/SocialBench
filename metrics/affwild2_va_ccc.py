"""
Metric : Concordance Correlation Coefficient (CCC)
Dataset: AffWild2
Task   : Valence-Arousal estimation

GT format  : {"valence": 0.42, "arousal": -0.18}   — values in [-1, 1]
Pred format: {"valence": 0.42, "arousal": -0.18}

CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)
Final score = (CCC_valence + CCC_arousal) / 2  (ABAW convention)
"""

from typing import Any

import numpy as np


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = np.mean((x - mean_x) * (y - mean_y))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom < 1e-10:
        return 0.0
    return float(2 * cov / denom)


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_v = np.array([g["valence"] for g in gts], dtype=np.float32)
    gt_a = np.array([g["arousal"] for g in gts], dtype=np.float32)
    pred_v = np.array([p["valence"] for p in preds], dtype=np.float32)
    pred_a = np.array([p["arousal"] for p in preds], dtype=np.float32)

    ccc_v = _ccc(pred_v, gt_v)
    ccc_a = _ccc(pred_a, gt_a)
    return {
        "ccc_valence": ccc_v,
        "ccc_arousal": ccc_a,
        "ccc_mean": (ccc_v + ccc_a) / 2,
    }
