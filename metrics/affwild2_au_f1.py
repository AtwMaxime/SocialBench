"""
Metric : Per-AU F1, macro-average AU F1
Dataset: AffWild2
Task   : Action unit detection

GT format  : {"action_units": ["AU1", "AU6", ...]}
Pred format: {"action_units": ["AU1", "AU6", ...]}

AU set: AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU15, AU23, AU24, AU25, AU26
Per-AU F1 and macro average are the official ABAW metrics.
"""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

AU_COLS = [
    "AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
    "AU12", "AU15", "AU23", "AU24", "AU25", "AU26",
]


def _to_binary(aus: list[str], classes: list[str]) -> list[int]:
    au_set = set(aus)
    return [1 if a in au_set else 0 for a in classes]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_matrix = np.array([_to_binary(g.get("action_units", []), AU_COLS) for g in gts])
    pred_matrix = np.array([_to_binary(p.get("action_units", []), AU_COLS) for p in preds])

    result = {}
    f1s = []
    for i, au in enumerate(AU_COLS):
        if gt_matrix[:, i].sum() == 0:
            continue
        f1 = f1_score(gt_matrix[:, i], pred_matrix[:, i], zero_division=0)
        result[f"f1_{au}"] = float(f1)
        f1s.append(f1)

    result["macro_au_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return result
