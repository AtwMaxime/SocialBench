"""
Metric : Per-AU F1, macro-average AU F1
Dataset: MMEW
Task   : Action unit detection (apex frame)

GT format  : {"action_units": ["AU6", "AU12", ...]}
Pred format: {"action_units": ["AU6", "AU12", ...]}

Each AU is treated as an independent binary classification.
AU set is inferred from the union of GT labels across the corpus.
"""

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

# MMEW action units seen in the dataset
AU_CLASSES = [
    "AU1", "AU2", "AU4", "AU5", "AU6", "AU7",
    "AU9", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU20", "AU23", "AU24", "AU25", "AU26",
]


def _to_binary(aus: list[str], classes: list[str]) -> list[int]:
    au_set = set(aus)
    return [1 if a in au_set else 0 for a in classes]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_matrix = np.array([_to_binary(g["action_units"], AU_CLASSES) for g in gts])
    pred_matrix = np.array([_to_binary(p["action_units"], AU_CLASSES) for p in preds])

    result = {}
    f1s = []
    for i, au in enumerate(AU_CLASSES):
        if gt_matrix[:, i].sum() == 0:
            continue
        f1 = f1_score(gt_matrix[:, i], pred_matrix[:, i], zero_division=0)
        result[f"f1_{au}"] = float(f1)
        f1s.append(f1)

    result["macro_au_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    return result
