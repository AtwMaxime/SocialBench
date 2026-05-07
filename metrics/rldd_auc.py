"""
Metric : AUC (Area Under ROC Curve)
Dataset: RLDD (Real-Life Deception Detection)
Task   : Deception detection (binary)

GT format  : {"label": "deceptive" | "truthful"}
Pred format: {"label": "deceptive" | "truthful"}
          OR {"label": "deceptive", "confidence": 0.87}  — if model outputs a score

Without a confidence score, AUC is computed from the hard binary prediction,
which degenerates to a single point on the ROC curve. Provide "confidence"
(probability of being deceptive) for a meaningful AUC.
"""

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _score(d: dict[str, Any]) -> float:
    if "confidence" in d:
        return float(d["confidence"])
    return 1.0 if d["label"] == "deceptive" else 0.0


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels = np.array([1 if g["label"] == "deceptive" else 0 for g in gts])
    pred_scores = np.array([_score(p) for p in preds])

    if gt_labels.sum() == 0 or gt_labels.sum() == len(gt_labels):
        return {"auc": float("nan")}

    return {"auc": float(roc_auc_score(gt_labels, pred_scores))}
