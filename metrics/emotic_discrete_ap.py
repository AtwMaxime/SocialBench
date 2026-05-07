"""
Metric : Average Precision (AP) per category, mAP
Dataset: EMOTIC
Task   : Discrete emotion recognition (multi-label, 26 categories)

GT format  (train)     : {"emotions": ["Happiness", "Engagement"]}
GT format  (val/test)  : {"emotions_per_annotator": [["Happiness"], ["Engagement", "Peace"], ...]}
Pred format            : {"emotions": ["Happiness", "Engagement"]}

For multi-annotator GT, the union of all annotators' labels is used as the positive set.
AP per class uses binary labels (no confidence scores from the model).
mAP = mean AP across all 26 categories.
"""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

CATEGORIES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment",
    "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem",
    "Excitement", "Fatigue", "Fear", "Happiness", "Pain", "Peace",
    "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise",
    "Sympathy", "Yearning",
]


def _gt_labels(gt: dict[str, Any]) -> set[str]:
    if "emotions_per_annotator" in gt:
        return set(e for ann in gt["emotions_per_annotator"] for e in ann)
    return set(gt["emotions"])


def _to_binary(labels: set[str], classes: list[str]) -> list[int]:
    return [1 if c in labels else 0 for c in classes]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_matrix = np.array([_to_binary(_gt_labels(g), CATEGORIES) for g in gts])
    pred_matrix = np.array([_to_binary(set(p.get("emotions", [])), CATEGORIES) for p in preds])

    aps = []
    result = {}
    for i, cat in enumerate(CATEGORIES):
        if gt_matrix[:, i].sum() == 0:
            continue
        ap = average_precision_score(gt_matrix[:, i], pred_matrix[:, i])
        f1 = f1_score(gt_matrix[:, i], pred_matrix[:, i], zero_division=0)
        key = cat.lower().replace("/", "_").replace(" ", "_")
        result[f"ap_{key}"] = float(ap)
        result[f"f1_{key}"] = float(f1)
        aps.append(ap)

    result["mAP"] = float(np.mean(aps)) if aps else float("nan")
    return result
