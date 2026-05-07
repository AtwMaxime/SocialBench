"""
Metric : mAP, per-class AP, per-class Recall
Dataset: PISC
Task   : Social relationship recognition (6-class)

GT format  : {"relationship": "Friends"}
Pred format: {"relationship": "Friends"}

mAP = mean of per-class AP computed from binary one-vs-rest classification.
AP uses hard binary labels (no confidence score).
"""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, recall_score, f1_score

CLASSES = ["Friends", "Family", "Couple", "Professional", "Commercial", "No relation"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels = [g["relationship"] for g in gts]
    pred_labels = [p.get("relationship") or p.get("social_relationship") for p in preds]

    aps = []
    result = {}
    for cls in CLASSES:
        gt_bin = [1 if l == cls else 0 for l in gt_labels]
        pred_bin = [1 if l == cls else 0 for l in pred_labels]
        if sum(gt_bin) == 0:
            continue
        ap = average_precision_score(gt_bin, pred_bin)
        recall = recall_score(gt_bin, pred_bin, zero_division=0)
        f1 = f1_score(gt_bin, pred_bin, zero_division=0)
        key = cls.lower().replace(" ", "_")
        result[f"ap_{key}"] = float(ap)
        result[f"recall_{key}"] = float(recall)
        result[f"f1_{key}"] = float(f1)
        aps.append(ap)

    result["mAP"] = float(np.mean(aps)) if aps else float("nan")
    return result
