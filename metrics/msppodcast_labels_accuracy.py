"""
Metric : Unweighted Accuracy (UA) and Weighted Accuracy (WA)
Dataset: MSP-Podcast
Task   : Speech emotion classification (labels)

GT format  (train)     : {"emotion": "Neutral"}
GT format  (val/test)  : {"emotions_per_annotator": ["Neutral", "Anger", ...]}

Pred format            : {"emotion": "Neutral"}

For multi-annotator GT the majority vote is used as the reference label.
UA = macro-average recall across all emotion classes.
WA = overall accuracy (weighted by class frequency).
"""

from collections import Counter
from typing import Any

import numpy as np


def _gt_label(gt: dict[str, Any]) -> str:
    if "emotions_per_annotator" in gt:
        votes = gt["emotions_per_annotator"]
        return Counter(votes).most_common(1)[0][0]
    return gt["emotion"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    y_pred = [p["emotion"] for p in preds]
    y_true = [_gt_label(g) for g in gts]

    classes = sorted(set(y_true))

    # Weighted accuracy (overall)
    correct = sum(p == t for p, t in zip(y_pred, y_true))
    wa = correct / len(y_true) if y_true else 0.0

    # Unweighted accuracy (macro recall)
    per_class_recall = []
    for cls in classes:
        cls_idx = [i for i, t in enumerate(y_true) if t == cls]
        if not cls_idx:
            continue
        recall = sum(1 for i in cls_idx if y_pred[i] == cls) / len(cls_idx)
        per_class_recall.append(recall)
    ua = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    return {"ua": ua, "wa": wa}
