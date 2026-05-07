"""
Metric : Weighted F1 Score
Dataset: MSP-Podcast
Task   : Speech emotion classification (labels)

GT format  (train)     : {"emotion": "Neutral"}
GT format  (val/test)  : {"emotions_per_annotator": ["Neutral", "Anger", ...]}

Pred format            : {"emotion": "Neutral"}

For multi-annotator GT the majority vote is used as the reference label.
Weighted F1 = sklearn f1_score(average='weighted').
"""

from collections import Counter
from typing import Any

from sklearn.metrics import f1_score


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
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"weighted_f1": float(wf1)}
