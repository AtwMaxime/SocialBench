"""
Metric : Accuracy, macro F1
Dataset: RLDD (Real-Life Deception Detection)
Task   : Deception detection (binary)

GT format  : {"label": "deceptive" | "truthful"}
Pred format: {"label": "deceptive" | "truthful"}
"""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["deceptive", "truthful"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels = [g["label"] for g in gts]
    pred_labels = [p["label"] for p in preds]
    return {
        "accuracy": accuracy_score(gt_labels, pred_labels),
        "macro_f1": f1_score(
            gt_labels, pred_labels, labels=CLASSES, average="macro", zero_division=0
        ),
        "f1_deceptive": f1_score(
            gt_labels, pred_labels, labels=["deceptive"], average="micro", zero_division=0
        ),
        "f1_truthful": f1_score(
            gt_labels, pred_labels, labels=["truthful"], average="micro", zero_division=0
        ),
    }
