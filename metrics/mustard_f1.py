"""
Metric : Precision, Recall, weighted F1, Accuracy
Dataset: MUStARD
Task   : Multimodal sarcasm detection (binary)

GT format  : {"sarcasm": true | false}
Pred format: {"sarcasm": true | false}
"""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    pairs = [(g.get("sarcasm"), p.get("sarcasm")) for g, p in zip(gts, preds)
             if g.get("sarcasm") is not None and p.get("sarcasm") is not None]
    if not pairs:
        return {"accuracy": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
    gt_labels = [int(g) for g, _ in pairs]
    pred_labels = [int(p) for _, p in pairs]
    return {
        "accuracy": accuracy_score(gt_labels, pred_labels),
        "weighted_f1": f1_score(gt_labels, pred_labels, average="weighted", zero_division=0),
        "macro_f1": f1_score(gt_labels, pred_labels, average="macro", zero_division=0),
        "precision": precision_score(gt_labels, pred_labels, zero_division=0),
        "recall": recall_score(gt_labels, pred_labels, zero_division=0),
    }
