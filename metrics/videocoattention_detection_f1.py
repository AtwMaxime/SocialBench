"""
Metric : F1, Precision, Recall, Accuracy
Dataset: VideoCoAttention
Task   : Co-attention detection (binary)

GT format  : {"co_attention": true | false}
Pred format: {"co_attention": true | false}
"""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    valid_gt, valid_pred = [], []
    for p, g in zip(preds, gts):
        try:
            gt_val = int(g["co_attention"])
            pred_val = int(p["co_attention"])
        except (KeyError, TypeError, ValueError):
            continue
        valid_gt.append(gt_val)
        valid_pred.append(pred_val)
    n_used = len(valid_pred)
    if not valid_pred:
        return {"accuracy": float("nan"), "f1": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "n_used": 0}
    return {
        "accuracy": accuracy_score(valid_gt, valid_pred),
        "f1": f1_score(valid_gt, valid_pred, zero_division=0),
        "precision": precision_score(valid_gt, valid_pred, zero_division=0),
        "recall": recall_score(valid_gt, valid_pred, zero_division=0),
        "n_used": n_used,
    }
