"""
Metric : Accuracy, macro F1
Dataset: PISC
Task   : Social relationship recognition (6-class)

GT format  : {"relationship": "Friends"}
Pred format: {"relationship": "Friends"}

Classes: Friends, Family, Couple, Professional, Commercial, No relation
"""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score

CLASSES = ["Friends", "Family", "Couple", "Professional", "Commercial", "No relation"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    valid_gt, valid_pred = [], []
    for p, g in zip(preds, gts):
        pred_label = p.get("relationship") or p.get("social_relationship")
        if pred_label is None:
            continue
        valid_gt.append(g["relationship"])
        valid_pred.append(pred_label)
    n_used = len(valid_pred)
    if not valid_pred:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "n_used": 0}
    return {
        "accuracy": accuracy_score(valid_gt, valid_pred),
        "macro_f1": f1_score(
            valid_gt, valid_pred, labels=CLASSES, average="macro", zero_division=0
        ),
        "n_used": n_used,
    }
