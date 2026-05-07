"""
Metric : Accuracy, macro F1
Dataset: UR-FUNNY
Task   : Humor detection (binary)

GT format  : {"funny": 1 | 0}
Pred format: {"funny": 1 | 0}
"""

from typing import Any

from sklearn.metrics import accuracy_score, f1_score


def _to_binary(val: Any) -> int | None:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "true", "yes", "funny"):
            return 1
        if v in ("0", "false", "no", "not funny"):
            return 0
    return None


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    valid_gt, valid_pred = [], []
    for p, g in zip(preds, gts):
        gt_val = _to_binary(g.get("funny"))
        pred_val = _to_binary(p.get("funny"))
        if gt_val is None or pred_val is None:
            continue
        valid_gt.append(gt_val)
        valid_pred.append(pred_val)
    n_used = len(valid_pred)
    if not valid_pred:
        return {"accuracy": float("nan"), "macro_f1": float("nan"),
                "f1_funny": float("nan"), "f1_not_funny": float("nan"), "n_used": 0}
    return {
        "accuracy": accuracy_score(valid_gt, valid_pred),
        "macro_f1": f1_score(valid_gt, valid_pred, average="macro", zero_division=0),
        "f1_funny": f1_score(valid_gt, valid_pred, pos_label=1, zero_division=0),
        "f1_not_funny": f1_score(valid_gt, valid_pred, pos_label=0, zero_division=0),
        "n_used": n_used,
    }
