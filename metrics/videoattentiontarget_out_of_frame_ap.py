"""
Metric : Out-of-Frame Average Precision (AP) and F1
Dataset: VideoAttentionTarget
Task   : Gaze target estimation — out-of-frame detection

GT format  : {"out_of_frame": true} | {"gaze_point": [...], "label": "gaze target"}
Pred format: {"out_of_frame": true} | {"gaze_point": [...], "label": "gaze target"}

Treats out-of-frame prediction as a binary classification task.
AP is computed from the precision-recall curve using sklearn.
If pred contains a confidence score key "out_of_frame_score" it is used as
the ranking score; otherwise the hard binary label is used (AP degenerates to F1).
"""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def _label(d: dict[str, Any]) -> int:
    return 1 if d.get("out_of_frame", False) else 0


def _score(d: dict[str, Any]) -> float:
    if "out_of_frame_score" in d:
        return float(d["out_of_frame_score"])
    return float(_label(d))


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels = np.array([_label(g) for g in gts])
    pred_scores = np.array([_score(p) for p in preds])
    pred_labels = np.array([_label(p) for p in preds])

    if gt_labels.sum() == 0:
        return {"out_of_frame_ap": float("nan"), "out_of_frame_f1": float("nan")}

    ap = average_precision_score(gt_labels, pred_scores)
    f1 = f1_score(gt_labels, pred_labels, zero_division=0)
    return {"out_of_frame_ap": float(ap), "out_of_frame_f1": float(f1)}
