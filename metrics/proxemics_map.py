"""
Metric : Mean Average Precision (mAP), per-class AP
Dataset: Proxemics
Task   : Body contact recognition (multi-label)

GT format  : {"touching": ["Hand touch hand", "Hand touch shoulder", ...]}
Pred format: {"touching": ["Hand touch hand", ...]}

Each class is treated as an independent binary classification.
AP per class is computed from binary labels (no confidence scores).
mAP = mean of per-class APs.
"""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

CLASSES = [
    "Hand touch hand",
    "Hand touch shoulder",
    "Shoulder touch shoulder",
    "Hand touch elbow",
    "Elbow touch shoulder",
    "Hand touch torso",
]


def _flatten_labels(val: Any) -> list[str] | None:
    """Return a flat list of strings, flattening nested lists. Returns None if unparseable."""
    if val is None:
        return []
    if not isinstance(val, list):
        return None
    flat: list[str] = []
    for item in val:
        if isinstance(item, list):
            flat.extend(str(x) for x in item)
        elif isinstance(item, str):
            flat.append(item)
    return flat


def _to_binary(labels: list[str], classes: list[str]) -> list[int]:
    label_set = set(labels)
    return [1 if c in label_set else 0 for c in classes]


def _get_touching(d: dict[str, Any]) -> list[str] | None:
    val = d.get("touching") or d.get("touching_body_parts")
    return _flatten_labels(val)


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    valid_preds, valid_gts = [], []
    for p, g in zip(preds, gts):
        pred_labels = _get_touching(p)
        gt_labels = _get_touching(g)
        if pred_labels is None or gt_labels is None:
            continue
        valid_preds.append(pred_labels)
        valid_gts.append(gt_labels)

    n_used = len(valid_preds)
    if not valid_preds:
        return {"mAP": float("nan"), "n_used": 0}

    gt_matrix = np.array([_to_binary(g, CLASSES) for g in valid_gts])
    pred_matrix = np.array([_to_binary(p, CLASSES) for p in valid_preds])

    aps = []
    result: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        if gt_matrix[:, i].sum() == 0:
            continue
        ap = average_precision_score(gt_matrix[:, i], pred_matrix[:, i])
        f1 = f1_score(gt_matrix[:, i], pred_matrix[:, i], zero_division=0)
        key = cls.lower().replace(" ", "_")
        result[f"ap_{key}"] = float(ap)
        result[f"f1_{key}"] = float(f1)
        aps.append(ap)

    result["mAP"] = float(np.mean(aps)) if aps else float("nan")
    result["n_used"] = n_used
    return result
