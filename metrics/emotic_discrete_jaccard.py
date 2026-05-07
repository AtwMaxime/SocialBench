"""
Metric : Jaccard Coefficient (Intersection over Union on label sets)
Dataset: EMOTIC
Task   : Discrete emotion recognition (multi-label)

GT format  (train)     : {"emotions": ["Happiness", "Engagement"]}
GT format  (val/test)  : {"emotions_per_annotator": [["Happiness"], ["Engagement", "Peace"], ...]}
Pred format            : {"emotions": ["Happiness", "Engagement"]}

For multi-annotator GT, union of all annotators' labels is used.
Jaccard per sample = |pred ∩ gt| / |pred ∪ gt|
Mean Jaccard = average over all samples.
"""

from typing import Any


def _gt_labels(gt: dict[str, Any]) -> set[str]:
    if "emotions_per_annotator" in gt:
        return set(e for ann in gt["emotions_per_annotator"] for e in ann)
    return set(gt["emotions"])


def compute_jaccard(pred: dict[str, Any], gt: dict[str, Any]) -> float:
    pred_set = set(pred["emotions"])
    gt_set = _gt_labels(gt)
    union = pred_set | gt_set
    if not union:
        return 1.0
    return len(pred_set & gt_set) / len(union)


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    scores = [compute_jaccard(p, g) for p, g in zip(preds, gts)]
    return {"mean_jaccard": sum(scores) / len(scores)}
