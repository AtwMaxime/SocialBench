"""
Metric : Weighted F1, per-class F1, macro F1
Dataset: MSP-Podcast
Task   : Speech emotion recognition (categorical)

GT format  : {"emotions_per_annotator": ["Neutral", "Happiness", ...]}
             Multiple annotators per sample — majority vote is used as GT.
             Ties are broken by the first label in EMOTIONS order.
Pred format: {"emotion": "Neutral"}

Classes: Neutral, Happiness, Anger, Sadness, Surprise, Fear, Disgust, Contempt, Other
"""

from collections import Counter
from typing import Any

from sklearn.metrics import classification_report, f1_score

EMOTIONS = ["Neutral", "Happiness", "Anger", "Sadness", "Surprise", "Fear", "Disgust", "Contempt", "Other"]
_EMOTION_RANK = {e: i for i, e in enumerate(EMOTIONS)}


def _majority_vote(labels: list[str]) -> str:
    """Return the most common label; ties broken by EMOTIONS order."""
    if not labels:
        return "Neutral"
    counts = Counter(l.title() for l in labels)
    return min(counts, key=lambda e: (-counts[e], _EMOTION_RANK.get(e, 99)))


def _parse_emotion(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0] if val else None
        if val is None:
            return None
    try:
        return str(val).strip().title()
    except Exception:
        return None


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    valid_gt, valid_pred = [], []
    for p, g in zip(preds, gts):
        gt_label = _majority_vote(g.get("emotions_per_annotator") or [])
        pred_label = _parse_emotion(p.get("emotion"))
        if pred_label is None:
            continue
        valid_gt.append(gt_label)
        valid_pred.append(pred_label)
    n_used = len(valid_pred)
    if not valid_pred:
        result = {"weighted_f1": float("nan"), "macro_f1": float("nan"), "n_used": 0}
        for cls in EMOTIONS:
            result[f"f1_{cls.lower()}"] = float("nan")
        return result

    report = classification_report(
        valid_gt, valid_pred, labels=EMOTIONS, output_dict=True, zero_division=0
    )
    result = {
        "weighted_f1": f1_score(
            valid_gt, valid_pred, labels=EMOTIONS, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(
            valid_gt, valid_pred, labels=EMOTIONS, average="macro", zero_division=0
        ),
        "n_used": n_used,
    }
    for cls in EMOTIONS:
        if cls in report:
            result[f"f1_{cls.lower()}"] = report[cls]["f1-score"]
    return result
