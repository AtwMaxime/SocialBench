"""
Metric : Weighted F1, per-class F1, macro F1
Dataset: MELD
Task   : Emotion recognition in conversations

GT format  : {"emotion": "Neutral"}
Pred format: {"emotion": "Neutral"}

Classes: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
Weighted F1 is the official MELD metric (accounts for class imbalance).
"""

from typing import Any

from sklearn.metrics import classification_report, f1_score

EMOTIONS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels = [g["emotion"].title() if g.get("emotion") else "Neutral" for g in gts]
    pred_labels = [p["emotion"].title() if p.get("emotion") else "Neutral" for p in preds]

    report = classification_report(
        gt_labels, pred_labels, labels=EMOTIONS, output_dict=True, zero_division=0
    )
    result = {
        "weighted_f1": f1_score(
            gt_labels, pred_labels, labels=EMOTIONS, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(
            gt_labels, pred_labels, labels=EMOTIONS, average="macro", zero_division=0
        ),
    }
    for cls in EMOTIONS:
        if cls in report:
            result[f"f1_{cls.lower()}"] = report[cls]["f1-score"]
    return result
