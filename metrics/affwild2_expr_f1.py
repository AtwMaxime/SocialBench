"""
Metric : Macro F1
Dataset: AffWild2
Task   : Facial expression recognition

GT format  : {"label": "Happiness"}
Pred format: {"label": "Happiness"}

Classes: Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise, Other
Macro F1 is the official ABAW metric.
"""

from typing import Any

from sklearn.metrics import f1_score, classification_report

EXPR_LABELS = [
    "Neutral", "Anger", "Disgust", "Fear",
    "Happiness", "Sadness", "Surprise", "Other",
]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    pairs = [(g["label"], p.get("label", "")) for g, p in zip(gts, preds) if "label" in g]
    if not pairs:
        return {"macro_f1": 0.0}
    gt_labels, pred_labels = zip(*pairs)

    report = classification_report(
        gt_labels, pred_labels, labels=EXPR_LABELS, output_dict=True, zero_division=0
    )
    result = {
        "macro_f1": f1_score(
            gt_labels, pred_labels, labels=EXPR_LABELS, average="macro", zero_division=0
        )
    }
    for cls in EXPR_LABELS:
        if cls in report:
            result[f"f1_{cls.lower()}"] = report[cls]["f1-score"]
    return result
