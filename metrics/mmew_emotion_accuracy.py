"""
Metric : Accuracy, macro F1, per-class F1
Dataset: MMEW
Task   : Micro/macro expression emotion classification

GT format  : {"emotion": "happiness"}
Pred format: {"emotion": "happiness"}

Classes (micro): happiness, surprise, disgust, fear, sadness, anger, others
Classes (macro): anger, disgust, fear, happiness, sadness, surprise
"""

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score

MICRO_CLASSES = [
    "happiness", "surprise", "disgust", "fear", "sadness", "anger", "others"
]
MACRO_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels = [g["emotion"] for g in gts]
    pred_labels = [p["emotion"] for p in preds]

    classes = MICRO_CLASSES if "others" in gt_labels else MACRO_CLASSES

    report = classification_report(
        gt_labels, pred_labels, labels=classes, output_dict=True, zero_division=0
    )
    result = {
        "accuracy": accuracy_score(gt_labels, pred_labels),
        "macro_f1": f1_score(
            gt_labels, pred_labels, labels=classes, average="macro", zero_division=0
        ),
        "uar": recall_score(
            gt_labels, pred_labels, labels=classes, average="macro", zero_division=0
        ),
    }
    for cls in classes:
        if cls in report:
            result[f"f1_{cls}"] = report[cls]["f1-score"]
    return result
