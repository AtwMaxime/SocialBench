"""
Metric : Accuracy, per-class F1, macro F1
Dataset: VocalSound
Task   : Vocal sound classification

GT format  : {"vocal_sound": "Laughter"}
Pred format: {"vocal_sound": "Laughter"}

Classes: Laughter, Sigh, Cough, Throat clearing, Sneeze, Sniff
"""

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, f1_score

CLASSES = ["Laughter", "Sigh", "Cough", "Throat clearing", "Sneeze", "Sniff"]


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    def _get_label(d):
        return d.get("vocal_sound") or d.get("sound") or d.get("Vocal Sound") or d.get("class") or ""

    gt_labels = [_get_label(g) for g in gts]
    pred_labels = [_get_label(p) for p in preds]

    report = classification_report(
        gt_labels, pred_labels, labels=CLASSES, output_dict=True, zero_division=0
    )
    result = {
        "accuracy": accuracy_score(gt_labels, pred_labels),
        "macro_f1": f1_score(gt_labels, pred_labels, average="macro", zero_division=0),
    }
    for cls in CLASSES:
        if cls in report:
            result[f"f1_{cls.lower().replace(' ', '_')}"] = report[cls]["f1-score"]
    return result
