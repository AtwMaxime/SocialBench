"""
Metric : UAR (Unweighted Average Recall = macro recall), accuracy, per-class F1
Dataset: MEVIEW (concealed micro-expression recognition)
Task   : 6-class emotion classification

GT format  : {"emotion": str}
Pred format: {"emotion": str}
"""

import re
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, recall_score

CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "surprise"]
_CLASSES_RE = re.compile(r"\b(" + "|".join(CLASSES) + r")\b", re.IGNORECASE)


def parse_text(response: str) -> dict | None:
    """Fallback for models that output free text instead of JSON.

    Scans for a quoted emotion word (preferred) then any bare match.
    Returns {"emotion": <word>} for the last match found, or None.
    """
    # Prefer word inside quotes: 'the emotion is "disgust"'
    quoted = re.findall(r'["\'](' + "|".join(CLASSES) + r')["\']', response, re.IGNORECASE)
    if quoted:
        return {"emotion": quoted[-1].lower()}
    # Bare word fallback
    bare = _CLASSES_RE.findall(response)
    if bare:
        return {"emotion": bare[-1].lower()}
    return None


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    gt_labels  = [g.get("emotion", "").lower() for g in gts]
    pred_labels = [p.get("emotion", "").lower() for p in preds]

    result = {
        "accuracy":  accuracy_score(gt_labels, pred_labels),
        "macro_f1":  f1_score(gt_labels, pred_labels, labels=CLASSES,
                              average="macro", zero_division=0),
        "uar":       recall_score(gt_labels, pred_labels, labels=CLASSES,
                                  average="macro", zero_division=0),
    }
    for cls in CLASSES:
        result[f"f1_{cls}"] = f1_score(
            gt_labels, pred_labels, labels=[cls], average="micro", zero_division=0
        )
    return result
