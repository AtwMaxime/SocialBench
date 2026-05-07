"""
Metric : Concordance Correlation Coefficient (CCC)
Dataset: MSP-Podcast
Task   : Speech emotion — Valence / Arousal / Dominance estimation

GT format  : {"vad_per_annotator": [{"arousal": x, "valence": x, "dominance": x}, ...]}
             Multiple annotators per sample — mean across annotators is used as GT.
             Scores are on a 1–7 scale.
Pred format: {"arousal": x.x, "valence": x.x, "dominance": x.x}

CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)
Final score = mean of CCC_arousal, CCC_valence, CCC_dominance.
"""

from typing import Any

import numpy as np


def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = np.mean((x - mean_x) * (y - mean_y))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom < 1e-10:
        return 0.0
    return float(2 * cov / denom)


def _mean_annotator(entry: dict, key: str) -> float:
    anns = entry.get("vad_per_annotator") or []
    vals = [a[key] for a in anns if key in a]
    return float(np.mean(vals)) if vals else 0.0


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0] if val else None
        if val is None:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    valid_gt_a, valid_gt_v, valid_gt_d = [], [], []
    valid_pred_a, valid_pred_v, valid_pred_d = [], [], []
    for p, g in zip(preds, gts):
        pa = _to_float(p.get("arousal"))
        pv = _to_float(p.get("valence"))
        pd = _to_float(p.get("dominance"))
        if pa is None or pv is None or pd is None:
            continue
        valid_gt_a.append(_mean_annotator(g, "arousal"))
        valid_gt_v.append(_mean_annotator(g, "valence"))
        valid_gt_d.append(_mean_annotator(g, "dominance"))
        valid_pred_a.append(pa)
        valid_pred_v.append(pv)
        valid_pred_d.append(pd)
    n_used = len(valid_pred_a)
    if not valid_pred_a:
        return {"ccc_arousal": float("nan"), "ccc_valence": float("nan"),
                "ccc_dominance": float("nan"), "ccc_mean": float("nan"), "n_used": 0}

    gt_a = np.array(valid_gt_a, dtype=np.float32)
    gt_v = np.array(valid_gt_v, dtype=np.float32)
    gt_d = np.array(valid_gt_d, dtype=np.float32)
    pred_a = np.array(valid_pred_a, dtype=np.float32)
    pred_v = np.array(valid_pred_v, dtype=np.float32)
    pred_d = np.array(valid_pred_d, dtype=np.float32)

    ccc_a = _ccc(pred_a, gt_a)
    ccc_v = _ccc(pred_v, gt_v)
    ccc_d = _ccc(pred_d, gt_d)

    return {
        "ccc_arousal":   ccc_a,
        "ccc_valence":   ccc_v,
        "ccc_dominance": ccc_d,
        "ccc_mean":      (ccc_a + ccc_v + ccc_d) / 3,
        "n_used":        n_used,
    }
