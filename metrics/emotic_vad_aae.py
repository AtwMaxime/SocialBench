"""
Metric : Average Absolute Error (AAE) for Valence, Arousal, Dominance
Dataset: EMOTIC
Task   : Continuous VAD emotion estimation

GT format  (train)     : {"valence": 6, "arousal": 4, "dominance": 7}        — integers 1–9
GT format  (val/test)  : {"vad_per_annotator": [{"valence": 6, ...}, ...]}   — multiple annotators

Pred format            : {"valence": 6, "arousal": 4, "dominance": 7}

For multi-annotator GT, the mean of annotators' values is used as the reference.
AAE per dimension = mean(|pred - gt|) over all samples.
"""

from typing import Any

import numpy as np


def _gt_vad(gt: dict[str, Any]) -> tuple[float, float, float]:
    if "vad_per_annotator" in gt:
        anns = gt["vad_per_annotator"]
        v = np.mean([a["valence"] for a in anns])
        a = np.mean([a["arousal"] for a in anns])
        d = np.mean([a["dominance"] for a in anns])
        return float(v), float(a), float(d)
    return float(gt["valence"]), float(gt["arousal"]), float(gt["dominance"])


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    errors_v, errors_a, errors_d = [], [], []
    for p, g in zip(preds, gts):
        try:
            gv, ga, gd = _gt_vad(g)
            errors_v.append(abs(float(p["valence"]) - gv))
            errors_a.append(abs(float(p["arousal"]) - ga))
            errors_d.append(abs(float(p["dominance"]) - gd))
        except (KeyError, TypeError, ValueError):
            continue
    n_used = len(errors_v)
    if not errors_v:
        return {"aae_valence": float("nan"), "aae_arousal": float("nan"),
                "aae_dominance": float("nan"), "aae_mean": float("nan"), "n_used": 0}
    return {
        "aae_valence": float(np.mean(errors_v)),
        "aae_arousal": float(np.mean(errors_a)),
        "aae_dominance": float(np.mean(errors_d)),
        "aae_mean": float(np.mean(errors_v + errors_a + errors_d)),
        "n_used": n_used,
    }
