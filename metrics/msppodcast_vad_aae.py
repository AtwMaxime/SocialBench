"""
Metric : Average Absolute Error (AAE) for Arousal, Valence, Dominance
Dataset: MSP-Podcast
Task   : Continuous VAD emotion estimation

GT format  (train)     : {"arousal": 2.2, "valence": 4.0, "dominance": 2.6}   — floats 1–7
GT format  (val/test)  : {"vad_per_annotator": [{"arousal": x, "valence": y, "dominance": z}, ...]}

Pred format            : {"arousal": 2.2, "valence": 4.0, "dominance": 2.6}

For multi-annotator GT the mean of annotators' values is used as the reference.
"""

from typing import Any

import numpy as np


def _gt_vad(gt: dict[str, Any]) -> tuple[float, float, float]:
    if "vad_per_annotator" in gt:
        anns = gt["vad_per_annotator"]
        a = np.mean([x["arousal"] for x in anns])
        v = np.mean([x["valence"] for x in anns])
        d = np.mean([x["dominance"] for x in anns])
        return float(a), float(v), float(d)
    return float(gt["arousal"]), float(gt["valence"]), float(gt["dominance"])


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    errors_a, errors_v, errors_d = [], [], []
    for p, g in zip(preds, gts):
        ga, gv, gd = _gt_vad(g)
        errors_a.append(abs(p["arousal"] - ga))
        errors_v.append(abs(p["valence"] - gv))
        errors_d.append(abs(p["dominance"] - gd))
    return {
        "aae_arousal": float(np.mean(errors_a)),
        "aae_valence": float(np.mean(errors_v)),
        "aae_dominance": float(np.mean(errors_d)),
        "aae_mean": float(np.mean(errors_a + errors_v + errors_d)),
    }
