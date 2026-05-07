"""
Metric : L2 Distance
Dataset: VideoAttentionTarget
Task   : Gaze target estimation (in-frame only)

GT format  : {"gaze_point": [gx, gy], "label": "gaze target"}
           | {"out_of_frame": true}
Pred format: {"gaze_point": [gx, gy], "label": "gaze target"}
           | {"out_of_frame": true}

Distance is only computed when both GT and pred are in-frame.
Coordinates in [0, 1000], normalized to [0, 1] internally.
"""

import math
from typing import Any


def _is_out_of_frame(d: dict[str, Any]) -> bool:
    return d.get("out_of_frame", False)


def compute_distance(
    pred: dict[str, Any], gt: dict[str, Any]
) -> dict[str, float | None]:
    if _is_out_of_frame(gt) or _is_out_of_frame(pred):
        return {"dist": None}
    try:
        gp = pred.get("gaze_point")
        if gp is None or not isinstance(gp, (list, tuple)) or len(gp) < 2:
            return {"dist": None}
        gp = [float(v) for v in gp]
        if len(gp) == 4:  # bbox [x1,y1,x2,y2] → centre
            gp = [(gp[0] + gp[2]) / 2, (gp[1] + gp[3]) / 2]
        px, py = [v / 1000.0 for v in gp]
        gt_gp = gt.get("gaze_point")
        if gt_gp is None or len(gt_gp) < 2:
            return {"dist": None}
        gx, gy = [float(v) / 1000.0 for v in gt_gp[:2]]
        return {"dist": math.sqrt((px - gx) ** 2 + (py - gy) ** 2)}
    except (TypeError, ValueError):
        return {"dist": None}


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    dists = [
        compute_distance(p, g)["dist"]
        for p, g in zip(preds, gts)
    ]
    valid = [d for d in dists if d is not None]
    n_used = len(valid)
    return {"dist": sum(valid) / n_used if n_used else float("nan"), "n_used": n_used}
