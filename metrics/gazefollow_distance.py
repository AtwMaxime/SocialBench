"""
Metric : L2 Distance (min and avg over annotators)
Dataset: GazeFollow
Task   : Gaze target estimation

GT format  (validation): {"gaze_points": [[x1, y1], [x2, y2], ...]}  — 10 annotators
GT format  (train)     : {"gaze_point": [x, y]}                       — single annotator
Pred format            : {"gaze_point": [x, y]}

Coordinates are in [0, 1000] (Qwen convention). Normalized to [0, 1] internally.
Returns:
    min_dist  — minimum L2 distance to any annotator (standard GazeFollow metric)
    avg_dist  — average L2 distance to all annotators
"""

import math
from typing import Any


def _to_points(gt: dict[str, Any]) -> list[tuple[float, float]]:
    if "gaze_points" in gt:
        return [(p[0] / 1000.0, p[1] / 1000.0) for p in gt["gaze_points"]]
    x, y = gt["gaze_point"]
    return [(x / 1000.0, y / 1000.0)]


def _l2(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_distance(
    pred: dict[str, Any], gt: dict[str, Any]
) -> dict[str, float]:
    gp = pred["gaze_point"]
    if len(gp) == 4:  # model output a bbox — use center
        px, py = (gp[0] + gp[2]) / 2, (gp[1] + gp[3]) / 2
    else:
        px, py = gp[0], gp[1]
    pred_pt = (px / 1000.0, py / 1000.0)
    gt_pts = _to_points(gt)
    dists = [_l2(pred_pt, g) for g in gt_pts]
    return {
        "min_dist": min(dists),
        "avg_dist": sum(dists) / len(dists),
    }


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    results = []
    for p, g in zip(preds, gts):
        gp = p.get("gaze_point", [])
        if not isinstance(gp, list) or len(gp) < 2:
            continue
        results.append(compute_distance(p, g))
    if not results:
        return {"min_dist": 0.0, "avg_dist": 0.0}
    return {
        "min_dist": sum(r["min_dist"] for r in results) / len(results),
        "avg_dist": sum(r["avg_dist"] for r in results) / len(results),
    }
