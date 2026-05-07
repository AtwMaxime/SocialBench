"""
Metric : Angular Error (degrees)
Dataset: GazeFollow
Task   : Gaze target estimation

GT format  (validation): {"gaze_points": [[x1, y1], [x2, y2], ...]}
GT format  (train)     : {"gaze_point": [x, y]}
Pred format            : {"gaze_point": [x, y]}
head_bbox              : [x1, y1, x2, y2] in [0, 1000] — from the user message prompt

The gaze direction is the vector from the head center to the gaze point.
Angular error = arccos(dot(pred_dir, gt_dir) / (|pred_dir| * |gt_dir|)), in degrees.
Min angular error is reported for multi-annotator GT.

Coordinates normalized to [0, 1] internally.
"""

import math
from typing import Any


def _head_center(head_bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [v / 1000.0 for v in head_bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _direction(origin: tuple[float, float], target: tuple[float, float]) -> tuple[float, float]:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    norm = math.sqrt(dx**2 + dy**2)
    if norm < 1e-8:
        return (0.0, 0.0)
    return (dx / norm, dy / norm)


def _angle_between(d1: tuple[float, float], d2: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
    return math.degrees(math.acos(dot))


def _to_points(gt: dict[str, Any]) -> list[tuple[float, float]]:
    if "gaze_points" in gt:
        return [(p[0] / 1000.0, p[1] / 1000.0) for p in gt["gaze_points"]]
    x, y = gt["gaze_point"]
    return [(x / 1000.0, y / 1000.0)]


def compute_angular_error(
    pred: dict[str, Any],
    gt: dict[str, Any],
    head_bbox: list[float],
) -> dict[str, float]:
    center = _head_center(head_bbox)
    gp = pred["gaze_point"]
    px, py = ((gp[0] + gp[2]) / 2, (gp[1] + gp[3]) / 2) if len(gp) == 4 else (gp[0], gp[1])
    pred_dir = _direction(center, (px / 1000.0, py / 1000.0))
    gt_pts = _to_points(gt)
    angles = [_angle_between(pred_dir, _direction(center, g)) for g in gt_pts]
    return {
        "min_angular_error": min(angles),
        "avg_angular_error": sum(angles) / len(angles),
    }


def aggregate(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    head_bboxes: list[list[float]],
) -> dict[str, float]:
    results = []
    for p, g, h in zip(preds, gts, head_bboxes):
        gp = p.get("gaze_point", [])
        if not isinstance(gp, list) or len(gp) < 2 or h is None:
            continue
        results.append(compute_angular_error(p, g, h))
    if not results:
        return {"min_angular_error": 0.0, "avg_angular_error": 0.0}
    return {
        "min_angular_error": sum(r["min_angular_error"] for r in results) / len(results),
        "avg_angular_error": sum(r["avg_angular_error"] for r in results) / len(results),
    }
