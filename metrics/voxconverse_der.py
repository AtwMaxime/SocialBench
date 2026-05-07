"""
Metric : Diarisation Error Rate (DER)
Dataset: VoxConverse
Task   : Speaker diarization

GT format  : {"timeline": ["A", "AB", "B", "-", ...]}   — 60 bins × 0.5s = 30s window
Pred format: {"timeline": ["A", "AB", "B", "-", ...]}

Speaker labels in the GT are canonical (A, B, C, ...) ordered by first appearance.
The predicted labels may use different letters, so an optimal permutation (Hungarian
matching) is applied before computing DER.

DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Speech Time

Bin duration: 0.5s
Overlap bins (e.g. "AB") are expanded to two speaker segments.
Silence bins ("-") contribute to False Alarm if predicted as speech.
"""

from itertools import permutations
from typing import Any

import numpy as np

BIN_DURATION = 0.5  # seconds per bin


def _parse_timeline(timeline: list[str]) -> dict[str, list[tuple[float, float]]]:
    """Convert timeline list to {speaker: [(start, end), ...]} segments."""
    segments: dict[str, list[tuple[float, float]]] = {}
    for i, bin_label in enumerate(timeline):
        start = i * BIN_DURATION
        end = start + BIN_DURATION
        if bin_label == "-":
            continue
        speakers = list(bin_label)  # "AB" → ["A", "B"]
        for spk in speakers:
            segments.setdefault(spk, [])
            # Merge with previous segment if contiguous
            if segments[spk] and segments[spk][-1][1] == start:
                segments[spk][-1] = (segments[spk][-1][0], end)
            else:
                segments[spk].append((start, end))
    return segments


def _total_duration(segments: dict[str, list[tuple[float, float]]]) -> float:
    return sum(e - s for segs in segments.values() for s, e in segs)


def _speaker_time_matrix(
    gt_segs: dict[str, list[tuple[float, float]]],
    pred_segs: dict[str, list[tuple[float, float]]],
    duration: float,
) -> np.ndarray:
    """Overlap matrix[i, j] = time gt speaker i and pred speaker j are active simultaneously."""
    gt_spks = sorted(gt_segs.keys())
    pred_spks = sorted(pred_segs.keys())
    mat = np.zeros((len(gt_spks), len(pred_spks)))
    for i, gs in enumerate(gt_spks):
        for j, ps in enumerate(pred_spks):
            for gs_start, gs_end in gt_segs[gs]:
                for ps_start, ps_end in pred_segs[ps]:
                    overlap = max(0.0, min(gs_end, ps_end) - max(gs_start, ps_start))
                    mat[i, j] += overlap
    return mat, gt_spks, pred_spks


def _compute_der_single(
    pred_timeline: list[str], gt_timeline: list[str]
) -> dict[str, float]:
    gt_segs = _parse_timeline(gt_timeline)
    pred_segs = _parse_timeline(pred_timeline)

    total_duration = len(gt_timeline) * BIN_DURATION
    ref_speech = _total_duration(gt_segs)

    if ref_speech == 0:
        return {"der": float("nan"), "missed_speech": 0.0, "false_alarm": 0.0, "speaker_confusion": 0.0}

    # False alarm: pred speech where GT is silence
    fa = 0.0
    for spk, segs in pred_segs.items():
        for ps, pe in segs:
            gt_speech_here = sum(
                max(0.0, min(pe, gs_e) - max(ps, gs_s))
                for gs_segs in gt_segs.values()
                for gs_s, gs_e in gs_segs
            )
            fa += max(0.0, (pe - ps) - gt_speech_here)

    # Missed speech: GT speech not covered by any pred speaker
    ms = 0.0
    for spk, segs in gt_segs.items():
        for gs, ge in segs:
            pred_speech_here = sum(
                max(0.0, min(ge, ps_e) - max(gs, ps_s))
                for ps_segs in pred_segs.values()
                for ps_s, ps_e in ps_segs
            )
            ms += max(0.0, (ge - gs) - pred_speech_here)

    # Speaker confusion: correct speech time under best Hungarian mapping
    if gt_segs and pred_segs:
        mat, gt_spks, pred_spks = _speaker_time_matrix(gt_segs, pred_segs, total_duration)
        # Greedy best assignment (Hungarian approximation for small N)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-mat)
        correct = mat[row_ind, col_ind].sum()
    else:
        correct = 0.0

    total_pred_speech = _total_duration(pred_segs)
    confusion = max(0.0, total_pred_speech - fa - correct)

    der = (ms + fa + confusion) / ref_speech
    return {
        "der": der,
        "missed_speech": ms / ref_speech,
        "false_alarm": fa / ref_speech,
        "speaker_confusion": confusion / ref_speech,
    }


def compute_der(pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, float] | None:
    if "timeline" not in pred or "timeline" not in gt:
        return None
    return _compute_der_single(pred["timeline"], gt["timeline"])


def aggregate(
    preds: list[dict[str, Any]], gts: list[dict[str, Any]]
) -> dict[str, float]:
    results = [compute_der(p, g) for p, g in zip(preds, gts)]
    results = [r for r in results if r is not None]
    if not results:
        return {"der": float("nan"), "missed_speech": float("nan"),
                "false_alarm": float("nan"), "speaker_confusion": float("nan")}
    keys = ["der", "missed_speech", "false_alarm", "speaker_confusion"]
    return {
        k: sum(r[k] for r in results if not np.isnan(r[k])) / len(results)
        for k in keys
    }
