#!/usr/bin/env python3
"""
Compute naive/algorithmic baselines for all SocialBench tasks.

Strategy per task type:
  - Classification       : predict majority class from train set
  - Multi-label (mAP/F1) : predict each label positive if it appears in >50% of train;
                            otherwise use always-positive (gives mAP = class prevalence)
  - Regression (CCC)     : predict train mean → CCC = 0 analytically
  - Regression (AAE)     : predict train mean → compute AAE on test set
  - Localization (dist)  : predict image center (500, 500) → compute L2 distance on test
  - Localization (bbox)  : predict center bbox (250,250,750,750) → compute IoU / L2 on test
  - Speaker count        : predict train mode
  - Diarization (DER)    : predict single speaker throughout

Outputs results as naive_baselines.json.

Usage:
    python compute_baselines.py
    python compute_baselines.py --parquets-dir parquets --output naive_baselines.json
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "metrics"))

PARQUETS_DIR = ROOT / "parquets"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_gt(parquet_path: Path) -> list[dict]:
    """Load assistant messages from a parquet file as parsed dicts."""
    table = pq.read_table(parquet_path, columns=["messages"])
    gts = []
    for row in table.to_pydict()["messages"]:
        msgs = json.loads(row) if isinstance(row, str) else row
        for m in msgs:
            if m["role"] == "assistant":
                content = m["content"]
                if "<think>" in content:
                    content = content[content.rfind("</think>") + len("</think>"):].strip()
                gts.append(json.loads(content))
                break
    return gts


def find_split(base: Path, split: str, prefix: str | None = None) -> Path:
    if prefix:
        p = base / f"{prefix}_{split}.parquet"
        if p.exists():
            return p
        raise FileNotFoundError(f"{p} not found")
    matches = sorted(base.glob(f"*_{split}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No {split}.parquet in {base}")


def run_metric(metric_name: str, preds: list[dict], gts: list[dict]) -> dict:
    import importlib
    mod = importlib.import_module(metric_name)
    return mod.aggregate(preds, gts)


# ── baseline builders ─────────────────────────────────────────────────────────

def majority_class_preds(train_gts: list[dict], key: str, n: int) -> list[dict]:
    counts = Counter()
    for g in train_gts:
        val = g.get(key)
        if val is not None:
            counts[val] += 1
    majority = counts.most_common(1)[0][0]
    return [{key: majority} for _ in range(n)]


def majority_multilabel_preds(train_gts: list[dict], key: str,
                               all_labels: list[str], n: int,
                               threshold: float = 0.5) -> list[dict]:
    label_counts = Counter()
    for g in train_gts:
        for lbl in g.get(key, []):
            label_counts[lbl] += 1
    total = len(train_gts)
    majority_set = [lbl for lbl in all_labels if label_counts[lbl] / total > threshold]
    return [{key: majority_set} for _ in range(n)]


def always_positive_multilabel_preds(key: str, all_labels: list[str], n: int) -> list[dict]:
    return [{key: list(all_labels)} for _ in range(n)]


def center_gaze_preds(n: int) -> list[dict]:
    return [{"gaze_point": [500, 500]} for _ in range(n)]


def center_bbox_preds(n: int) -> list[dict]:
    return [{"bbox": [250, 250, 750, 750]} for _ in range(n)]


def train_mean_preds(train_gts: list[dict], keys: list[str], n: int) -> list[dict]:
    means = {k: float(np.mean([g[k] for g in train_gts if k in g])) for k in keys}
    return [means.copy() for _ in range(n)]


def mode_count_preds(train_gts: list[dict], key: str, n: int) -> list[dict]:
    counts = Counter(g[key] for g in train_gts if key in g)
    mode = counts.most_common(1)[0][0]
    return [{key: mode} for _ in range(n)]


def single_speaker_diarization_preds(test_gts: list[dict]) -> list[dict]:
    preds = []
    for g in test_gts:
        n_bins = len(g.get("timeline", []))
        preds.append({"timeline": ["A"] * n_bins})
    return preds


# ── per-task baseline computation ────────────────────────────────────────────

def compute_all_baselines(parquets_dir: Path) -> dict:
    results = {}

    # ── GazeFollow ───────────────────────────────────────────────────────────
    print("GazeFollow...")
    base = parquets_dir / "gazefollow"
    test_gt = load_gt(find_split(base, "validation", prefix="gazefollow"))
    preds = center_gaze_preds(len(test_gt))
    r = run_metric("gazefollow_distance", preds, test_gt)
    results["gazefollow"] = r
    print(f"  {r}")

    # ── VAT frame ────────────────────────────────────────────────────────────
    print("VAT frame...")
    base = parquets_dir / "videoattentiontarget" / "frame"
    test_gt = load_gt(find_split(base, "test", prefix="vat_frame"))
    preds = center_gaze_preds(len(test_gt))
    r = run_metric("videoattentiontarget_distance", preds, test_gt)
    results["vat_frame"] = r
    print(f"  {r}")

    # ── VAT video ────────────────────────────────────────────────────────────
    print("VAT video...")
    base = parquets_dir / "videoattentiontarget" / "video"
    test_gt = load_gt(find_split(base, "test", prefix="vat_video"))
    preds = center_gaze_preds(len(test_gt))
    r = run_metric("videoattentiontarget_distance", preds, test_gt)
    results["vat_video"] = r
    print(f"  {r}")

    # ── VocalSound ───────────────────────────────────────────────────────────
    print("VocalSound...")
    base = parquets_dir / "vocalsound"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "vocal_sound", len(test_gt))
    r = run_metric("vocalsound_accuracy", preds, test_gt)
    results["vocalsound"] = r
    print(f"  {r}")

    # ── VoxConverse speaker count ─────────────────────────────────────────────
    print("VoxConverse speaker count...")
    base = parquets_dir / "voxconverse" / "speaker"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = mode_count_preds(train_gt, "num_speakers", len(test_gt))
    r = run_metric("voxconverse_speaker_count_mae", preds, test_gt)
    results["voxconverse_speaker"] = r
    print(f"  {r}")

    # ── Proxemics no_skeleton ────────────────────────────────────────────────
    print("Proxemics (no skeleton)...")
    import proxemics_map as prox_mod
    base = parquets_dir / "proxemics" / "no_skeleton"
    test_gt = load_gt(find_split(base, "test"))
    preds = always_positive_multilabel_preds("touching", prox_mod.CLASSES, len(test_gt))
    r = run_metric("proxemics_map", preds, test_gt)
    results["proxemics_no_skel"] = r
    print(f"  {r}")

    # ── Proxemics skeleton ───────────────────────────────────────────────────
    print("Proxemics (skeleton)...")
    base = parquets_dir / "proxemics" / "skeleton"
    test_gt = load_gt(find_split(base, "test"))
    preds = always_positive_multilabel_preds("touching", prox_mod.CLASSES, len(test_gt))
    r = run_metric("proxemics_map", preds, test_gt)
    results["proxemics_skel"] = r
    print(f"  {r}")

    # ── MMEW AU detection ────────────────────────────────────────────────────
    print("MMEW AU detection...")
    import mmew_au_f1 as mmew_au_mod
    base = parquets_dir / "mmew" / "apex_au"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "val"))
    au_counts = Counter()
    for g in train_gt:
        for au in g.get("action_units", []):
            au_counts[au] += 1
    total = len(train_gt)
    majority_aus = [au for au in mmew_au_mod.AU_CLASSES if au_counts[au] / total > 0.5]
    preds = [{"action_units": majority_aus} for _ in range(len(test_gt))]
    r = run_metric("mmew_au_f1", preds, test_gt)
    results["mmew_au"] = r
    print(f"  {r}")

    # ── MMEW emotion (apex) ──────────────────────────────────────────────────
    print("MMEW emotion (apex)...")
    base = parquets_dir / "mmew" / "apex_emotion"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "val"))
    preds = majority_class_preds(train_gt, "emotion", len(test_gt))
    r = run_metric("mmew_emotion_accuracy", preds, test_gt)
    results["mmew_apex_emotion"] = r
    print(f"  {r}")

    # ── MMEW emotion (clip) ──────────────────────────────────────────────────
    print("MMEW emotion (clip)...")
    base = parquets_dir / "mmew" / "clip_emotion"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "val"))
    preds = majority_class_preds(train_gt, "emotion", len(test_gt))
    r = run_metric("mmew_emotion_accuracy", preds, test_gt)
    results["mmew_clip_emotion"] = r
    print(f"  {r}")

    # ── AffWild2 AU ──────────────────────────────────────────────────────────
    print("AffWild2 AU...")
    import affwild2_au_f1 as aff_au_mod
    base = parquets_dir / "affwild2" / "au"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "val"))
    au_counts = Counter()
    for g in train_gt:
        for au in g.get("action_units", []):
            au_counts[au] += 1
    total = len(train_gt)
    majority_aus = [au for au in aff_au_mod.AU_COLS if au_counts[au] / total > 0.5]
    preds = [{"action_units": majority_aus} for _ in range(len(test_gt))]
    r = run_metric("affwild2_au_f1", preds, test_gt)
    results["affwild2_au"] = r
    print(f"  {r}")

    # ── AffWild2 VA ──────────────────────────────────────────────────────────
    print("AffWild2 VA (predict train mean → CCC = 0 analytically)...")
    base = parquets_dir / "affwild2" / "va"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "val"))
    preds = train_mean_preds(train_gt, ["valence", "arousal"], len(test_gt))
    r = run_metric("affwild2_va_ccc", preds, test_gt)
    results["affwild2_va"] = r
    print(f"  {r}")

    # ── AffWild2 Expression ──────────────────────────────────────────────────
    print("AffWild2 Expression...")
    base = parquets_dir / "affwild2" / "expr"
    train_gt = load_gt(find_split(base, "train", prefix="affwild2_expr"))
    test_gt = load_gt(find_split(base, "val", prefix="affwild2_expr"))
    preds = majority_class_preds(train_gt, "label", len(test_gt))
    r = run_metric("affwild2_expr_f1", preds, test_gt)
    results["affwild2_expr"] = r
    print(f"  {r}")

    # ── MSP-Podcast VAD ──────────────────────────────────────────────────────
    print("MSP-Podcast VAD (predict train mean → CCC = 0 analytically)...")
    base = parquets_dir / "msppodcast" / "vad"
    train_gt = load_gt(find_split(base, "train", prefix="msppodcast_vad"))
    test_gt = load_gt(find_split(base, "test", prefix="msppodcast_vad"))
    preds = train_mean_preds(train_gt, ["valence", "arousal", "dominance"], len(test_gt))
    r = run_metric("msppodcast_vad_ccc", preds, test_gt)
    results["msppodcast_vad"] = r
    print(f"  {r}")

    # ── MSP-Podcast labels ───────────────────────────────────────────────────
    print("MSP-Podcast labels...")
    base = parquets_dir / "msppodcast" / "labels"
    train_gt = load_gt(find_split(base, "train", prefix="msppodcast_labels"))
    test_gt = load_gt(find_split(base, "test", prefix="msppodcast_labels"))
    preds = majority_class_preds(train_gt, "emotion", len(test_gt))
    r = run_metric("msppodcast_labels_f1", preds, test_gt)
    results["msppodcast_labels"] = r
    print(f"  {r}")

    # ── VoxConverse Diarization ───────────────────────────────────────────────
    print("VoxConverse diarization (single speaker baseline)...")
    base = parquets_dir / "voxconverse" / "diarization"
    test_gt = load_gt(find_split(base, "test"))
    preds = single_speaker_diarization_preds(test_gt)
    r = run_metric("voxconverse_der", preds, test_gt)
    results["voxconverse_diar"] = r
    print(f"  {r}")

    # ── MELD video_audio ─────────────────────────────────────────────────────
    print("MELD video+audio...")
    base = parquets_dir / "meld" / "video_audio"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "emotion", len(test_gt))
    r = run_metric("meld_weighted_f1", preds, test_gt)
    results["meld_video_audio"] = r
    print(f"  {r}")

    # ── MELD audio_only ──────────────────────────────────────────────────────
    print("MELD audio only...")
    base = parquets_dir / "meld" / "audio_only"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "emotion", len(test_gt))
    r = run_metric("meld_weighted_f1", preds, test_gt)
    results["meld_audio_only"] = r
    print(f"  {r}")

    # ── MELD video_transcript ────────────────────────────────────────────────
    print("MELD video+transcript...")
    base = parquets_dir / "meld" / "video_transcript"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "emotion", len(test_gt))
    r = run_metric("meld_weighted_f1", preds, test_gt)
    results["meld_video_transcript"] = r
    print(f"  {r}")

    # ── EMOTIC discrete ───────────────────────────────────────────────────────
    print("EMOTIC discrete (always predict all categories)...")
    import emotic_discrete_ap as emotic_mod
    base = parquets_dir / "emotic" / "discrete"
    test_gt = load_gt(find_split(base, "test", prefix="emotic_discrete"))
    preds = always_positive_multilabel_preds("emotions", emotic_mod.CATEGORIES, len(test_gt))
    r = run_metric("emotic_discrete_ap", preds, test_gt)
    results["emotic_discrete"] = r
    print(f"  {r}")

    # ── EMOTIC VAD ───────────────────────────────────────────────────────────
    print("EMOTIC VAD (predict train mean)...")
    base = parquets_dir / "emotic" / "vad"
    train_gt = load_gt(find_split(base, "train", prefix="emotic_vad"))
    test_gt = load_gt(find_split(base, "test", prefix="emotic_vad"))
    preds = train_mean_preds(train_gt, ["valence", "arousal", "dominance"], len(test_gt))
    r = run_metric("emotic_vad_aae", preds, test_gt)
    results["emotic_vad"] = r
    print(f"  {r}")

    # ── PISC ─────────────────────────────────────────────────────────────────
    print("PISC...")
    base = parquets_dir / "pisc"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "relationship", len(test_gt))
    r = run_metric("pisc_accuracy", preds, test_gt)
    r.update(run_metric("pisc_map", preds, test_gt))
    results["pisc"] = r
    print(f"  {r}")

    # ── VideoCoAttention detection ─────────────────────────────────────────
    print("VideoCoAttention detection...")
    base = parquets_dir / "videocoattention" / "detection"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "co_attention", len(test_gt))
    r = run_metric("videocoattention_detection_f1", preds, test_gt)
    results["videocoatt_detection"] = r
    print(f"  {r}")

    # ── VideoCoAttention localization ──────────────────────────────────────
    print("VideoCoAttention localization (center bbox)...")
    base = parquets_dir / "videocoattention" / "localization"
    test_gt = load_gt(find_split(base, "test"))
    preds = center_bbox_preds(len(test_gt))
    r = run_metric("videocoattention_localization_iou", preds, test_gt)
    results["videocoatt_localization"] = r
    print(f"  {r}")

    # ── MUStARD with context ───────────────────────────────────────────────
    print("MUStARD (with context)...")
    base = parquets_dir / "mustard" / "video_context"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "sarcasm", len(test_gt))
    r = run_metric("mustard_f1", preds, test_gt)
    results["mustard_context"] = r
    print(f"  {r}")

    # ── MUStARD without context ────────────────────────────────────────────
    print("MUStARD (no context)...")
    base = parquets_dir / "mustard" / "video_no_context"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "sarcasm", len(test_gt))
    r = run_metric("mustard_f1", preds, test_gt)
    results["mustard_no_context"] = r
    print(f"  {r}")

    # ── UR-Funny video+context ─────────────────────────────────────────────
    print("UR-Funny (video+context)...")
    base = parquets_dir / "urfunny" / "video_context"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "funny", len(test_gt))
    r = run_metric("urfunny_f1", preds, test_gt)
    results["urfunny_context"] = r
    print(f"  {r}")

    # ── UR-Funny video+audio ───────────────────────────────────────────────
    print("UR-Funny (video+audio)...")
    base = parquets_dir / "urfunny" / "video_audio"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "funny", len(test_gt))
    r = run_metric("urfunny_f1", preds, test_gt)
    results["urfunny_audio"] = r
    print(f"  {r}")

    # ── RLDD ──────────────────────────────────────────────────────────────
    print("RLDD...")
    base = parquets_dir / "rldd"
    train_gt = load_gt(find_split(base, "train"))
    test_gt = load_gt(find_split(base, "test"))
    preds = majority_class_preds(train_gt, "label", len(test_gt))
    r = run_metric("rldd_accuracy", preds, test_gt)
    results["rldd"] = r
    print(f"  {r}")

    # ── MEVIEW ────────────────────────────────────────────────────────────────
    # No train set, no metric file. 6 classes → UAR (majority class) = 1/6 analytically.
    print("MEVIEW (analytical: UAR = 1/N_classes = 1/6)...")
    results["meview"] = {"uar": 1 / 6}
    print(f"  {results['meview']}")

    return results


# ── pretty print ──────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("NAIVE BASELINE SUMMARY")
    print("=" * 70)

    TASKS = [
        ("gazefollow",            "GazeFollow — gaze target",          "min_dist",        "Dist↓"),
        ("vat_frame",             "VAT — att. target (frame)",          "dist",            "Dist↓"),
        ("vat_video",             "VAT — att. target (video)",          "dist",            "Dist↓"),
        ("vocalsound",            "VocalSound — vocal sound clf",       "accuracy",        "Acc↑"),
        ("voxconverse_speaker",   "VoxConverse — speaker count",        "exact_accuracy",  "ExAcc↑"),
        ("proxemics_no_skel",     "Proxemics — touch (no skel)",        "mAP",             "mAP↑"),
        ("proxemics_skel",        "Proxemics — touch (+skel)",          "mAP",             "mAP↑"),
        ("mmew_au",               "MMEW — AU detection",                "macro_f1",        "macF1↑"),
        ("mmew_apex_emotion",     "MMEW — emotion (apex)",              "uar",             "UAR↑"),
        ("mmew_clip_emotion",     "MMEW — emotion (clip)",              "uar",             "UAR↑"),
        ("affwild2_au",           "AffWild2 — AU detection",            "macro_f1",        "macF1↑"),
        ("affwild2_va",           "AffWild2 — valence-arousal",         "ccc_avg",         "CCC↑"),
        ("affwild2_expr",         "AffWild2 — expression",              "macro_f1",        "macF1↑"),
        ("msppodcast_vad",        "MSP-Podcast — VAD",                  "ccc_avg",         "CCC↑"),
        ("msppodcast_labels",     "MSP-Podcast — emotion",              "macro_f1",        "macF1↑"),
        ("voxconverse_diar",      "VoxConverse — diarization",          "DER",             "DER↓"),
        ("meld_video_audio",      "MELD — emotion (V+A)",               "weighted_f1",     "wF1↑"),
        ("meld_audio_only",       "MELD — emotion (Audio)",             "weighted_f1",     "wF1↑"),
        ("meld_video_transcript", "MELD — emotion (V+T)",               "weighted_f1",     "wF1↑"),
        ("emotic_discrete",       "EMOTIC — emotion (discrete)",        "mAP",             "mAP↑"),
        ("emotic_vad",            "EMOTIC — VAD",                       "AAE",             "AAE↓"),
        ("pisc",                  "PISC — social relation",             "mAP",             "mAP↑"),
        ("videocoatt_detection",  "VideoCoAtt — co-att. detection",     "accuracy",        "Acc2↑"),
        ("videocoatt_localization","VideoCoAtt — co-att. localization", "l2_dist",         "L2↓"),
        ("mustard_context",       "MUStARD — sarcasm (w/ ctx)",         "accuracy",        "Acc2↑"),
        ("mustard_no_context",    "MUStARD — sarcasm (w/o ctx)",        "accuracy",        "Acc2↑"),
        ("urfunny_context",       "UR-Funny — humor (V+A+ctx)",         "accuracy",        "Acc2↑"),
        ("urfunny_audio",         "UR-Funny — humor (V+A)",             "accuracy",        "Acc2↑"),
        ("rldd",                  "RLDD — deception",                   "accuracy",        "Acc↑"),
        ("meview",                "MEVIEW — concealed emotion",          "uar",             "UAR↑"),
    ]

    for key, label, metric_key, display in TASKS:
        if key not in results:
            print(f"  {label:<52} {'MISSING':>10}  ({display})")
            continue
        r = results[key]
        val = r.get(metric_key)
        if val is None:
            val = next((v for v in r.values() if isinstance(v, float)), "?")
        val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
        print(f"  {label:<52} {val_str:>10}  ({display})")

    print()
    print("Strategies used:")
    print("  Classification  → majority class from train set")
    print("  Multi-label     → always-positive (mAP = mean class prevalence)")
    print("  Regression CCC  → predict train mean → CCC = 0.000 by definition")
    print("  Regression AAE  → predict train mean")
    print("  Localization    → predict image center (500,500) or bbox (250,250,750,750)")
    print("  Speaker count   → predict train mode")
    print("  Diarization     → predict single speaker throughout")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute naive baselines for all SocialBench tasks.")
    parser.add_argument("--parquets-dir", default="parquets", help="Root parquet directory (default: parquets).")
    parser.add_argument("--output", default="naive_baselines.json", help="Output file (default: naive_baselines.json).")
    args = parser.parse_args()

    parquets_dir = ROOT / args.parquets_dir
    results = compute_all_baselines(parquets_dir)
    print_summary(results)

    out_path = ROOT / args.output
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
