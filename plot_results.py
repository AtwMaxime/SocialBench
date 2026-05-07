#!/usr/bin/env python3
"""
Radar/spider charts — SocialBench model comparison.

Absolute normalisation: outer ring = theoretical maximum of the metric.
  - ↑ metrics : score = value / theoretical_max  (all are [0,1] → score = value)
  - ↓ metrics : score = max(0, 1 - value / theoretical_max)

Three subplots: L1 · Perception | L2 · Understanding | L3 · Theory of Mind

Usage:
    python plot_results.py --results-dir results/adjusted
    python plot_results.py --results-dir results/adjusted --output figures/radar.pdf
    python plot_results.py --results-dir results/adjusted --no-sota

Env-var toggles (set to 0 to hide, default 1 = show):
    ZERO_SHOT_QWEN3       Qwen3-Omni
    ZERO_SHOT_QWEN25      qwen25omni
    ZERO_SHOT_MINICPMO45  minicpmo45
    ZERO_SHOT_MINICPMO26  minicpmo26
    ZERO_SHOT_GEMMA4      gemma4
    ZERO_SHOT_UNIMOE2     unimoe2
    SINGLE_TASK           1epoch_qwen3omni
    JOINT                 joint_sqrt_final
    SOTA                  dashed SOTA reference polygon
    BASELINE              dotted naive-baseline polygon (requires --baselines)

Input: JSON files produced by compute_adjusted_metrics.py.
Each file should have a "results" dict with task keys mapping to:
    {"metrics_adjusted": {...}, "metrics_no_baseline": {...}}
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})

ROOT = Path(__file__).parent

# Default color palette (up to 8 models)
DEFAULT_COLORS = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA3377", "#33BBEE", "#BBCC33", "#EE3377"]

# ── Env-var toggles ────────────────────────────────────────────────────────────
# Maps env-var name → JSON file stem (model name without .json)
MODEL_ENV = {
    "ZERO_SHOT_QWEN3":      "Qwen3-Omni",
    "ZERO_SHOT_QWEN25":     "qwen25omni",
    "ZERO_SHOT_MINICPMO45": "minicpmo45",
    "ZERO_SHOT_MINICPMO26": "minicpmo26",
    "ZERO_SHOT_GEMMA4":     "gemma4",
    "SINGLE_TASK":          "1epoch_qwen3omni",
    "JOINT":                "joint_sqrt_final",
}

MODEL_DISPLAY_NAMES = {
    "Qwen3-Omni":        "Qwen3-Omni (30B)",
    "qwen25omni":        "Qwen2.5-Omni (7B)",
    "minicpmo45":        "MiniCPM-o 4.5 (9B)",
    "minicpmo26":        "MiniCPM-o 2.6 (8B)",
    "gemma4":            "Gemma-4 (4B)",
    "1epoch_qwen3omni":  "Single-task SFT",
    "joint_sqrt_final":  "Joint SFT",
}


def env_flag(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip() != "0"


def is_model_enabled(stem: str) -> bool:
    for env_var, model_stem in MODEL_ENV.items():
        if model_stem == stem:
            return env_flag(env_var)
    return True  # unknown models shown by default


# ── Level axes ─────────────────────────────────────────────────────────────────
# (task_key, metric, direction, sota, weight, theoretical_max)
# direction : "up" → higher is better,  "down" → lower is better
# sota      : published SOTA value (in original metric space), or None
# theoretical_max : upper bound in original metric space
LEVELS = [
    ("L1 · Perception", [
        ("GazeFollow",
            [("gazefollow-vlm_val",                   "avg_dist",      "down", 0.099,  1.0, 1.0)]),
        ("VAT\n(frame)",
            [("vat-omni_frame_test",                  "dist",          "down", 0.076,  1.0, 1.0)]),
        ("VAT\n(video)",
            [("vat-omni_video_test",                  "dist",          "down", 0.076,  1.0, 1.0)]),
        ("VocalSound",
            [("vocalsound-omni_test",                 "accuracy",      "up",   0.980,  1.0, 1.0)]),
        ("Proxemics",
            [("proxemics-vlm_no-skeleton_test",       "mAP",           "up",   0.738,  1.0, 1.0)]),
        ("Proxemics\n(+skel.)",
            [("proxemics-vlm_skeleton_test",          "mAP",           "up",   0.738,  1.0, 1.0)]),
        ("AffWild2\n(AU)",
            [("affwild2-omni_au_val",                 "macro_au_f1",   "up",   0.595,  1.0, 1.0)]),
        ("AffWild2\n(VA)",
            [("affwild2-omni_va_val",                 "ccc_mean",      "up",   0.6695, 1.0, 1.0)]),
        ("MSP-Podcast\n(VAD)",
            [("msppodcast-omni_vad_test",             "ccc_mean",      "up",   0.6852, 1.0, 1.0)]),
    ]),
    ("L2 · Understanding", [
        ("VoxConverse\n(diarization)",
            [("voxconverse-omni_diarization_test",    "der",           "down", 0.082,  1.0, 1.0)]),
        ("MMEW\n(apex)",
            [("mmew_apex-emotion_val",                "uar",           "up",   0.836,  1.0, 1.0)]),
        ("MMEW\n(clip)",
            [("mmew_clip-emotion_val",                "uar",           "up",   0.836,  1.0, 1.0)]),
        ("AffWild2\n(expr.)",
            [("affwild2-omni_expr_val",               "macro_f1",      "up",   0.537,  1.0, 1.0)]),
        ("MELD\n(V+A)",
            [("meld-omni_video-audio_test",           "weighted_f1",   "up",   0.718,  1.0, 1.0)]),
        ("MELD\n(audio)",
            [("meld-omni_audio-only_test",            "weighted_f1",   "up",   0.601,  1.0, 1.0)]),
        ("MELD\n(V+T)",
            [("meld-omni_video-transcript_test",      "weighted_f1",   "up",   0.718,  1.0, 1.0)]),
        ("EMOTIC\n(discrete)",
            [("emotic-vlm_discrete_test",             "mAP",           "up",   0.381,  1.0, 1.0)]),
        ("EMOTIC\n(VAD)",
            [("emotic-vlm_vad_test",                  "aae_mean",      "down", 0.926,  1.0, 2.0)]),
        ("PISC",
            [("pisc-vlm_test",                        "mAP",           "up",   0.805,  1.0, 1.0)]),
        ("VideoCoAtt\n(det.)",
            [("videocoattention-vlm_detection_test",  "accuracy",      "up",   0.781,  1.0, 1.0)]),
        ("VideoCoAtt\n(loc.)",
            [("videocoattention-vlm_localization_test","l2_dist",      "down", 0.0628, 1.0, 1.0)]),
        ("MSP-Podcast\n(labels)",
            [("msppodcast-omni_labels_test",          "macro_f1",      "up",   0.472,  1.0, 1.0)]),
    ]),
    ("L3 · Theory of Mind", [
        ("MuStARD\n(w/ ctx)",
            [("mustard-omni_video-context_test",      "accuracy",      "up",   0.774,  1.0, 1.0)]),
        ("MuStARD\n(w/o ctx)",
            [("mustard-omni_video-no-context_test",   "accuracy",      "up",   0.774,  1.0, 1.0)]),
        ("UR-Funny\n(V+ctx)",
            [("urfunny-omni_video-context_test",      "accuracy",      "up",   0.736,  1.0, 1.0)]),
        ("UR-Funny\n(V+A)",
            [("urfunny-omni_video-audio_test",        "accuracy",      "up",   0.736,  1.0, 1.0)]),
        ("RLDD",
            [("rldd-omni_test",                       "accuracy",      "up",   0.964,  1.0, 1.0)]),
        ("MEVIEW",
            [("meview-omni_test",                     "uar",           "up",   0.685,  1.0, 1.0)]),
    ]),
]

# Reverse mapping: task_key → baseline_key (for BASELINE polygon)
TASK_TO_BASELINE = {
    "affwild2-omni_au_val":              "affwild2_au",
    "affwild2-omni_expr-think_val":      "affwild2_expr",
    "affwild2-omni_expr_val":            "affwild2_expr",
    "affwild2-omni_va_val":              "affwild2_va",
    "emotic-vlm_discrete_test":          "emotic_discrete",
    "emotic-vlm_vad_test":               "emotic_vad",
    "gazefollow-vlm_val":                "gazefollow",
    "meld-omni_audio-only_test":         "meld_audio_only",
    "meld-omni_video-audio_test":        "meld_video_audio",
    "meld-omni_video-transcript_test":   "meld_video_transcript",
    "meview-omni_test":                  "meview",
    "mmew_apex-au_val":                  "mmew_au",
    "mmew_apex-emotion_val":             "mmew_apex_emotion",
    "mmew_clip-emotion-think_val":       "mmew_clip_emotion",
    "mmew_clip-emotion_val":             "mmew_clip_emotion",
    "msppodcast-omni_labels_test":       "msppodcast_labels",
    "msppodcast-omni_vad_test":          "msppodcast_vad",
    "mustard-omni_video-context_test":   "mustard_context",
    "mustard-omni_video-no-context_test":"mustard_no_context",
    "pisc-vlm_test":                     "pisc",
    "proxemics-vlm_no-skeleton_test":    "proxemics_no_skel",
    "proxemics-vlm_skeleton_test":       "proxemics_skel",
    "rldd-omni_test":                    "rldd",
    "urfunny-omni_video-audio_test":     "urfunny_audio",
    "urfunny-omni_video-context_test":   "urfunny_context",
    "vat-omni_frame_test":               "vat_frame",
    "vat-omni_video_test":               "vat_video",
    "videocoattention-vlm_detection_test":    "videocoatt_detection",
    "videocoattention-vlm_localization_test": "videocoatt_localization",
    "vocalsound-omni_test":              "vocalsound",
    "voxconverse-omni_diarization_test": "voxconverse_diar",
    "voxconverse-omni_speaker-count_test": "voxconverse_speaker",
}


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise_abs(value, direction, theoretical_max):
    if value is None:
        return 0.0
    if direction == "up":
        return max(0.0, min(value / theoretical_max, 1.0))
    else:
        return max(0.0, 1.0 - value / theoretical_max)


def axis_score(results, tasks):
    total_w = total_s = 0.0
    for task_key, metric, direction, _sota, weight, theoretical_max in tasks:
        task = results.get(task_key, {})
        val = (task.get("metrics_adjusted", {}).get(metric)
               or task.get("metrics_no_baseline", {}).get(metric))
        total_s += weight * normalise_abs(val, direction, theoretical_max)
        total_w += weight
    return total_s / total_w if total_w > 0 else 0.0


def sota_score(tasks):
    total_w = total_s = 0.0
    has_any = False
    for _, _, direction, sota, weight, theoretical_max in tasks:
        if sota is None:
            return None
        total_s += weight * normalise_abs(sota, direction, theoretical_max)
        total_w += weight
        has_any = True
    return total_s / total_w if has_any and total_w > 0 else None


def baseline_axis_score(naive_baselines, tasks):
    """Compute per-axis score using naive baseline values."""
    total_w = total_s = 0.0
    for task_key, metric, direction, _sota, weight, theoretical_max in tasks:
        bk = TASK_TO_BASELINE.get(task_key)
        val = naive_baselines.get(bk, {}).get(metric) if bk else None
        total_s += weight * normalise_abs(val, direction, theoretical_max)
        total_w += weight
    return total_s / total_w if total_w > 0 else 0.0


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)["results"]


# ── Radar drawing ──────────────────────────────────────────────────────────────

def draw_radar(ax, labels, scores_per_model, sota_scores, baseline_scores, colors, title, show_sota, show_baseline):
    N = len(labels)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles_closed = angles + angles[:1]

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1.08)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6.5, color="#999999")
    ax.set_rlabel_position(12)
    ax.grid(color="#DDDDDD", linewidth=0.6, linestyle="--")
    ax.spines["polar"].set_color("#CCCCCC")
    ax.set_facecolor("#FAFAFA")

    outer_ring = [1.0] * (N + 1)
    ax.plot(angles_closed, outer_ring, color="#AAAAAA", linewidth=1.2, linestyle="-", zorder=3, alpha=0.7)

    if show_sota and sota_scores is not None:
        sota_vals = sota_scores + sota_scores[:1]
        ax.plot(angles_closed, sota_vals, color="#222222", linewidth=2.0, linestyle="--", zorder=5, alpha=0.9)

    if show_baseline and baseline_scores is not None:
        baseline_vals = baseline_scores + baseline_scores[:1]
        ax.plot(angles_closed, baseline_vals, color="#888888", linewidth=1.5, linestyle=":", zorder=5, alpha=0.85)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color="#222222", ha="center", va="center")

    for scores, color in zip(scores_per_model, colors):
        vals = scores + scores[:1]
        ax.plot(angles_closed, vals, color=color, linewidth=2.0, zorder=4, solid_capstyle="round")

    ax.set_title(title, pad=20, fontsize=15, fontweight="bold", color="#111111")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot SocialBench radar charts from adjusted metrics.")
    parser.add_argument(
        "--results-dir",
        default="results/adjusted",
        help="Directory containing adjusted model JSON files (default: results/adjusted).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: {results_dir}/radar.pdf).",
    )
    parser.add_argument(
        "--no-sota",
        action="store_true",
        help="Disable the SOTA reference polygon.",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help="Path to naive_baselines.json for the BASELINE polygon (default: none).",
    )
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    output_path = Path(args.output) if args.output else results_dir / "radar.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Env-var overrides
    show_sota     = (not args.no_sota) and env_flag("SOTA", True)
    show_baseline = env_flag("BASELINE", False)

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    models = []
    color_idx = 0
    for path in json_files:
        if not is_model_enabled(path.stem):
            print(f"[SKIP] {path.stem} (disabled by env var)")
            continue
        try:
            results = load_results(path)
            color = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            color_idx += 1
            models.append((path.stem, results, color))
        except Exception as e:
            print(f"[SKIP] {path.name}: {e}")

    if not models:
        print("No valid model files loaded.")
        return

    print(f"Loaded {len(models)} model(s): {[m[0] for m in models]}")

    # Load naive baselines for BASELINE polygon
    naive_baselines = {}
    if show_baseline:
        baselines_path = Path(args.baselines) if args.baselines else ROOT / "results/naive_baselines.json"
        if baselines_path.exists():
            with open(baselines_path) as f:
                naive_baselines = json.load(f)
            print(f"Loaded baselines from {baselines_path}")
        else:
            print(f"[WARN] BASELINE=1 but baselines file not found: {baselines_path}")
            show_baseline = False

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5.8),
        subplot_kw={"polar": True},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for ax, (level_title, axes_def) in zip(axes, LEVELS):
        labels = [a[0] for a in axes_def]
        colors = [c for _, _, c in models]

        scores_per_model = [
            [axis_score(results, tasks) for _, tasks in axes_def]
            for _, results, _ in models
        ]

        sota_scores_list = None
        if show_sota:
            sota_list = [sota_score(tasks) for _, tasks in axes_def]
            if None not in sota_list:
                sota_scores_list = sota_list
            else:
                sota_scores_list = [s if s is not None else 0.0 for s in sota_list]

        baseline_scores_list = None
        if show_baseline and naive_baselines:
            baseline_scores_list = [
                baseline_axis_score(naive_baselines, tasks)
                for _, tasks in axes_def
            ]

        draw_radar(ax, labels, scores_per_model, sota_scores_list, baseline_scores_list,
                   colors, level_title, show_sota, show_baseline)

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor=color,
                       label=MODEL_DISPLAY_NAMES.get(name, name), alpha=0.85)
        for name, _, color in models
    ]
    if show_sota:
        legend_handles.append(
            mlines.Line2D([], [], color="#222222", linewidth=2.0, linestyle="--", label="SOTA")
        )
    if show_baseline:
        legend_handles.append(
            mlines.Line2D([], [], color="#888888", linewidth=1.5, linestyle=":", label="Baseline")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        fontsize=16,
        handlelength=1.8,
        handleheight=0.9,
    )

    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved → {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
