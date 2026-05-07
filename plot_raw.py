#!/usr/bin/env python3
"""
Radar/spider charts — SocialBench model comparison (raw metrics, no baseline correction).

Same layout as plot_results.py but reads raw metrics files directly
(the "metrics" key, without adjusted/no_baseline split).

Usage:
    python plot_raw.py --results-dir results/raw
    python plot_raw.py --results-dir results/raw --output figures/radar_raw.pdf
    python plot_raw.py --results-dir results/raw --no-sota

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

# Import all shared definitions from plot_results
from plot_results import (
    ROOT, DEFAULT_COLORS, MODEL_ENV, LEVELS, TASK_TO_BASELINE,
    matplotlib as _mpl_settings,
    env_flag, is_model_enabled,
    normalise_abs, sota_score, baseline_axis_score,
    draw_radar,
)


def axis_score(results, tasks):
    """Score an axis from raw metrics (no baseline correction)."""
    total_w = total_s = 0.0
    for task_key, metric, direction, _sota, weight, theoretical_max in tasks:
        task = results.get(task_key, {})
        val = task.get("metrics", {}).get(metric)
        total_s += weight * normalise_abs(val, direction, theoretical_max)
        total_w += weight
    return total_s / total_w if total_w > 0 else 0.0


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)["results"]


def main():
    parser = argparse.ArgumentParser(description="Plot SocialBench radar charts from raw metrics.")
    parser.add_argument(
        "--results-dir",
        default="results/raw",
        help="Directory containing raw model JSON files (default: results/raw).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: {results_dir}/radar_raw.pdf).",
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
    output_path = Path(args.output) if args.output else results_dir / "radar_raw.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        mpatches.Patch(facecolor=color, edgecolor=color, label=name, alpha=0.85)
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
