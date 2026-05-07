#!/usr/bin/env python3
"""
Compute adjusted metrics using the formula:
    adjusted = (metric * n_valid + baseline_metric * n_invalid) / n_samples

For each task, metrics are adjusted only when the same key exists in the baseline.
Results are saved per model in {output_dir}/{model_name}.json.

Usage:
    python compute_adjusted_metrics.py --results-dir output/my_model
    python compute_adjusted_metrics.py --results-dir output/ --baselines naive_baselines.json
"""

import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent

# Mapping from task key (in metrics JSON) to baseline key (in naive_baselines.json)
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

SKIP_KEYS = {"n_used"}


def adjust_metrics(task_metrics: dict, baseline: dict, n_valid: int, n_samples: int) -> tuple[dict, dict]:
    """Apply the valid/invalid correction formula to all shared metric keys."""
    n_invalid = n_samples - n_valid
    adjusted = {}
    not_adjusted = {}

    for metric_key, value in task_metrics.items():
        if not isinstance(value, (int, float)):
            continue
        if metric_key in baseline and metric_key not in SKIP_KEYS:
            baseline_val = baseline[metric_key]
            if isinstance(baseline_val, (int, float)):
                adj = (value * n_valid + baseline_val * n_invalid) / n_samples
                adjusted[metric_key] = adj
            else:
                not_adjusted[metric_key] = value
        else:
            not_adjusted[metric_key] = value

    return adjusted, not_adjusted


def process_metrics_file(metrics_path: Path, model_name: str, baselines: dict) -> dict:
    with open(metrics_path) as f:
        data = json.load(f)

    output = {
        "model": model_name,
        "source": str(metrics_path),
        "results": {}
    }

    for task_key, task_data in data["results"].items():
        if "error" in task_data and "n_samples" not in task_data:
            output["results"][task_key] = task_data
            continue

        n_samples = task_data.get("n_samples", 0)
        n_valid = task_data.get("n_valid", 0)
        n_invalid = n_samples - n_valid
        raw_metrics = task_data.get("metrics", {})

        baseline_key = TASK_TO_BASELINE.get(task_key)
        baseline = baselines.get(baseline_key, {}) if baseline_key else {}

        adjusted, not_adjusted = adjust_metrics(raw_metrics, baseline, n_valid, n_samples)

        entry = {
            "n_samples": n_samples,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "format_compliance_rate": task_data.get("format_compliance_rate"),
            "baseline_key": baseline_key,
            "metrics_adjusted": adjusted,
            "metrics_no_baseline": not_adjusted,
        }

        if n_invalid > 0:
            print(f"  [{model_name}] {task_key}: {n_invalid}/{n_samples} invalid samples corrected")

        output["results"][task_key] = entry

    return output


def main():
    parser = argparse.ArgumentParser(description="Apply baseline correction to raw evaluation metrics.")
    parser.add_argument(
        "--results-dir",
        default="output",
        help="Directory containing model subdirs, each with a metrics.json (default: output).",
    )
    parser.add_argument(
        "--metrics-file",
        default="metrics.json",
        help="Name of the metrics file inside each model dir (default: metrics.json).",
    )
    parser.add_argument(
        "--baselines",
        default="naive_baselines.json",
        help="Path to naive_baselines.json (default: naive_baselines.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for adjusted JSON files (default: {results_dir}/adjusted).",
    )
    args = parser.parse_args()

    results_dir = ROOT / args.results_dir
    baselines_path = ROOT / args.baselines
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "adjusted"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(baselines_path) as f:
        baselines = json.load(f)

    # Collect metrics files: flat *.json in results_dir take priority over subdirs
    metrics_files = {}  # model_name -> Path

    flat_jsons = sorted(results_dir.glob("*.json"))
    if flat_jsons:
        for p in flat_jsons:
            metrics_files[p.stem] = p
    else:
        # Fall back to subdirectory mode: each subdir contains a metrics.json
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and d.name != "adjusted":
                mp = d / args.metrics_file
                if mp.exists():
                    metrics_files[d.name] = mp
                else:
                    print(f"[SKIP] {d.name}: {args.metrics_file} not found")

    if not metrics_files:
        print(f"No metrics files found in {results_dir}")
        return

    for model_name, metrics_path in sorted(metrics_files.items()):
        print(f"\nProcessing {model_name}...")
        result = process_metrics_file(metrics_path, model_name, baselines)

        out_path = output_dir / f"{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
