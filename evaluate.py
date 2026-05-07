#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluate model predictions for all SocialBench tasks.

Scans --pred-dir for PRED_*.jsonl files produced by `swift infer`, loads the
matching ground-truth from the local parquet files, runs the appropriate
metrics, and writes a consolidated metrics.json.

PRED filename convention:
    PRED_{dataset}_{split}.jsonl                 (single-subset datasets)
    PRED_{dataset}_{subset}_{split}.jsonl        (multi-subset datasets)

    e.g.  PRED_gazefollow-vlm_val.jsonl
          PRED_meld-omni_video-audio_test.jsonl
          PRED_affwild2-omni_expr_val.jsonl

Splits: use "test" when available, "val" otherwise (affwild2, mmew).

Usage:
    python evaluate.py --pred-dir output/my_model
    python evaluate.py --pred-dir output/my_model --parquets-dir parquets
    python evaluate.py --pred-dir output/my_model --output output/my_model/metrics.json
"""

import argparse
import importlib.util
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BBOX_RE = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")


def parse_response(text: str) -> dict | None:
    """Strip <think> blocks and extract the first JSON object from model output."""
    text = _THINK_RE.sub("", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    last = None
    for chunk in re.findall(r"\{[^{}]*\}", text):
        try:
            last = json.loads(chunk)
        except json.JSONDecodeError:
            pass
    return last


def extract_gt_from_messages(messages: list[dict]) -> dict | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return parse_response(content)
    return None


def extract_head_bbox(messages: list[dict]) -> list[float] | None:
    for msg in messages:
        if msg.get("role") == "user":
            m = _BBOX_RE.search(msg.get("content", ""))
            if m:
                return [float(m.group(i)) for i in range(1, 5)]
    return None


# ---------------------------------------------------------------------------
# Parquet loader
# ---------------------------------------------------------------------------

def find_parquet_files(base_dir: Path, split: str, prefix: str | None = None) -> list[Path]:
    if prefix:
        exact = base_dir / f"{prefix}_{split}.parquet"
        if exact.exists():
            return [exact]
        raise FileNotFoundError(f"Expected {exact} but file not found.")

    matches = sorted(base_dir.glob(f"*_{split}.parquet"))
    if matches:
        return matches

    existing = [p.name for p in base_dir.iterdir()] if base_dir.exists() else ["(dir not found)"]
    raise FileNotFoundError(
        f"No parquet found for split='{split}' in {base_dir}.\n"
        f"  Files present: {existing}"
    )


def load_parquet_rows(base_dir: Path, split: str, prefix: str | None = None) -> list[Any]:
    files = find_parquet_files(base_dir, split, prefix=prefix)
    rows = []
    for f in files:
        table = pq.read_table(f, columns=["messages"])
        for item in table.column("messages").to_pylist():
            if isinstance(item, str):
                item = json.loads(item)
            rows.append(item)
    return rows


# ---------------------------------------------------------------------------
# Metric loader
# ---------------------------------------------------------------------------

def load_metric_module(name: str):
    path = ROOT / "metrics" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Metric file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Key: "{dataset}_{subset}" or "{dataset}" for single-subset datasets.
# Matches the PRED filename stem after stripping PRED_ prefix and _{split}.jsonl suffix.
#
# parquet_dir    : path relative to --parquets-dir
# split_alias    : maps the split name in the PRED filename to the actual parquet split
# parquet_prefix : disambiguates when multiple files share the same split suffix
# needs_head_bbox: if True, head bbox is extracted from the user message (GazeFollow / VAT)
# text_parser    : optional fallback parser for models that output free text (e.g. MEVIEW)

REGISTRY: dict[str, dict] = {

    # ── Gaze ──────────────────────────────────────────────────────────────────

    "gazefollow-vlm": {
        "parquet_dir": "gazefollow",
        "split_alias": {"test": "validation", "val": "validation"},
        "parquet_prefix": "gazefollow",
        "metrics": ["gazefollow_distance", "gazefollow_auc", "gazefollow_angular_error"],
        "needs_head_bbox": True,
    },
    "vat-omni_video": {
        "parquet_dir": "videoattentiontarget/video",
        "parquet_prefix": "vat_video",
        "metrics": ["videoattentiontarget_distance", "videoattentiontarget_out_of_frame_ap"],
    },
    "vat-omni_frame": {
        "parquet_dir": "videoattentiontarget/frame",
        "parquet_prefix": "vat_frame",
        "metrics": ["videoattentiontarget_distance", "videoattentiontarget_out_of_frame_ap"],
    },
    "videocoattention-vlm_detection": {
        "parquet_dir": "videocoattention/detection",
        "metrics": ["videocoattention_detection_f1"],
    },
    "videocoattention-vlm_localization": {
        "parquet_dir": "videocoattention/localization",
        "metrics": ["videocoattention_localization_iou"],
    },

    # ── Emotion ───────────────────────────────────────────────────────────────

    "meld-omni_video-audio": {
        "parquet_dir": "meld/video_audio",
        "metrics": ["meld_weighted_f1"],
    },
    "meld-omni_audio-only": {
        "parquet_dir": "meld/audio_only",
        "metrics": ["meld_weighted_f1"],
    },
    "meld-omni_video-transcript": {
        "parquet_dir": "meld/video_transcript",
        "metrics": ["meld_weighted_f1"],
    },
    "affwild2-omni_expr": {
        "parquet_dir": "affwild2/expr",
        "split_alias": {"test": "val"},
        "parquet_prefix": "affwild2_expr",
        "metrics": ["affwild2_expr_f1"],
    },
    "affwild2-omni_expr-think": {
        "parquet_dir": "affwild2/expr",
        "split_alias": {"test": "val"},
        "parquet_prefix": "affwild2_expr_think",
        "metrics": ["affwild2_expr_f1"],
    },
    "affwild2-omni_va": {
        "parquet_dir": "affwild2/va",
        "split_alias": {"test": "val"},
        "metrics": ["affwild2_va_ccc"],
    },
    "affwild2-omni_au": {
        "parquet_dir": "affwild2/au",
        "split_alias": {"test": "val"},
        "metrics": ["affwild2_au_f1"],
    },
    "mmew_apex-au": {
        "parquet_dir": "mmew/apex_au",
        "split_alias": {"test": "val"},
        "metrics": ["mmew_au_f1"],
    },
    "mmew_apex-emotion": {
        "parquet_dir": "mmew/apex_emotion",
        "split_alias": {"test": "val"},
        "metrics": ["mmew_emotion_accuracy"],
    },
    "mmew_clip-emotion": {
        "parquet_dir": "mmew/clip_emotion",
        "split_alias": {"test": "val"},
        "metrics": ["mmew_emotion_accuracy"],
    },
    "mmew_clip-emotion-think": {
        "parquet_dir": "mmew/clip_emotion_think",
        "split_alias": {"test": "val"},
        "metrics": ["mmew_emotion_accuracy"],
    },

    # ── Social context ────────────────────────────────────────────────────────

    "emotic-vlm_discrete": {
        "parquet_dir": "emotic/discrete",
        "parquet_prefix": "emotic_discrete",
        "metrics": ["emotic_discrete_ap", "emotic_discrete_jaccard"],
    },
    "emotic-vlm_vad": {
        "parquet_dir": "emotic/vad",
        "parquet_prefix": "emotic_vad",
        "metrics": ["emotic_vad_aae"],
    },
    "pisc-vlm": {
        "parquet_dir": "pisc",
        "metrics": ["pisc_accuracy", "pisc_map"],
    },
    "proxemics-vlm_no-skeleton": {
        "parquet_dir": "proxemics/no_skeleton",
        "metrics": ["proxemics_accuracy", "proxemics_map"],
    },
    "proxemics-vlm_skeleton": {
        "parquet_dir": "proxemics/skeleton",
        "metrics": ["proxemics_accuracy", "proxemics_map"],
    },

    # ── High-level social understanding ──────────────────────────────────────

    "mustard-omni_video-no-context": {
        "parquet_dir": "mustard/video_no_context",
        "metrics": ["mustard_f1"],
    },
    "mustard-omni_video-context": {
        "parquet_dir": "mustard/video_context",
        "metrics": ["mustard_f1"],
    },
    "urfunny-omni_video-audio": {
        "parquet_dir": "urfunny/video_audio",
        "metrics": ["urfunny_f1"],
    },
    "urfunny-omni_video-context": {
        "parquet_dir": "urfunny/video_context",
        "metrics": ["urfunny_f1"],
    },
    "rldd-omni": {
        "parquet_dir": "rldd",
        "metrics": ["rldd_accuracy", "rldd_auc"],
    },

    # ── Audio ─────────────────────────────────────────────────────────────────

    "vocalsound-omni": {
        "parquet_dir": "vocalsound",
        "metrics": ["vocalsound_accuracy"],
    },
    "voxconverse-omni_diarization": {
        "parquet_dir": "voxconverse/diarization",
        "metrics": ["voxconverse_der"],
    },
    "voxconverse-omni_speaker-count": {
        "parquet_dir": "voxconverse/speaker",
        "metrics": ["voxconverse_speaker_count_mae"],
    },

    # ── MSP-Podcast ───────────────────────────────────────────────────────────

    "msppodcast-omni_labels": {
        "parquet_dir": "msppodcast/labels",
        "parquet_prefix": "msppodcast_labels",
        "metrics": ["msppodcast_labels_f1"],
    },
    "msppodcast-omni_vad": {
        "parquet_dir": "msppodcast/vad",
        "parquet_prefix": "msppodcast_vad",
        "metrics": ["msppodcast_vad_ccc"],
    },

    # ── MEVIEW ────────────────────────────────────────────────────────────────

    "meview-omni": {
        "parquet_dir": "meview",
        "parquet_prefix": "meview_recognition",
        "metrics": ["meview_uar"],
        "text_parser": "meview_uar.parse_text",
    },
}

# ---------------------------------------------------------------------------
# PRED filename parser
# ---------------------------------------------------------------------------

_KNOWN_SPLITS = {"train", "test", "val", "validation", "dev"}


def parse_pred_filename(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem.removeprefix("PRED_")
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in _KNOWN_SPLITS:
        return parts[0], parts[1]
    return stem, "test"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_file(
    pred_path: Path,
    parquets_dir: Path,
    registry_key: str,
    split: str,
    max_samples: int | None = None,
) -> dict:
    config = REGISTRY[registry_key]
    actual_split = config.get("split_alias", {}).get(split, split)
    parquet_dir = parquets_dir / config["parquet_dir"]
    parquet_prefix = config.get("parquet_prefix", None)
    needs_head_bbox = config.get("needs_head_bbox", False)

    pred_lines = []
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            pred_lines.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    gt_rows = load_parquet_rows(parquet_dir, actual_split, prefix=parquet_prefix)

    n_total = min(len(pred_lines), len(gt_rows))
    if max_samples is not None and n_total > max_samples:
        pred_lines = pred_lines[:max_samples]
        gt_rows = gt_rows[:max_samples]
        n_total = max_samples
    if len(pred_lines) != len(gt_rows):
        print(
            f"  [WARN] PRED has {len(pred_lines)} lines, parquet has {len(gt_rows)} rows"
            f" — using {n_total}"
        )

    text_parser_fn = None
    if "text_parser" in config:
        mod_name, fn_name = config["text_parser"].rsplit(".", 1)
        text_parser_fn = getattr(load_metric_module(mod_name), fn_name)

    preds: list[dict] = []
    gts: list[dict] = []
    head_bboxes: list[list[float]] = []
    n_parse_failures = 0

    for i in range(n_total):
        response = pred_lines[i].get("response", "")
        pred_dict = parse_response(response)
        if pred_dict is None and text_parser_fn is not None:
            pred_dict = text_parser_fn(response)
        gt_dict = extract_gt_from_messages(gt_rows[i])

        if pred_dict is None or not isinstance(pred_dict, dict):
            n_parse_failures += 1
            continue
        if gt_dict is None:
            continue

        preds.append(pred_dict)
        gts.append(gt_dict)

        if needs_head_bbox:
            messages = pred_lines[i].get("messages") or gt_rows[i]
            bbox = extract_head_bbox(messages)
            head_bboxes.append(bbox)

    n_valid = len(preds)
    if n_valid == 0:
        return {
            "n_samples": n_total,
            "n_valid": 0,
            "n_parse_failures": n_parse_failures,
            "metrics": {},
            "error": "no valid samples after parsing",
        }

    metrics_out: dict[str, Any] = {}
    n_used_per_metric: list[int] = []

    for metric_name in config["metrics"]:
        try:
            mod = load_metric_module(metric_name)
            if metric_name == "gazefollow_angular_error":
                result = mod.aggregate(preds, gts, head_bboxes)
            else:
                result = mod.aggregate(preds, gts)
            n_used_per_metric.append(result.pop("n_used", n_valid))
            metrics_out.update(result)
        except Exception as e:
            metrics_out[f"{metric_name}_error"] = str(e)
            print(f"  [ERROR] {metric_name}: {e}")
            traceback.print_exc()

    n_metric_valid = min(n_used_per_metric) if n_used_per_metric else n_valid
    format_compliance_rate = round(n_metric_valid / n_total, 4) if n_total > 0 else 0.0

    return {
        "n_samples": n_total,
        "n_valid": n_valid,
        "n_metric_valid": n_metric_valid,
        "n_parse_failures": n_parse_failures,
        "format_compliance_rate": format_compliance_rate,
        "metrics": metrics_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate social benchmark predictions against local parquet ground-truth."
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing PRED_*.jsonl files (output of swift infer).",
    )
    parser.add_argument(
        "--parquets-dir",
        default="parquets",
        help="Root directory of built parquet files (default: parquets).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: {pred_dir}/metrics.json).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap evaluation to the first N samples per dataset (default: no cap).",
    )
    args = parser.parse_args()

    pred_dir = ROOT / args.pred_dir
    parquets_dir = ROOT / args.parquets_dir
    output_path = Path(args.output) if args.output else pred_dir / "metrics.json"

    pred_files = sorted(pred_dir.glob("PRED_*.jsonl"))
    if not pred_files:
        print(f"No PRED_*.jsonl files found in {pred_dir}")
        sys.exit(1)

    print(f"Found {len(pred_files)} prediction file(s) in {pred_dir}\n")

    all_results: dict[str, Any] = {}

    for pred_path in pred_files:
        registry_key, split = parse_pred_filename(pred_path.name)
        result_key = f"{registry_key}_{split}"

        print(f"── {pred_path.name}")
        print(f"   registry_key={registry_key}  split={split}")

        if registry_key not in REGISTRY:
            print(f"   [SKIP] No registry entry for '{registry_key}'\n")
            all_results[result_key] = {"error": f"unknown dataset key: {registry_key}"}
            continue

        try:
            result = evaluate_file(pred_path, parquets_dir, registry_key, split,
                                   max_samples=args.max_samples)
            all_results[result_key] = result
            print(
                f"   n_valid={result['n_valid']}/{result['n_samples']}"
                f"  parse_failures={result['n_parse_failures']}"
            )
            for k, v in result["metrics"].items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.4f}")
                else:
                    print(f"   {k}: {v}")
        except Exception as e:
            print(f"   [ERROR] {e}")
            traceback.print_exc()
            all_results[result_key] = {"error": str(e)}

        print()

    for r in all_results.values():
        if "n_samples" in r and r["n_samples"] > 0:
            r["parse_failure_pct"] = round(
                100.0 * r["n_parse_failures"] / r["n_samples"], 1
            )

    print("\n── Format compliance summary (n_metric_valid / n_samples) ────")
    for key, r in sorted(all_results.items()):
        if "n_samples" in r and r["n_samples"] > 0:
            rate = r.get("format_compliance_rate", 0.0)
            parse_fail = r.get("parse_failure_pct", 0.0)
            flag = "  !" if rate < 0.80 else ""
            print(f"  {key:<50}  {rate*100:5.1f}%  (parse_fail={parse_fail:.1f}%){flag}")

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pred_dir": str(pred_dir),
            "parquets_dir": str(parquets_dir),
        },
        "results": all_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
