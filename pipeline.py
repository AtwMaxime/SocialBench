import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
from config import MERGE_TEST


def _p(*parts):
    return os.path.join(ROOT_DIR, *parts)


DATASETS = [
    {
        "name": "gazefollow",
        "required_paths": [
            _p("dataset", "Gazefollow", "data_new", "train_annotations.txt")
        ],
        "builder": _p("builders", "gazefollow_parquet.py"),
        "checker": _p("checks", "check_gazefollow.py"),
        "test_parquets": [
            _p("parquets", "gazefollow", "gazefollow_validation.parquet")
        ],
    },
    {
        "name": "videoattentiontarget",
        "required_paths": [
            _p("dataset", "videoattentiontarget", "annotations")
        ],
        "builder": _p("builders", "videoattentiontarget_parquet.py"),
        "checker": _p("checks", "check_videoattentiontarget.py"),
        "test_parquets": [
            _p("parquets", "videoattentiontarget", "frame", "vat_frame_test.parquet")
        ],
    },
    {
        "name": "videocoattention",
        "required_paths": [_p("dataset", "VideoCoAtt", "VideoCoAtt_Dataset")],
        "builder": _p("builders", "videocoattention_parquet.py"),
        "checker": _p("checks", "check_videocoattention.py"),
        "test_parquets": [
            _p(
                "parquets",
                "videocoattention",
                "localization",
                "videocoattention_localization_test.parquet",
            )
        ],
    },
    {
        "name": "vocalsound",
        "required_paths": [_p("dataset", "VocalSound_release_16k", "audio_16k")],
        "builder": _p("builders", "vocalsound_parquet.py"),
        "checker": _p("checks", "check_vocalsound.py"),
        "test_parquets": [_p("parquets", "vocalsound", "vocalsound_test.parquet")],
    },
    {
        "name": "proxemics",
        "required_paths": [_p("dataset", "dataset_proxemics", "images")],
        "builder": _p("builders", "proxemics_parquet.py"),
        "checker": _p("checks", "check_proxemics.py"),
        "test_parquets": [
            _p(
                "parquets",
                "proxemics",
                "no_skeleton",
                "proxemics_no_skeleton_test.parquet",
            )
        ],
    },
    {
        "name": "mmew",
        "required_paths": [_p("dataset", "MMEW", "MMEW_Final")],
        "builder": _p("builders", "mmew_parquet.py"),
        "checker": _p("checks", "check_mmew.py"),
        # no test split — use val
        "test_parquets": [
            _p("parquets", "mmew", "apex_emotion", "mmew_apex_emotion_val.parquet")
        ],
    },
    {
        "name": "affwild2",
        "required_paths": [_p("dataset", "AffWild2", "ABAW Annotations")],
        "builder": _p("builders", "affwild2_parquet.py"),
        "checker": _p("checks", "check_affwild2.py"),
        # no test split — use val
        "test_parquets": [
            _p("parquets", "affwild2", "expr", "affwild2_expr_val.parquet")
        ],
    },
    {
        "name": "voxconverse",
        "required_paths": [_p("dataset", "voxconverse", "dev")],
        "builder": _p("builders", "voxconverse_parquet.py"),
        "checker": _p("checks", "check_voxconverse.py"),
        "test_parquets": [
            _p("parquets", "voxconverse", "diarization", "voxconverse_diarization_test.parquet")
        ],
    },
    {
        "name": "meld",
        "required_paths": [_p("dataset", "MELD-FAIR")],
        "builder": _p("builders", "meld_parquet.py"),
        "checker": _p("checks", "check_meld.py"),
        "test_parquets": [
            _p("parquets", "meld", "video_audio", "meld_video_audio_test.parquet")
        ],
    },
    {
        "name": "emotic",
        "required_paths": [
            _p("dataset", "EMOTIC", "Annotations", "Annotations.mat")
        ],
        "builder": _p("builders", "emotic_parquet.py"),
        "checker": _p("checks", "check_emotic.py"),
        "test_parquets": [
            _p("parquets", "emotic", "discrete", "emotic_discrete_test.parquet")
        ],
    },
    {
        "name": "pisc",
        "required_paths": [_p("dataset", "PISC", "image")],
        "builder": _p("builders", "pisc_parquet.py"),
        "checker": _p("checks", "check_pisc.py"),
        "test_parquets": [_p("parquets", "pisc", "pisc_test.parquet")],
    },
    {
        "name": "msppodcast",
        "required_paths": [_p("dataset", "msp-podcast", "Audios")],
        "builder": _p("builders", "msppodcast_parquet.py"),
        "checker": _p("checks", "check_msppodcast.py"),
        "test_parquets": [
            _p("parquets", "msppodcast", "labels", "msppodcast_labels_test.parquet")
        ],
    },
    {
        "name": "mustard",
        "required_paths": [_p("dataset", "MUStARD", "utterances_final")],
        "builder": _p("builders", "mustard_parquet.py"),
        "checker": _p("checks", "check_mustard.py"),
        "test_parquets": [
            _p(
                "parquets",
                "mustard",
                "video_no_context",
                "mustard_video_no_context_test.parquet",
            )
        ],
    },
    {
        "name": "urfunny",
        "required_paths": [_p("dataset", "UR-FUNNY-V2", "urfunny2_videos")],
        "builder": _p("builders", "urfunny_parquet.py"),
        "checker": _p("checks", "check_urfunny.py"),
        "test_parquets": [
            _p("parquets", "urfunny", "video_audio", "urfunny_video_audio_test.parquet")
        ],
    },
    {
        "name": "rldd",
        "required_paths": [_p("dataset", "RealLifeDeceptionDetection.2016")],
        "builder": _p("builders", "rldd_parquet.py"),
        "checker": _p("checks", "check_rldd.py"),
        "test_parquets": [_p("parquets", "rldd", "rldd_test.parquet")],
    },
    {
        "name": "meview",
        "required_paths": [_p("dataset", "MEVIEW", "me-cuts", "cuts")],
        "builder": _p("builders", "meview_parquet.py"),
        "checker": _p("checks", "check_meview.py"),
        "test_parquets": [
            _p("parquets", "meview", "meview_recognition_test.parquet")
        ],
    },
]


def run_script(path):
    result = subprocess.run([sys.executable, path], cwd=ROOT_DIR)
    return result.returncode == 0


def print_summary(results):
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    w = max(len(r["name"]) for r in results)
    fmt = f"  {{:<{w}}}  {{:<8}}  {{}}"
    print(fmt.format("DATASET", "STATUS", "NOTES"))
    print(f"  {'-' * (w + 22)}")
    for r in results:
        label = {"ok": "OK", "skipped": "SKIPPED", "failed": "FAILED"}[r["status"]]
        print(fmt.format(r["name"], label, r["reason"]))
    ok = sum(1 for r in results if r["status"] == "ok")
    sk = sum(1 for r in results if r["status"] == "skipped")
    fa = sum(1 for r in results if r["status"] == "failed")
    print(f"\n  Total: {len(results)}  OK: {ok}  Skipped: {sk}  Failed: {fa}")
    print(f"{'='*60}\n")


def merge_test_parquets(results):
    print("[MERGE] Assembling benchmark_test.parquet...")
    succeeded = {r["name"] for r in results if r["status"] == "ok"}
    tables = []
    for ds in DATASETS:
        if ds["name"] not in succeeded:
            continue
        for path in ds["test_parquets"]:
            if not os.path.exists(path):
                print(f"  [WARN] not found, skipping: {path}")
                continue
            tbl = pq.read_table(path)
            dataset_col = pa.array([ds["name"]] * tbl.num_rows, type=pa.string())
            tbl = tbl.append_column(pa.field("dataset", pa.string()), dataset_col)
            tables.append(tbl)
            print(f"  + {ds['name']} ({tbl.num_rows} rows)")
    if not tables:
        print("  [WARN] Nothing to merge.")
        return
    merged = pa.concat_tables(tables, promote_options="default")
    out = _p("parquets", "benchmark_test.parquet")
    pq.write_table(merged, out)
    print(f"  [OK] {merged.num_rows} total rows → {out}")


def main():
    results = []

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")

        missing = [p for p in ds["required_paths"] if not os.path.exists(p)]
        if missing:
            for p in missing:
                print(f"  [SKIP] Missing: {p}")
            results.append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": f"missing {len(missing)} path(s)",
                }
            )
            continue

        print(f"  [BUILD] {os.path.basename(ds['builder'])}...")
        t0 = time.time()
        if not run_script(ds["builder"]):
            results.append({"name": name, "status": "failed", "reason": "builder error"})
            continue

        print(f"  [CHECK] {os.path.basename(ds['checker'])}...")
        if not run_script(ds["checker"]):
            results.append({"name": name, "status": "failed", "reason": "checker error"})
            continue

        results.append(
            {"name": name, "status": "ok", "reason": f"{time.time() - t0:.0f}s"}
        )

    print_summary(results)

    if MERGE_TEST:
        merge_test_parquets(results)


if __name__ == "__main__":
    main()
