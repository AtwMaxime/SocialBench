"""
AffWild2 audio+video parquet builder (video_audio condition).

Unlike affwild2_parquet.py — which trims exactly N_FRAMES source frames, retimes
the clip to N_FRAMES fps and drops audio (`-an`) — this builder extracts a
*time-based* window of `--window-seconds` centered on the same center frame, at
NATIVE fps, keeping the audio muxed in the mp4 AND (optionally) writing a separate
16 kHz mono WAV into the `audios` field.

Window SELECTION (which center frames become samples), the per-frame labels, the
diversity filter and the splits are all reused unchanged from affwild2_parquet.py,
so this dataset is directly comparable to affwild2_swift — only the clip payload is
richer (audio + wider temporal context).

Usage:
  python affwild2_av_parquet.py --tasks expr va expr_think --splits train val \
      --max-half 2.0 --min-dur 1.0 --audio both
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# affwild2_parquet.py imports `from datasets import ...` at module top for a FEATURES
# object that is never used on the streaming-writer path. Stub it so this builder runs
# in minimal envs (e.g. without huggingface `datasets` installed).
if "datasets" not in sys.modules:
    import types as _types

    _stub = _types.ModuleType("datasets")
    _stub.Features = dict
    _stub.Sequence = lambda *a, **k: None
    _stub.Value = lambda *a, **k: None
    sys.modules["datasets"] = _stub

import affwild2_parquet as base  # noqa: E402

# ---- reused, unchanged ----
TASK_DIRS = base.TASK_DIRS
EXPR_LABELS = base.EXPR_LABELS
AU_COLS = base.AU_COLS
N_FRAMES = base.N_FRAMES  # label/selection window (16) — keep for comparability
CENTER = base.CENTER
STRIDE = base.STRIDE
is_valid = base.is_valid
make_filter_state = base.make_filter_state
passes_filter = base.passes_filter
update_filter_state = base.update_filter_state
load_annotation_by_task = base.load_annotation_by_task
load_expr, load_va, load_au = base.load_expr, base.load_va, base.load_au
make_answer = base.make_answer
get_think_stems = base.get_think_stems
build_video_index = base.build_video_index

ROOT_DIR = base.ROOT_DIR
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "affwild2_av")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".av_cache")

# Prompts: original text hardcodes "16-frame video clip", which is no longer true
# once we widen to a time window. Generalise the wording; JSON schema unchanged.
SYSTEM = {
    "expr": (
        "You are an expert in facial expression recognition. "
        "Given a short video clip with audio, predict the facial expression of the "
        "person at the center of the clip. "
        f"Classify into one of: {json.dumps(EXPR_LABELS)}. "
        'Provide your answer as a valid JSON object: {"label": "Expression"}.'
    ),
    "va": (
        "You are an expert in affective computing. "
        "Given a short video clip with audio, predict the valence and arousal of the "
        "person at the center of the clip. Both values are continuous in [-1, 1]. "
        'Provide your answer as a valid JSON object: {"valence": x.xxx, "arousal": x.xxx}.'
    ),
    "au": (
        "You are an expert in facial action unit detection. "
        "Given a short video clip with audio, predict which action units are active "
        "at the center of the clip. "
        f"Possible action units: {json.dumps(AU_COLS)}. "
        'Provide your answer as a valid JSON object: {"action_units": ["AU1", ...]} '
        'or {"action_units": []} if none are active.'
    ),
    "expr_think": (
        "You are an expert in facial expression recognition and affective computing. "
        "Given a short video clip with audio, analyze the center of the clip: first "
        "provide the valence/arousal and active action units inside <think> tags, then "
        "classify the facial expression. "
        f"Expression must be one of: {json.dumps(EXPR_LABELS)}. "
        "Use the format:\n<think>\n"
        '{"valence": x.xxx, "arousal": x.xxx}\n'
        '{"action_units": ["AU1", ...]}\n</think>\n'
        '{"label": "Expression"}'
    ),
}
USER = {
    "expr": "<video>\nWhat is the facial expression of the person at the center of this clip?",
    "va": "<video>\nWhat are the valence and arousal values at the center of this clip?",
    "au": "<video>\nWhich action units are active at the center of this clip?",
    "expr_think": (
        "<video>\nFor the center of this clip, provide the valence/arousal and active "
        "action units, then give the expression label."
    ),
}

_PA_SCHEMA = pa.schema(
    [
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.string()),
                    ]
                )
            ),
        ),
        ("videos", pa.list_(pa.large_binary())),
        ("audios", pa.list_(pa.large_binary())),
    ]
)

# runtime config, set in main
MAX_HALF = 2.0  # max half-window (s): clip spans at most [center-2s, center+2s]
MIN_DUR = 1.0  # drop sample if the pure span is shorter than this (s)
VA_TOL = 0.2  # va: stay in window while |v-vc|<=tol AND |a-ac|<=tol
AUDIO_MODE = "both"  # "muxed" | "both"
WAV_SR = 16000
WAV_CH = 1

_probe_cache = {}


def probe_video(path):
    """Return (fps, duration_seconds). ffprobe is authoritative (fps varies per video)."""
    if path in _probe_cache:
        return _probe_cache[path]
    fps, dur = 30.0, None
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.split()
        # first token = r_frame_rate (e.g. 30/1), second = duration
        if out:
            num, den = out[0].split("/")
            if float(den) > 0:
                fps = float(num) / float(den)
        if len(out) > 1:
            dur = float(out[1])
    except Exception as e:
        print(f"  ⚠️  probe failed for {os.path.basename(path)}: {e}")
    _probe_cache[path] = (fps, dur)
    return fps, dur


def compute_pure_span(task, ann, center, fps, n):
    """
    Grow a window outward from `center` while the annotation stays on the same
    EXPR label (task='expr'/'expr_think') or within VA_TOL of the center VA value
    (task='va'), capped at ±MAX_HALF seconds. Returns (t0, dur) in seconds, or
    None if the pure span is shorter than MIN_DUR (drop the sample).
    """
    cap = round(MAX_HALF * fps)
    lo_cap = max(0, center - cap)
    hi_cap = min(n - 1, center + cap)

    if task in ("expr", "expr_think"):
        cval = ann[center]

        def match(j):
            return ann[j] != -1 and ann[j] == cval

    else:  # va
        cv, ca = ann[center]

        def match(j):
            v, a = ann[j]
            return (
                -1.0 <= v <= 1.0
                and -1.0 <= a <= 1.0
                and abs(v - cv) <= VA_TOL
                and abs(a - ca) <= VA_TOL
            )

    left = center
    while left - 1 >= lo_cap and match(left - 1):
        left -= 1
    right = center
    while right + 1 <= hi_cap and match(right + 1):
        right += 1

    t0 = left / fps
    dur = (right + 1 - left) / fps
    if dur < MIN_DUR:
        return None
    return t0, dur


def build_av_clip(vid_path, t0, dur):
    """
    Extract [t0, t0+dur] at native fps, keeping audio muxed.
    Returns (mp4_bytes, wav_bytes_or_None). Cached on disk.
    """
    stem = os.path.splitext(os.path.basename(vid_path))[0]
    h = hashlib.md5(f"{vid_path}_{t0:.3f}_{dur:.3f}".encode()).hexdigest()[:12]
    mp4_cache = os.path.join(CACHE_DIR, f"{stem}_{h}.mp4")
    wav_cache = os.path.join(CACHE_DIR, f"{stem}_{h}.wav")

    mp4_bytes = wav_bytes = None

    if os.path.exists(mp4_cache):
        with open(mp4_cache, "rb") as f:
            mp4_bytes = f.read()
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t0:.3f}",
            "-i",
            vid_path,
            "-t",
            f"{dur:.3f}",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:1",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=180)
            if r.returncode == 0 and r.stdout:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(mp4_cache, "wb") as f:
                    f.write(r.stdout)
                mp4_bytes = r.stdout
            else:
                print(
                    f"  ⚠️  ffmpeg(mp4) failed {os.path.basename(vid_path)}@{t0:.2f}s"
                )
        except Exception as e:
            print(f"  ⚠️  ffmpeg(mp4) error {os.path.basename(vid_path)}: {e}")

    if AUDIO_MODE == "both":
        if os.path.exists(wav_cache):
            with open(wav_cache, "rb") as f:
                wav_bytes = f.read()
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{t0:.3f}",
                "-i",
                vid_path,
                "-t",
                f"{dur:.3f}",
                "-vn",
                "-ac",
                str(WAV_CH),
                "-ar",
                str(WAV_SR),
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "pipe:1",
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=180)
                if r.returncode == 0 and r.stdout:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    with open(wav_cache, "wb") as f:
                        f.write(r.stdout)
                    wav_bytes = r.stdout
                else:
                    print(
                        f"  ⚠️  ffmpeg(wav) failed {os.path.basename(vid_path)}@{t0:.2f}s"
                    )
            except Exception as e:
                print(f"  ⚠️  ffmpeg(wav) error {os.path.basename(vid_path)}: {e}")

    return mp4_bytes, wav_bytes


def make_sliding_generator(split, task, video_index, stride=STRIDE):
    ann_dir = TASK_DIRS[task][split]

    def generator():
        skipped_no_video = skipped_ffmpeg = skipped_short = 0
        for ann_file in sorted(f for f in os.listdir(ann_dir) if f.endswith(".txt")):
            stem = os.path.splitext(ann_file)[0]
            vid_path = video_index.get(stem)
            if vid_path is None:
                skipped_no_video += 1
                continue
            ann = load_annotation_by_task(task, os.path.join(ann_dir, ann_file))
            n = len(ann)
            if n < N_FRAMES:
                continue
            fps, duration = probe_video(vid_path)
            filter_state = make_filter_state(task)
            for w in range(0, n - N_FRAMES + 1, stride):
                center = w + CENTER
                if not is_valid(task, ann, center):
                    continue
                val = ann[center]
                if not passes_filter(task, val, filter_state):
                    continue
                span = compute_pure_span(task, ann, center, fps, n)
                if span is None:
                    skipped_short += 1
                    continue
                mp4_bytes, wav_bytes = build_av_clip(vid_path, span[0], span[1])
                if mp4_bytes is None:
                    skipped_ffmpeg += 1
                    continue
                update_filter_state(task, filter_state, val)
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM[task]},
                        {"role": "user", "content": USER[task]},
                        {"role": "assistant", "content": make_answer(task, val)},
                    ],
                    "videos": [mp4_bytes],
                    "audios": [wav_bytes] if wav_bytes is not None else [],
                }
        if skipped_no_video:
            print(f"  ⚠️  No video found: {skipped_no_video}")
        if skipped_short:
            print(f"  ℹ️  dropped (pure span < {MIN_DUR}s): {skipped_short}")
        if skipped_ffmpeg:
            print(f"  ⚠️  ffmpeg failures: {skipped_ffmpeg}")

    return generator


def make_think_generator(split, video_index, stride=STRIDE):
    train_stems, val_stems = get_think_stems()
    common_stems = train_stems if split == "train" else val_stems

    def _ann_paths(stem):
        for s in ("train", "val"):
            e = os.path.join(TASK_DIRS["expr"][s], f"{stem}.txt")
            v = os.path.join(TASK_DIRS["va"][s], f"{stem}.txt")
            a = os.path.join(TASK_DIRS["au"][s], f"{stem}.txt")
            if os.path.exists(e) and os.path.exists(v) and os.path.exists(a):
                return e, v, a
        return None, None, None

    def generator():
        skipped_no_video = skipped_ffmpeg = skipped_short = 0
        for stem in common_stems:
            vid_path = video_index.get(stem)
            if vid_path is None:
                skipped_no_video += 1
                continue
            e_path, v_path, a_path = _ann_paths(stem)
            if e_path is None:
                continue
            ann_expr, ann_va, ann_au = (
                load_expr(e_path),
                load_va(v_path),
                load_au(a_path),
            )
            n = min(len(ann_expr), len(ann_va), len(ann_au))
            if n < N_FRAMES:
                continue
            fps, duration = probe_video(vid_path)
            filter_state = make_filter_state("expr")
            for w in range(0, n - N_FRAMES + 1, stride):
                center = w + CENTER
                if not (
                    is_valid("expr", ann_expr, center)
                    and is_valid("va", ann_va, center)
                    and is_valid("au", ann_au, center)
                ):
                    continue
                expr_val = ann_expr[center]
                if not passes_filter("expr", expr_val, filter_state):
                    continue
                span = compute_pure_span("expr", ann_expr, center, fps, n)
                if span is None:
                    skipped_short += 1
                    continue
                mp4_bytes, wav_bytes = build_av_clip(vid_path, span[0], span[1])
                if mp4_bytes is None:
                    skipped_ffmpeg += 1
                    continue
                update_filter_state("expr", filter_state, expr_val)
                vv, aa = ann_va[center]
                active = [AU_COLS[i] for i, b in enumerate(ann_au[center]) if b == 1]
                think = (
                    json.dumps({"valence": round(vv, 3), "arousal": round(aa, 3)})
                    + "\n"
                    + json.dumps({"action_units": active})
                )
                answer = f"<think>\n{think}\n</think>\n" + json.dumps(
                    {"label": EXPR_LABELS[expr_val]}
                )
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM["expr_think"]},
                        {"role": "user", "content": USER["expr_think"]},
                        {"role": "assistant", "content": answer},
                    ],
                    "videos": [mp4_bytes],
                    "audios": [wav_bytes] if wav_bytes is not None else [],
                }
        if skipped_no_video:
            print(f"  ⚠️  No video found: {skipped_no_video}")
        if skipped_short:
            print(f"  ℹ️  dropped (pure span < {MIN_DUR}s): {skipped_short}")
        if skipped_ffmpeg:
            print(f"  ⚠️  ffmpeg failures: {skipped_ffmpeg}")

    return generator


def _write_parquet_streaming(generator_fn, output_path, batch_size=100):
    count = 0
    bm, bv, ba = [], [], []
    with pq.ParquetWriter(output_path, _PA_SCHEMA) as writer:
        for row in generator_fn():
            bm.append(
                [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
            )
            bv.append(row.get("videos") or [])
            ba.append(row.get("audios") or [])
            count += 1
            if count % batch_size == 0:
                writer.write_table(
                    pa.table(
                        {"messages": bm, "videos": bv, "audios": ba}, schema=_PA_SCHEMA
                    )
                )
                bm, bv, ba = [], [], []
                print(f"  wrote {count} examples...", end="\r", flush=True)
        if bm:
            writer.write_table(
                pa.table(
                    {"messages": bm, "videos": bv, "audios": ba}, schema=_PA_SCHEMA
                )
            )
    print(f"✅ {output_path} ({count} examples)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build AffWild2 audio+video parquets.")
    # "au" is not supported: compute_pure_span() only handles expr / va annotations.
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["expr", "va"],
        choices=["expr", "va", "expr_think"],
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val"]
    )
    p.add_argument(
        "--max-half",
        type=float,
        default=2.0,
        help="Max half-window in seconds (clip spans at most center ± this).",
    )
    p.add_argument(
        "--min-dur",
        type=float,
        default=1.0,
        help="Drop the sample if the pure span is shorter than this (s).",
    )
    p.add_argument(
        "--va-tol",
        type=float,
        default=0.2,
        help="VA: grow window while |v-vc|<=tol and |a-ac|<=tol.",
    )
    p.add_argument(
        "--audio",
        choices=["muxed", "both"],
        default="both",
        help="'muxed'=audio only inside mp4; 'both'=also a separate 16k mono WAV in audios[].",
    )
    p.add_argument("--wav-sr", type=int, default=16000)
    p.add_argument(
        "--limit-smoke",
        type=int,
        default=0,
        help="If >0, stop each split after N examples (smoke test).",
    )
    args = p.parse_args()

    MAX_HALF = args.max_half
    MIN_DUR = args.min_dur
    VA_TOL = args.va_tol
    AUDIO_MODE = args.audio
    WAV_SR = args.wav_sr
    print(
        f"ℹ️  pure-span mode: max_half={MAX_HALF}s  min_dur={MIN_DUR}s  va_tol={VA_TOL}"
    )
    print(f"ℹ️  audio={AUDIO_MODE}  wav_sr={WAV_SR}")
    print(f"ℹ️  OUTPUT_DIR → {OUTPUT_DIR}")

    print("📂 Building video index...")
    video_index = build_video_index()
    print(f"  Found {len(video_index)} videos")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def maybe_smoke(gen_fn):
        if args.limit_smoke <= 0:
            return gen_fn

        def wrapped():
            for i, row in enumerate(gen_fn()):
                if i >= args.limit_smoke:
                    break
                yield row

        return wrapped

    for task in [t for t in args.tasks if t != "expr_think"]:
        for split in args.splits:
            print(f"\n🚀 {task}/{split} (pure-span)...")
            task_dir = os.path.join(OUTPUT_DIR, task)
            os.makedirs(task_dir, exist_ok=True)
            out = os.path.join(task_dir, f"affwild2_{task}_{split}.parquet")
            _write_parquet_streaming(
                maybe_smoke(make_sliding_generator(split, task, video_index)), out
            )

    if "expr_think" in args.tasks:
        for split in args.splits:
            print(f"\n🚀 expr_think/{split} (pure-span)...")
            expr_dir = os.path.join(OUTPUT_DIR, "expr")
            os.makedirs(expr_dir, exist_ok=True)
            out = os.path.join(expr_dir, f"affwild2_expr_think_{split}.parquet")
            _write_parquet_streaming(
                maybe_smoke(make_think_generator(split, video_index)), out
            )

    print("\n✨ Done!")
