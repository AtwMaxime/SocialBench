import io
import json
import os
import random
import subprocess
import sys
import tempfile
import wave
from collections import Counter

import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT_DIR, "parquets", "affwild2_av")

EXPR_LABELS = [
    "Neutral",
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
    "Other",
]

# (task, subfolder, filename)
EXPECTED = [
    ("expr", "expr", "affwild2_expr_train.parquet"),
    ("expr", "expr", "affwild2_expr_val.parquet"),
    ("va", "va", "affwild2_va_train.parquet"),
    ("va", "va", "affwild2_va_val.parquet"),
]

# Builder defaults (affwild2_av_parquet.py): pure span in [MIN_DUR, 2 * MAX_HALF] seconds
MIN_DUR = 1.0
MAX_DUR = 4.0
DUR_TOL = 0.25  # container/codec rounding
WAV_SR = 16000
WAV_CH = 1

# Files are large (up to ~10 GB): scan the text column fully,
# decode binaries only for a few random row groups.
N_SAMPLE_ROW_GROUPS = 3
N_PROBE_VIDEOS = 3
# Some source videos have an audio track that ends before the video, giving a
# truncated (or empty) WAV. Tolerated up to this fraction of sampled rows.
MAX_SHORT_AUDIO_FRAC = 0.05


def parse_answer(task, content):
    parsed = json.loads(content)
    if task == "expr":
        assert "label" in parsed, "missing 'label'"
        assert parsed["label"] in EXPR_LABELS, f"unknown label '{parsed['label']}'"
    elif task == "va":
        assert "valence" in parsed and "arousal" in parsed, "missing valence/arousal"
        assert (
            -1.0 <= parsed["valence"] <= 1.0
        ), f"valence out of range: {parsed['valence']}"
        assert (
            -1.0 <= parsed["arousal"] <= 1.0
        ), f"arousal out of range: {parsed['arousal']}"
    return parsed


def check_wav(wav_bytes, where):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        # ffmpeg piped to stdout can't seek back to patch the header sizes
        # (nframes is a 0xFFFFFFFF placeholder) — derive duration from the payload.
        n = len(w.readframes(-1)) // (ch * sw)
    assert sr == WAV_SR, f"{where}: WAV sample rate {sr} != {WAV_SR}"
    assert ch == WAV_CH, f"{where}: WAV channels {ch} != {WAV_CH}"
    dur = n / sr
    assert dur <= MAX_DUR + DUR_TOL, f"{where}: WAV duration {dur:.2f}s > {MAX_DUR}s"
    return dur


def probe_video(mp4_bytes):
    """Return (has_video, has_audio, duration) via ffprobe."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4_bytes)
        f.flush()
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                f.name,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    info = json.loads(out.stdout or "{}")
    types = {s.get("codec_type") for s in info.get("streams", [])}
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    return "video" in types, "audio" in types, dur


def check_file(path, task):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    print(f"✅ {os.path.relpath(path, PARQUET_DIR)} — {n_rows} rows")
    assert n_rows > 0, "Empty parquet"

    required_cols = {"messages", "videos", "audios"}
    missing = required_cols - set(pf.schema_arrow.names)
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)

    # ---- full scan of the text column ----
    answers = []
    for i, msgs in enumerate(
        pf.read(columns=["messages"]).column("messages").to_pylist()
    ):
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"Row {i}: bad roles {roles}"
        assert msgs[1]["content"].startswith("<video>"), f"Row {i}: missing <video> tag"
        try:
            answers.append(parse_answer(task, msgs[2]["content"]))
        except (AssertionError, json.JSONDecodeError) as e:
            raise AssertionError(f"Row {i}: {e}")
    print("   Message structure: OK")

    if task == "expr":
        dist = Counter(a["label"] for a in answers)
        print("   Label distribution:")
        for lbl in EXPR_LABELS:
            print(f"     {lbl}: {dist.get(lbl, 0)}")
    else:
        vs = [a["valence"] for a in answers]
        ars = [a["arousal"] for a in answers]
        print(
            f"   Valence  min={min(vs):.3f}  max={max(vs):.3f}  mean={sum(vs)/len(vs):.3f}"
        )
        print(
            f"   Arousal  min={min(ars):.3f}  max={max(ars):.3f}  mean={sum(ars)/len(ars):.3f}"
        )

    # ---- binary payloads on sampled row groups ----
    rng = random.Random(0)
    rgs = rng.sample(
        range(pf.num_row_groups), min(N_SAMPLE_ROW_GROUPS, pf.num_row_groups)
    )
    videos, wav_durs = [], []
    for rg in rgs:
        tbl = pf.read_row_group(rg, columns=["videos", "audios"])
        for j, (vids, auds) in enumerate(
            zip(tbl.column("videos").to_pylist(), tbl.column("audios").to_pylist())
        ):
            where = f"row group {rg} row {j}"
            assert len(vids) == 1, f"{where}: expected 1 video, got {len(vids)}"
            assert len(vids[0]) > 1000, f"{where}: video bytes suspiciously small"
            assert len(auds) == 1, f"{where}: expected 1 audio, got {len(auds)}"
            wav_durs.append(check_wav(auds[0], where))
            videos.append(vids[0])
    print(f"   Sampled {len(videos)} rows from {len(rgs)} row groups: video/audio OK")
    short = sum(d < MIN_DUR - DUR_TOL for d in wav_durs)
    if short:
        print(
            f"   ⚠️  {short}/{len(wav_durs)} sampled WAVs shorter than {MIN_DUR}s "
            f"(audio track ends before video)"
        )
    assert (
        short / len(wav_durs) <= MAX_SHORT_AUDIO_FRAC
    ), f"Too many short WAVs: {short}/{len(wav_durs)}"
    print(
        f"   WAV duration  min={min(wav_durs):.2f}s  max={max(wav_durs):.2f}s  "
        f"mean={sum(wav_durs)/len(wav_durs):.2f}s"
    )

    for k, mp4 in enumerate(rng.sample(videos, min(N_PROBE_VIDEOS, len(videos)))):
        has_v, has_a, dur = probe_video(mp4)
        assert has_v, f"Probe {k}: no video stream"
        assert has_a, f"Probe {k}: no muxed audio stream"
        assert (
            dur == 0 or dur <= MAX_DUR + DUR_TOL
        ), f"Probe {k}: duration {dur:.2f}s too long"
    print(
        f"   ffprobe on {min(N_PROBE_VIDEOS, len(videos))} clips: video+audio streams OK"
    )
    print(f"   Sample video size: {len(videos[0]) / 1024:.1f} KB")


def check():
    for task, subfolder, fname in EXPECTED:
        path = os.path.join(PARQUET_DIR, subfolder, fname)
        check_file(path, task)
        print()

    print("✅ check_affwild2_av passed!")


if __name__ == "__main__":
    check()
