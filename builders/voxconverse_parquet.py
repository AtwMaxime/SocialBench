import json
import os
import subprocess

from datasets import Dataset, Features, Sequence, Value

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOXCONVERSE_DIR = os.path.join(ROOT_DIR, "dataset", "voxconverse")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "voxconverse")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".audio_cache")

SPLITS = {
    "train": {
        "audio_dir": os.path.join(VOXCONVERSE_DIR, "dev"),
        "rttm_dir": os.path.join(VOXCONVERSE_DIR, "annotations", "dev"),
    },
    "test": {
        "audio_dir": os.path.join(VOXCONVERSE_DIR, "test"),
        "rttm_dir": os.path.join(VOXCONVERSE_DIR, "annotations", "test"),
    },
}

# 30s windows, 15s stride (50% overlap enables cross-window speaker stitching)
# 0.5s bins → 60 entries per window (captures short backchannels)
WINDOW_SEC = 30
STRIDE_SEC = 15
BIN_SEC = 0.5
N_BINS = int(WINDOW_SEC / BIN_SEC)  # 60
MIN_SPEECH_BINS = 6  # skip windows with < 3s of annotated speech

FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "videos": Sequence(Value("binary")),
        "audios": Sequence(Value("binary")),
        # Metadata for inference-time cross-window speaker stitching
        "clip_id": Value("string"),
        "win_start": Value("float32"),
    }
)

SYSTEM = {
    "diarization": (
        "You are an expert in speaker diarization. "
        f"Given a {WINDOW_SEC}-second audio clip, identify who speaks in each 0.5-second bin. "
        "Assign each distinct speaker a letter (A, B, C, ...) in order of first appearance. "
        "Use '-' for silence and combined letters (e.g. 'AB') for simultaneous speech. "
        f"Provide your answer as a valid JSON object with exactly {N_BINS} entries: "
        '{"timeline": ["A", "A", "AB", "B", "-", ...]}.'
    ),
    "speaker_count": (
        "You are an expert in speaker diarization. "
        f"Given a {WINDOW_SEC}-second audio clip, count the number of distinct speakers present. "
        'Provide your answer as a valid JSON object: {"num_speakers": N}.'
    ),
}

USER = {
    "diarization": (
        "<audio>\n"
        f"For each of the {N_BINS} half-second bins in this clip, indicate which speaker(s) "
        "are active. Use letters (A, B, ...) in order of first appearance, '-' for silence, "
        "combined letters (e.g. 'AB') for overlap."
    ),
    "speaker_count": (
        f"<audio>\nHow many distinct speakers are present in this {WINDOW_SEC}-second clip?"
    ),
}


# ==========================================
# RTTM PARSING
# ==========================================


def parse_rttm(rttm_path):
    """Return list of (start, end, speaker) sorted by start time."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append((start, start + duration, speaker))
    segments.sort(key=lambda x: x[0])
    return segments


def get_duration(wav_path):
    """Return WAV duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception:
        return None


# ==========================================
# BIN-BASED ANNOTATION
# ==========================================


def build_timeline(segments, win_start):
    """
    Build N_BINS x BIN_SEC labels for the window starting at win_start.

    A speaker is active in a bin if their overlap exceeds half the bin
    duration (0.25s), matching the DER forgiveness collar from the paper.
    Speaker IDs normalized to A/B/C... by order of first appearance.
    """
    win_end = win_start + WINDOW_SEC

    # Clip segments to window, compute relative times
    window_segs = []
    for start, end, spk in segments:
        cs = max(start, win_start) - win_start
        ce = min(end, win_end) - win_start
        if ce > cs:
            window_segs.append((cs, ce, spk))

    # Normalize speaker IDs by first appearance
    mapping = {}
    labels = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for s, _, spk in sorted(window_segs, key=lambda x: x[0]):
        if spk not in mapping:
            mapping[spk] = next(labels)

    # Fill bins
    timeline = []
    for i in range(N_BINS):
        bin_start = i * BIN_SEC
        bin_end = bin_start + BIN_SEC
        active = set()
        for s, e, spk in window_segs:
            if min(e, bin_end) - max(s, bin_start) > BIN_SEC / 2:
                active.add(mapping[spk])
        timeline.append("".join(sorted(active)) if active else "-")

    return timeline, mapping


# ==========================================
# AUDIO TRIMMING
# ==========================================


def trim_audio(wav_path, start_sec, end_sec, clip_id):
    """Trim WAV to [start_sec, end_sec], resample to 16kHz mono. Cached."""
    cache_name = f"{clip_id}_{start_sec}_{end_sec}.wav"
    cache_path = os.path.join(CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec), "-to", str(end_sec),
        "-i", wav_path,
        "-ar", "16000", "-ac", "1",
        "-f", "wav", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and result.stdout:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(result.stdout)
            return result.stdout
        print(f"  ⚠️  ffmpeg trim failed: {clip_id} [{start_sec}-{end_sec}]")
    except Exception as e:
        print(f"  ⚠️  ffmpeg error: {clip_id}: {e}")
    return None


# ==========================================
# GENERATORS
# ==========================================


def make_generator(audio_dir, rttm_dir, task):
    def generator():
        skipped = 0
        rttm_files = sorted(f for f in os.listdir(rttm_dir) if f.endswith(".rttm"))

        for rttm_file in rttm_files:
            clip_id = rttm_file[: -len(".rttm")]
            wav_path = os.path.join(audio_dir, f"{clip_id}.wav")

            if not os.path.exists(wav_path):
                skipped += 1
                continue

            duration = get_duration(wav_path)
            if duration is None:
                skipped += 1
                continue

            segments = parse_rttm(os.path.join(rttm_dir, rttm_file))
            if not segments:
                skipped += 1
                continue

            win_start = 0
            while win_start + WINDOW_SEC <= int(duration):
                timeline, mapping = build_timeline(segments, win_start)

                active_bins = sum(1 for b in timeline if b != "-")
                if active_bins < MIN_SPEECH_BINS:
                    win_start += STRIDE_SEC
                    continue

                audio_bytes = trim_audio(wav_path, win_start, win_start + WINDOW_SEC, clip_id)
                if audio_bytes is None:
                    win_start += STRIDE_SEC
                    skipped += 1
                    continue

                if task == "diarization":
                    answer = json.dumps({"timeline": timeline})
                else:
                    answer = json.dumps({"num_speakers": len(mapping)})

                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM[task]},
                        {"role": "user", "content": USER[task]},
                        {"role": "assistant", "content": answer},
                    ],
                    "videos": [],
                    "audios": [audio_bytes],
                    "clip_id": clip_id,
                    "win_start": float(win_start),
                }

                win_start += STRIDE_SEC

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, paths in SPLITS.items():
        for task in ["diarization", "speaker_count"]:
            print(f"\n🚀 {task}/{split_name}...")
            ds = Dataset.from_generator(
                make_generator(paths["audio_dir"], paths["rttm_dir"], task),
                features=FEATURES,
            )
            task_subdir = "speaker" if task == "speaker_count" else task
            task_dir = os.path.join(OUTPUT_DIR, task_subdir)
            os.makedirs(task_dir, exist_ok=True)
            out_path = os.path.join(task_dir, f"voxconverse_{task}_{split_name}.parquet")
            ds.to_parquet(out_path)
            print(f"✅ {out_path} ({len(ds)} examples)")

    print("\n✨ Done!")
