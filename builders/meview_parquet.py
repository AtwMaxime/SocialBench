import json
import os
import subprocess
import tempfile

import pandas as pd
from datasets import Dataset, Features, Sequence, Value

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEVIEW_DIR = os.path.join(ROOT_DIR, "dataset", "MEVIEW")
CUTS_DIR = os.path.join(MEVIEW_DIR, "me-cuts", "cuts")
ANNOT_PATH = os.path.join(ROOT_DIR, "dataset", "MEVIEW", "MEVIEW_v2.xlsx")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "meview")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".video_cache")

SOURCE_FPS = 30
N_FRAMES = 16
FRAME_BUFFER = 5  # frames before onset and after offset to include

EMOTION_LABELS = ["contempt", "disgust", "surprise", "happy", "anger", "fear"]

SYSTEM = (
    "You are an expert in micro-expression recognition. "
    "Micro-expressions are brief, involuntary facial expressions lasting less than "
    "half a second that reveal suppressed or concealed emotions. "
    "Given a short video clip of a person's face, identify the concealed emotion. "
    f"Choose from: {json.dumps(EMOTION_LABELS)}. "
    'Provide your answer as a valid JSON object: {"emotion": "..."}.'
)

FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "videos": Sequence(Value("binary")),
        "audios": Sequence(Value("binary")),
    }
)


def load_annotations():
    """Load MEVIEW_v2.xlsx and return a list of annotation dicts."""
    df = pd.read_excel(ANNOT_PATH, engine="openpyxl", header=0)
    # Row 0 is the sub-header ("Subject", "ID", "Onset" ...) — skip it
    df = df.iloc[1:].reset_index(drop=True)

    # Column mapping (from the multi-level header Excel structure):
    #   Unnamed: 0  → Subject
    #   Unnamed: 1  → ID
    #   FINAL       → FINAL onset (within-clip frame, 1-indexed)
    #   Unnamed: 14 → FINAL offset (within-clip frame, 1-indexed)
    #   Unnamed: 19 → Emotion label
    rows = []
    for _, row in df.iterrows():
        subject = row["Unnamed: 0"]
        clip_id = row["Unnamed: 1"]
        onset = row["FINAL"]
        offset = row["Unnamed: 14"]
        emotion_raw = row["Unnamed: 19"]

        if pd.isna(subject) or pd.isna(emotion_raw):
            continue
        try:
            subject = int(subject)
            clip_id = int(clip_id)
            onset = int(onset)
            offset = int(offset)
        except (ValueError, TypeError):
            continue

        emotion = str(emotion_raw).strip().lower()
        emotion = _normalize_emotion(emotion)
        if emotion is None:
            continue

        rows.append(
            {
                "subject": subject,
                "clip_id": clip_id,
                "onset": onset,
                "offset": offset,
                "emotion": emotion,
                "filename": f"sub{subject:02d}-{clip_id}.mp4",
            }
        )
    return rows


def _normalize_emotion(emotion):
    """Map raw emotion string to one of the canonical labels."""
    mapping = {
        "contempt": "contempt",
        "contempt/disgust": "disgust",
        "disgust": "disgust",
        "surprise": "surprise",
        "happy": "happy",
        "happiness": "happy",
        "anger": "anger",
        "fear": "fear",
    }
    return mapping.get(emotion)


def get_frame_count(vid_path):
    """Return total frame count via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "csv=p=0",
        vid_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return int(result.stdout.strip())
    except Exception:
        return None


def encode_clip(vid_path, onset, offset, total_frames):
    """
    Trim clip to [onset - BUFFER, offset + BUFFER] (1-indexed) then
    encode at N_FRAMES frames, H.264, no audio. Result is cached to disk.
    """
    start_frame = max(1, onset - FRAME_BUFFER)
    end_frame = min(total_frames, offset + FRAME_BUFFER)

    cache_name = (
        os.path.basename(vid_path).replace(".mp4", "")
        + f"_on{onset}_off{offset}.mp4"
    )
    cache_path = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    # Compute time range (0-indexed for ffmpeg)
    start_sec = (start_frame - 1) / SOURCE_FPS
    end_sec = end_frame / SOURCE_FPS
    duration = end_sec - start_sec

    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed = os.path.join(tmpdir, "trimmed.mp4")

        # Step 1: Trim
        cmd_trim = [
            "ffmpeg", "-y",
            "-ss", str(start_sec), "-to", str(end_sec),
            "-i", vid_path,
            "-c:v", "copy", "-an",
            trimmed,
        ]
        r = subprocess.run(cmd_trim, capture_output=True, timeout=30)
        if r.returncode != 0 or not os.path.exists(trimmed):
            print(f"  ⚠️  Trim failed for {vid_path}")
            return None

        # Step 2: Re-encode at N_FRAMES frames
        vf = f"fps={N_FRAMES}/{duration},scale=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd_enc = [
            "ffmpeg", "-y",
            "-i", trimmed,
            "-vf", vf,
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
            "-an",
            "-movflags", "frag_keyframe+empty_moov",
            "-f", "mp4", "pipe:1",
        ]
        try:
            result = subprocess.run(cmd_enc, capture_output=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(result.stdout)
                return result.stdout
            print(f"  ⚠️  Encode failed for {vid_path}")
        except Exception as e:
            print(f"  ⚠️  ffmpeg error for {vid_path}: {e}")
    return None


def make_generator(rows):
    def generator():
        skipped = 0
        for r in rows:
            vid_path = os.path.join(CUTS_DIR, r["filename"])
            if not os.path.exists(vid_path):
                print(f"  ⚠️  Missing clip: {r['filename']}")
                skipped += 1
                continue

            total_frames = get_frame_count(vid_path)
            if total_frames is None:
                skipped += 1
                continue

            video_bytes = encode_clip(vid_path, r["onset"], r["offset"], total_frames)
            if video_bytes is None:
                skipped += 1
                continue

            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": "<video>\nWhat concealed emotion is shown in this micro-expression clip?",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"emotion": r["emotion"]}),
                    },
                ],
                "videos": [video_bytes],
                "audios": [],
            }

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


if __name__ == "__main__":
    print("📂 Loading MEVIEW v2 annotations...")
    rows = load_annotations()
    print(f"  Found {len(rows)} valid micro-expression clips")

    from collections import Counter

    counts = Counter(r["emotion"] for r in rows)
    for emotion, n in sorted(counts.items()):
        print(f"    {emotion}: {n}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n🚀 recognition/test ({len(rows)} samples)...")
    ds = Dataset.from_generator(make_generator(rows), features=FEATURES)
    out_path = os.path.join(OUTPUT_DIR, "meview_recognition_test.parquet")
    ds.to_parquet(out_path)
    print(f"✅ {out_path} ({len(ds)} examples)")

    print("\n✨ Done!")
