import os
import sys
import io
import json
import subprocess
import pandas as pd
from datasets import Dataset, Features, Sequence, Value

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import VIDEO_MODE, TARGET_FPS, FIXED_FRAMES

MELD_FAIR_DIR = os.path.join(ROOT_DIR, "dataset", "MELD-FAIR")
BASE_DIR      = os.path.join(MELD_FAIR_DIR, "MELD", "realigned")
OUTPUT_DIR    = os.path.join(ROOT_DIR, "parquets", "meld")
CACHE_DIR     = os.path.join(OUTPUT_DIR, ".video_cache")
EMOTIONS   = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]

# split_name → (folder_name_in_realigned, csv_filename)
SPLITS = {
    "train":      ("train", "realigned_train_sent_emo.csv"),
    "validation": ("dev",   "realigned_dev_sent_emo.csv"),
    "test":       ("test",  "realigned_test_sent_emo.csv"),
}

FEATURES = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "videos":   Sequence(Value("binary")),
    "audios":   Sequence(Value("binary")),
})

# ==========================================
# SYSTEM PROMPTS
# ==========================================

SYSTEM = {
    "video_audio": (
        "You are an expert in multimodal emotion recognition. "
        "Given a video clip and an audio recording of a speaker in a conversation, "
        f"classify their emotion into one of: {json.dumps(EMOTIONS)}. "
        'Provide your answer as a valid JSON object: {"emotion": "Category"}.'
    ),
    "audio_only": (
        "You are an expert in emotion recognition from speech. "
        "Given an audio recording of a speaker in a conversation, "
        f"classify their emotion into one of: {json.dumps(EMOTIONS)}. "
        'Provide your answer as a valid JSON object: {"emotion": "Category"}.'
    ),
    "video_transcript": (
        "You are an expert in multimodal emotion recognition. "
        "Given a video clip of a speaker and the transcript of what they said, "
        f"classify their emotion into one of: {json.dumps(EMOTIONS)}. "
        'Provide your answer as a valid JSON object: {"emotion": "Category"}.'
    ),
}

# ==========================================
# HELPERS
# ==========================================

def load_bbox_df():
    csv_path = os.path.join(BASE_DIR, "MELD_active_speaker_face_bboxes.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Dialogue ID"]  = df["Dialogue ID"].astype(int)
    df["Utterance ID"] = df["Utterance ID"].astype(int)
    df["Frame Number"] = df["Frame Number"].astype(int)
    df.set_index(["Split", "Dialogue ID", "Utterance ID", "Frame Number"], inplace=True)
    return df


def get_video_info(vid_path):
    """Return (total_frames, width, height) via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", vid_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        info = json.loads(result.stdout)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                w   = int(s.get("width", 1))
                h   = int(s.get("height", 1))
                nb  = int(s.get("nb_frames", 0))
                if nb == 0:
                    r   = s.get("r_frame_rate", "24/1").split("/")
                    fps = float(r[0]) / max(float(r[1]), 1)
                    nb  = max(1, int(float(s.get("duration", 1)) * fps))
                return nb, w, h
    except Exception:
        pass
    return 0, 1, 1


def get_bbox(bbox_df, split_key, dia_id, utt_id, frame_idx, w, h):
    """Look up Qwen-normalized bbox [x1,y1,x2,y2] in [0,1000] for the given frame."""
    try:
        row = bbox_df.loc[(split_key, dia_id, utt_id, frame_idx)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        x1 = max(0, min(1000, int(row["X Left"]   / w * 1000)))
        y1 = max(0, min(1000, int(row["Y Top"]     / h * 1000)))
        x2 = max(0, min(1000, int(row["X Right"]   / w * 1000)))
        y2 = max(0, min(1000, int(row["Y Bottom"]  / h * 1000)))
        return [x1, y1, x2, y2]
    except KeyError:
        return None


def get_duration(vid_path):
    """Return video duration in seconds via ffprobe."""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "csv=p=0", vid_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip())
    except Exception:
        return None


def downsample_video(vid_path):
    """Downsample video (no audio). Caches result to disk."""
    rel = os.path.relpath(vid_path, BASE_DIR).replace(os.sep, "_")

    if VIDEO_MODE == "framerate":
        suffix = f"_{TARGET_FPS}fps"
        vf     = f"fps={TARGET_FPS}"
    else:  # fixed_number
        duration = get_duration(vid_path)
        if duration is None:
            return None
        suffix = f"_{FIXED_FRAMES}frames"
        vf     = f"fps={FIXED_FRAMES}/{duration}"

    cache_path = os.path.join(CACHE_DIR, rel.replace(".mp4", f"{suffix}.mp4"))

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    cmd = [
        "ffmpeg", "-y", "-i", vid_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-an",
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(result.stdout)
            return result.stdout
        print(f"  ⚠️  ffmpeg failed for {vid_path}")
    except Exception as e:
        print(f"  ⚠️  ffmpeg error for {vid_path}: {e}")
    return None


# ==========================================
# GENERATOR
# ==========================================

def make_generator(rows, split_key, bbox_df, modality):
    needs_video = modality in ("video_audio", "video_transcript")
    needs_audio = modality in ("video_audio", "audio_only")

    def generator():
        skipped = 0
        for _, row in rows.iterrows():
            try:
                dia_id    = int(row.get("Corrected Dialogue_ID", row["Dialogue_ID"]))
                utt_id    = int(row.get("Corrected Utterance_ID", row["Utterance_ID"]))
                emotion   = str(row["Emotion"])
                utterance = str(row["Utterance"])
            except Exception:
                skipped += 1
                continue

            folder   = f"{dia_id:04d}"
            vid_path = os.path.join(BASE_DIR, split_key, "videos", folder,
                                    f"dia{dia_id}_utt{utt_id}.mp4")
            aud_path = os.path.join(BASE_DIR, split_key, "audio", "16000", folder,
                                    f"dia{dia_id}_utt{utt_id}.wav")

            if needs_video and not os.path.exists(vid_path):
                skipped += 1
                continue
            if needs_audio and not os.path.exists(aud_path):
                skipped += 1
                continue

            # --- BBox: look up middle frame of original video ---
            bbox = None
            if needs_video:
                total, w, h = get_video_info(vid_path)
                if total > 0:
                    bbox = get_bbox(bbox_df, split_key, dia_id, utt_id, total // 2, w, h)

            # --- Downsample video (cached) ---
            video_bytes = None
            if needs_video:
                video_bytes = downsample_video(vid_path)
                if video_bytes is None:
                    skipped += 1
                    continue

            # --- Audio bytes ---
            audio_bytes = None
            if needs_audio:
                with open(aud_path, "rb") as f:
                    audio_bytes = f.read()

            # --- Build prompt ---
            bbox_line = (
                f"Speaker face bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n"
                if bbox else ""
            )

            if modality == "video_audio":
                user_content = (
                    f"<video>\n"
                    f"{bbox_line}"
                    f"<audio>\n"
                    f"What is the emotion of the speaker?"
                )
            elif modality == "audio_only":
                user_content = "<audio>\nWhat is the emotion of the speaker?"
            else:  # video_transcript
                user_content = (
                    f"<video>\n"
                    f"{bbox_line}"
                    f"Transcript: \"{utterance}\"\n"
                    f"What is the emotion of the speaker?"
                )

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM[modality]},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": json.dumps({"emotion": emotion})},
                ],
                "videos": [video_bytes] if video_bytes else [],
                "audios": [audio_bytes] if audio_bytes else [],
            }

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("📂 Loading bbox data...")
    bbox_df = load_bbox_df()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, (split_key, csv_file) in SPLITS.items():
        csv_path = os.path.join(BASE_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"❌ CSV not found: {csv_path}")
            continue

        rows = pd.read_csv(csv_path)
        print(f"\n📋 {split_name}: {len(rows)} clips")

        for modality in ["video_audio", "audio_only", "video_transcript"]:
            print(f"\n🚀 {modality}/{split_name}...")
            ds = Dataset.from_generator(
                make_generator(rows, split_key, bbox_df, modality),
                features=FEATURES,
            )
            modality_dir = os.path.join(OUTPUT_DIR, modality)
            os.makedirs(modality_dir, exist_ok=True)
            output_path = os.path.join(modality_dir, f"meld_{modality}_{split_name}.parquet")
            ds.to_parquet(output_path)
            print(f"✅ {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
