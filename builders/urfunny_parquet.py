import os
import sys
import json
import pickle
import subprocess
from datasets import Dataset, Features, Sequence, Value

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import VIDEO_MODE, TARGET_FPS, FIXED_FRAMES

URFUNNY_DIR = os.path.join(ROOT_DIR, "dataset", "UR-FUNNY-V2")
VIDEOS_DIR  = os.path.join(URFUNNY_DIR, "urfunny2_videos")
SDK_DIR     = os.path.join(URFUNNY_DIR, "sdk_features")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "parquets", "urfunny")
CACHE_DIR   = os.path.join(OUTPUT_DIR, ".video_cache")

FEATURES = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "videos":   Sequence(Value("binary")),
    "audios":   Sequence(Value("binary")),
})

SYSTEM = {
    "video_audio": (
        "You are an expert at detecting humor in video. "
        "Given a video clip and audio recording of a speaker delivering a punchline, "
        "determine whether it is funny. "
        'Provide your answer as a valid JSON object: {"funny": 1} or {"funny": 0}.'
    ),
    "video_context": (
        "You are an expert at detecting humor in video. "
        "Given a video clip and audio recording of a speaker delivering a punchline, "
        "along with the preceding context sentences, determine whether it is funny. "
        'Provide your answer as a valid JSON object: {"funny": 1} or {"funny": 0}.'
    ),
}


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
    """Downsample video keeping the audio track. Caches result to disk."""
    if VIDEO_MODE == "framerate":
        suffix = f"_{TARGET_FPS}fps"
        vf     = f"fps={TARGET_FPS}"
    else:  # fixed_number
        duration = get_duration(vid_path)
        if duration is None:
            return None
        suffix = f"_{FIXED_FRAMES}frames"
        vf     = f"fps={FIXED_FRAMES}/{duration}"

    safe_name  = os.path.basename(vid_path).replace(".mp4", f"{suffix}.mp4")
    cache_path = os.path.join(CACHE_DIR, safe_name)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    cmd = [
        "ffmpeg", "-y", "-i", vid_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-c:a", "aac", "-ac", "2",
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


def format_context(context_sentences):
    return "\n".join(f'  - "{s}"' for s in context_sentences)


def make_generator(ids, labels, lang, modality):
    def generator():
        skipped = 0
        for clip_id in ids:
            vid_path = os.path.join(VIDEOS_DIR, f"{clip_id}.mp4")
            if not os.path.exists(vid_path):
                skipped += 1
                continue

            video_bytes = downsample_video(vid_path)
            if video_bytes is None:
                skipped += 1
                continue

            label = labels[clip_id]
            entry = lang.get(clip_id, {})

            if modality == "video_audio":
                user_content = "<video>\nIs the punchline funny?"
            else:  # video_context
                context = entry.get("context_sentences", [])
                ctx_text = format_context(context)
                user_content = (
                    f"<video>\n"
                    f"Context:\n{ctx_text}\n"
                    f"Is the punchline funny?"
                )

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM[modality]},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": json.dumps({"funny": label})},
                ],
                "videos": [video_bytes],
                "audios": [],
            }

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


if __name__ == "__main__":
    print("📂 Loading annotations...")
    with open(os.path.join(SDK_DIR, "humor_label_sdk.pkl"), "rb") as f:
        labels = pickle.load(f, encoding="latin1")
    with open(os.path.join(SDK_DIR, "data_folds.pkl"), "rb") as f:
        folds = pickle.load(f, encoding="latin1")
    with open(os.path.join(SDK_DIR, "language_sdk.pkl"), "rb") as f:
        lang = pickle.load(f, encoding="latin1")

    for split_name, ids in folds.items():
        funny     = sum(1 for i in ids if labels.get(i) == 1)
        not_funny = sum(1 for i in ids if labels.get(i) == 0)
        print(f"  {split_name}: {len(ids)} clips  (funny={funny}, not_funny={not_funny})")

    for modality in ["video_audio", "video_context"]:
        modality_dir = os.path.join(OUTPUT_DIR, modality)
        os.makedirs(modality_dir, exist_ok=True)
        for split_name, ids in folds.items():
            print(f"\n🚀 {modality}/{split_name} ({len(ids)} samples)...")
            ds = Dataset.from_generator(
                make_generator(ids, labels, lang, modality),
                features=FEATURES,
            )
            output_path = os.path.join(modality_dir, f"urfunny_{modality}_{split_name}.parquet")
            ds.to_parquet(output_path)
            print(f"✅ {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
