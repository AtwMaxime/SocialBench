import os
import sys
import json
import random
import subprocess
from datasets import Dataset, Features, Sequence, Value

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import VIDEO_MODE, TARGET_FPS, FIXED_FRAMES

RLDD_DIR   = os.path.join(ROOT_DIR, "dataset", "RealLifeDeceptionDetection.2016",
                           "Real-life_Deception_Detection_2016")
CLIPS_DIR  = os.path.join(RLDD_DIR, "Clips")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "rldd")
CACHE_DIR  = os.path.join(OUTPUT_DIR, ".video_cache")

RANDOM_SEED = 42

FEATURES = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "videos":   Sequence(Value("binary")),
    "audios":   Sequence(Value("binary")),
})

SYSTEM = (
    "You are an expert at detecting deception in video. "
    "Given a video clip and audio recording of a person, determine whether "
    "they are being deceptive or truthful. "
    'Provide your answer as a valid JSON object: {"label": "deceptive"} or {"label": "truthful"}.'
)


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


def make_generator(samples):
    def generator():
        skipped = 0
        for vid_path, label in samples:
            video_bytes = downsample_video(vid_path)
            if video_bytes is None:
                skipped += 1
                continue

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": "<video>\nIs this person being deceptive or truthful?"},
                    {"role": "assistant", "content": json.dumps({"label": label})},
                ],
                "videos": [video_bytes],
                "audios": [],
            }

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


if __name__ == "__main__":
    # Collect all clips with their labels
    all_samples = []
    for label, folder in [("deceptive", "Deceptive"), ("truthful", "Truthful")]:
        clip_dir = os.path.join(CLIPS_DIR, folder)
        for fname in sorted(os.listdir(clip_dir)):
            if fname.endswith(".mp4"):
                all_samples.append((os.path.join(clip_dir, fname), label))

    print(f"📂 Found {len(all_samples)} clips  "
          f"({sum(1 for _, l in all_samples if l == 'deceptive')} deceptive, "
          f"{sum(1 for _, l in all_samples if l == 'truthful')} truthful)")

    # 80/20 split with fixed seed
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    splits = {
        "train": all_samples[:split_idx],
        "test":  all_samples[split_idx:],
    }
    print(f"  train: {len(splits['train'])}  test: {len(splits['test'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, samples in splits.items():
        print(f"\n🚀 {split_name} ({len(samples)} samples)...")
        ds = Dataset.from_generator(make_generator(samples), features=FEATURES)
        output_path = os.path.join(OUTPUT_DIR, f"rldd_{split_name}.parquet")
        ds.to_parquet(output_path)
        print(f"✅ {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
