import os
import sys
import io
import json
import random
import subprocess
from datasets import Dataset, Features, Sequence, Value

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import VIDEO_MODE, TARGET_FPS, FIXED_FRAMES

MUSTARD_DIR = os.path.join(ROOT_DIR, "dataset", "MUStARD")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "parquets", "mustard")
CACHE_DIR   = os.path.join(OUTPUT_DIR, ".video_cache")

RANDOM_SEED = 42

FEATURES = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "videos":   Sequence(Value("binary")),
    "audios":   Sequence(Value("binary")),
})

SYSTEM = {
    "video_no_context": (
        "You are an expert at detecting sarcasm in multimodal communication. "
        "Given a video clip and audio recording of a speaker, determine whether "
        "their utterance is sarcastic. "
        'Provide your answer as a valid JSON object: {"sarcasm": true} or {"sarcasm": false}.'
    ),
    "video_context": (
        "You are an expert at detecting sarcasm in multimodal communication. "
        "Given a video clip and audio recording of a speaker, along with the preceding "
        "dialogue context, determine whether their utterance is sarcastic. "
        'Provide your answer as a valid JSON object: {"sarcasm": true} or {"sarcasm": false}.'
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
    rel = os.path.relpath(vid_path, MUSTARD_DIR).replace(os.sep, "_")

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


def format_context(context, context_speakers):
    lines = [f'  - {spk}: "{utt}"' for spk, utt in zip(context_speakers, context)]
    return "\n".join(lines)


def make_generator(samples, modality):
    def generator():
        skipped = 0
        for key, entry in samples:
            vid_path = os.path.join(MUSTARD_DIR, "utterances_final", f"{key}.mp4")

            if not os.path.exists(vid_path):
                skipped += 1
                continue

            video_bytes = downsample_video(vid_path)
            if video_bytes is None:
                skipped += 1
                continue

            if modality == "video_no_context":
                user_content = "<video>\nIs the speaker being sarcastic?"
            else:  # video_context
                ctx_text = format_context(entry["context"], entry["context_speakers"])
                user_content = (
                    f"<video>\n"
                    f"Dialogue context:\n{ctx_text}\n"
                    f"Is the speaker being sarcastic?"
                )

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM[modality]},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": json.dumps({"sarcasm": entry["sarcasm"]})},
                ],
                "videos": [video_bytes],
                "audios": [],
            }

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


if __name__ == "__main__":
    print("📂 Loading annotations...")
    with open(os.path.join(MUSTARD_DIR, "sarcasm_data.json")) as f:
        data = json.load(f)

    # 80/20 split with fixed seed
    keys = list(data.keys())
    rng  = random.Random(RANDOM_SEED)
    rng.shuffle(keys)
    split_idx = int(len(keys) * 0.8)
    splits = {
        "train": [(k, data[k]) for k in keys[:split_idx]],
        "test":  [(k, data[k]) for k in keys[split_idx:]],
    }
    print(f"  train: {len(splits['train'])}  test: {len(splits['test'])}")

    for modality in ["video_no_context", "video_context"]:
        modality_dir = os.path.join(OUTPUT_DIR, modality)
        os.makedirs(modality_dir, exist_ok=True)
        for split_name, samples in splits.items():
            print(f"\n🚀 {modality}/{split_name} ({len(samples)} samples)...")
            ds = Dataset.from_generator(
                make_generator(samples, modality),
                features=FEATURES,
            )
            output_path = os.path.join(modality_dir, f"mustard_{modality}_{split_name}.parquet")
            ds.to_parquet(output_path)
            print(f"✅ {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
