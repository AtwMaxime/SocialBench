import json
import os
import random
import subprocess
import tempfile
from collections import Counter

from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR        = os.path.join(ROOT_DIR, "parquets", "mmew")
_APEX_AU_DIR        = os.path.join(PARQUETS_DIR, "apex_au")
_APEX_EMOTION_DIR   = os.path.join(PARQUETS_DIR, "apex_emotion")
_CLIP_EMOTION_DIR   = os.path.join(PARQUETS_DIR, "clip_emotion")
_CLIP_THINK_DIR     = os.path.join(PARQUETS_DIR, "clip_emotion_think")


def play_video(video_bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        print("▶️  Playing video... [close window to continue]")
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
    finally:
        os.unlink(tmp_path)


def show_image(pil_img):
    print(f"   Image size: {pil_img.width}x{pil_img.height}")
    print("🖼️  Displaying image... [close window to continue]")
    pil_img.show()


def print_sample(example, is_video=False):
    msgs = {m["role"]: m["content"] for m in example["messages"]}
    print(f"\n--- 📝 User ---\n{msgs['user']}")
    print(f"\n--- 💬 Assistant ---\n{msgs['assistant']}\n")
    if is_video:
        play_video(example["videos"][0])
    else:
        show_image(example["images"][0])


# ============================================================
# 1. apex_au  —  apex frame → action unit prediction
# ============================================================
for split in ["train", "val"]:
    parquet_file = os.path.join(_APEX_AU_DIR, f"mmew_apex_au_{split}.parquet")
    print(f"\n{'='*60}")
    print(f"📂 apex_au / {split}")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        continue

    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples")

    # AU frequency across all samples
    au_counts = Counter()
    none_count = 0
    for ex in ds:
        aus = json.loads(ex["messages"][-1]["content"]).get("action_units", [])
        if aus:
            au_counts.update(aus)
        else:
            none_count += 1
    print(f"   (no AUs): {none_count}")
    for au, count in sorted(au_counts.items()):
        print(f"   {au:<6} {count:>5}")

    idx = random.randint(0, len(ds) - 1)
    print(f"\n🎲 Sample #{idx}")
    print_sample(ds[idx], is_video=False)


# ============================================================
# 2. apex_emotion  —  apex/still frame → emotion label
# ============================================================
for split in ["train", "val"]:
    parquet_file = os.path.join(_APEX_EMOTION_DIR, f"mmew_apex_emotion_{split}.parquet")
    print(f"\n{'='*60}")
    print(f"📂 apex_emotion / {split}")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        continue

    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples")

    emotion_counts = Counter(
        json.loads(ex["messages"][-1]["content"]).get("emotion")
        for ex in ds
    )
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"   {emotion:<12} {count:>5}")

    idx = random.randint(0, len(ds) - 1)
    print(f"\n🎲 Sample #{idx}")
    print_sample(ds[idx], is_video=False)


# ============================================================
# 3. clip_emotion  —  16-frame clip → emotion (direct)
# ============================================================
for split in ["train", "val"]:
    parquet_file = os.path.join(_CLIP_EMOTION_DIR, f"mmew_clip_emotion_{split}.parquet")
    print(f"\n{'='*60}")
    print(f"📂 clip_emotion / {split}")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        continue

    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples")

    emotion_counts = Counter(
        json.loads(ex["messages"][-1]["content"]).get("emotion")
        for ex in ds
    )
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"   {emotion:<12} {count:>5}")

    idx = random.randint(0, len(ds) - 1)
    print(f"\n🎲 Sample #{idx}")
    print_sample(ds[idx], is_video=True)


# ============================================================
# 4. clip_emotion_think  —  16-frame clip → <think> + emotion
# ============================================================
for split in ["train", "val"]:
    parquet_file = os.path.join(
        _CLIP_THINK_DIR, f"mmew_clip_emotion_think_{split}.parquet"
    )
    print(f"\n{'='*60}")
    print(f"📂 clip_emotion_think / {split}")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        continue

    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples")

    # Parse emotion from the JSON part after </think>
    emotion_counts = Counter()
    for ex in ds:
        answer = ex["messages"][-1]["content"]
        json_part = answer.split("</think>")[-1].strip()
        try:
            emotion_counts[json.loads(json_part).get("emotion")] += 1
        except json.JSONDecodeError:
            pass
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"   {emotion:<12} {count:>5}")

    idx = random.randint(0, len(ds) - 1)
    print(f"\n🎲 Sample #{idx}")
    print_sample(ds[idx], is_video=True)
