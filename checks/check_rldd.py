import os
import random
import tempfile
import subprocess
from datasets import load_dataset

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "rldd")


def play_video(video_bytes):
    """Play video (with embedded audio) via ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
    finally:
        os.unlink(tmp_path)


for split in ["train", "test"]:
    parquet_file = os.path.join(PARQUETS_DIR, f"rldd_{split}.parquet")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        continue

    print(f"\n{'='*60}")
    print(f"📂 rldd / {split}")
    ds = load_dataset("parquet", data_files={split: parquet_file}, split=split)
    print(f"✅ {len(ds)} examples")

    deceptive = [i for i, ex in enumerate(ds) if '"deceptive"' in ex["messages"][-1]["content"]]
    truthful  = [i for i, ex in enumerate(ds) if '"truthful"'  in ex["messages"][-1]["content"]]
    print(f"   deceptive: {len(deceptive)}  truthful: {len(truthful)}")

    idx     = random.choice(deceptive + truthful)
    example = ds[idx]
    msgs    = {m["role"]: m["content"] for m in example["messages"]}

    print(f"\n🎲 Sample #{idx}")
    print(f"\n--- 📝 User ---\n{msgs['user']}")
    print(f"\n--- 💬 Assistant ---\n{msgs['assistant']}\n")

    print("▶️  Playing video + audio... [close window to continue]")
    play_video(example["videos"][0])
