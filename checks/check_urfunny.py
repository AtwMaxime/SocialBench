import os
import random
import tempfile
import subprocess
from datasets import load_dataset

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "urfunny")


def play_video(video_bytes):
    """Play video (with embedded audio) via ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
    finally:
        os.unlink(tmp_path)


for modality in ["video_audio", "video_context"]:
    parquet_file = os.path.join(PARQUETS_DIR, modality, f"urfunny_{modality}_train.parquet")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        continue

    print(f"\n{'='*60}")
    print(f"📂 {modality}")
    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples")

    funny     = sum(1 for ex in ds if '"funny": 1' in ex["messages"][-1]["content"])
    not_funny = sum(1 for ex in ds if '"funny": 0' in ex["messages"][-1]["content"])
    print(f"   funny={funny}  not_funny={not_funny}")

    idx     = random.randint(0, len(ds) - 1)
    example = ds[idx]
    msgs    = {m["role"]: m["content"] for m in example["messages"]}

    print(f"\n🎲 Sample #{idx}")
    print(f"\n--- 📝 User ---\n{msgs['user']}")
    print(f"\n--- 💬 Assistant ---\n{msgs['assistant']}\n")

    print("▶️  Playing video + audio... [close window to continue]")
    play_video(example["videos"][0])
