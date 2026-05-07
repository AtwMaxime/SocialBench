import os
import json
import random
import tempfile
import subprocess
from datasets import load_dataset

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "mustard")


def play_video(video_bytes):
    """Play video (with embedded audio) via ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        print("▶️  Playing video + audio... [close window to continue]")
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
    finally:
        os.unlink(tmp_path)


for modality in ["video_no_context", "video_context"]:
    parquet_file = os.path.join(PARQUETS_DIR, modality, f"mustard_{modality}_train.parquet")
    print(f"\n{'='*60}")
    print(f"📂 {modality}")

    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples")

    idx      = random.randint(0, len(ds) - 1)
    example  = ds[idx]
    user_msg = next(m for m in example["messages"] if m["role"] == "user")
    asst_msg = next(m for m in example["messages"] if m["role"] == "assistant")

    print(f"🎲 Index: {idx}")
    print(f"\n--- 📝 User ---\n{user_msg['content']}")
    print(f"\n--- 💬 Assistant ---\n{asst_msg['content']}\n")

    play_video(example["videos"][0])
