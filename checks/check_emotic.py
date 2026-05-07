import io
import json
import os
import re
import random
from collections import Counter
from PIL import Image, ImageDraw
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "emotic")


def denorm(v, size):
    return round(v / 1000 * size)


def draw_bbox(img, bb_norm, color="red", label=None):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    x1 = denorm(bb_norm[0], W)
    y1 = denorm(bb_norm[1], H)
    x2 = denorm(bb_norm[2], W)
    y2 = denorm(bb_norm[3], H)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=color)
    return img


def check_task(task):
    parquet_file = os.path.join(PARQUETS_DIR, task, f"emotic_{task}_train.parquet")
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        return

    print(f"\n{'='*60}")
    print(f"📂 {task}")
    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples loaded.")

    if task == "discrete":
        # Emotion distribution
        label_counts = Counter()
        for ex in ds:
            answer = json.loads(
                next(m for m in ex["messages"] if m["role"] == "assistant")["content"]
            )
            for e in answer["emotions"]:
                label_counts[e] += 1
        print(f"\n--- Emotion distribution (top 10) ---")
        for label, count in label_counts.most_common(10):
            print(f"  {label}: {count}")
        print(f"  Total unique emotions seen: {len(label_counts)}")
    else:
        # VAD stats
        vs, ar, do = [], [], []
        for ex in ds:
            answer = json.loads(
                next(m for m in ex["messages"] if m["role"] == "assistant")["content"]
            )
            vs.append(answer["valence"])
            ar.append(answer["arousal"])
            do.append(answer["dominance"])
        print(f"\n--- VAD statistics ---")
        for name, vals in [("valence", vs), ("arousal", ar), ("dominance", do)]:
            print(
                f"  {name}: min={min(vals)}  max={max(vals)}  "
                f"mean={sum(vals)/len(vals):.2f}"
            )

    # Random sample
    idx = random.randint(0, len(ds) - 1)
    example = ds[idx]
    user_msg = next(m for m in example["messages"] if m["role"] == "user")
    asst_msg = next(m for m in example["messages"] if m["role"] == "assistant")

    print(f"\n🎲 Sample #{idx}")
    print(f"\n--- 📝 User ---\n{user_msg['content']}")
    print(f"\n--- 💬 Assistant ---\n{asst_msg['content']}\n")

    # Annotate and show image
    m = re.search(
        r"Bounding box: \[(\d+), (\d+), (\d+), (\d+)\]", user_msg["content"]
    )
    if m and example.get("images"):
        bb = [int(v) for v in m.groups()]
        img_raw = example["images"][0]
        if isinstance(img_raw, dict):
            img = Image.open(io.BytesIO(img_raw["bytes"])).copy()
        else:
            img = img_raw.copy()

        answer = json.loads(asst_msg["content"])
        if task == "discrete":
            label = ", ".join(answer["emotions"][:2])
        else:
            label = f"V={answer['valence']} A={answer['arousal']} D={answer['dominance']}"

        img = draw_bbox(img, bb, color="red", label=label)
        img.show()


for task in ["discrete", "vad"]:
    check_task(task)
