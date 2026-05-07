import json
import os
import random
import re
from PIL import ImageDraw
from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "proxemics")

COLORS = ["#FF4444", "#4488FF"]  # red = Person 1, blue = Person 2


def denorm(v_norm, size):
    return round(v_norm / 1000 * size)


def check_variant(variant):
    parquet_file = os.path.join(
        PARQUETS_DIR, variant, f"proxemics_{variant}_train.parquet"
    )
    if not os.path.exists(parquet_file):
        print(f"❌ Not found: {parquet_file}")
        return

    print(f"\n{'='*60}")
    print(f"📂 {variant}")
    ds = load_dataset("parquet", data_files={"train": parquet_file}, split="train")
    print(f"✅ {len(ds)} examples loaded.")

    # Label distribution
    from collections import Counter

    label_counts = Counter()
    none_count = 0
    for ex in ds:
        answer = json.loads(
            next(m for m in ex["messages"] if m["role"] == "assistant")["content"]
        )
        touching = answer["touching"]
        if not touching:
            none_count += 1
        for t in touching:
            label_counts[t] += 1
    print(f"\n--- Label distribution ---")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print(f"  (no contact): {none_count}")

    # Random sample
    idx = random.randint(0, len(ds) - 1)
    example = ds[idx]
    user_msg = next(m for m in example["messages"] if m["role"] == "user")
    assistant_msg = next(m for m in example["messages"] if m["role"] == "assistant")

    answer = json.loads(assistant_msg["content"])
    touching = answer["touching"]
    print(f"\n🎲 Sample #{idx}")
    print(f"\n--- 💬 Active classes ---")
    if touching:
        for t in touching:
            print(f"  ✅ {t}")
    else:
        print("  (none — no body parts touching)")

    # Annotate image
    img = example["images"][0]
    imgW, imgH = img.size
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    raw_boxes = re.findall(
        r"Person \d+ bounding box: \[(\d+), (\d+), (\d+), (\d+)\]",
        user_msg["content"],
    )
    for i, (raw, color) in enumerate(zip(raw_boxes, COLORS)):
        x1, y1, x2, y2 = (int(v) for v in raw)
        box = (denorm(x1, imgW), denorm(y1, imgH), denorm(x2, imgW), denorm(y2, imgH))
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), f"Person {i+1}", fill=color)

    if variant == "skeleton":
        kp_pattern = re.compile(r"- [^:]+: \[(\d+), (\d+)\]")
        person_idx = -1
        for line in user_msg["content"].split("\n"):
            if "Person 1 skeleton:" in line:
                person_idx = 0
            elif "Person 2 skeleton:" in line:
                person_idx = 1
            elif person_idx >= 0:
                m = kp_pattern.search(line)
                if m:
                    kp_x = denorm(int(m.group(1)), imgW)
                    kp_y = denorm(int(m.group(2)), imgH)
                    c = COLORS[person_idx]
                    draw.ellipse([kp_x - 4, kp_y - 4, kp_x + 4, kp_y + 4], fill=c)

    annotated.show()


for variant in ["skeleton", "no_skeleton"]:
    check_variant(variant)
