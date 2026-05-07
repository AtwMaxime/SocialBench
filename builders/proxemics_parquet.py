import os
import json
import random
from datasets import Dataset, Features, Sequence, Value, Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXEMICS_DIR = os.path.join(ROOT_DIR, "dataset", "dataset_proxemics")
IMAGE_DIR     = os.path.join(PROXEMICS_DIR, "images", "release")
LABELS_FILE   = os.path.join(PROXEMICS_DIR, "labels_6classes_pair.json")
BB_FILE       = os.path.join(PROXEMICS_DIR, "labels_6classes_pair_BBs.json")
OUTPUT_DIR    = os.path.join(ROOT_DIR, "parquets", "proxemics")

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
IMAGE_SIZE  = 224  # All images are 224×224

CLASSES = [
    "Hand touch hand",
    "Hand touch shoulder",
    "Shoulder touch shoulder",
    "Hand touch elbow",
    "Elbow touch shoulder",
    "Hand touch torso",
]

KEYPOINT_NAMES = [
    "Left Head",
    "Right Head",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Hand",
    "Right Hand",
    "Left Torso",
    "Right Torso",
]

SYSTEM_PROMPT = (
    'You are an expert in analyzing physical contact between people. '
    'Given an image of two individuals and their body information, identify which body '
    'parts are touching between them. Choose from: '
    '["Hand touch hand", "Hand touch shoulder", "Shoulder touch shoulder", '
    '"Hand touch elbow", "Elbow touch shoulder", "Hand touch torso"]. '
    'Provide your answer as a valid JSON string with a list of touching body part pairs. '
    'The list may be empty if no body parts are touching.'
)

# ==========================================
# 2. HELPERS
# ==========================================

def norm(val):
    """Normalize a pixel coordinate in [0, IMAGE_SIZE] to [0, 1000]."""
    return round(val / IMAGE_SIZE * 1000)

def bb_to_qwen(bb):
    """Convert dataset BB format [y_min, y_max, x_min, x_max] to Qwen [x1, y1, x2, y2] in [0, 1000]."""
    y_min, y_max, x_min, x_max = bb
    return [norm(x_min), norm(y_min), norm(x_max), norm(y_max)]

def format_skeleton(kps, person_label):
    """Format 10 keypoints as named lines, coordinates normalized to [0, 1000]."""
    lines = [f"{person_label} skeleton:"]
    for name, (x, y) in zip(KEYPOINT_NAMES, kps):
        lines.append(f"  - {name}: [{norm(x)}, {norm(y)}]")
    return "\n".join(lines)

# ==========================================
# 3. GENERATOR
# ==========================================

def make_generator(entries, with_skeleton):
    def generator():
        skipped = 0
        for fname, label_data, bb_data in entries:
            img_path = os.path.join(IMAGE_DIR, fname)
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                skipped += 1
                continue

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            # Bounding boxes → Qwen [x1, y1, x2, y2] in [0, 1000]
            b0 = bb_to_qwen(bb_data["p0"])
            b1 = bb_to_qwen(bb_data["p1"])

            # Active contact classes
            vec = label_data["proxemics"]["p0-p1"]
            touching = [CLASSES[i] for i, v in enumerate(vec) if v == 1]

            if with_skeleton:
                kps0 = label_data["coordinates"]["p0"]
                kps1 = label_data["coordinates"]["p1"]
                skel0 = format_skeleton(kps0, "Person 1")
                skel1 = format_skeleton(kps1, "Person 2")
                user_content = (
                    f"<image>\n"
                    f"Here are the bounding boxes of the 2 persons:\n"
                    f"Person 1 bounding box: [{b0[0]}, {b0[1]}, {b0[2]}, {b0[3]}]\n"
                    f"Person 2 bounding box: [{b1[0]}, {b1[1]}, {b1[2]}, {b1[3]}]\n\n"
                    f"{skel0}\n\n"
                    f"{skel1}\n\n"
                    f"Evaluate which body parts are touching between Person 1 and Person 2."
                )
            else:
                user_content = (
                    f"<image>\n"
                    f"Here are the bounding boxes of the 2 persons:\n"
                    f"Person 1 bounding box: [{b0[0]}, {b0[1]}, {b0[2]}, {b0[3]}]\n"
                    f"Person 2 bounding box: [{b1[0]}, {b1[1]}, {b1[2]}, {b1[3]}]\n\n"
                    f"Evaluate which body parts are touching between Person 1 and Person 2."
                )

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": json.dumps({"touching": touching})},
                ],
                "images": [{"bytes": img_bytes, "path": None}],
            }

        if skipped:
            print(f"⚠️  Total skipped: {skipped}")

    return generator

# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    print("📂 Loading annotations...")
    with open(LABELS_FILE) as f:
        labels = json.load(f)
    with open(BB_FILE) as f:
        bbs = json.load(f)

    # Build entry list — only images present in both annotation files
    all_entries = []
    for fname in sorted(labels.keys()):
        if fname not in bbs:
            print(f"⚠️  No BB for {fname}, skipping.")
            continue
        all_entries.append((fname, labels[fname], bbs[fname]))

    print(f"Total entries: {len(all_entries)}")

    # Reproducible 80/20 train/test split
    random.seed(RANDOM_SEED)
    shuffled = all_entries[:]
    random.shuffle(shuffled)
    split_idx    = int(len(shuffled) * TRAIN_RATIO)
    train_entries = shuffled[:split_idx]
    test_entries  = shuffled[split_idx:]
    print(f"Train: {len(train_entries)}, Test: {len(test_entries)}")

    features = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images":   Sequence(Image(decode=True)),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for with_skeleton, variant in [(True, "skeleton"), (False, "no_skeleton")]:
        variant_dir = os.path.join(OUTPUT_DIR, variant)
        os.makedirs(variant_dir, exist_ok=True)
        for split_name, entries in [("train", train_entries), ("test", test_entries)]:
            print(f"\n🚀 Processing {variant}/{split_name}...")
            ds = Dataset.from_generator(
                make_generator(entries, with_skeleton),
                features=features,
            )
            output_path = os.path.join(
                variant_dir, f"proxemics_{variant}_{split_name}.parquet"
            )
            ds.to_parquet(output_path)
            print(f"✅ Saved: {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
