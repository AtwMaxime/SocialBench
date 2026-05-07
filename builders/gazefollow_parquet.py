import json
import os
from collections import defaultdict

from datasets import Dataset, Features, Image, Sequence, Value

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAZEFOLLOW_DIR = os.path.join(ROOT_DIR, "dataset", "Gazefollow", "data_new")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "gazefollow")

TRAIN_ANN = os.path.join(GAZEFOLLOW_DIR, "train_annotations.txt")
TEST_ANN = os.path.join(GAZEFOLLOW_DIR, "test_annotation.txt")

# ==========================================
# 2. PROMPTS
# ==========================================

SYSTEM_PROMPT = (
    "You are an expert in gaze estimation. "
    "Given an image and the bounding box of a person's head, predict where they are looking. "
    "Bounding box coordinates are normalized to [0, 1000] in (x1, y1, x2, y2) format. "
    'Provide your answer as a valid JSON object: {"gaze_point": [x, y]}.'
)

# ==========================================
# 3. HELPERS
# ==========================================


def norm1000(v):
    """Normalize a [0, 1] coordinate to Qwen's [0, 1000] range."""
    return round(v * 1000)


def bbox_to_qwen(x, y, w, h):
    """Convert (x, y, w, h) in [0,1] to [x1, y1, x2, y2] in [0, 1000]."""
    return [norm1000(x), norm1000(y), norm1000(x + w), norm1000(y + h)]


def make_example(img_bytes, head_bbox, gaze_x, gaze_y):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "<image>\n"
                    f"Person's head bounding box: [{head_bbox[0]}, {head_bbox[1]}, "
                    f"{head_bbox[2]}, {head_bbox[3]}]\n"
                    "Where is this person looking?"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"gaze_point": [norm1000(gaze_x), norm1000(gaze_y)]}
                ),
            },
        ],
        "images": [{"bytes": img_bytes, "path": None}],
    }


def make_example_multi(img_bytes, head_bbox, gazes):
    """Validation variant: stores all annotator gaze points for multi-GT metrics."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "<image>\n"
                    f"Person's head bounding box: [{head_bbox[0]}, {head_bbox[1]}, "
                    f"{head_bbox[2]}, {head_bbox[3]}]\n"
                    "Where is this person looking?"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "gaze_points": [
                            [norm1000(gx), norm1000(gy)] for gx, gy in gazes
                        ]
                    }
                ),
            },
        ],
        "images": [{"bytes": img_bytes, "path": None}],
    }


def load_image(img_path):
    full_path = os.path.join(GAZEFOLLOW_DIR, img_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path, "rb") as f:
        return f.read()


# ==========================================
# 4. GENERATORS
# ==========================================


def train_generator():
    """
    One example per annotation row.
    Each row: one person in one image with their head bbox and gaze target.
    """
    skipped = 0
    with open(TRAIN_ANN) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            img_path = parts[0].strip()
            try:
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                gaze_x = float(parts[8])
                gaze_y = float(parts[9])
            except ValueError:
                continue

            img_bytes = load_image(img_path)
            if img_bytes is None:
                skipped += 1
                continue

            yield make_example(img_bytes, bbox_to_qwen(x, y, w, h), gaze_x, gaze_y)

    if skipped:
        print(f"  ⚠️  Skipped (image not found): {skipped}")


def validation_generator():
    """
    Test set: 10 annotators per person.
    Groups by (image_path, person_id) and averages gaze coordinates across annotators.
    """
    # First pass: collect all annotations grouped by (image_path, person_id)
    groups = defaultdict(list)
    head_bboxes = {}

    with open(TEST_ANN) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            img_path = parts[0].strip()
            idx = parts[1].strip()  # format "{person_id}-{annotator_id}"
            person_id = idx.split("-")[0]
            key = (img_path, person_id)

            try:
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                gaze_x = float(parts[8])
                gaze_y = float(parts[9])
            except ValueError:
                continue

            if key not in head_bboxes:
                head_bboxes[key] = (x, y, w, h)
            groups[key].append((gaze_x, gaze_y))

    # Second pass: yield one example per person (all annotator gaze points)
    skipped = 0
    for (img_path, person_id), gazes in sorted(groups.items()):
        img_bytes = load_image(img_path)
        if img_bytes is None:
            skipped += 1
            continue

        x, y, w, h = head_bboxes[(img_path, person_id)]
        yield make_example_multi(img_bytes, bbox_to_qwen(x, y, w, h), gazes)

    if skipped:
        print(f"  ⚠️  Skipped (image not found): {skipped}")


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features = Features(
        {
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "images": Sequence(Image(decode=True)),
        }
    )

    for split_name, gen in [("train", train_generator), ("validation", validation_generator)]:
        print(f"\n🚀 {split_name}...")
        ds = Dataset.from_generator(gen, features=features)
        out_path = os.path.join(OUTPUT_DIR, f"gazefollow_{split_name}.parquet")
        ds.to_parquet(out_path)
        print(f"✅ {out_path} ({len(ds)} examples)")

    print("\n✨ Done!")
