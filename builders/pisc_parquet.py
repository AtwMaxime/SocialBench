import os
import json
from datasets import Dataset, Features, Sequence, Value, Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PISC_DIR  = os.path.join(ROOT_DIR, "dataset", "PISC")
IMAGE_DIR = os.path.join(PISC_DIR, "image")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "pisc")

RELATIONSHIP_LABELS = {
    1: "Friends",
    2: "Family",
    3: "Couple",
    4: "Professional",
    5: "Commercial",
    6: "No relation",
}

SPLITS = {
    "train":      "relationship_split/relation_trainidx.json",
    "validation": "relationship_split/relation_validx.json",
    "test":       "relationship_split/relation_testidx.json",
}

SYSTEM_PROMPT = (
    'You are a vision-language model specialized in social relationship recognition. '
    'Given an image and the bounding boxes of two individuals, classify their fine-grained '
    'social relationship into one of: ["Friends", "Family", "Couple", "Professional", '
    '"Commercial", "No relation"]. Provide your answer as a valid JSON string.'
)

# ==========================================
# 2. HELPERS
# ==========================================

def normalize_bbox(bbox, imgW, imgH):
    """Normalize pixel bbox [x1, y1, x2, y2] to Qwen's [0, 1000] range."""
    x1, y1, x2, y2 = bbox
    return (
        round(x1 / imgW * 1000),
        round(y1 / imgH * 1000),
        round(x2 / imgW * 1000),
        round(y2 / imgH * 1000),
    )

def make_generator(split_ids, annotation_map, relationship_data):
    def generator():
        for img_id in split_ids:
            if img_id not in relationship_data or img_id not in annotation_map:
                continue

            ann = annotation_map[img_id]
            imgW, imgH = ann["imgW"], ann["imgH"]
            bboxes = ann["bbox"]

            img_path = os.path.join(IMAGE_DIR, f"{ann['id']:05d}.jpg")
            if not os.path.exists(img_path):
                print(f"⚠️  Image not found: {img_path}")
                continue

            # Read image once, yield one example per pair
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            for pair_str, label_int in relationship_data[img_id].items():
                i, j = (int(x) - 1 for x in pair_str.split())

                if i >= len(bboxes) or j >= len(bboxes):
                    print(f"⚠️  Invalid pair '{pair_str}' for image {img_id} "
                          f"({len(bboxes)} persons)")
                    continue

                b1 = normalize_bbox(bboxes[i], imgW, imgH)
                b2 = normalize_bbox(bboxes[j], imgW, imgH)
                label = RELATIONSHIP_LABELS[label_int]

                user_content = (
                    f"<image>\n"
                    f"Two individuals are visible in this image.\n"
                    f"Person 1 bounding box: [{b1[0]}, {b1[1]}, {b1[2]}, {b1[3]}]\n"
                    f"Person 2 bounding box: [{b2[0]}, {b2[1]}, {b2[2]}, {b2[3]}]\n"
                    f"What is the social relationship between Person 1 and Person 2?"
                )

                yield {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": json.dumps({"relationship": label})},
                    ],
                    "images": [{"bytes": img_bytes, "path": None}],
                }

    return generator

# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":
    print("📂 Loading annotations...")
    with open(os.path.join(PISC_DIR, "annotation_image_info.json")) as f:
        annotations = json.load(f)
    annotation_map = {str(ann["id"]): ann for ann in annotations}

    with open(os.path.join(PISC_DIR, "relationship.json")) as f:
        relationship_data = json.load(f)

    features = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images":   Sequence(Image(decode=True)),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, split_file in SPLITS.items():
        print(f"\n🚀 Processing {split_name}...")

        with open(os.path.join(PISC_DIR, split_file)) as f:
            split_ids = json.load(f)

        ds = Dataset.from_generator(
            make_generator(split_ids, annotation_map, relationship_data),
            features=features,
        )

        output_path = os.path.join(OUTPUT_DIR, f"pisc_{split_name}.parquet")
        ds.to_parquet(output_path)
        print(f"✅ Saved: {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
