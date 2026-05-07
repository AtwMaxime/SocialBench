import math
import os
import json
import numpy as np
import scipy.io
from datasets import Dataset, Features, Sequence, Value, Image as HFImage

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTIC_DIR = os.path.join(ROOT_DIR, "dataset", "EMOTIC")
ANN_FILE = os.path.join(EMOTIC_DIR, "Annotations", "Annotations.mat")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "emotic")

CATEGORIES = [
    "Affection",
    "Anger",
    "Annoyance",
    "Anticipation",
    "Aversion",
    "Confidence",
    "Disapproval",
    "Disconnection",
    "Disquietment",
    "Doubt/Confusion",
    "Embarrassment",
    "Engagement",
    "Esteem",
    "Excitement",
    "Fatigue",
    "Fear",
    "Happiness",
    "Pain",
    "Peace",
    "Pleasure",
    "Sadness",
    "Sensitivity",
    "Suffering",
    "Surprise",
    "Sympathy",
    "Yearning",
]

SYSTEM_DISCRETE = (
    "You are an expert in context-based emotion recognition. "
    "Given an image and the bounding box of a person, identify their emotional state "
    "from visual context, scene, and body language. "
    "Choose from: [" + ", ".join(CATEGORIES) + "]. "
    'Provide your answer as a valid JSON object: {"emotions": ["Emotion1", "Emotion2"]}. '
    "The list must contain at least one emotion."
)

SYSTEM_VAD = (
    "You are an expert in context-based emotion recognition. "
    "Given an image and the bounding box of a person, predict their emotional state "
    "as valence, arousal, and dominance scores. "
    "Each score is an integer from 1 (very low) to 9 (very high). "
    'Provide your answer as a valid JSON object: {"valence": x, "arousal": x, "dominance": x}.'
)

FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(HFImage(decode=True)),
    }
)


# ==========================================
# HELPERS
# ==========================================


def parse_image_size(entry):
    """Return (W, H) from image_size field — sz[0]=W, sz[1]=H."""
    sz = entry["image_size"][0, 0]
    return int(sz[0].flat[0]), int(sz[1].flat[0])


def norm(val, total):
    return max(0, min(1000, round(val / total * 1000)))


def norm_bbox(bbox, w, h):
    """Normalize [x1, y1, x2, y2] pixels → Qwen [0, 1000]."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return [norm(x1, w), norm(y1, h), norm(x2, w), norm(y2, h)]


def parse_categories(ann_cats, ann_idx=0):
    """Extract list of category strings from annotations_categories at ann_idx.
    Returns None if annotations are missing."""
    if ann_cats.shape[1] == 0 or ann_idx >= ann_cats.shape[1]:
        return None
    ann = ann_cats[0, ann_idx]
    cats = ann["categories"]
    if cats.size == 0:
        return None
    return [str(item.flat[0]) for item in cats.flat]


def parse_vad(ann_cont, ann_idx=0):
    """Extract (valence, arousal, dominance) ints from annotations_continuous at ann_idx.
    Returns None if annotations are missing."""
    if ann_cont.shape[1] == 0 or ann_idx >= ann_cont.shape[1]:
        return None
    ann = ann_cont[0, ann_idx]
    v, a, d = ann["valence"].flat[0], ann["arousal"].flat[0], ann["dominance"].flat[0]
    try:
        if any(math.isnan(float(x)) for x in (v, a, d)):
            return None
    except (TypeError, ValueError):
        return None
    return int(v), int(a), int(d)


def parse_all_categories(ann_cats):
    """Extract category lists from all annotators. Returns None if no valid annotation found."""
    results = [
        parse_categories(ann_cats, i)
        for i in range(ann_cats.shape[1])
        if parse_categories(ann_cats, i) is not None
    ]
    return results if results else None


def parse_all_vad(ann_cont):
    """Extract VAD dicts from all annotators. Returns None if no valid annotation found."""
    results = []
    for i in range(ann_cont.shape[1]):
        vad = parse_vad(ann_cont, i)
        if vad is not None:
            v, a, d = vad
            results.append({"valence": v, "arousal": a, "dominance": d})
    return results if results else None


def person_description(p):
    """Build a natural-language description like 'female adult', 'male person', 'teenager'."""
    gender = str(p["gender"][0]) if p["gender"].size > 0 else "Unknown"
    age = str(p["age"][0]) if p["age"].size > 0 else "Unknown"
    g = None if gender == "Unknown" else gender.lower()
    a = None if age == "Unknown" else age.lower()
    if g and a:
        return f"{g} {a}"
    elif g:
        return f"{g} person"
    elif a:
        return a
    return "person"


# ==========================================
# GENERATOR
# ==========================================


def make_generator(mat_split, task, multi_gt=False):
    def generator():
        skipped = 0
        for i in range(mat_split.shape[1]):
            entry = mat_split[0, i]
            fname = str(entry["filename"][0])
            folder = str(entry["folder"][0])
            img_path = os.path.join(EMOTIC_DIR, folder, fname)

            if not os.path.exists(img_path):
                skipped += 1
                continue

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            w, h = parse_image_size(entry)
            persons = entry["person"]

            for j in range(persons.shape[1]):
                p = persons[0, j]
                bb = norm_bbox(p["body_bbox"][0], w, h)
                desc = person_description(p)

                if task == "discrete":
                    if multi_gt:
                        all_cats = parse_all_categories(p["annotations_categories"])
                        if not all_cats:
                            continue
                        answer = json.dumps({"emotions_per_annotator": all_cats})
                    else:
                        categories = parse_categories(p["annotations_categories"])
                        if not categories:
                            continue
                        answer = json.dumps({"emotions": categories})
                    user_content = (
                        f"<image>\n"
                        f"Bounding box: [{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}]\n"
                        f"What are the emotions of the {desc} in this bounding box?"
                    )
                    system = SYSTEM_DISCRETE
                else:  # vad
                    if multi_gt:
                        all_vad = parse_all_vad(p["annotations_continuous"])
                        if not all_vad:
                            continue
                        answer = json.dumps({"vad_per_annotator": all_vad})
                    else:
                        vad = parse_vad(p["annotations_continuous"])
                        if vad is None:
                            continue
                        v, a, d = vad
                        answer = json.dumps({"valence": v, "arousal": a, "dominance": d})
                    user_content = (
                        f"<image>\n"
                        f"Bounding box: [{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}]\n"
                        f"What are the valence, arousal, and dominance scores "
                        f"of the {desc} in this bounding box?"
                    )
                    system = SYSTEM_VAD

                yield {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer},
                    ],
                    "images": [{"bytes": img_bytes, "path": None}],
                }

        if skipped:
            print(f"  ⚠️  Skipped {skipped} missing images")

    return generator


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("📂 Loading Annotations.mat...")
    mat = scipy.io.loadmat(ANN_FILE)

    splits = {
        "train": mat["train"],
        "val": mat["val"],
        "test": mat["test"],
    }

    for task in ["discrete", "vad"]:
        task_dir = os.path.join(OUTPUT_DIR, task)
        os.makedirs(task_dir, exist_ok=True)

        for split_name, mat_split in splits.items():
            multi_gt = split_name in ("val", "test")
            print(f"\n🚀 {task}/{split_name} ({mat_split.shape[1]} images)...")
            ds = Dataset.from_generator(
                make_generator(mat_split, task, multi_gt=multi_gt),
                features=FEATURES,
            )
            output_path = os.path.join(
                task_dir, f"emotic_{task}_{split_name}.parquet"
            )
            ds.to_parquet(output_path)
            print(f"✅ {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
