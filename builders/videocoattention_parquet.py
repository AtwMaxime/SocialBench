import json
import os
import random
from bisect import bisect_left
from collections import defaultdict

from datasets import Dataset, Features, Image, Sequence, Value

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(
    ROOT_DIR, "dataset", "VideoCoAtt", "VideoCoAtt_Dataset"
)
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "videocoattention")

# Images are 480 px wide × 320 px tall
IMG_W = 480
IMG_H = 320

SPLITS = {
    "train": "train",
    "val": "validate",
    "test": "test",
}

RANDOM_SEED = 42

# ==========================================
# 2. PROMPTS
# ==========================================

SYSTEM_DETECTION = (
    "You are an expert in social scene understanding. "
    "Given an image and the bounding boxes of people in the scene, "
    "determine whether they share a common visual attention target. "
    "Provide your answer as a valid JSON object: "
    '{"co_attention": true} or {"co_attention": false}.'
)

SYSTEM_LOCALIZATION = (
    "You are an expert in social scene understanding. "
    "Given an image and the bounding boxes of people sharing a visual attention target, "
    "locate the shared attention region. "
    "Bounding box coordinates are normalized to [0, 1000] in (x1, y1, x2, y2) format. "
    'Provide your answer as a valid JSON object: {"bbox": [x1, y1, x2, y2]}.'
)

# ==========================================
# 3. HELPERS
# ==========================================


def normalize_bbox(x1, y1, x2, y2, w=IMG_W, h=IMG_H):
    """Normalize pixel bbox to Qwen [0, 1000] range."""
    return [
        round(x1 / w * 1000),
        round(y1 / h * 1000),
        round(x2 / w * 1000),
        round(y2 / h * 1000),
    ]


def parse_annotation_file(ann_path):
    """
    Parse a VideoCoAtt annotation file.

    Returns:
        dict: frame_id (int) -> list of events, each event is a dict:
              {"ca_bbox": [x1,y1,x2,y2], "face_bboxes": [[x1,y1,x2,y2], ...]}
    """
    frame_data = defaultdict(list)
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            # Need at least: co_att_id, frame_id, ca_bbox(4), 1 face(4) = 10 values
            if len(parts) < 10:
                continue
            frame_id = int(parts[1])
            ca_bbox = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
            face_bboxes = []
            i = 6
            while i + 3 < len(parts):
                face_bboxes.append(
                    [int(parts[i]), int(parts[i + 1]), int(parts[i + 2]), int(parts[i + 3])]
                )
                i += 4
            if face_bboxes:
                frame_data[frame_id].append(
                    {"ca_bbox": ca_bbox, "face_bboxes": face_bboxes}
                )
    return dict(frame_data)


def face_str(face_bboxes_norm):
    """Format a list of normalized face bboxes as a human-readable string."""
    lines = []
    for i, b in enumerate(face_bboxes_norm):
        lines.append(f"Person {i + 1}: [{b[0]}, {b[1]}, {b[2]}, {b[3]}]")
    return "\n".join(lines)


def load_image_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def img_path(split_dir, video_id, frame_id):
    return os.path.join(
        IMAGES_DIR, split_dir, str(video_id), f"{frame_id:05d}_{video_id}.jpg"
    )


# ==========================================
# 4. GENERATORS
# ==========================================


def make_localization_generator(split_dir):
    """
    One example per annotation row.
    Given frame + face bboxes → predict co-attention target bbox.
    """

    def generator():
        ann_dir = os.path.join(ANNOTATIONS_DIR, split_dir)
        video_ids = sorted(
            int(f[:-4]) for f in os.listdir(ann_dir) if f.endswith(".txt")
        )

        for video_id in video_ids:
            ann_path = os.path.join(ann_dir, f"{video_id}.txt")
            frame_data = parse_annotation_file(ann_path)

            for frame_id in sorted(frame_data):
                p = img_path(split_dir, video_id, frame_id)
                if not os.path.exists(p):
                    continue
                img_bytes = load_image_bytes(p)

                for event in frame_data[frame_id]:
                    faces_norm = [normalize_bbox(*b) for b in event["face_bboxes"]]
                    ca_norm = normalize_bbox(*event["ca_bbox"])

                    user_content = (
                        "<image>\n"
                        "The following people are present in the image:\n"
                        f"{face_str(faces_norm)}\n"
                        "Where is the shared visual attention target?"
                    )
                    yield {
                        "messages": [
                            {"role": "system", "content": SYSTEM_LOCALIZATION},
                            {"role": "user", "content": user_content},
                            {
                                "role": "assistant",
                                "content": json.dumps({"bbox": ca_norm}),
                            },
                        ],
                        "images": [{"bytes": img_bytes, "path": None}],
                    }

    return generator


def make_detection_generator(split_dir, rng):
    """
    One example per unique frame.
    Positives: annotated frames → {"co_attention": true}
    Negatives: unannotated frames, face bboxes from nearest annotated frame → {"co_attention": false}
    Negatives are subsampled to 1:1 balance with positives.
    """

    def generator():
        ann_dir = os.path.join(ANNOTATIONS_DIR, split_dir)
        video_ids = sorted(
            int(f[:-4]) for f in os.listdir(ann_dir) if f.endswith(".txt")
        )

        for video_id in video_ids:
            ann_path = os.path.join(ann_dir, f"{video_id}.txt")
            frame_data = parse_annotation_file(ann_path)
            annotated_frames = sorted(frame_data)

            # All frames available in the images folder
            vid_img_dir = os.path.join(IMAGES_DIR, split_dir, str(video_id))
            if not os.path.isdir(vid_img_dir):
                continue
            all_frame_ids = sorted(
                int(fname.split("_")[0])
                for fname in os.listdir(vid_img_dir)
                if fname.endswith(".jpg")
            )

            # --- Positives ---
            positives = []
            for frame_id in annotated_frames:
                p = img_path(split_dir, video_id, frame_id)
                if not os.path.exists(p):
                    continue
                # Use first co-att event's face bboxes
                event = frame_data[frame_id][0]
                faces_norm = [normalize_bbox(*b) for b in event["face_bboxes"]]
                positives.append((frame_id, p, faces_norm, True))

            if not positives:
                continue

            # --- Negatives from unannotated frames ---
            annotated_set = set(annotated_frames)
            unannotated = [f for f in all_frame_ids if f not in annotated_set]
            neg_count = min(len(positives), len(unannotated))
            sampled_neg = rng.sample(unannotated, neg_count)

            negatives = []
            for frame_id in sampled_neg:
                p = img_path(split_dir, video_id, frame_id)
                if not os.path.exists(p):
                    continue
                # Face bboxes from nearest annotated frame (proxy for face positions)
                idx = bisect_left(annotated_frames, frame_id)
                idx = min(idx, len(annotated_frames) - 1)
                nearest = annotated_frames[idx]
                event = frame_data[nearest][0]
                faces_norm = [normalize_bbox(*b) for b in event["face_bboxes"]]
                negatives.append((frame_id, p, faces_norm, False))

            # Combine, shuffle
            all_examples = positives + negatives
            rng.shuffle(all_examples)

            for frame_id, p, faces_norm, label in all_examples:
                img_bytes = load_image_bytes(p)
                user_content = (
                    "<image>\n"
                    "The following people are present in the image:\n"
                    f"{face_str(faces_norm)}\n"
                    "Are these people sharing a common visual attention target?"
                )
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM_DETECTION},
                        {"role": "user", "content": user_content},
                        {
                            "role": "assistant",
                            "content": json.dumps({"co_attention": label}),
                        },
                    ],
                    "images": [{"bytes": img_bytes, "path": None}],
                }

    return generator


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

    for split_name, split_dir in SPLITS.items():
        for task in ["localization", "detection"]:
            print(f"\n🚀 {task}/{split_name}...")

            rng = random.Random(RANDOM_SEED)

            if task == "localization":
                gen = make_localization_generator(split_dir)
            else:
                gen = make_detection_generator(split_dir, rng)

            ds = Dataset.from_generator(gen, features=features)
            task_dir = os.path.join(OUTPUT_DIR, task)
            os.makedirs(task_dir, exist_ok=True)
            out_path = os.path.join(
                task_dir, f"videocoattention_{task}_{split_name}.parquet"
            )
            ds.to_parquet(out_path)
            print(f"✅ {out_path} ({len(ds)} examples)")

    print("\n✨ Done!")
