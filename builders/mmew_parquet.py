import hashlib
import json
import os
import subprocess
import tempfile

import openpyxl
from datasets import Dataset, Features, Image, Sequence, Value

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MMEW_DIR = os.path.join(ROOT_DIR, "dataset", "MMEW", "MMEW_Final")
MACRO_DIR = os.path.join(MMEW_DIR, "Macro_Expression")
MICRO_DIR = os.path.join(MMEW_DIR, "Micro_Expression")
EXCEL_PATH = os.path.join(ROOT_DIR, "dataset", "MMEW", "MMEW_Micro_Exp (20).xlsx")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "mmew")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".video_cache")

N_FRAMES = 16

EMOTION_LABELS = [
    "happiness",
    "surprise",
    "disgust",
    "fear",
    "sadness",
    "anger",
    "others",
]
MACRO_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Subject-based split: S01–S24 → train, S25–S30 → val
TRAIN_SUBJECTS = set(range(1, 25))
VAL_SUBJECTS = set(range(25, 31))

FEATURES_IMAGE = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images": Sequence(Image(decode=True)),
    }
)

FEATURES_VIDEO = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "videos": Sequence(Value("binary")),
        "audios": Sequence(Value("binary")),
    }
)

SYSTEM_AU = (
    "You are an expert in facial action unit analysis. "
    "Given an apex frame of a micro-expression, identify the active facial action units. "
    'Provide your answer as a valid JSON object: {"action_units": ["AU6", "AU12"]} '
    'or {"action_units": []} if none are active.'
)

SYSTEM_APEX_EMOTION = (
    "You are an expert in facial expression recognition. "
    f"Given a face image, classify the emotion into one of: {json.dumps(EMOTION_LABELS)}. "
    'Provide your answer as a valid JSON object: {"emotion": "happiness"}.'
)

SYSTEM_CLIP_EMOTION = (
    "You are an expert in micro-expression recognition. "
    "Given a video clip of a face, classify the micro-expression emotion into one of: "
    f"{json.dumps(EMOTION_LABELS)}. "
    'Provide your answer as a valid JSON object: {"emotion": "happiness"}.'
)

SYSTEM_CLIP_THINK = (
    "You are an expert in micro-expression recognition. "
    "Given a video clip of a face, reason step by step inside <think> tags: spot the apex "
    "frame, identify the active action units, and deduce the emotion. Then output the final "
    f"answer. The emotion must be one of: {json.dumps(EMOTION_LABELS)}. "
    "Use the format:\n"
    "<think>\n"
    "1. Apex Spotting: The movement begins from a neutral state (Onset), reaches its maximum "
    "muscular intensity during the sequence (Apex), and then fades (Offset).\n"
    "2. Action Units: At the peak intensity (Apex), the activated Action Units are <AUs>.\n"
    "3. Deduction: The dynamic activation of these specific Action Units is a physical "
    "signature characteristic of <emotion>.\n"
    '</think>\n{"emotion": "<emotion>"}'
)


# ==========================================
# HELPERS
# ==========================================


def read_excel():
    """Load MMEW micro-expression annotations from Excel."""
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb.active
    rows = []
    for row in ws.iter_rows(values_only=True):
        if row[0] == "Subject":
            continue
        subject, filename, onset, apex, offset, au_raw, emotion, _ = row
        if subject is None or emotion is None:
            continue
        rows.append(
            {
                "subject": int(subject),
                "filename": str(filename).strip(),
                "onset": int(onset),
                "apex": int(apex),
                "offset": int(offset),
                "au_raw": au_raw,
                "emotion": str(emotion).strip(),
            }
        )
    return rows


def parse_aus(au_raw):
    """Parse AU string like '6+12' or integer 4 into ['AU6', 'AU12']."""
    if au_raw is None:
        return []
    s = str(au_raw).strip()
    if not s:
        return []
    result = []
    for part in s.split("+"):
        part = part.strip()
        if part:
            try:
                result.append(f"AU{int(float(part))}")
            except ValueError:
                pass
    return result


def au_think_str(au_raw):
    """Format AU raw value for <think> tag: '6+12' → 'AU6+12', None → 'none'."""
    if au_raw is None:
        return "none"
    s = str(au_raw).strip()
    return f"AU{s}" if s else "none"


def select_frames(onset, apex, offset, n=N_FRAMES):
    """
    Select up to n frame indices from [onset, offset] (1-indexed),
    always including the apex frame. Returns a sorted list.
    """
    total = offset - onset + 1
    if total <= n:
        return list(range(onset, offset + 1))
    # Uniformly sample n frames across the range
    indices = [onset + round(k * (total - 1) / (n - 1)) for k in range(n)]
    # Guarantee apex is included — swap the closest sampled frame if needed
    if apex not in indices:
        closest = min(range(len(indices)), key=lambda i: abs(indices[i] - apex))
        indices[closest] = apex
        indices.sort()
    return indices


def build_clip(seq_name, frame_paths):
    """
    Build a 16fps H.264 MP4 from a list of JPEG frame paths.
    Result is cached to disk by a hash of the frame list.
    """
    key = hashlib.md5("".join(frame_paths).encode()).hexdigest()[:12]
    cache_path = os.path.join(CACHE_DIR, f"{seq_name}_{key}.mp4")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        list_path = os.path.join(tmpdir, "frames.txt")
        with open(list_path, "w") as f:
            for fp in frame_paths:
                f.write(f"file '{fp}'\n")
                f.write(f"duration {1 / N_FRAMES}\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-vf",
            f"fps={N_FRAMES},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "ultrafast",
            "-an",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:1",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(result.stdout)
                return result.stdout
            print(f"  ⚠️  ffmpeg failed for {seq_name}")
        except Exception as e:
            print(f"  ⚠️  ffmpeg error for {seq_name}: {e}")
    return None


# ==========================================
# GENERATORS
# ==========================================


def gen_apex_au(rows):
    """Apex frame image → action unit prediction."""

    def generator():
        skipped = 0
        for r in rows:
            seq_dir = os.path.join(MICRO_DIR, r["emotion"], r["filename"])
            apex_path = os.path.join(seq_dir, f"{r['apex']}.jpg")
            if not os.path.exists(apex_path):
                skipped += 1
                continue
            with open(apex_path, "rb") as f:
                img_bytes = f.read()
            aus = parse_aus(r["au_raw"])
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_AU},
                    {
                        "role": "user",
                        "content": "<image>\nWhich facial action units are active in this apex frame?",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"action_units": aus}),
                    },
                ],
                "images": [{"bytes": img_bytes, "path": None}],
            }
        if skipped:
            print(f"  ⚠️  Skipped (missing apex frame): {skipped}")

    return generator


def gen_apex_emotion(rows, subject_ids):
    """
    Apex frame images (micro) + still face images (macro) → emotion label.
    Micro sequences come from `rows`; macro images from subjects in `subject_ids`.
    """

    def generator():
        # --- Micro: apex frame from each annotated sequence ---
        skipped = 0
        for r in rows:
            seq_dir = os.path.join(MICRO_DIR, r["emotion"], r["filename"])
            apex_path = os.path.join(seq_dir, f"{r['apex']}.jpg")
            if not os.path.exists(apex_path):
                skipped += 1
                continue
            with open(apex_path, "rb") as f:
                img_bytes = f.read()
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_APEX_EMOTION},
                    {
                        "role": "user",
                        "content": "<image>\nWhat emotion is expressed in this face image?",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"emotion": r["emotion"]}),
                    },
                ],
                "images": [{"bytes": img_bytes, "path": None}],
            }
        if skipped:
            print(f"  ⚠️  Skipped micro (missing apex): {skipped}")

        # --- Macro: all still images for the given subjects ---
        for subj_id in sorted(subject_ids):
            subj_dir = os.path.join(MACRO_DIR, f"S{subj_id:02d}")
            if not os.path.isdir(subj_dir):
                continue
            for emotion in MACRO_EMOTIONS:
                emot_dir = os.path.join(subj_dir, emotion)
                if not os.path.isdir(emot_dir):
                    continue
                for fname in sorted(os.listdir(emot_dir)):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    with open(os.path.join(emot_dir, fname), "rb") as f:
                        img_bytes = f.read()
                    yield {
                        "messages": [
                            {"role": "system", "content": SYSTEM_APEX_EMOTION},
                            {
                                "role": "user",
                                "content": "<image>\nWhat emotion is expressed in this face image?",
                            },
                            {
                                "role": "assistant",
                                "content": json.dumps({"emotion": emotion}),
                            },
                        ],
                        "images": [{"bytes": img_bytes, "path": None}],
                    }

    return generator


def gen_clip_emotion(rows, think=False):
    """16-frame video clip (apex included) → emotion, optionally with AU <think> reasoning."""

    def generator():
        skipped = 0
        for r in rows:
            seq_dir = os.path.join(MICRO_DIR, r["emotion"], r["filename"])
            if not os.path.isdir(seq_dir):
                skipped += 1
                continue

            indices = select_frames(r["onset"], r["apex"], r["offset"])
            frame_paths = [
                os.path.join(seq_dir, f"{i}.jpg")
                for i in indices
                if os.path.exists(os.path.join(seq_dir, f"{i}.jpg"))
            ]
            if len(frame_paths) < 4:
                skipped += 1
                continue

            video_bytes = build_clip(r["filename"], frame_paths)
            if video_bytes is None:
                skipped += 1
                continue

            if think:
                au_str = au_think_str(r["au_raw"])
                emotion_label = r["emotion"]
                answer = f"""<think>
1. Apex Spotting: The movement begins from a neutral state (Onset), reaches its maximum muscular intensity during the sequence (Apex), and then fades (Offset).
2. Action Units: At the peak intensity (Apex), the activated Action Units are {au_str}.
3. Deduction: The dynamic activation of these specific Action Units is a physical signature characteristic of {emotion_label}.
</think>
{json.dumps({"emotion": emotion_label})}"""
                system = SYSTEM_CLIP_THINK
            else:
                answer = json.dumps({"emotion": r["emotion"]})
                system = SYSTEM_CLIP_EMOTION

            yield {
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": "<video>\nWhat micro-expression emotion is shown in this clip?",
                    },
                    {"role": "assistant", "content": answer},
                ],
                "videos": [video_bytes],
                "audios": [],
            }
        if skipped:
            print(f"  ⚠️  Skipped: {skipped}")

    return generator


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("📂 Loading annotations...")
    all_rows = read_excel()
    train_rows = [r for r in all_rows if r["subject"] in TRAIN_SUBJECTS]
    val_rows = [r for r in all_rows if r["subject"] in VAL_SUBJECTS]
    print(f"  Total: {len(all_rows)}  train: {len(train_rows)}  val: {len(val_rows)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = [
        (
            "apex_au",
            FEATURES_IMAGE,
            [
                ("train", gen_apex_au(train_rows)),
                ("val", gen_apex_au(val_rows)),
            ],
        ),
        (
            "apex_emotion",
            FEATURES_IMAGE,
            [
                ("train", gen_apex_emotion(train_rows, TRAIN_SUBJECTS)),
                ("val", gen_apex_emotion(val_rows, VAL_SUBJECTS)),
            ],
        ),
        (
            "clip_emotion",
            FEATURES_VIDEO,
            [
                ("train", gen_clip_emotion(train_rows, think=False)),
                ("val", gen_clip_emotion(val_rows, think=False)),
            ],
        ),
        (
            "clip_emotion_think",
            FEATURES_VIDEO,
            [
                ("train", gen_clip_emotion(train_rows, think=True)),
                ("val", gen_clip_emotion(val_rows, think=True)),
            ],
        ),
    ]

    for task_name, features, splits in tasks:
        task_dir = os.path.join(OUTPUT_DIR, task_name)
        os.makedirs(task_dir, exist_ok=True)
        for split_name, gen_fn in splits:
            print(f"\n🚀 {task_name}/{split_name}...")
            ds = Dataset.from_generator(gen_fn, features=features)
            out_path = os.path.join(task_dir, f"mmew_{task_name}_{split_name}.parquet")
            ds.to_parquet(out_path)
            print(f"✅ {out_path} ({len(ds)} examples)")

    print("\n✨ Done!")
