import hashlib
import json
import math
import os
import subprocess
from datasets import Dataset, Features, Sequence, Value


# ==========================================
# FAST FILTER STATE HELPERS
# ==========================================


class VaGrid:
    """
    O(1) amortized spatial hash for VA diversity filter.
    Cells of size `threshold` × `threshold`; only 3×3 neighbor cells are
    checked, guaranteeing all points within Euclidean distance `threshold`
    are found without a full list scan.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.cell_size = threshold
        self.cells = {}  # (ix, iy) -> list[(v, a)]

    def _cell(self, v, a):
        return (
            int((v + 1.0) / self.cell_size),
            int((a + 1.0) / self.cell_size),
        )

    def is_too_close(self, v, a):
        cx, cy = self._cell(v, a)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for sv, sa in self.cells.get((cx + dx, cy + dy), []):
                    if math.sqrt((v - sv) ** 2 + (a - sa) ** 2) < self.threshold:
                        return True
        return False

    def add(self, v, a):
        key = self._cell(v, a)
        self.cells.setdefault(key, []).append((v, a))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(ROOT_DIR, "dataset", "AffWild2", "ABAW Annotations")
VIDEOS_ROOT = os.path.join(ROOT_DIR, "dataset", "AffWild2")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "affwild2")
CACHE_DIR = os.path.join(OUTPUT_DIR, ".video_cache")

TASK_DIRS = {
    "expr": {
        "train": os.path.join(ANN_DIR, "EXPR_Recognition_Challenge", "Train_Set"),
        "val": os.path.join(ANN_DIR, "EXPR_Recognition_Challenge", "Validation_Set"),
    },
    "va": {
        "train": os.path.join(ANN_DIR, "VA_Estimation_Challenge", "Train_Set"),
        "val": os.path.join(ANN_DIR, "VA_Estimation_Challenge", "Validation_Set"),
    },
    "au": {
        "train": os.path.join(ANN_DIR, "AU_Detection_Challenge", "Train_Set"),
        "val": os.path.join(ANN_DIR, "AU_Detection_Challenge", "Validation_Set"),
    },
}

THRESHOLDS = {
    "expr": 5,     # max windows per expression label per video
    "va": 0.05,    # min Euclidean distance in [-1,1]²
    "au": 1,       # min Hamming distance on 12 AUs
}

N_FRAMES = 16
CENTER = N_FRAMES // 2   # index 8 within the window = center frame
STRIDE = 16              # frames to advance between windows
THINK_VAL_BORROW = 20   # train videos to move into expr_think val

EXPR_LABELS = [
    "Neutral", "Anger", "Disgust", "Fear",
    "Happiness", "Sadness", "Surprise", "Other",
]
AU_COLS = [
    "AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
    "AU12", "AU15", "AU23", "AU24", "AU25", "AU26",
]

FEATURES = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "videos": Sequence(Value("binary")),
    "audios": Sequence(Value("binary")),
})

SYSTEM = {
    "expr": (
        "You are an expert in facial expression recognition. "
        "Given a 16-frame video clip, predict the facial expression at the center frame. "
        f"Classify into one of: {json.dumps(EXPR_LABELS)}. "
        'Provide your answer as a valid JSON object: {"label": "Expression"}.'
    ),
    "va": (
        "You are an expert in affective computing. "
        "Given a 16-frame video clip, predict the valence and arousal at the center frame. "
        "Both values are continuous in [-1, 1]. "
        'Provide your answer as a valid JSON object: {"valence": x.xxx, "arousal": x.xxx}.'
    ),
    "au": (
        "You are an expert in facial action unit detection. "
        "Given a 16-frame video clip, predict which action units are active at the center frame. "
        f"Possible action units: {json.dumps(AU_COLS)}. "
        'Provide your answer as a valid JSON object: {"action_units": ["AU1", ...]} '
        'or {"action_units": []} if none are active.'
    ),
    "expr_think": (
        "You are an expert in facial expression recognition and affective computing. "
        "Given a 16-frame video clip, analyze the center frame: first provide the valence/arousal "
        "and active action units inside <think> tags, then classify the facial expression. "
        f"Expression must be one of: {json.dumps(EXPR_LABELS)}. "
        "Use the format:\n"
        "<think>\n"
        '{"valence": x.xxx, "arousal": x.xxx}\n'
        '{"action_units": ["AU1", ...]}\n'
        "</think>\n"
        '{"label": "Expression"}'
    ),
}

USER = {
    "expr": "<video>\nWhat is the facial expression at the center frame of this clip?",
    "va": "<video>\nWhat are the valence and arousal values at the center frame of this clip?",
    "au": "<video>\nWhich action units are active at the center frame of this clip?",
    "expr_think": (
        "<video>\nFor the center frame of this clip, provide the valence/arousal and "
        "active action units, then give the expression label."
    ),
}


# ==========================================
# VIDEO INDEX
# ==========================================


def build_video_index():
    """Scan batch1/batch2/new_vids and return {stem: path}. Prefer .mp4 over .avi."""
    index = {}
    for folder in ["batch1", "batch2", "new_vids"]:
        folder_path = os.path.join(VIDEOS_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in (".mp4", ".avi"):
                if stem not in index or ext.lower() == ".mp4":
                    index[stem] = os.path.join(folder_path, fname)
    return index


# ==========================================
# ANNOTATION LOADERS
# ==========================================


def load_expr(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    result = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            try:
                result.append(int(line))
            except ValueError:
                pass
    return result


def load_va(path):
    result = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            result.append((float(parts[0]), float(parts[1])))
        except ValueError:
            pass
    return result


def load_au(path):
    result = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            result.append([int(x) for x in line.split(",")])
        except ValueError:
            pass
    return result


# ==========================================
# VALIDITY & DIVERSITY FILTER
# ==========================================


def is_valid(task, ann, idx):
    if idx >= len(ann):
        return False
    val = ann[idx]
    if task == "expr":
        return val != -1
    elif task == "va":
        v, a = val
        return -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0
    else:  # au
        return -1 not in val


def make_filter_state(task):
    """Return a per-video filter state object for the given task."""
    if task == "expr":
        return {}  # label (int) -> count
    elif task == "va":
        return VaGrid(THRESHOLDS["va"])
    else:  # au — threshold=1 means exact duplicates only, so a set suffices
        return set()  # set of tuples


def passes_filter(task, val, state):
    if task == "expr":
        return state.get(val, 0) < THRESHOLDS["expr"]
    elif task == "va":
        return not state.is_too_close(val[0], val[1])
    else:  # au
        return tuple(val) not in state


def update_filter_state(task, state, val):
    if task == "expr":
        state[val] = state.get(val, 0) + 1
    elif task == "va":
        state.add(val[0], val[1])
    else:  # au
        state.add(tuple(val))


# ==========================================
# VIDEO CLIP BUILDING
# ==========================================


def build_window_clip(vid_path, start_frame):
    """
    Extract N_FRAMES consecutive frames starting at start_frame using the
    trim filter (efficient — avoids decoding the whole video). Cached.
    """
    h = hashlib.md5(f"{vid_path}_{start_frame}".encode()).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(vid_path))[0]
    cache_path = os.path.join(CACHE_DIR, f"{stem}_{h}.mp4")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    vf = (
        f"trim=start_frame={start_frame}:end_frame={start_frame + N_FRAMES},"
        "setpts=PTS-STARTPTS,"
        f"fps={N_FRAMES}"
    )

    cmd = [
        "ffmpeg", "-y", "-i", vid_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
        "-an",
        "-movflags", "frag_keyframe+empty_moov",
        "-f", "mp4", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and result.stdout:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(result.stdout)
            return result.stdout
        print(f"  ⚠️  ffmpeg failed for {os.path.basename(vid_path)} frame {start_frame}")
    except Exception as e:
        print(f"  ⚠️  ffmpeg error for {os.path.basename(vid_path)}: {e}")
    return None


# ==========================================
# ANSWER FORMATTING
# ==========================================


def make_answer(task, val):
    if task == "expr":
        return json.dumps({"label": EXPR_LABELS[val]})
    elif task == "va":
        v, a = val
        return json.dumps({"valence": round(v, 3), "arousal": round(a, 3)})
    else:  # au
        active = [AU_COLS[i] for i, b in enumerate(val) if b == 1]
        return json.dumps({"action_units": active})


# ==========================================
# GENERATORS
# ==========================================


def make_sliding_generator(split, task, video_index, stride=STRIDE):
    """Sliding window generator for expr / va / au tasks."""
    ann_dir = TASK_DIRS[task][split]

    def generator():
        skipped_no_video = 0
        skipped_ffmpeg = 0

        for ann_file in sorted(f for f in os.listdir(ann_dir) if f.endswith(".txt")):
            stem = os.path.splitext(ann_file)[0]
            vid_path = video_index.get(stem)
            if vid_path is None:
                skipped_no_video += 1
                continue

            ann = load_annotation_by_task(task, os.path.join(ann_dir, ann_file))
            n = len(ann)
            if n < N_FRAMES:
                continue

            filter_state = make_filter_state(task)
            for w in range(0, n - N_FRAMES + 1, stride):
                center = w + CENTER
                if not is_valid(task, ann, center):
                    continue
                val = ann[center]
                if not passes_filter(task, val, filter_state):
                    continue

                video_bytes = build_window_clip(vid_path, w)
                if video_bytes is None:
                    skipped_ffmpeg += 1
                    continue

                update_filter_state(task, filter_state, val)
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM[task]},
                        {"role": "user", "content": USER[task]},
                        {"role": "assistant", "content": make_answer(task, val)},
                    ],
                    "videos": [video_bytes],
                    "audios": [],
                }

        if skipped_no_video:
            print(f"  ⚠️  No video found: {skipped_no_video}")
        if skipped_ffmpeg:
            print(f"  ⚠️  ffmpeg failures: {skipped_ffmpeg}")

    return generator


def get_think_stems():
    """
    Returns (train_stems, val_stems) for expr_think.
    Borrows THINK_VAL_BORROW videos from train to augment the small official val set.
    The borrowed stems are removed from train to prevent leakage.
    """
    def common(split):
        dirs = [TASK_DIRS[t][split] for t in ("expr", "va", "au")]
        sets = [
            {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".txt")}
            for d in dirs
        ]
        return sorted(sets[0] & sets[1] & sets[2])

    official_train = common("train")
    official_val = common("val")

    borrowed = official_train[-THINK_VAL_BORROW:]
    final_train = official_train[:-THINK_VAL_BORROW]
    final_val = official_val + borrowed
    return final_train, final_val


def make_think_generator(split, video_index, stride=STRIDE):
    """
    expr_think task: only for videos annotated in all 3 tasks.
    Center-frame answer includes VA + AU in <think>, EXPR label outside.
    Val set is augmented with THINK_VAL_BORROW videos borrowed from train.
    """
    train_stems, val_stems = get_think_stems()
    common_stems = train_stems if split == "train" else val_stems

    def _ann_dirs(stem):
        """Return (expr_dir, va_dir, au_dir) for a given stem, searching train then val."""
        for s in ("train", "val"):
            expr_path = os.path.join(TASK_DIRS["expr"][s], f"{stem}.txt")
            va_path = os.path.join(TASK_DIRS["va"][s], f"{stem}.txt")
            au_path = os.path.join(TASK_DIRS["au"][s], f"{stem}.txt")
            if os.path.exists(expr_path) and os.path.exists(va_path) and os.path.exists(au_path):
                return expr_path, va_path, au_path
        return None, None, None

    def generator():
        skipped_no_video = 0
        skipped_ffmpeg = 0

        for stem in common_stems:
            vid_path = video_index.get(stem)
            if vid_path is None:
                skipped_no_video += 1
                continue

            expr_path, va_path, au_path = _ann_dirs(stem)
            if expr_path is None:
                continue

            ann_expr = load_expr(expr_path)
            ann_va = load_va(va_path)
            ann_au = load_au(au_path)

            n = min(len(ann_expr), len(ann_va), len(ann_au))
            if n < N_FRAMES:
                continue

            filter_state = make_filter_state("expr")
            for w in range(0, n - N_FRAMES + 1, stride):
                center = w + CENTER
                if not is_valid("expr", ann_expr, center):
                    continue
                if not is_valid("va", ann_va, center):
                    continue
                if not is_valid("au", ann_au, center):
                    continue

                expr_val = ann_expr[center]
                if not passes_filter("expr", expr_val, filter_state):
                    continue

                video_bytes = build_window_clip(vid_path, w)
                if video_bytes is None:
                    skipped_ffmpeg += 1
                    continue

                update_filter_state("expr", filter_state, expr_val)

                v, a = ann_va[center]
                active_aus = [AU_COLS[i] for i, b in enumerate(ann_au[center]) if b == 1]
                think_content = (
                    json.dumps({"valence": round(v, 3), "arousal": round(a, 3)}) + "\n"
                    + json.dumps({"action_units": active_aus})
                )
                answer = (
                    f"<think>\n{think_content}\n</think>\n"
                    + json.dumps({"label": EXPR_LABELS[expr_val]})
                )

                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM["expr_think"]},
                        {"role": "user", "content": USER["expr_think"]},
                        {"role": "assistant", "content": answer},
                    ],
                    "videos": [video_bytes],
                    "audios": [],
                }

        if skipped_no_video:
            print(f"  ⚠️  No video found: {skipped_no_video}")
        if skipped_ffmpeg:
            print(f"  ⚠️  ffmpeg failures: {skipped_ffmpeg}")

    return generator


def load_annotation_by_task(task, path):
    if task == "expr":
        return load_expr(path)
    elif task == "va":
        return load_va(path)
    else:
        return load_au(path)


# ==========================================
# STATS MODE
# ==========================================


def run_stats():
    strides = [4, 8, 16, 32, 64]
    print(f"  Window size: {N_FRAMES} frames  |  Center frame index: {CENTER}\n")

    for task in ["expr", "va", "au"]:
        print(f"\n{'='*58}\nTASK: {task.upper()}\n{'='*58}")
        for split in ["train", "val"]:
            files = sorted(
                f for f in os.listdir(TASK_DIRS[task][split]) if f.endswith(".txt")
            )
            print(f"\n  {split}: {len(files)} sequences")
            print(f"  {'Stride':<8} {'Raw windows':>12} {'After diversity':>16} {'avg/seq':>8}")
            print("  " + "-" * 46)
            for stride in strides:
                raw = 0
                total = 0
                for fname in files:
                    ann = load_annotation_by_task(
                        task, os.path.join(TASK_DIRS[task][split], fname)
                    )
                    n = len(ann)
                    filter_state = make_filter_state(task)
                    for w in range(0, n - N_FRAMES + 1, stride):
                        center = w + CENTER
                        if not is_valid(task, ann, center):
                            continue
                        raw += 1
                        val = ann[center]
                        if passes_filter(task, val, filter_state):
                            update_filter_state(task, filter_state, val)
                            total += 1
                print(f"  {stride:<8} {raw:>12,} {total:>16,} {total/len(files):>8.1f}")

    # expr_think: use adjusted splits (borrowed videos)
    train_stems, val_stems = get_think_stems()
    print(f"\n{'='*58}\nTASK: EXPR_THINK (intersection of all 3, {THINK_VAL_BORROW} train→val)\n{'='*58}")
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        print(f"\n  {split_name}: {len(stems)} videos")
        print(f"  {'Stride':<8} {'After diversity':>16} {'avg/seq':>8}")
        print("  " + "-" * 34)
        for stride in strides:
            total = 0
            for stem in stems:
                for s in ("train", "val"):
                    expr_path = os.path.join(TASK_DIRS["expr"][s], f"{stem}.txt")
                    va_path = os.path.join(TASK_DIRS["va"][s], f"{stem}.txt")
                    au_path = os.path.join(TASK_DIRS["au"][s], f"{stem}.txt")
                    if os.path.exists(expr_path):
                        break
                ann_expr = load_expr(expr_path)
                ann_va = load_va(va_path)
                ann_au = load_au(au_path)
                n = min(len(ann_expr), len(ann_va), len(ann_au))
                filter_state = make_filter_state("expr")
                for w in range(0, n - N_FRAMES + 1, stride):
                    center = w + CENTER
                    if not (
                        is_valid("expr", ann_expr, center)
                        and is_valid("va", ann_va, center)
                        and is_valid("au", ann_au, center)
                    ):
                        continue
                    val = ann_expr[center]
                    if passes_filter("expr", val, filter_state):
                        update_filter_state("expr", filter_state, val)
                        total += 1
            avg = total / len(stems) if stems else 0
            print(f"  {stride:<8} {total:>16,} {avg:>8.1f}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import sys

    if "--stats" in sys.argv:
        print("📊 AffWild2 sliding window simulation...\n")
        run_stats()
        print("\n✨ Done.")
        sys.exit(0)

    print("📂 Building video index...")
    video_index = build_video_index()
    print(f"  Found {len(video_index)} videos")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for task in ["expr", "va", "au"]:
        for split in ["train", "val"]:
            print(f"\n🚀 {task}/{split} (stride={STRIDE})...")
            ds = Dataset.from_generator(
                make_sliding_generator(split, task, video_index, stride=STRIDE),
                features=FEATURES,
            )
            task_dir = os.path.join(OUTPUT_DIR, task)
            os.makedirs(task_dir, exist_ok=True)
            output_path = os.path.join(task_dir, f"affwild2_{task}_{split}.parquet")
            ds.to_parquet(output_path)
            print(f"✅ {output_path} ({len(ds)} examples)")

    for split in ["train", "val"]:
        print(f"\n🚀 expr_think/{split} (stride={STRIDE})...")
        ds = Dataset.from_generator(
            make_think_generator(split, video_index, stride=STRIDE),
            features=FEATURES,
        )
        expr_dir = os.path.join(OUTPUT_DIR, "expr")
        os.makedirs(expr_dir, exist_ok=True)
        output_path = os.path.join(expr_dir, f"affwild2_expr_think_{split}.parquet")
        ds.to_parquet(output_path)
        print(f"✅ {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
