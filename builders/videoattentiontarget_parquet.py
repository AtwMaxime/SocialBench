import os
import json
import random
import subprocess
from PIL import Image as PILImage
from datasets import Dataset, Features, Sequence, Value, Image as HFImage

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAT_DIR    = os.path.join(ROOT_DIR, "dataset", "videoattentiontarget")
ANN_DIR    = os.path.join(VAT_DIR, "annotations")
IMG_DIR    = os.path.join(VAT_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "videoattentiontarget")
CACHE_DIR  = os.path.join(OUTPUT_DIR, ".video_cache")

ORIG_FPS        = 30
TARGET_FPS      = 16
SAMPLES_PER_SEQ = 5
RANDOM_SEED     = 42

# Spatial diversity filter for frame variant:
# minimum Euclidean distance (in [0,1000] space) between any two selected gaze points.
# ~10% of the frame width — allows ~100 non-overlapping points across the full image.
MIN_GAZE_DIST   = 25

FEATURES_VIDEO = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "videos":   Sequence(Value("binary")),
})

FEATURES_FRAME = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "images":   Sequence(HFImage(decode=True)),
})

SYSTEM_VIDEO = (
    "You are an expert in gaze target estimation in video. "
    "Given a video clip and the bounding box of a person's head, predict where they "
    "are looking at the specified timestamp. "
    "Coordinates are normalized to [0, 1000]. "
    'If the gaze target is within the frame provide: {"gaze_point": [x, y], "label": "gaze target"}. '
    'If the target is outside the frame provide: {"out_of_frame": true}.'
)

SYSTEM_FRAME = (
    "You are an expert in gaze target estimation. "
    "Given an image and the bounding box of a person's head, predict where they are looking. "
    "Coordinates are normalized to [0, 1000]. "
    'If the gaze target is within the frame provide: {"gaze_point": [x, y], "label": "gaze target"}. '
    'If the target is outside the frame provide: {"out_of_frame": true}.'
)


# ==========================================
# HELPERS
# ==========================================

def parse_sequence(seq_path):
    entries = []
    with open(seq_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            entries.append({
                'frame': parts[0],
                'head':  (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])),
                'gaze':  (int(parts[5]), int(parts[6])),
            })
    return entries


def get_image_size(show, clip_id):
    frames_dir = os.path.join(IMG_DIR, show, clip_id)
    first = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))[0]
    with PILImage.open(os.path.join(frames_dir, first)) as img:
        return img.size  # (W, H)


def norm(v, size):
    return max(0, min(1000, round(v / size * 1000)))


def norm_bbox(x1, y1, x2, y2, w, h):
    return [norm(x1, w), norm(y1, h), norm(x2, w), norm(y2, h)]


def make_answer(gaze, w, h):
    gx, gy = gaze
    if gx == -1 and gy == -1:
        return json.dumps({"out_of_frame": True})
    return json.dumps({"gaze_point": [norm(gx, w), norm(gy, h)], "label": "gaze target"})


def reconstruct_video(show, clip_id):
    """Reconstruct clip from JPEG frames at TARGET_FPS. Caches to disk."""
    safe_key  = f"{show}_{clip_id}".replace(" ", "_").replace("'", "")
    cache_path = os.path.join(CACHE_DIR, f"{safe_key}_{TARGET_FPS}fps.mp4")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()

    frames_dir = os.path.join(IMG_DIR, show, clip_id)
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith('.jpg'))
    if not frames:
        return None

    start_number = int(frames[0].replace(".jpg", ""))

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(ORIG_FPS),
        "-start_number", str(start_number),
        "-f", "image2",
        "-i", os.path.join(frames_dir, "%08d.jpg"),
        "-vf", f"fps={TARGET_FPS}",
        "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast",
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
        print(f"  ⚠️  ffmpeg failed for {show}/{clip_id}")
    except Exception as e:
        print(f"  ⚠️  ffmpeg error for {show}/{clip_id}: {e}")
    return None


def iter_sequences(split):
    """Yield (show, clip_id, entries) for every sequence file in the split."""
    split_dir = os.path.join(ANN_DIR, split)
    for show in sorted(os.listdir(split_dir)):
        show_dir = os.path.join(split_dir, show)
        if not os.path.isdir(show_dir):
            continue
        for clip_id in sorted(os.listdir(show_dir)):
            clip_dir = os.path.join(show_dir, clip_id)
            if not os.path.isdir(clip_dir):
                continue
            for seq_name in sorted(os.listdir(clip_dir)):
                seq_path = os.path.join(clip_dir, seq_name)
                entries = parse_sequence(seq_path)
                if entries:
                    yield show, clip_id, entries


# ==========================================
# VIDEO GENERATOR — 5 random frames per sequence
# ==========================================

def make_video_generator(split):
    rng = random.Random(RANDOM_SEED)

    def generator():
        skipped = 0
        for show, clip_id, entries in iter_sequences(split):
            try:
                w, h = get_image_size(show, clip_id)
            except Exception:
                skipped += len(entries)
                continue

            video_bytes = reconstruct_video(show, clip_id)
            if video_bytes is None:
                skipped += SAMPLES_PER_SEQ
                continue

            indices = rng.sample(range(len(entries)), min(SAMPLES_PER_SEQ, len(entries)))

            for idx in indices:
                entry = entries[idx]
                t  = idx / ORIG_FPS
                hb = norm_bbox(*entry['head'], w, h)

                user_content = (
                    f"<video>\n"
                    f"Person head bounding box: [{hb[0]}, {hb[1]}, {hb[2]}, {hb[3]}]\n"
                    f"At t={t:.2f}s, where is this person looking?"
                )

                yield {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_VIDEO},
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": make_answer(entry['gaze'], w, h)},
                    ],
                    "videos": [video_bytes],
                }

        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


# ==========================================
# FRAME GENERATOR — spatially diverse frames
# ==========================================

def _gaze_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _too_close(gaze_norm, selected):
    return any(_gaze_dist(gaze_norm, s) < MIN_GAZE_DIST for s in selected)


def make_frame_generator(split):
    def generator():
        skipped = 0
        filtered = 0
        for show, clip_id, entries in iter_sequences(split):
            try:
                w, h = get_image_size(show, clip_id)
            except Exception:
                skipped += len(entries)
                continue

            selected_gaze = []   # normalized gaze points accepted so far for this sequence
            out_of_frame_seen = False

            for entry in entries:
                gx, gy = entry['gaze']
                is_oof = (gx == -1 and gy == -1)

                # Out-of-frame: allow at most one per sequence
                if is_oof:
                    if out_of_frame_seen:
                        filtered += 1
                        continue
                    out_of_frame_seen = True
                else:
                    # Normalize gaze to [0,1000] for distance check
                    gaze_norm = (norm(gx, w), norm(gy, h))
                    if _too_close(gaze_norm, selected_gaze):
                        filtered += 1
                        continue
                    selected_gaze.append(gaze_norm)

                frame_path = os.path.join(IMG_DIR, show, clip_id, entry['frame'])
                if not os.path.exists(frame_path):
                    skipped += 1
                    continue

                with open(frame_path, "rb") as f:
                    img_bytes = f.read()

                hb = norm_bbox(*entry['head'], w, h)

                user_content = (
                    f"<image>\n"
                    f"Person head bounding box: [{hb[0]}, {hb[1]}, {hb[2]}, {hb[3]}]\n"
                    f"Where is this person looking?"
                )

                yield {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_FRAME},
                        {"role": "user",      "content": user_content},
                        {"role": "assistant", "content": make_answer(entry['gaze'], w, h)},
                    ],
                    "images": [{"bytes": img_bytes, "path": None}],
                }

        print(f"  📊 Filtered out {filtered:,} near-duplicate frames  (MIN_GAZE_DIST={MIN_GAZE_DIST})")
        if skipped:
            print(f"  ⚠️  Total skipped: {skipped}")

    return generator


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_dir = os.path.join(OUTPUT_DIR, "video")
    frame_dir = os.path.join(OUTPUT_DIR, "frame")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)

    for split in ["train", "test"]:
        print(f"\n🚀 video/{split}...")
        ds = Dataset.from_generator(make_video_generator(split), features=FEATURES_VIDEO)
        path = os.path.join(video_dir, f"vat_video_{split}.parquet")
        ds.to_parquet(path)
        print(f"✅ {path} ({len(ds)} examples)")

        print(f"\n🚀 frame/{split}...")
        ds = Dataset.from_generator(make_frame_generator(split), features=FEATURES_FRAME)
        path = os.path.join(frame_dir, f"vat_frame_{split}.parquet")
        ds.to_parquet(path)
        print(f"✅ {path} ({len(ds)} examples)")

    print("\n✨ Done!")
