import os
import io
import re
import json
import random
import tempfile
import subprocess
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "videoattentiontarget")


def read_random_rows(parquet_file, n=1):
    """Read n consecutive rows from a random row group. No disk cache."""
    pf      = pq.ParquetFile(parquet_file)
    num_rgs = pf.metadata.num_row_groups
    for _ in range(20):
        rg_idx = random.randint(0, num_rgs - 1)
        table  = pf.read_row_group(rg_idx)
        if table.num_rows >= n:
            start = random.randint(0, table.num_rows - n)
            return [
                {col: table[col][start + i].as_py() for col in table.schema.names}
                for i in range(n)
            ]
    return None


def get_msg(row, role):
    return next(m for m in row["messages"] if m["role"] == role)


def denorm(v, size):
    return round(v / 1000 * size)


def extract_frame_at(video_bytes, t_sec, tmpdir):
    """Extract the nearest frame to t_sec. Returns PIL Image or None."""
    vid_path   = os.path.join(tmpdir, "video.mp4")
    frame_path = os.path.join(tmpdir, f"frame_{t_sec:.3f}.jpg")
    with open(vid_path, "wb") as f:
        f.write(video_bytes)
    cmd = ["ffmpeg", "-y", "-i", vid_path, "-ss", str(t_sec),
           "-vframes", "1", "-q:v", "2", frame_path]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(frame_path):
        with open(frame_path, "rb") as f:
            return Image.open(io.BytesIO(f.read())).copy()
    return None


def parse_sample(user_content, asst_content):
    m    = re.search(r"bounding box: \[(\d+), (\d+), (\d+), (\d+)\]", user_content)
    bbox = [int(v) for v in m.groups()] if m else None
    t_m  = re.search(r"t=(\d+\.\d+)s", user_content)
    t    = float(t_m.group(1)) if t_m else None
    return bbox, t, json.loads(asst_content)


def annotate_frame(img, bbox_norm, answer, label):
    draw = ImageDraw.Draw(img)
    W, H = img.size
    x1 = denorm(bbox_norm[0], W)
    y1 = denorm(bbox_norm[1], H)
    x2 = denorm(bbox_norm[2], W)
    y2 = denorm(bbox_norm[3], H)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if "gaze_point" in answer:
        gx = denorm(answer["gaze_point"][0], W)
        gy = denorm(answer["gaze_point"][1], H)
        draw.line([cx, cy, gx, gy], fill="white", width=2)
        r = 8
        draw.ellipse([gx - r, gy - r, gx + r, gy + r], fill="lime", outline="white", width=2)
        draw.text((gx + r + 4, gy - 8), label, fill="lime")
    else:
        draw.text((x1, max(0, y1 - 18)), f"out of frame | {label}", fill="orange")
    return img


# ============================================================
# 1. Video variant — annotated collage of 5 frames + play video
# ============================================================
print("\n" + "=" * 60)
print("📂 vat_video")
parquet_file = os.path.join(PARQUETS_DIR, "video", "vat_video_train.parquet")
pf_meta = pq.ParquetFile(parquet_file).metadata
print(f"✅ {pf_meta.num_rows} examples  ({pf_meta.num_row_groups} row groups)")

rows = read_random_rows(parquet_file, n=5)
print(f"🎲 Sampled 5 consecutive rows from a random row group")

video_bytes      = rows[0]["videos"][0]
frames_annotated = []

with tempfile.TemporaryDirectory() as tmpdir:
    for i, row in enumerate(rows):
        user_content = get_msg(row, "user")["content"]
        asst_content = get_msg(row, "assistant")["content"]
        bbox, t, answer = parse_sample(user_content, asst_content)
        print(f"  [{i}] t={t:.2f}s  {asst_content}")

        frame = extract_frame_at(video_bytes, t, tmpdir)
        if frame and bbox:
            frames_annotated.append(annotate_frame(frame, bbox, answer, f"t={t:.2f}s"))

if frames_annotated:
    fw, fh  = frames_annotated[0].size
    collage = Image.new("RGB", (fw * len(frames_annotated), fh))
    for i, f in enumerate(frames_annotated):
        collage.paste(f, (i * fw, 0))
    scale   = min(1.0, 1800 / collage.width)
    collage = collage.resize(
        (int(collage.width * scale), int(collage.height * scale)), Image.LANCZOS
    )

print("\n▶️  Playing full video... [close window to continue]")
with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
    tmp.write(video_bytes)
    tmp_path = tmp.name
try:
    subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
finally:
    os.unlink(tmp_path)


# ============================================================
# 2. Frame variant — head bbox + gaze point + vector
# ============================================================
print("\n" + "=" * 60)
print("📂 vat_frame")
parquet_file = os.path.join(PARQUETS_DIR, "frame", "vat_frame_train.parquet")
pf_meta = pq.ParquetFile(parquet_file).metadata
print(f"✅ {pf_meta.num_rows} examples  ({pf_meta.num_row_groups} row groups)")

rows = read_random_rows(parquet_file, n=1)
row  = rows[0]
user_content = get_msg(row, "user")["content"]
asst_content = get_msg(row, "assistant")["content"]
bbox, _, answer = parse_sample(user_content, asst_content)

print(f"\n--- 📝 User ---\n{user_content}")
print(f"\n--- 💬 Assistant ---\n{asst_content}\n")

img_bytes = row["images"][0]["bytes"]
img = Image.open(io.BytesIO(img_bytes)).copy()
img = annotate_frame(img, bbox, answer, "gaze")

