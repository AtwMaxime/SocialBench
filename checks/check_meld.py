import os
import io
import json
import re
import random
import tempfile
import subprocess
import soundfile as sf
from datasets import load_dataset

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR          = os.path.join(ROOT_DIR, "parquets", "meld")
_VIDEO_AUDIO_DIR      = os.path.join(PARQUETS_DIR, "video_audio")
_AUDIO_ONLY_DIR       = os.path.join(PARQUETS_DIR, "audio_only")
_VIDEO_TRANSCRIPT_DIR = os.path.join(PARQUETS_DIR, "video_transcript")


def extract_bbox(user_content):
    """Parse 'Speaker face bounding box: [x1, y1, x2, y2]' from user message."""
    m = re.search(r"Speaker face bounding box: \[(\d+), (\d+), (\d+), (\d+)\]", user_content)
    if m:
        return [int(m.group(i)) for i in range(1, 5)]
    return None


def get_video_dims(vid_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", vid_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return int(s["width"]), int(s["height"])
    return None, None


def drawbox_vf(bbox_norm, w, h):
    """Build ffmpeg drawbox filter string from normalized [0,1000] bbox."""
    if not bbox_norm or not w or not h:
        return None
    x1, y1, x2, y2 = bbox_norm
    px = int(x1 / 1000 * w)
    py = int(y1 / 1000 * h)
    bw = int((x2 - x1) / 1000 * w)
    bh = int((y2 - y1) / 1000 * h)
    return f"drawbox=x={px}:y={py}:w={bw}:h={bh}:color=red@0.8:t=3"


def play_video_audio_with_bbox(video_bytes, audio_bytes, bbox_norm):
    """Mux video + audio, overlay bbox, play with ffplay."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "video.mp4")
        aud_path = os.path.join(tmpdir, "audio.wav")
        out_path = os.path.join(tmpdir, "output.mp4")

        with open(vid_path, "wb") as f:
            f.write(video_bytes)
        with open(aud_path, "wb") as f:
            f.write(audio_bytes)

        w, h = get_video_dims(vid_path)
        vf = drawbox_vf(bbox_norm, w, h)

        cmd = ["ffmpeg", "-y", "-i", vid_path, "-i", aud_path]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-shortest", out_path]

        subprocess.run(cmd, capture_output=True)
        print("▶️  Playing video + audio (bbox in red)... [close window to continue]")
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", out_path])


def play_video_with_bbox(video_bytes, bbox_norm):
    """Overlay bbox on silent video and play with ffplay."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_path = os.path.join(tmpdir, "video.mp4")
        out_path = os.path.join(tmpdir, "output.mp4")

        with open(vid_path, "wb") as f:
            f.write(video_bytes)

        w, h = get_video_dims(vid_path)
        vf = drawbox_vf(bbox_norm, w, h)

        if vf:
            cmd = ["ffmpeg", "-y", "-i", vid_path, "-vf", vf,
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast", out_path]
            subprocess.run(cmd, capture_output=True)
            play_path = out_path
        else:
            play_path = vid_path

        print("▶️  Playing video (bbox in red)... [close window to continue]")
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", play_path])


def play_audio(audio_bytes):
    """Play raw WAV bytes via ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(data) / sr
        channels = 1 if data.ndim == 1 else data.shape[1]
        print(f"   Rate: {sr} Hz  Duration: {duration:.2f}s  Channels: {channels}")
        print("▶️  Playing audio... [close window to continue]")
        subprocess.run(["ffplay", "-autoexit", "-loglevel", "quiet", tmp_path])
    finally:
        os.unlink(tmp_path)


# ============================================================
# 1. video_audio — watch video + audio, bbox overlaid
# ============================================================
print("\n" + "=" * 60)
print("📂 video_audio")
ds = load_dataset("parquet",
                  data_files={"train": os.path.join(_VIDEO_AUDIO_DIR, "meld_video_audio_train.parquet")},
                  split="train")
print(f"✅ {len(ds)} examples")

idx     = random.randint(0, len(ds) - 1)
example = ds[idx]
user_msg = next(m for m in example["messages"] if m["role"] == "user")
asst_msg = next(m for m in example["messages"] if m["role"] == "assistant")
print(f"🎲 Index: {idx}")
print(f"\n--- 📝 User ---\n{user_msg['content']}")
print(f"\n--- 💬 Assistant ---\n{asst_msg['content']}\n")

bbox = extract_bbox(user_msg["content"])
play_video_audio_with_bbox(example["videos"][0], example["audios"][0], bbox)


# ============================================================
# 2. video_transcript — read transcript, watch video with bbox
# ============================================================
print("\n" + "=" * 60)
print("📂 video_transcript")
ds = load_dataset("parquet",
                  data_files={"train": os.path.join(_VIDEO_TRANSCRIPT_DIR, "meld_video_transcript_train.parquet")},
                  split="train")
print(f"✅ {len(ds)} examples")

idx     = random.randint(0, len(ds) - 1)
example = ds[idx]
user_msg = next(m for m in example["messages"] if m["role"] == "user")
asst_msg = next(m for m in example["messages"] if m["role"] == "assistant")
print(f"🎲 Index: {idx}")
print(f"\n--- 📝 User ---\n{user_msg['content']}")
print(f"\n--- 💬 Assistant ---\n{asst_msg['content']}\n")

bbox = extract_bbox(user_msg["content"])
play_video_with_bbox(example["videos"][0], bbox)


# ============================================================
# 3. audio_only — listen to the clip
# ============================================================
print("\n" + "=" * 60)
print("📂 audio_only")
ds = load_dataset("parquet",
                  data_files={"train": os.path.join(_AUDIO_ONLY_DIR, "meld_audio_only_train.parquet")},
                  split="train")
print(f"✅ {len(ds)} examples")

idx     = random.randint(0, len(ds) - 1)
example = ds[idx]
user_msg = next(m for m in example["messages"] if m["role"] == "user")
asst_msg = next(m for m in example["messages"] if m["role"] == "assistant")
print(f"🎲 Index: {idx}")
print(f"\n--- 📝 User ---\n{user_msg['content']}")
print(f"\n--- 💬 Assistant ---\n{asst_msg['content']}\n")

play_audio(example["audios"][0])
