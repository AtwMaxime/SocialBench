import hashlib
import io
import os
from typing import Dict, Any

import soundfile as sf
from swift.llm.dataset import DatasetMeta, SubsetDataset, register_dataset, MessagesPreprocessor

AUDIO_CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "/tmp"), "swift_decoded_audios")
VIDEO_CACHE_DIR = os.path.join(os.environ.get("SCRATCH", "/tmp"), "swift_decoded_videos")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)
os.makedirs(VIDEO_CACHE_DIR, exist_ok=True)

# La chaîne EXACTE demandée par MS-Swift pour le loss_scale
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"

class MeldThinkingPreprocessor(MessagesPreprocessor):
    """MELD audio-only: audio bytes → temp WAV, think block."""
    def __call__(self, dataset, **kwargs):
        return _cast_to_string_columns(super().__call__(dataset, **kwargs), ["videos", "audios"])

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Traitement Audio Binaire — doit être fait AVANT super() qui déclenche template.encode()
        processed_audios = []
        if "audios" in row and row["audios"]:
            for audio_data in row["audios"]:
                audio_bytes = audio_data['bytes'] if isinstance(audio_data, dict) else audio_data
                file_hash = hashlib.md5(audio_bytes).hexdigest()
                wav_path = os.path.join(AUDIO_CACHE_DIR, f"audio_{file_hash}.wav")
                if not os.path.exists(wav_path):
                    data, sr = sf.read(io.BytesIO(audio_bytes))
                    sf.write(wav_path, data, sr)
                processed_audios.append(wav_path)
        row["audios"] = processed_audios

        # 2. Injection du bloc Think — doit être fait AVANT super() qui encode les messages
        for msg in row["messages"]:
            if msg["role"] == "assistant":
                if not msg["content"].startswith("<think>"):
                    msg["content"] = EMPTY_THINK_BLOCK + msg["content"]

        return super().preprocess(row)

def _cast_to_string_columns(dataset, columns):
    """Cast binary-typed media columns to string so path strings are not returned as bytes."""
    from datasets import Sequence, Value
    for col in columns:
        if col in dataset.features:
            dataset = dataset.cast_column(col, Sequence(Value("string")))
    return dataset


def _audio_bytes_to_path(audio_data):
    """Write raw audio bytes to a temp WAV file and return its path."""
    audio_bytes = audio_data['bytes'] if isinstance(audio_data, dict) else audio_data
    file_hash = hashlib.md5(audio_bytes).hexdigest()
    wav_path = os.path.join(AUDIO_CACHE_DIR, f"audio_{file_hash}.wav")
    if not os.path.exists(wav_path):
        data, sr = sf.read(io.BytesIO(audio_bytes))
        sf.write(wav_path, data, sr)
    return wav_path


def _video_bytes_to_path(video_data):
    """Write raw video bytes to a temp MP4 file and return its path."""
    video_bytes = video_data['bytes'] if isinstance(video_data, dict) else video_data
    file_hash = hashlib.md5(video_bytes).hexdigest()
    mp4_path = os.path.join(VIDEO_CACHE_DIR, f"video_{file_hash}.mp4")
    if not os.path.exists(mp4_path):
        with open(mp4_path, 'wb') as f:
            f.write(video_bytes)
    return mp4_path

class MeldVideoAudioPreprocessor(MessagesPreprocessor):
    """MELD video+audio: video bytes → temp MP4, audio bytes → temp WAV, think block."""
    def __call__(self, dataset, **kwargs):
        return _cast_to_string_columns(super().__call__(dataset, **kwargs), ["videos", "audios"])

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["videos"] = [_video_bytes_to_path(v) for v in row.get("videos") or []]
        row["audios"] = [_audio_bytes_to_path(a) for a in row.get("audios") or []]
        for msg in row["messages"]:
            if msg["role"] == "assistant" and not msg["content"].startswith("<think>"):
                msg["content"] = EMPTY_THINK_BLOCK + msg["content"]
        return super().preprocess(row)

class MeldVideoTranscriptPreprocessor(MessagesPreprocessor):
    """MELD video+transcript: video bytes → temp MP4, think block (no audio)."""
    def __call__(self, dataset, **kwargs):
        return _cast_to_string_columns(super().__call__(dataset, **kwargs), ["videos", "audios"])

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["videos"] = [_video_bytes_to_path(v) for v in row.get("videos") or []]
        for msg in row["messages"]:
            if msg["role"] == "assistant" and not msg["content"].startswith("<think>"):
                msg["content"] = EMPTY_THINK_BLOCK + msg["content"]
        return super().preprocess(row)

class GazeThinkingPreprocessor(MessagesPreprocessor):
    """Préprocesseur pour GazeFollow : Injecte juste le bloc de pensée vide."""
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        for msg in row["messages"]:
            if msg["role"] == "assistant":
                if not msg["content"].startswith("<think>"):
                    msg["content"] = EMPTY_THINK_BLOCK + msg["content"]
        return super().preprocess(row)

# ==========================================
# REGISTRE DES DATASETS
# ==========================================

register_dataset(
    DatasetMeta(
        dataset_name='meld-omni',
        hf_dataset_id='AtwMaxime/meld_swift',
        subsets=[
            SubsetDataset(name='video-audio',      subset='video_audio',      preprocess_func=MeldVideoAudioPreprocessor()),
            SubsetDataset(name='audio-only',       subset='audio_only',       preprocess_func=MeldThinkingPreprocessor()),
            SubsetDataset(name='video-transcript', subset='video_transcript', preprocess_func=MeldVideoTranscriptPreprocessor()),
        ],
        split=['train', 'validation', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='mmew-omni',
        hf_dataset_id='AtwMaxime/mmew_swift',
        subsets=[
            SubsetDataset(name='apex-au',           subset='apex_au',           preprocess_func=GazeThinkingPreprocessor()),
            SubsetDataset(name='apex-emotion',      subset='apex_emotion',      preprocess_func=GazeThinkingPreprocessor()),
            SubsetDataset(name='clip-emotion',      subset='clip_emotion',      preprocess_func=MeldVideoTranscriptPreprocessor()),
            SubsetDataset(name='clip-emotion-think', subset='clip_emotion_think', preprocess_func=MeldVideoTranscriptPreprocessor()),
        ],
        split=['train', 'val'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='gazefollow-vlm',
        hf_dataset_id='AtwMaxime/gazefollow_swift',
        preprocess_func=GazeThinkingPreprocessor()
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='pisc-vlm',
        hf_dataset_id='AtwMaxime/pisc_swift',
        preprocess_func=GazeThinkingPreprocessor()
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='vocalsound-omni',
        hf_dataset_id='AtwMaxime/vocalsound_swift',
        preprocess_func=MeldThinkingPreprocessor()
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='proxemics-omni',
        hf_dataset_id='AtwMaxime/proxemics_swift',
        subsets=[
            SubsetDataset(
                name='skeleton',
                subset='skeleton',
                preprocess_func=GazeThinkingPreprocessor(),
            ),
            SubsetDataset(
                name='no-skeleton',
                subset='no_skeleton',
                preprocess_func=GazeThinkingPreprocessor(),
            ),
        ],
        split=['train', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='mustard-omni',
        hf_dataset_id='AtwMaxime/mustard_swift',
        subsets=[
            SubsetDataset(name='video-no-context', subset='video_no_context', preprocess_func=MeldVideoTranscriptPreprocessor()),
            SubsetDataset(name='video-context',    subset='video_context',    preprocess_func=MeldVideoTranscriptPreprocessor()),
        ],
        split=['train', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='vat-omni',
        hf_dataset_id='AtwMaxime/vat_swift',
        subsets=[
            SubsetDataset(
                name='video',
                subset='video',
                preprocess_func=MeldVideoTranscriptPreprocessor(),
            ),
            SubsetDataset(
                name='frame',
                subset='frame',
                preprocess_func=GazeThinkingPreprocessor(),
            ),
        ],
        split=['train', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='rldd-video',
        hf_dataset_id='AtwMaxime/rldd_swift',
        preprocess_func=MeldVideoTranscriptPreprocessor()
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='urfunny-omni',
        hf_dataset_id='AtwMaxime/urfunny_swift',
        subsets=[
            SubsetDataset(name='video-audio',   subset='video_audio',   preprocess_func=MeldVideoTranscriptPreprocessor()),
            SubsetDataset(name='video-context', subset='video_context', preprocess_func=MeldVideoTranscriptPreprocessor()),
        ],
        split=['train', 'dev', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='emotic-vlm',
        hf_dataset_id='AtwMaxime/emotic_swift',
        subsets=[
            SubsetDataset(
                name='discrete',
                subset='discrete',
                preprocess_func=GazeThinkingPreprocessor(),
            ),
            SubsetDataset(
                name='vad',
                subset='vad',
                preprocess_func=GazeThinkingPreprocessor(),
            ),
        ],
        split=['train', 'val', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='affwild2-omni',
        hf_dataset_id='AtwMaxime/affwild2_swift',
        subsets=[
            SubsetDataset(name='expr', subset='expr', preprocess_func=MeldVideoTranscriptPreprocessor()),
            SubsetDataset(name='va',   subset='va',   preprocess_func=MeldVideoTranscriptPreprocessor()),
            SubsetDataset(name='au',   subset='au',   preprocess_func=MeldVideoTranscriptPreprocessor()),
        ],
        split=['train', 'val'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='videocoattention-vlm',
        hf_dataset_id='AtwMaxime/videocoattention_swift',
        subsets=[
            SubsetDataset(name='localization', subset='localization', preprocess_func=GazeThinkingPreprocessor()),
            SubsetDataset(name='detection',    subset='detection',    preprocess_func=GazeThinkingPreprocessor()),
        ],
        split=['train', 'val', 'test'],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name='meview-omni',
        hf_dataset_id='AtwMaxime/meview_swift',
        subsets=[
            SubsetDataset(
                name='recognition',
                subset='recognition',
                preprocess_func=MeldVideoTranscriptPreprocessor(),
            ),
        ],
        split=['test'],
    )
)


class VoxConversePreprocessor(MessagesPreprocessor):
    """VoxConverse: audio bytes → temp WAV, strip stitching metadata columns."""

    def __call__(self, dataset, **kwargs):
        return _cast_to_string_columns(super().__call__(dataset, **kwargs), ["videos", "audios"])

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row["audios"] = [_audio_bytes_to_path(a) for a in row.get("audios") or []]
        # Strip metadata columns not expected by MS-Swift
        row.pop("clip_id", None)
        row.pop("win_start", None)
        for msg in row["messages"]:
            if msg["role"] == "assistant" and not msg["content"].startswith("<think>"):
                msg["content"] = EMPTY_THINK_BLOCK + msg["content"]
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        dataset_name="voxconverse-omni",
        hf_dataset_id="AtwMaxime/voxconverse_swift",
        subsets=[
            SubsetDataset(
                name="diarization",
                subset="diarization",
                preprocess_func=VoxConversePreprocessor(),
            ),
            SubsetDataset(
                name="speaker-count",
                subset="speaker_count",
                preprocess_func=VoxConversePreprocessor(),
            ),
        ],
        split=["train", "test"],
    )
)

register_dataset(
    DatasetMeta(
        dataset_name="msppodcast-omni",
        hf_dataset_id="AtwMaxime/msppodcast_swift",
        subsets=[
            SubsetDataset(
                name="labels",
                subset="labels",
                preprocess_func=MeldThinkingPreprocessor(),
            ),
            SubsetDataset(
                name="vad",
                subset="vad",
                preprocess_func=MeldThinkingPreprocessor(),
            ),
        ],
        split=["train", "validation", "test"],
    )
)