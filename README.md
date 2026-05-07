# SocialBench

**SocialBench** is a multimodal benchmark for social scene understanding in MLLMs. It spans 16 datasets and 30 tasks organized across three cognitive levels: Perception (L1), Understanding (L2), and Theory of Mind (L3). Radar charts cover 28 of these tasks (VoxConverse speaker counting and MMEW AU apex are reported in the results tables but excluded from the radar as they have no published SOTA reference).

This repo contains the data pipeline, MS-Swift plugins, evaluation scripts, and pre-computed results from the paper.

## Table of Contents

- [Benchmark Overview](#benchmark-overview)
- [Quickstart](#quickstart)
- [Prerequisites](#prerequisites)
- [Step 1 — Download Datasets](#step-1--download-datasets)
  - [GazeFollow](#gazefollow)
  - [VideoAttentionTarget (VAT)](#videoattentiontarget-vat)
  - [VideoCoAttention](#videocoattention)
  - [VocalSound](#vocalsound)
  - [Proxemics](#proxemics)
  - [MMEW](#mmew)
  - [AffWild2](#affwild2)
  - [VoxConverse](#voxconverse)
  - [MELD](#meld)
  - [EMOTIC](#emotic)
  - [PISC](#pisc)
  - [MSP-Podcast](#msp-podcast)
  - [MUStARD](#mustard)
  - [UR-FUNNY](#ur-funny)
  - [RLDD](#rldd)
  - [MEVIEW](#meview-test-only)
- [Step 2 — Build Parquet Files](#step-2--build-parquet-files)
- [Step 3 — Zero-Shot Evaluation](#step-3--zero-shot-evaluation)
- [Step 4 — Compute Metrics](#step-4--compute-metrics)
- [Step 5 — Plot Results](#step-5--plot-results)
- [Fine-Tuning](#fine-tuning)
- [Parquet Schema](#parquet-schema)

---

## Benchmark Overview

| Level | Capability | Datasets |
|---|---|---|
| **L1 — Perception** | Gaze, sound, micro-expressions, body contact | GazeFollow, VAT, VideoCoAtt, VocalSound, Proxemics, MMEW, AffWild2, VoxConverse |
| **L2 — Understanding** | Emotion in context, relationships, diarization | MELD, EMOTIC, PISC, MSP-Podcast |
| **L3 — Theory of Mind** | Sarcasm, humor, deception, concealed emotion | MUStARD, UR-FUNNY, RLDD, MEVIEW |

---

## Quickstart

```
Step 1 → Download raw datasets        dataset/
Step 2 → Build Parquet files          bash run_pipeline.sh
Step 3 → Run zero-shot inference      bash scripts/run_eval.sh
Step 4 → Compute metrics              python evaluate.py --pred-dir output/my_model
Step 5 → Plot results                 python plot_results.py
```

---

## Prerequisites

- Python 3.9+
- `ffmpeg`:
  ```bash
  sudo apt install ffmpeg   # Ubuntu/Debian
  brew install ffmpeg       # macOS
  ```
- ~1TB free disk space for all datasets
- [MS-Swift](https://github.com/modelscope/ms-swift) for inference/training

All Python dependencies are installed automatically on first `bash run_pipeline.sh`.

---

## Step 1 — Download Datasets

Place raw datasets in `dataset/`. Only datasets that are present will be built — the rest are silently skipped.

**11 datasets can be downloaded automatically:**

```bash
bash download_datasets.sh                        # download all available
bash download_datasets.sh gazefollow vocalsound  # or pick specific ones
bash download_datasets.sh --list                 # see what's available
```

The following 5 datasets require a manual registration or license request and cannot be automated: **AffWild2**, **EMOTIC**, **MSP-Podcast**, **MMEW**, **VideoCoAttention**. Download instructions are below.

---

### GazeFollow
**Task:** Given an image and a person's head bounding box, predict where they are looking.

**Download:** [gazefollow_extended.zip](http://gazefollow.csail.mit.edu/download.html)

```
dataset/Gazefollow/data_new/
├── train_annotations.txt
├── test_annotation.txt
└── (image folders)
```

---

### VideoAttentionTarget (VAT)
**Task:** Given a video clip and a person's head bounding box, predict their gaze target.

**Download:** [VideoAttentionTarget](https://github.com/ejcgt/attention-target-detection)

```
dataset/videoattentiontarget/
├── annotations/
│   ├── train/
│   └── test/
└── images/
    └── (show name)/(clip id)/(frames)
```

---

### VideoCoAttention
**Task:** Localize the shared visual attention target of multiple people, or detect whether one exists.

**Download:** Request via email to lfan@g.ucla.edu — include the statement: *"I promise to use the dataset for non-commercial, academic, and research purposes only."* See [VideoCoAtt GitHub](https://github.com/LifengFan/Shared-Attention) for details.

```
dataset/VideoCoAtt/VideoCoAtt_Dataset/
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
└── images/
```

---

### VocalSound
**Task:** Classify a vocal sound (laughter, sigh, cough, throat clearing, sneeze, sniff).

**Download:** [VocalSound](https://github.com/YuanGongND/vocalsound) — `VocalSound_release_16k.tar.gz`

```
dataset/VocalSound_release_16k/
├── audio_16k/
└── datafiles/
    ├── tr.json
    ├── val.json
    └── te.json
```

---

### Proxemics
**Task:** Given an image and two people's bounding boxes, identify which body parts are touching.

**Download:** Contact the authors via the [Proxemics paper](https://arxiv.org/abs/1709.09455)

```
dataset/dataset_proxemics/
├── images/release/
├── labels_6classes_pair.json
└── labels_6classes_pair_BBs.json
```

---

### MMEW
**Task:** Recognize micro/macro-expressions from face images or video clips; predict action units.

**Download:** [MMEW Dataset](https://github.com/benxianyeteam/MMEW-Dataset?tab=readme-ov-file)

```
dataset/MMEW/
├── MMEW_Final/
│   ├── Macro_Expression/
│   └── Micro_Expression/
└── MMEW_Micro_Exp (20).xlsx
```

---

### AffWild2
**Task:** Given a 16-frame face video clip, predict expression, valence/arousal, or action units.

**Download:** [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/) (requires registration)

```
dataset/AffWild2/
├── ABAW Annotations/
│   ├── EXPR_Recognition_Challenge/
│   ├── VA_Estimation_Challenge/
│   └── AU_Detection_Challenge/
└── (video batches: batch1/, batch2/, new_vids/, ...)
```

---

### VoxConverse
**Task:** Speaker diarization and speaker counting from 30-second audio windows.

**Download:**
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
wget https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
git clone --depth 1 https://github.com/joonson/voxconverse dataset/voxconverse/annotations
```
Extract dev WAVs → `dataset/voxconverse/dev/`, test WAVs → `dataset/voxconverse/test/`.

```
dataset/voxconverse/
├── dev/           (WAV files)
├── test/          (WAV files)
└── annotations/
    ├── dev/       (RTTM files)
    └── test/      (RTTM files)
```

---

### MELD
**Task:** Classify a speaker's emotion from a conversational clip (Friends TV show). Three modality variants: video+audio, audio-only, video+transcript.

**Download:** [MELD-FAIR](https://github.com/knowledgetechnologyuhh/MELD-FAIR) — place in `dataset/MELD-FAIR/`. The raw videos can be downloaded directly:
```bash
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
```

```
dataset/MELD-FAIR/MELD/realigned/
├── train/ · dev/ · test/
│   ├── videos/
│   └── audio/16000/
├── realigned_*_sent_emo.csv
└── MELD_active_speaker_face_bboxes.csv
```

---

### EMOTIC
**Task:** Given a scene image and a person's bounding box, predict their emotional state (26 categories or VAD scores).

**Download:** [EMOTIC](https://s3.sunai.uoc.edu/emotic/index.html) (requires registration)

```
dataset/EMOTIC/
├── Annotations/Annotations.mat
├── mscoco/
├── ade20k/
├── framesdb/
└── emodb_small/
```

---

### PISC
**Task:** Given an image and two people's bounding boxes, classify their social relationship (6 categories).

**Download:** [PISC Dataset](https://zenodo.org/record/1059155)

```bash
cat images-00 images-01 images-02 images-03 | tar xz -C dataset/PISC/
```

```
dataset/PISC/
├── image/
└── relationship_split/
    ├── relation_trainidx.json
    ├── relation_validx.json
    └── relation_testidx.json
```

---

### MSP-Podcast
**Task:** Classify a speaker's emotion (9 categories) or predict VAD scores from podcast audio.

**Download:** [MSP Lab](https://www.lab-msp.com/MSP/MSP-Podcast.html) (requires registration). Extract `Audios.tar.gz` manually:

```bash
tar -xzf dataset/msp-podcast/Audios.tar.gz -C dataset/msp-podcast/
```

```
dataset/msp-podcast/
├── Audios/
└── Labels/Labels/
    ├── labels_consensus.csv
    └── labels_detailed.csv
```

---

### MUStARD
**Task:** Detect whether a spoken utterance is sarcastic, with or without preceding dialogue context.

**Download:** [MUStARD](https://github.com/soujanyaporia/MUStARD) — place in `dataset/MUStARD/`.

```
dataset/MUStARD/
├── utterances_final/    (MP4 files)
└── sarcasm_data.json
```

---

### UR-FUNNY
**Task:** Detect whether a speaker's punchline is funny, with or without context.

**Download:** [UR-FUNNY V2](https://github.com/ROC-HCI/UR-FUNNY) — place in `dataset/UR-FUNNY-V2/`.

```
dataset/UR-FUNNY-V2/
├── urfunny2_videos/
└── sdk_features/
```

---

### RLDD
**Task:** Classify whether a person is being deceptive or truthful from real-world video footage.

**Download:** [RLDD](http://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip) — place in `dataset/RealLifeDeceptionDetection.2016/`.

```
dataset/RealLifeDeceptionDetection.2016/Real-life_Deception_Detection_2016/
└── Clips/
```

---

### MEVIEW *(test-only)*
**Task:** Identify the concealed emotion revealed by a micro-expression in a short face video.

**Download:** Contact the authors via the [MEVIEW paper](https://arxiv.org/abs/2209.07486) — place in `dataset/MEVIEW/`.

```
dataset/MEVIEW/
├── me-cuts/cuts/    (MP4 clips)
└── MEVIEW_v2.xlsx
```

---

## Step 2 — Build Parquet Files

```bash
bash run_pipeline.sh
```

The pipeline creates and activates `.venv_dataset/` on first run, installs all dependencies, then builds and checks each available dataset in sequence.

```
parquets/
├── gazefollow/
├── videoattentiontarget/frame/  video/
├── videocoattention/localization/  detection/
├── vocalsound/
├── proxemics/skeleton/  no_skeleton/
├── mmew/
├── affwild2/expr/  va/  au/
├── voxconverse/diarization/  speaker_count/
├── meld/video_audio/  audio_only/  video_transcript/
├── emotic/discrete/  vad/
├── pisc/
├── msppodcast/labels/  vad/
├── mustard/video_context/  video_no_context/
├── urfunny/video_audio/  video_context/
├── rldd/
└── meview/
```

**Configuration** (`config.py`):
```python
VIDEO_MODE   = "fixed_number"   # "framerate" or "fixed_number"
TARGET_FPS   = 16
FIXED_FRAMES = 16
MERGE_TEST   = False            # set True to produce parquets/benchmark_test.parquet
```

---

## Step 3 — Zero-Shot Evaluation

The inference plugin strips ground-truth answers so the model generates its own predictions. Each `-infer` alias corresponds to a training alias (e.g. `vocalsound-omni-infer` for `vocalsound-omni`).

### Example script

```bash
#!/bin/bash
# Zero-shot evaluation with MS-Swift
set -euo pipefail

MODEL="Qwen/Qwen3-Omni-7B"
OUTPUT="output/Qwen3-Omni"
PLUGIN="plugins/omni_dataset_plugin_infer.py"

mkdir -p "$OUTPUT"

DATASETS=(
    "gazefollow-vlm-infer"
    "vat-omni-infer/frame"
    "vat-omni-infer/video"
    "vocalsound-omni-infer"
    "proxemics-vlm-infer/no-skeleton"
    "proxemics-vlm-infer/skeleton"
    "mmew-infer/apex-emotion"
    "mmew-infer/clip-emotion"
    "affwild2-omni-infer/au"
    "affwild2-omni-infer/va"
    "affwild2-omni-infer/expr"
    "voxconverse-omni-infer/diarization"
    "voxconverse-omni-infer/speaker-count"
    "meld-omni-infer/video-audio"
    "meld-omni-infer/audio-only"
    "meld-omni-infer/video-transcript"
    "emotic-vlm-infer/discrete"
    "emotic-vlm-infer/vad"
    "pisc-vlm-infer"
    "videocoattention-vlm-infer/detection"
    "videocoattention-vlm-infer/localization"
    "msppodcast-omni-infer/labels"
    "msppodcast-omni-infer/vad"
    "mustard-omni-infer/video-context"
    "mustard-omni-infer/video-no-context"
    "urfunny-omni-infer/video-audio"
    "urfunny-omni-infer/video-context"
    "rldd-omni-infer"
    "meview-omni-infer/recognition"
)

for DATASET in "${DATASETS[@]}"; do
    echo "=== $DATASET ==="
    swift infer \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --custom_plugin "$PLUGIN" \
        --result_dir "$OUTPUT" \
        --max_new_tokens 512
done

echo "Done. Results in $OUTPUT"
```

Each run writes a `PRED_{dataset}_{subset}_{split}.jsonl` file to `$OUTPUT`.

### Full dataset aliases

| Training alias | Inference alias | Modality |
|---|---|---|
| `gazefollow-vlm` | `gazefollow-vlm-infer` | Image |
| `vat-omni/frame` | `vat-omni-infer/frame` | Image |
| `vat-omni/video` | `vat-omni-infer/video` | Video |
| `videocoattention-vlm/localization` | `videocoattention-vlm-infer/localization` | Image |
| `videocoattention-vlm/detection` | `videocoattention-vlm-infer/detection` | Image |
| `vocalsound-omni` | `vocalsound-omni-infer` | Audio |
| `proxemics-vlm/skeleton` | `proxemics-vlm-infer/skeleton` | Image |
| `proxemics-vlm/no-skeleton` | `proxemics-vlm-infer/no-skeleton` | Image |
| `mmew-omni/apex-au` | `mmew-infer/apex-au` | Image |
| `mmew-omni/apex-emotion` | `mmew-infer/apex-emotion` | Image |
| `mmew-omni/clip-emotion` | `mmew-infer/clip-emotion` | Video |
| `affwild2-omni/expr` | `affwild2-omni-infer/expr` | Video |
| `affwild2-omni/va` | `affwild2-omni-infer/va` | Video |
| `affwild2-omni/au` | `affwild2-omni-infer/au` | Video |
| `voxconverse-omni/diarization` | `voxconverse-omni-infer/diarization` | Audio |
| `voxconverse-omni/speaker-count` | `voxconverse-omni-infer/speaker-count` | Audio |
| `meld-omni/video-audio` | `meld-omni-infer/video-audio` | Video + Audio |
| `meld-omni/audio-only` | `meld-omni-infer/audio-only` | Audio |
| `meld-omni/video-transcript` | `meld-omni-infer/video-transcript` | Video + Text |
| `emotic-vlm/discrete` | `emotic-vlm-infer/discrete` | Image |
| `emotic-vlm/vad` | `emotic-vlm-infer/vad` | Image |
| `pisc-vlm` | `pisc-vlm-infer` | Image |
| `msppodcast-omni/labels` | `msppodcast-omni-infer/labels` | Audio |
| `msppodcast-omni/vad` | `msppodcast-omni-infer/vad` | Audio |
| `mustard-omni/video-context` | `mustard-omni-infer/video-context` | Video + Audio |
| `mustard-omni/video-no-context` | `mustard-omni-infer/video-no-context` | Video + Audio |
| `urfunny-omni/video-audio` | `urfunny-omni-infer/video-audio` | Video + Audio |
| `urfunny-omni/video-context` | `urfunny-omni-infer/video-context` | Video + Audio |
| `rldd-omni` | `rldd-omni-infer` | Video + Audio |
| `meview-omni/recognition` | `meview-omni-infer/recognition` | Video |

---

## Step 4 — Compute Metrics

Once inference is done, compute metrics from the `PRED_*.jsonl` files:

```bash
python evaluate.py \
    --pred-dir output/Qwen3-Omni \
    --parquets-dir parquets
```

This writes `output/Qwen3-Omni/metrics.json` with per-task results.

### Baseline correction (optional)

Some models fail to format answers on a fraction of samples. The adjusted metric corrects for this:

```
adjusted = (metric × n_valid + baseline × n_invalid) / n_samples
```

First, compute naive baselines from your Parquet files (one-time):

```bash
python compute_baselines.py \
    --parquets-dir parquets \
    --output results/naive_baselines.json
```

Then apply the correction across all models in `output/`:

```bash
python compute_adjusted_metrics.py \
    --results-dir output \
    --baselines results/naive_baselines.json \
    --output-dir results/adjusted
```

Pre-computed baselines and adjusted results for the paper's models are in `results/`.

---

## Step 5 — Plot Results

```bash
# From adjusted metrics (baseline-corrected)
python plot_results.py --results-dir results/adjusted

# From raw metrics
python plot_raw.py --results-dir output
```

Both scripts save a `radar.pdf` / `radar_raw.pdf` next to the results.

### Selective display

Control which models appear via environment variables (set to `0` to hide):

```bash
ZERO_SHOT_QWEN3=1 ZERO_SHOT_QWEN25=1 JOINT=0 SINGLE_TASK=0 \
    python plot_results.py --results-dir results/adjusted
```

| Variable | Model |
|---|---|
| `ZERO_SHOT_QWEN3` | Qwen3-Omni (zero-shot) |
| `ZERO_SHOT_QWEN25` | Qwen2.5-Omni (zero-shot) |
| `ZERO_SHOT_MINICPMO45` | MiniCPM-o 4.5 (zero-shot) |
| `ZERO_SHOT_MINICPMO26` | MiniCPM-o 2.6 (zero-shot) |
| `ZERO_SHOT_GEMMA4` | Gemma-4 (zero-shot) |
| `SINGLE_TASK` | Single-task fine-tuned |
| `JOINT` | Joint fine-tuned |
| `SOTA` | SOTA reference polygon (default: 1) |
| `BASELINE` | Naive baseline polygon (default: 0) |

---

## Fine-Tuning

```bash
swift sft \
    --model Qwen/Qwen3-Omni-7B \
    --dataset vocalsound-omni meld-omni/video-audio affwild2-omni/expr \
    --custom_plugin plugins/omni_dataset_plugin.py \
    --output_dir output/my_finetuned
```

Use `--dataset all` (not a real flag — list all aliases explicitly) to train jointly on all 16 datasets, which corresponds to the **joint** model from the paper.

---

## Parquet Schema

All datasets share a unified schema:

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | System / user / assistant turns |
| `images` | `list[Image]` | PIL images as bytes |
| `audios` | `list[binary]` | Raw 16kHz mono WAV bytes |
| `videos` | `list[binary]` | Raw MP4 bytes (H.264, 16fps) |

Bounding boxes follow Qwen's convention: `(x1, y1, x2, y2)` normalized to `[0, 1000]`. Answers are always valid JSON strings.

---

## Dataset Licenses

All datasets are used strictly for academic research in accordance with their respective terms.

| Dataset | License / Terms | Access |
|---|---|---|
| GazeFollow | Academic / Public | Public download |
| VideoAttentionTarget (VAT) | Public (no explicit license stated) | Public download |
| VideoCoAttention | Academic / Non-commercial research only | Request via email to lfan@g.ucla.edu (free for research purposes) |
| VocalSound | Public / Academic | Public download |
| Proxemics | Public (no explicit license stated) | Public download |
| MMEW | Academic (Agreement required) | Registration required |
| AffWild2 | Academic (Agreement required) | Registration required |
| VoxConverse | CC BY 4.0 International | Public download |
| MELD | Academic / Public | Public download |
| EMOTIC | Academic / Public | Registration required |
| PISC | Academic / Public | Public download (Zenodo) |
| MSP-Podcast | Academic License (institution signature required, free of cost) | Registration required |
| MUStARD | Public | Public download |
| UR-FUNNY | Public (no explicit license stated) | Public download |
| RLDD | Academic / Public | Public download |
| MEVIEW | Public (no explicit license stated) | Contact authors |

The benchmark harness and evaluation code in this repository are released under the **MIT License**.

