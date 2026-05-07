import json
import os
import random
from collections import Counter

from datasets import load_dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUETS_DIR = os.path.join(ROOT_DIR, "parquets", "msppodcast")


def check_task(task):
    task_dir = os.path.join(PARQUETS_DIR, task)
    if not os.path.isdir(task_dir):
        print(f"❌ Directory not found: {task_dir}")
        return

    print(f"\n{'='*60}")
    print(f"📂 {task}")

    for split in ["train", "validation", "test"]:
        parquet_file = os.path.join(task_dir, f"msppodcast_{task}_{split}.parquet")
        if not os.path.exists(parquet_file):
            print(f"  ❌ Not found: {parquet_file}")
            continue

        ds = load_dataset("parquet", data_files={split: parquet_file}, split=split)
        print(f"\n  ✅ {split}: {len(ds):,} examples")

        if task == "labels":
            label_counts = Counter()
            for ex in ds:
                answer = json.loads(
                    next(m for m in ex["messages"] if m["role"] == "assistant")[
                        "content"
                    ]
                )
                if "emotions_per_annotator" in answer:
                    for e in answer["emotions_per_annotator"]:
                        label_counts[e] += 1
                else:
                    label_counts[answer["emotion"]] += 1
            print(f"  --- Emotion distribution ---")
            for label, count in label_counts.most_common():
                print(f"    {label}: {count:,}")
        else:
            arousals, valences, dominances = [], [], []
            for ex in ds:
                answer = json.loads(
                    next(m for m in ex["messages"] if m["role"] == "assistant")[
                        "content"
                    ]
                )
                if "vad_per_annotator" in answer:
                    for vad in answer["vad_per_annotator"]:
                        arousals.append(vad["arousal"])
                        valences.append(vad["valence"])
                        dominances.append(vad["dominance"])
                else:
                    arousals.append(answer["arousal"])
                    valences.append(answer["valence"])
                    dominances.append(answer["dominance"])
            print(f"  --- VAD statistics ---")
            for name, vals in [
                ("arousal", arousals),
                ("valence", valences),
                ("dominance", dominances),
            ]:
                print(
                    f"    {name}: min={min(vals):.2f}  max={max(vals):.2f}  "
                    f"mean={sum(vals)/len(vals):.2f}"
                )

        # Audio check
        idx = random.randint(0, len(ds) - 1)
        example = ds[idx]
        audios = example.get("audios", [])
        audio_ok = audios and audios[0] is not None and len(audios[0]) > 0
        print(f"  🎲 Sample #{idx} — audio present: {'✅' if audio_ok else '❌'}")
        asst = next(m for m in example["messages"] if m["role"] == "assistant")
        print(f"  💬 Answer: {asst['content'][:120]}")


for task in ["labels", "vad"]:
    check_task(task)
