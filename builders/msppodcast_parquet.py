import csv
import json
import os

from datasets import Dataset, Features, Sequence, Value

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSP_DIR = os.path.join(ROOT_DIR, "dataset", "msp-podcast")
AUDIO_DIR = os.path.join(MSP_DIR, "Audios")
LABELS_DIR = os.path.join(MSP_DIR, "Labels", "Labels")

CONSENSUS_CSV = os.path.join(LABELS_DIR, "labels_consensus.csv")
DETAILED_CSV = os.path.join(LABELS_DIR, "labels_detailed.csv")

OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "msppodcast")

# Consensus emotion code → human-readable label
EMO_MAP = {
    "N": "Neutral",
    "H": "Happiness",
    "A": "Anger",
    "S": "Sadness",
    "U": "Surprise",
    "F": "Fear",
    "D": "Disgust",
    "C": "Contempt",
    "O": "Other",
}
EMO_LABELS = list(EMO_MAP.values())

# Split_Set values → parquet split names
SPLIT_MAP = {
    "Train": "train",
    "Development": "validation",
    "Test1": "test",
    "Test2": "test",
    # Test3 has hidden labels — skipped
}

# ==========================================
# 2. PROMPTS
# ==========================================

SYSTEM_LABELS = (
    "You are an expert in speech emotion recognition. "
    "Given an audio clip of a speaker, classify their emotional state. "
    f"Choose from: {json.dumps(EMO_LABELS)}. "
    'Provide your answer as a valid JSON object: {"emotion": "Neutral"}.'
)

USER_LABELS = "<audio>\nWhat is the emotional state of the speaker in this audio clip?"

SYSTEM_VAD = (
    "You are an expert in speech emotion recognition. "
    "Given an audio clip of a speaker, predict their emotional state "
    "as arousal, valence, and dominance scores. "
    "Each score is a float from 1 (very low) to 7 (very high). "
    'Provide your answer as a valid JSON object: {"arousal": x.x, "valence": x.x, "dominance": x.x}.'
)

USER_VAD = "<audio>\nWhat are the arousal, valence, and dominance scores of the speaker in this audio clip?"

# ==========================================
# 3. HELPERS
# ==========================================

FEATURES = Features(
    {
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "audios": Sequence(Value("binary")),
    }
)


def normalize_emo(label: str) -> str:
    """Normalize 'Other-xxx' variants to 'Other'."""
    if label.lower().startswith("other"):
        return "Other"
    return label


def load_consensus() -> dict:
    """filename → {emo, arousal, valence, dominance, split}"""
    data = {}
    with open(CONSENSUS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            split = SPLIT_MAP.get(row["Split_Set"])
            if split is None:
                continue  # Test3 — skip
            emo_code = row["EmoClass"]
            data[row["FileName"]] = {
                "emo_code": emo_code,
                "emo": EMO_MAP.get(emo_code),  # None for X (no agreement)
                "arousal": float(row["EmoAct"]),
                "valence": float(row["EmoVal"]),
                "dominance": float(row["EmoDom"]),
                "split": split,
            }
    return data


def load_detailed() -> dict:
    """filename → list of {emo, arousal, valence, dominance} per annotator"""
    data: dict[str, list] = {}
    with open(DETAILED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if not row["EmoAct"] or not row["EmoVal"] or not row["EmoDom"]:
                continue
            fname = row["FileName"]
            data.setdefault(fname, []).append(
                {
                    "emo": normalize_emo(row["EmoClass_Major"]),
                    "arousal": float(row["EmoAct"]),
                    "valence": float(row["EmoVal"]),
                    "dominance": float(row["EmoDom"]),
                }
            )
    return data


def load_audio(filename: str) -> bytes | None:
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


# ==========================================
# 4. GENERATORS
# ==========================================


def make_labels_generator(consensus: dict, detailed: dict, target_split: str):
    def generator():
        skipped_missing = 0
        skipped_no_consensus = 0
        for filename, meta in consensus.items():
            if meta["split"] != target_split:
                continue

            # Train: single consensus label — skip X (no agreement)
            if target_split == "train":
                if meta["emo"] is None:
                    skipped_no_consensus += 1
                    continue
                answer = json.dumps({"emotion": meta["emo"]})
            else:
                # Val / Test: all per-annotator primary emotions
                anns = detailed.get(filename, [])
                if not anns:
                    skipped_no_consensus += 1
                    continue
                answer = json.dumps(
                    {"emotions_per_annotator": [a["emo"] for a in anns]}
                )

            audio_bytes = load_audio(filename)
            if audio_bytes is None:
                skipped_missing += 1
                continue

            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_LABELS},
                    {"role": "user", "content": USER_LABELS},
                    {"role": "assistant", "content": answer},
                ],
                "audios": [audio_bytes],
            }

        if skipped_missing:
            print(f"  ⚠️  Skipped (audio not found): {skipped_missing}")
        if skipped_no_consensus:
            print(f"  ⚠️  Skipped (no consensus / no annotations): {skipped_no_consensus}")

    return generator


def make_vad_generator(consensus: dict, detailed: dict, target_split: str):
    def generator():
        skipped_missing = 0
        for filename, meta in consensus.items():
            if meta["split"] != target_split:
                continue

            # Train: consensus averaged VAD
            if target_split == "train":
                answer = json.dumps(
                    {
                        "arousal": round(meta["arousal"], 3),
                        "valence": round(meta["valence"], 3),
                        "dominance": round(meta["dominance"], 3),
                    }
                )
            else:
                # Val / Test: all per-annotator VAD values
                anns = detailed.get(filename, [])
                if not anns:
                    continue
                answer = json.dumps(
                    {
                        "vad_per_annotator": [
                            {
                                "arousal": a["arousal"],
                                "valence": a["valence"],
                                "dominance": a["dominance"],
                            }
                            for a in anns
                        ]
                    }
                )

            audio_bytes = load_audio(filename)
            if audio_bytes is None:
                skipped_missing += 1
                continue

            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM_VAD},
                    {"role": "user", "content": USER_VAD},
                    {"role": "assistant", "content": answer},
                ],
                "audios": [audio_bytes],
            }

        if skipped_missing:
            print(f"  ⚠️  Skipped (audio not found): {skipped_missing}")

    return generator


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    print("📂 Loading annotation files...")
    consensus = load_consensus()
    detailed = load_detailed()
    print(f"  Consensus entries : {len(consensus):,}")
    print(f"  Detailed entries  : {len(detailed):,}")

    for task, gen_fn, subfolder in [
        ("labels", make_labels_generator, "labels"),
        ("vad", make_vad_generator, "vad"),
    ]:
        task_dir = os.path.join(OUTPUT_DIR, subfolder)
        os.makedirs(task_dir, exist_ok=True)

        for split in ["train", "validation", "test"]:
            print(f"\n🚀 {task}/{split}...")
            gen = gen_fn(consensus, detailed, split)
            ds = Dataset.from_generator(gen, features=FEATURES)
            out_path = os.path.join(task_dir, f"msppodcast_{task}_{split}.parquet")
            ds.to_parquet(out_path)
            print(f"✅ {out_path} ({len(ds):,} examples)")

    print("\n✨ Done!")
