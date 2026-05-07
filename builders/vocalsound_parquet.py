import os
import json
from datasets import Dataset, Features, Sequence, Value, Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS_DIR     = os.path.join(ROOT_DIR, "dataset", "VocalSound_release_16k")
AUDIO_DIR  = os.path.join(VS_DIR, "audio_16k")
OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "vocalsound")

LABEL_MAP = {
    '/m/01j3sz': 'Laughter',
    '/m/07plz5l': 'Sigh',
    '/m/01b_21':  'Cough',
    '/m/0dl9sf8': 'Throat clearing',
    '/m/01hsr_':  'Sneeze',
    '/m/07ppn3j': 'Sniff',
}

SPLITS = {
    "train":      "datafiles/tr.json",
    "validation": "datafiles/val.json",
    "test":       "datafiles/te.json",
}

SYSTEM_PROMPT = (
    'You are an expert in vocal sound recognition. '
    'Given an audio clip of a person producing a vocal sound, '
    'classify it into one of the following categories: '
    '["Laughter", "Sigh", "Cough", "Throat clearing", "Sneeze", "Sniff"]. '
    'Provide your answer as a valid JSON string.'
)

USER_PROMPT = (
    '<audio>\n'
    'For this given audio, identify the vocal sound and give the corresponding class '
    'out of the following: ["Laughter", "Sigh", "Cough", "Throat clearing", "Sneeze", "Sniff"].'
)

# ==========================================
# 2. GENERATOR
# ==========================================

def make_generator(entries):
    def generator():
        skipped = 0
        for entry in entries:
            # Remap absolute path from original machine to local audio_16k folder
            filename  = os.path.basename(entry['wav'])
            audio_path = os.path.join(AUDIO_DIR, filename)

            if not os.path.exists(audio_path):
                print(f"⚠️  Audio not found: {audio_path}")
                skipped += 1
                continue

            label_mid = entry['labels']
            if label_mid not in LABEL_MAP:
                print(f"⚠️  Unknown label '{label_mid}' for {filename}, skipping.")
                skipped += 1
                continue

            label = LABEL_MAP[label_mid]

            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": USER_PROMPT},
                    {"role": "assistant", "content": json.dumps({"vocal_sound": label})},
                ],
                "images": [],
                "audios": [audio_bytes],
            }

        if skipped:
            print(f"⚠️  Total skipped: {skipped}")

    return generator

# ==========================================
# 3. MAIN
# ==========================================

if __name__ == "__main__":
    features = Features({
        "messages": [{"role": Value("string"), "content": Value("string")}],
        "images":   Sequence(Image(decode=True)),
        "audios":   Sequence(Value("binary")),
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name, split_file in SPLITS.items():
        print(f"\n🚀 Processing {split_name}...")

        with open(os.path.join(VS_DIR, split_file)) as f:
            entries = json.load(f)['data']

        ds = Dataset.from_generator(
            make_generator(entries),
            features=features,
        )

        output_path = os.path.join(OUTPUT_DIR, f"vocalsound_{split_name}.parquet")
        ds.to_parquet(output_path)
        print(f"✅ Saved: {output_path} ({len(ds)} examples)")

    print("\n✨ Done!")
