import json
import os
import random
import sys

import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT_DIR, "parquets", "voxconverse")

EXPECTED_FILES = [
    ("diarization",    "voxconverse_diarization_train.parquet"),
    ("diarization",    "voxconverse_diarization_test.parquet"),
    ("speaker",        "voxconverse_speaker_count_train.parquet"),
    ("speaker",        "voxconverse_speaker_count_test.parquet"),
]


def check_file(path, task):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    table = pq.read_table(path)
    df = table.to_pandas()
    print(f"✅ {os.path.basename(path)} — {len(df)} rows")

    # Schema
    required_cols = {"messages", "audios", "videos", "clip_id", "win_start"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)

    # Message structure & answer format
    for i, row in df.iterrows():
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"Row {i}: bad roles {roles}"

        content = msgs[2]["content"]
        parsed = json.loads(content)

        if task == "diarization":
            assert "timeline" in parsed, f"Row {i}: missing 'timeline'"
            tl = parsed["timeline"]
            assert len(tl) == 60, f"Row {i}: timeline length {len(tl)}, expected 60"
            for bin_label in tl:
                assert isinstance(bin_label, str), f"Row {i}: bin not a string: {bin_label}"
        else:
            assert "num_speakers" in parsed, f"Row {i}: missing 'num_speakers'"
            assert isinstance(parsed["num_speakers"], int), f"Row {i}: num_speakers not int"

        assert len(row["audios"]) == 1, f"Row {i}: expected 1 audio"

    print(f"   Message structure: OK")

    # Sample answer
    sample = df.iloc[random.randint(0, len(df) - 1)]
    answer = json.loads(sample["messages"][2]["content"])

    if task == "diarization":
        print(f"   Sample timeline: {answer['timeline']}")
        # Distribution of unique speakers per window
        from collections import Counter
        spk_counts = Counter(
            len({b for b in json.loads(r["messages"][2]["content"])["timeline"] if b != "-"})
            for _, r in df.iterrows()
        )
        print(f"   Unique speaker-bins distribution:")
        for k, v in sorted(spk_counts.items()):
            print(f"     {k} unique label(s): {v} windows")
    else:
        from collections import Counter
        counts = Counter(
            json.loads(r["messages"][2]["content"])["num_speakers"]
            for _, r in df.iterrows()
        )
        print(f"   Speaker count distribution:")
        for n, c in sorted(counts.items()):
            print(f"     {n} speaker(s): {c}")

    # Audio size check
    audio_data = sample["audios"][0]
    audio_bytes = audio_data["bytes"] if isinstance(audio_data, dict) else bytes(audio_data)
    assert len(audio_bytes) > 1000, "Audio bytes suspiciously small"
    print(f"   Sample audio size: {len(audio_bytes) / 1024:.1f} KB")


def check():
    for subdir, fname in EXPECTED_FILES:
        task = "diarization" if "diarization" in fname else "speaker_count"
        check_file(os.path.join(PARQUET_DIR, subdir, fname), task)
        print()

    print("✅ check_voxconverse passed!")


if __name__ == "__main__":
    check()
