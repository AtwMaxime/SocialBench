import io
import json
import os
import random
import sys

import pyarrow.parquet as pq
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(ROOT_DIR, "parquets", "meview", "meview_recognition_test.parquet")


def check():
    if not os.path.exists(PARQUET_PATH):
        print(f"❌ Not found: {PARQUET_PATH}")
        sys.exit(1)

    table = pq.read_table(PARQUET_PATH)
    df = table.to_pandas()
    print(f"✅ Loaded {len(df)} rows from {PARQUET_PATH}")

    # Schema check
    required_cols = {"messages", "videos", "audios"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)
    print(f"   Columns: {list(df.columns)}")

    # Message structure
    for i, row in df.iterrows():
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"Row {i}: bad roles {roles}"

        content = msgs[2]["content"]
        parsed = json.loads(content)
        assert "emotion" in parsed, f"Row {i}: missing 'emotion' key in {content}"

    print("   Message structure: OK")

    # Emotion distribution
    from collections import Counter

    emotions = [json.loads(row["messages"][2]["content"])["emotion"] for _, row in df.iterrows()]
    counts = Counter(emotions)
    print(f"\n   Emotion distribution ({len(df)} total):")
    for emotion, n in sorted(counts.items()):
        print(f"    {emotion}: {n}")

    # Video check — sample one random row
    sample_row = df.iloc[random.randint(0, len(df) - 1)]
    video_data = sample_row["videos"][0]
    video_bytes = video_data["bytes"] if isinstance(video_data, dict) else bytes(video_data)
    assert len(video_bytes) > 1000, "Video bytes suspiciously small"
    print(f"\n   Sample video size: {len(video_bytes) / 1024:.1f} KB")
    print(f"   Sample emotion: {json.loads(sample_row['messages'][2]['content'])['emotion']}")


    print("\n✅ check_meview passed!")


if __name__ == "__main__":
    check()
