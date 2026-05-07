import json
import os
import random
import sys

import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT_DIR, "parquets", "videocoattention")

EXPECTED_FILES = [
    "videocoattention_localization_train.parquet",
    "videocoattention_localization_val.parquet",
    "videocoattention_localization_test.parquet",
    "videocoattention_detection_train.parquet",
    "videocoattention_detection_val.parquet",
    "videocoattention_detection_test.parquet",
]


def check_file(path, task):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    table = pq.read_table(path)
    df = table.to_pandas()
    print(f"✅ {os.path.basename(path)} — {len(df)} rows")

    # Schema
    required_cols = {"messages", "images"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)

    # Message structure
    for i, row in df.iterrows():
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"Row {i}: bad roles {roles}"

        content = msgs[2]["content"]
        parsed = json.loads(content)

        if task == "localization":
            assert "bbox" in parsed, f"Row {i}: missing 'bbox'"
            bbox = parsed["bbox"]
            assert len(bbox) == 4, f"Row {i}: bbox length {len(bbox)}, expected 4"
            for coord in bbox:
                assert 0 <= coord <= 1000, f"Row {i}: coord out of range: {coord}"
        else:
            assert "co_attention" in parsed, f"Row {i}: missing 'co_attention'"
            assert isinstance(
                parsed["co_attention"], bool
            ), f"Row {i}: co_attention not bool"

        assert len(row["images"]) == 1, f"Row {i}: expected 1 image"

    print("   Message structure: OK")

    # Balance check for detection
    if task == "detection":
        from collections import Counter

        labels = Counter(
            json.loads(r["messages"][2]["content"])["co_attention"]
            for _, r in df.iterrows()
        )
        total = sum(labels.values())
        print(
            f"   Label balance: true={labels[True]} ({labels[True]/total*100:.1f}%), "
            f"false={labels[False]} ({labels[False]/total*100:.1f}%)"
        )

    # Sample check
    sample = df.iloc[random.randint(0, len(df) - 1)]
    answer = json.loads(sample["messages"][2]["content"])
    if task == "localization":
        print(f"   Sample bbox: {answer['bbox']}")
    else:
        print(f"   Sample co_attention: {answer['co_attention']}")

    # Image size check
    img_data = sample["images"][0]
    img_bytes = img_data["bytes"] if isinstance(img_data, dict) else bytes(img_data)
    assert len(img_bytes) > 1000, "Image bytes suspiciously small"
    print(f"   Sample image size: {len(img_bytes) / 1024:.1f} KB")


def check():
    for fname in EXPECTED_FILES:
        task = "localization" if "localization" in fname else "detection"
        check_file(os.path.join(PARQUET_DIR, task, fname), task)
        print()

    print("✅ check_videocoattention passed!")


if __name__ == "__main__":
    check()
