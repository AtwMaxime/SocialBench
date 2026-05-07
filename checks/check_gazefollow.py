import json
import os
import random
import sys

import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT_DIR, "parquets", "gazefollow")

EXPECTED_FILES = [
    "gazefollow_train.parquet",
    "gazefollow_validation.parquet",
]


def check_file(path):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    table = pq.read_table(path)
    df = table.to_pandas()
    print(f"✅ {os.path.basename(path)} — {len(df)} rows")

    required_cols = {"messages", "images"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)

    for i, row in df.iterrows():
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"Row {i}: bad roles {roles}"

        user = msgs[1]["content"]
        assert "<image>" in user, f"Row {i}: missing <image> tag"
        assert "bounding box" in user, f"Row {i}: missing bbox in user prompt"

        parsed = json.loads(msgs[2]["content"])
        # train: {"gaze_point": [x, y]}  |  validation: {"gaze_points": [[x,y], ...]}
        if "gaze_point" in parsed:
            gp = parsed["gaze_point"]
            assert len(gp) == 2, f"Row {i}: gaze_point length {len(gp)}, expected 2"
            for coord in gp:
                assert 0 <= coord <= 1000, f"Row {i}: gaze coord out of range: {coord}"
        else:
            assert "gaze_points" in parsed, f"Row {i}: missing 'gaze_point' or 'gaze_points'"
            for gp in parsed["gaze_points"]:
                assert len(gp) == 2, f"Row {i}: gaze_points entry length {len(gp)}, expected 2"
                for coord in gp:
                    assert 0 <= coord <= 1000, f"Row {i}: gaze coord out of range: {coord}"

        assert len(row["images"]) == 1, f"Row {i}: expected 1 image"

    print("   Message structure: OK")

    sample = df.iloc[random.randint(0, len(df) - 1)]
    parsed = json.loads(sample["messages"][2]["content"])
    gp = parsed.get("gaze_point") or parsed.get("gaze_points", [None])[0]
    print(f"   Sample gaze_point: {gp}")

    img_data = sample["images"][0]
    img_bytes = img_data["bytes"] if isinstance(img_data, dict) else bytes(img_data)
    assert len(img_bytes) > 1000, "Image bytes suspiciously small"
    print(f"   Sample image size: {len(img_bytes) / 1024:.1f} KB")


def check():
    for fname in EXPECTED_FILES:
        check_file(os.path.join(PARQUET_DIR, fname))
        print()

    print("✅ check_gazefollow passed!")


if __name__ == "__main__":
    check()
