import json
import os
import random
import sys
from collections import Counter

import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT_DIR, "parquets", "affwild2")

EXPR_LABELS = [
    "Neutral", "Anger", "Disgust", "Fear",
    "Happiness", "Sadness", "Surprise", "Other",
]
AU_COLS = [
    "AU1", "AU2", "AU4", "AU6", "AU7", "AU10",
    "AU12", "AU15", "AU23", "AU24", "AU25", "AU26",
]

# (task, subfolder, filename)
EXPECTED = [
    ("expr",       "expr", "affwild2_expr_train.parquet"),
    ("expr",       "expr", "affwild2_expr_val.parquet"),
    ("va",         "va",   "affwild2_va_train.parquet"),
    ("va",         "va",   "affwild2_va_val.parquet"),
    ("au",         "au",   "affwild2_au_train.parquet"),
    ("au",         "au",   "affwild2_au_val.parquet"),
    ("expr_think", "expr", "affwild2_expr_think_train.parquet"),
    ("expr_think", "expr", "affwild2_expr_think_val.parquet"),
]


def check_file(path, task):
    if not os.path.exists(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    table = pq.read_table(path)
    df = table.to_pandas()
    print(f"✅ {os.path.relpath(path, PARQUET_DIR)} — {len(df)} rows")

    required_cols = {"messages", "videos", "audios"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"❌ Missing columns: {missing}")
        sys.exit(1)

    for i, row in df.iterrows():
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"], f"Row {i}: bad roles {roles}"

        content = msgs[2]["content"]

        if task == "expr_think":
            assert "<think>" in content and "</think>" in content, \
                f"Row {i}: missing think block"
            json_part = content.split("</think>")[-1].strip()
            parsed = json.loads(json_part)
            assert "label" in parsed, f"Row {i}: missing 'label'"
            assert parsed["label"] in EXPR_LABELS, \
                f"Row {i}: unknown label '{parsed['label']}'"
        elif task == "expr":
            parsed = json.loads(content)
            assert "label" in parsed, f"Row {i}: missing 'label'"
            assert parsed["label"] in EXPR_LABELS, \
                f"Row {i}: unknown label '{parsed['label']}'"
        elif task == "va":
            parsed = json.loads(content)
            assert "valence" in parsed and "arousal" in parsed, \
                f"Row {i}: missing valence/arousal"
            assert -1.0 <= parsed["valence"] <= 1.0, \
                f"Row {i}: valence out of range: {parsed['valence']}"
            assert -1.0 <= parsed["arousal"] <= 1.0, \
                f"Row {i}: arousal out of range: {parsed['arousal']}"
        elif task == "au":
            parsed = json.loads(content)
            assert "action_units" in parsed, f"Row {i}: missing 'action_units'"
            for au in parsed["action_units"]:
                assert au in AU_COLS, f"Row {i}: unknown AU '{au}'"

        assert len(row["videos"]) == 1, f"Row {i}: expected 1 video"

    print("   Message structure: OK")

    # Per-task stats
    if task in ("expr", "expr_think"):
        def get_label(r):
            c = r["messages"][2]["content"]
            return json.loads(c.split("</think>")[-1].strip())["label"] \
                if task == "expr_think" else json.loads(c)["label"]

        dist = Counter(get_label(r) for _, r in df.iterrows())
        print("   Label distribution:")
        for lbl in EXPR_LABELS:
            print(f"     {lbl}: {dist.get(lbl, 0)}")

    elif task == "va":
        valences = [json.loads(r["messages"][2]["content"])["valence"] for _, r in df.iterrows()]
        arousals = [json.loads(r["messages"][2]["content"])["arousal"] for _, r in df.iterrows()]
        print(f"   Valence  min={min(valences):.3f}  max={max(valences):.3f}  "
              f"mean={sum(valences)/len(valences):.3f}")
        print(f"   Arousal  min={min(arousals):.3f}  max={max(arousals):.3f}  "
              f"mean={sum(arousals)/len(arousals):.3f}")

    elif task == "au":
        all_aus = []
        none_count = 0
        for _, r in df.iterrows():
            aus = json.loads(r["messages"][2]["content"])["action_units"]
            if not aus:
                none_count += 1
            all_aus.extend(aus)
        dist = Counter(all_aus)
        top = sorted(dist.items(), key=lambda x: -x[1])[:5]
        print(f"   No AUs active: {none_count}")
        print(f"   Top AUs: {', '.join(f'{au}={n}' for au, n in top)}")

    # Video size check
    sample = df.iloc[random.randint(0, len(df) - 1)]
    vid_data = sample["videos"][0]
    vid_bytes = vid_data["bytes"] if isinstance(vid_data, dict) else bytes(vid_data)
    assert len(vid_bytes) > 1000, "Video bytes suspiciously small"
    print(f"   Sample video size: {len(vid_bytes) / 1024:.1f} KB")


def check():
    for task, subfolder, fname in EXPECTED:
        path = os.path.join(PARQUET_DIR, subfolder, fname)
        check_file(path, task)
        print()

    print("✅ check_affwild2 passed!")


if __name__ == "__main__":
    check()
