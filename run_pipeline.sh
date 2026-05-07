#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv_dataset"

# Detect python binary
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "❌ No python or python3 found. Please install Python 3.9+." >&2
    exit 1
fi

# Bootstrap venv on first run
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[SETUP] Creating virtual environment..."
    "$PYTHON" -m venv "$VENV"
    source "$VENV/bin/activate"
    echo "[SETUP] Installing dependencies..."
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV/bin/activate"
fi

python "$SCRIPT_DIR/pipeline.py" "$@"
