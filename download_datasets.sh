#!/usr/bin/env bash
# download_datasets.sh — Download all publicly available datasets.
#
# Usage:
#   bash download_datasets.sh              # download everything
#   bash download_datasets.sh gazefollow vocalsound rldd
#   bash download_datasets.sh --list       # show available datasets
#
# Gated datasets (require licence / registration) are NOT handled here:
#   AffWild2, EMOTIC, MSP-Podcast, MMEW, VideoCoAttention

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/dataset"
mkdir -p "$DATA_DIR"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
skip()  { echo -e "${YELLOW}[SKIP]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── helpers ───────────────────────────────────────────────────────────────────
require() {
    command -v "$1" &>/dev/null || error "'$1' is required but not installed."
}

# ── dataset functions ─────────────────────────────────────────────────────────

download_gazefollow() {
    local dest="$DATA_DIR/Gazefollow"
    # Check for actual annotation file, not just the directory (guards against partial extractions)
    [[ -f "$dest/data_new/train_annotations.txt" ]] && { skip "GazeFollow already present"; return; }
    info "GazeFollow — downloading (~2.6 GB)..."
    rm -rf "$dest"
    mkdir -p "$dest"
    wget -q --show-progress -O "$dest/data.zip" \
        "http://gazefollow.csail.mit.edu/downloads/data.zip"
    # unzip can fail on zip64 files (>4GB) — use python as fallback
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$dest/data.zip" "$dest" || unzip -q "$dest/data.zip" -d "$dest"
    rm "$dest/data.zip"
    info "GazeFollow done → $dest"
}

download_vocalsound() {
    local dest="$DATA_DIR/VocalSound_release_16k"
    [[ -d "$dest/audio_16k" ]] && { skip "VocalSound already present"; return; }
    info "VocalSound — downloading (~2.5 GB)..."
    wget -q --show-progress -O "$DATA_DIR/vocalsound_16k.zip" \
        "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1"
    mkdir -p "$dest"
    unzip -q "$DATA_DIR/vocalsound_16k.zip" -d "$dest"
    rm "$DATA_DIR/vocalsound_16k.zip"
    info "VocalSound done → $dest"
}

download_voxconverse() {
    local dest="$DATA_DIR/voxconverse"
    [[ -d "$dest/dev" ]] && { skip "VoxConverse already present"; return; }
    info "VoxConverse — cloning annotations..."
    git clone --depth 1 https://github.com/joonson/voxconverse "$dest/annotations"
    info "VoxConverse — downloading dev WAVs (~1.4 GB)..."
    wget -q --show-progress -O "$DATA_DIR/vox_dev.zip" \
        "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
    unzip -q "$DATA_DIR/vox_dev.zip" -d "$dest"
    # normalise folder name regardless of what the zip uses
    find "$dest" -maxdepth 1 -type d -name "*dev*" ! -name "dev" -exec mv {} "$dest/dev" \;
    rm "$DATA_DIR/vox_dev.zip"
    info "VoxConverse — downloading test WAVs (~1.4 GB)..."
    wget -q --show-progress -O "$DATA_DIR/vox_test.zip" \
        "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
    unzip -q "$DATA_DIR/vox_test.zip" -d "$dest"
    find "$dest" -maxdepth 1 -type d -name "*test*" ! -name "test" -exec mv {} "$dest/test" \;
    rm "$DATA_DIR/vox_test.zip"
    info "VoxConverse done → $dest"
}

download_meld() {
    local dest="$DATA_DIR/MELD-FAIR"
    [[ -d "$dest/MELD/realigned" ]] && { skip "MELD already processed"; return; }
    info "MELD — cloning MELD-FAIR (includes precomputed CSVs)..."
    git clone --depth 1 https://github.com/knowledgetechnologyuhh/MELD-FAIR "$dest"
    info "MELD — downloading raw clips (~8 GB)..."
    wget -q --show-progress -O "$dest/MELD.Raw.tar.gz" \
        "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
    mkdir -p "$dest/MELD"
    tar -xzf "$dest/MELD.Raw.tar.gz" -C "$dest/MELD"
    rm "$dest/MELD.Raw.tar.gz"
    info "MELD — installing MELD-FAIR dependencies..."
    pip install -q -r "$dest/requirements.txt" 2>/dev/null || \
        pip install -q moviepy pandas tqdm
    info "MELD — running video assembler (this may take a while)..."
    # Run from inside the repo so relative paths in config.py resolve correctly
    (cd "$dest" && python3 realigner/realigned_video_assembler.py)
    info "MELD done → $dest/MELD/realigned/"
    echo ""
    echo "  NOTE: audio extraction (audio/16000/) may require a separate step."
    echo "  Check the MELD-FAIR README if the builder reports missing audio files."
    echo ""
}

download_mustard() {
    local dest="$DATA_DIR/MUStARD"
    [[ -d "$dest/utterances_final" ]] && { skip "MUStARD already present"; return; }
    info "MUStARD — downloading raw video clips (~3 GB)..."
    mkdir -p "$dest"
    wget -q --show-progress -O "$DATA_DIR/mustard_raw.zip" \
        "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
    unzip -q "$DATA_DIR/mustard_raw.zip" -d "$dest"
    # The zip may extract to a subdirectory — flatten if needed
    extracted=$(find "$dest" -maxdepth 1 -mindepth 1 -type d | head -1)
    if [[ -n "$extracted" && "$extracted" != "$dest/utterances_final" ]]; then
        mv "$extracted" "$dest/utterances_final" 2>/dev/null || true
    fi
    rm "$DATA_DIR/mustard_raw.zip"
    info "MUStARD — fetching sarcasm_data.json..."
    wget -q -O "$dest/sarcasm_data.json" \
        "https://raw.githubusercontent.com/soujanyaporia/MUStARD/master/data/sarcasm_data.json"
    info "MUStARD done → $dest"
}

download_urfunny() {
    local dest="$DATA_DIR/UR-FUNNY-V2"
    [[ -d "$dest/urfunny2_videos" ]] && { skip "UR-FUNNY already present"; return; }
    info "UR-FUNNY — downloading videos (~7 GB)..."
    mkdir -p "$dest"
    wget -q --show-progress -O "$DATA_DIR/urfunny_videos.zip" \
        "https://www.dropbox.com/s/lg7kjx0kul3ansq/urfunny2_videos.zip?dl=1"
    unzip -q "$DATA_DIR/urfunny_videos.zip" -d "$dest"
    rm "$DATA_DIR/urfunny_videos.zip"
    info "UR-FUNNY — downloading SDK features / annotations..."
    wget -q --show-progress -O "$DATA_DIR/urfunny_sdk.zip" \
        "https://www.dropbox.com/sh/9h0pcqmqoplx9p2/AAC8yYikSBVYCSFjm3afFHQva?dl=1"
    mkdir -p "$dest/sdk_features"
    unzip -q "$DATA_DIR/urfunny_sdk.zip" -d "$dest/sdk_features"
    rm "$DATA_DIR/urfunny_sdk.zip"
    info "UR-FUNNY done → $dest"
}

download_rldd() {
    local dest="$DATA_DIR/RealLifeDeceptionDetection.2016"
    [[ -d "$dest/Real-life_Deception_Detection_2016" ]] && { skip "RLDD already present"; return; }
    info "RLDD — downloading (~500 MB)..."
    wget -q --show-progress -O "$DATA_DIR/rldd.zip" \
        "https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip"
    mkdir -p "$dest"
    unzip -q "$DATA_DIR/rldd.zip" -d "$dest"
    rm "$DATA_DIR/rldd.zip"
    info "RLDD done → $dest"
}

download_vat() {
    local dest="$DATA_DIR/videoattentiontarget"
    [[ -d "$dest/annotations" ]] && { skip "VideoAttentionTarget already present"; return; }
    info "VideoAttentionTarget — downloading (~15 GB)..."
    wget -q --show-progress -O "$DATA_DIR/vat.zip" \
        "https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=1"
    unzip -q "$DATA_DIR/vat.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/vat.zip"
    info "VideoAttentionTarget done → $dest"
}

download_proxemics() {
    local dest="$DATA_DIR/dataset_proxemics"
    [[ -d "$dest/images" ]] && { skip "Proxemics already present"; return; }
    info "Proxemics — downloading (~112 MB)..."
    wget -q --show-progress -O "$DATA_DIR/proxemics.zip" \
        "https://zenodo.org/records/11184513/files/dataset_proxemics.zip?download=1"
    unzip -q "$DATA_DIR/proxemics.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/proxemics.zip"
    info "Proxemics done → $dest"
}

download_pisc() {
    local dest="$DATA_DIR/PISC"
    [[ -d "$dest/image" ]] && { skip "PISC already present"; return; }
    info "PISC — downloading (~3.8 GB)..."
    wget -q --show-progress -O "$DATA_DIR/pisc.zip" \
        "https://zenodo.org/records/11184513/files/dataset_pisc.zip?download=1"
    unzip -q "$DATA_DIR/pisc.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/pisc.zip"
    info "PISC done → $dest"
}

download_meview() {
    local dest="$DATA_DIR/MEVIEW"
    [[ -d "$dest/me-cuts/cuts" ]] && { skip "MEVIEW already present"; return; }
    info "MEVIEW — downloading (~500 MB)..."
    mkdir -p "$dest/me-cuts"
    wget -q --show-progress -O "$DATA_DIR/mecuts.zip" \
        "https://cmp.felk.cvut.cz/~cechj/ME/me-cuts.zip"
    unzip -q "$DATA_DIR/mecuts.zip" -d "$dest/me-cuts"
    rm "$DATA_DIR/mecuts.zip"
    info "MEVIEW done → $dest"
}

# ── dispatch ──────────────────────────────────────────────────────────────────

ALL_DATASETS=(gazefollow vocalsound voxconverse meld mustard urfunny rldd vat proxemics pisc meview)

list_datasets() {
    echo "Available datasets:"
    for ds in "${ALL_DATASETS[@]}"; do echo "  $ds"; done
    echo ""
    echo "Gated (not included): affwild2 emotic msppodcast mmew videocoattention"
}

main() {
    require wget
    require unzip
    require git

    if [[ $# -eq 0 || "$1" == "all" ]]; then
        targets=("${ALL_DATASETS[@]}")
    elif [[ "$1" == "--list" ]]; then
        list_datasets; exit 0
    else
        targets=("$@")
    fi

    failed=()
    for ds in "${targets[@]}"; do
        echo ""
        echo "════════════════════════════════════════"
        echo "  ${ds^^}"
        echo "════════════════════════════════════════"
        if ! (
            case "$ds" in
                gazefollow)   download_gazefollow ;;
                vocalsound)   download_vocalsound ;;
                voxconverse)  download_voxconverse ;;
                meld)         download_meld ;;
                mustard)      download_mustard ;;
                urfunny)      download_urfunny ;;
                rldd)         download_rldd ;;
                vat)          download_vat ;;
                proxemics)    download_proxemics ;;
                pisc)         download_pisc ;;
                meview)       download_meview ;;
                *)            echo -e "${RED}[ERROR]${NC}  Unknown dataset: $ds"; exit 1 ;;
            esac
        ); then
            echo -e "${RED}[FAILED]${NC}  $ds — skipping, continuing with next dataset"
            failed+=("$ds")
        fi
    done

    if [[ ${#failed[@]} -gt 0 ]]; then
        echo ""
        echo -e "${RED}[WARN]${NC}  Failed datasets: ${failed[*]}"
    fi

    echo ""
    echo "════════════════════════════════════════"
    echo "  Done. Run: bash run_pipeline.sh"
    echo "════════════════════════════════════════"
}

main "$@"
