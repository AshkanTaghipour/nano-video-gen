#!/bin/bash
# Download the DiffSynth-Studio example video dataset from ModelScope

set -e

DATA_DIR="./data/example_video_dataset"

echo "=== Downloading DiffSynth-Studio Example Video Dataset ==="
echo ""

if [ -d "${DATA_DIR}" ]; then
    echo "Dataset already exists at ${DATA_DIR}"
    echo "To re-download, remove it first: rm -rf ${DATA_DIR}"
    exit 0
fi

# Ensure git-lfs is available
if ! command -v git-lfs &> /dev/null && ! git lfs version &> /dev/null 2>&1; then
    echo "git-lfs not found. Please install it: https://git-lfs.github.com"
    exit 1
fi

git lfs install

echo "Cloning dataset to ${DATA_DIR} (~27 MB)..."
git clone https://www.modelscope.cn/datasets/DiffSynth-Studio/example_video_dataset.git "${DATA_DIR}"

# Ensure LFS files are fully pulled
echo "Pulling LFS files..."
cd "${DATA_DIR}" && git lfs pull && cd -

# Validate: count real video files (not LFS pointers)
VIDEO_COUNT=$(find "${DATA_DIR}" -name "*.mp4" ! -path '*/.git/*' -size +1k | wc -l)
echo ""
echo "=== Download Complete ==="
echo "Dataset saved to: ${DATA_DIR}"
echo "Valid video files: ${VIDEO_COUNT}"
echo ""

if [ "${VIDEO_COUNT}" -eq 0 ]; then
    echo "WARNING: No video files downloaded. LFS pull may have failed."
    echo "You can generate synthetic data instead:"
    echo "  python -m nano_video_gen.data.generate_synthetic"
    exit 1
fi
