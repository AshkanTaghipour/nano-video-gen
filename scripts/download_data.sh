#!/bin/bash
# Download the DiffSynth-Studio example video dataset from ModelScope

set -e

DATA_DIR="./data/example_video_dataset"

echo "=== Downloading Example Video Dataset ==="
echo ""

# Check if modelscope CLI is available
if ! command -v modelscope &> /dev/null; then
    echo "modelscope CLI not found. Installing..."
    pip install modelscope
fi

echo "Downloading dataset to ${DATA_DIR}..."
modelscope download --dataset DiffSynth-Studio/example_video_dataset --local_dir "${DATA_DIR}"

echo ""
echo "=== Download Complete ==="
echo "Dataset saved to: ${DATA_DIR}"
echo ""
echo "If this fails, you can generate synthetic data instead:"
echo "  python -m nano_video_gen.data.generate_synthetic"
