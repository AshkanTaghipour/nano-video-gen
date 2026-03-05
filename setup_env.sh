#!/bin/bash
# Setup script for Nano Video Generation Tutorial
# Creates a conda environment with all required dependencies

set -e

ENV_NAME="tut_vide_gen"
ENV_DIR="$(pwd)/.conda_envs/${ENV_NAME}"

echo "=== Nano Video Generation Tutorial - Environment Setup ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment in the current directory
echo "Creating conda environment '${ENV_NAME}' at ${ENV_DIR}..."
conda create -y -p "${ENV_DIR}" python=3.10

echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

# Install PyTorch (CUDA 12.1)
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
echo ""
echo "Installing additional dependencies..."
pip install -r requirements.txt

# Register Jupyter kernel
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_DIR}"
echo ""
echo "To download the example dataset:"
echo "  bash scripts/download_data.sh"
echo ""
echo "To start the notebooks:"
echo "  jupyter notebook notebooks/"
