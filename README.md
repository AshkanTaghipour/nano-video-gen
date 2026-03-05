# Nano Video Generation Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AshkanTaghipour/nano-video-gen/blob/main/notebooks/nano_video_gen_colab.ipynb)

A standalone, educational codebase that teaches the concepts behind modern video diffusion models (Wan 2.1/2.2 architecture) with minimal code.

Every architectural concept from Wan 14B is preserved in a ~1.5M parameter "nano" model that you can train on a single GPU in under an hour.

## Architecture: Wan 14B vs Nano

| Component | Wan 14B | Wan 1.3B | **Nano** |
|-----------|---------|----------|----------|
| DiT dim | 5120 | 1536 | **128** |
| Attention heads | 40 | 12 | **4** |
| Transformer layers | 40 | 30 | **2** |
| FFN dim | 13824 | 8960 | **512** |
| Latent channels | 16 | 16 | **4** |
| Text dim | 4096 | 4096 | **64** |
| Freq dim | 256 | 256 | **64** |
| Patch size | [1,2,2] | [1,2,2] | **[1,2,2]** |
| Parameters | ~14B | ~1.3B | **~1.5M** |

Every layer type is preserved:
- Sinusoidal time embedding + MLP
- Text embedding projection (Linear + GELU + Linear)
- Time projection to 6 modulation params per block
- 3D RoPE (temporal + height + width)
- Self-attention with Q/K RMSNorm
- Cross-attention for text conditioning
- AdaIN modulation (shift/scale/gate)
- Feed-forward network (Linear + GELU + Linear)
- Learnable per-block modulation parameters
- Output head with modulation + unpatchify

## Quick Start

### Option A: Run in Google Colab (no local setup needed)

Click the **Open in Colab** badge above to launch the all-in-one notebook. It covers every section of this tutorial -- from VAE basics to training and inference -- in a single runnable notebook. Works with Colab's free T4 GPU.

### Option B: Run Locally

#### 1. Setup Environment

```bash
bash setup_env.sh
conda activate ./.conda_envs/tut_vide_gen
```

#### 2. Get Data

**Option A:** Download DiffSynth-Studio example dataset
```bash
bash scripts/download_data.sh
```

**Option B:** Generate synthetic moving-shapes dataset
```bash
python -m nano_video_gen.data.generate_synthetic
```

#### 3. Train

```bash
python scripts/train.py --data_dir ./data/synthetic_dataset --epochs 50
```

#### 4. Generate

```bash
python scripts/generate.py --checkpoint outputs/checkpoint_final.pt --num_samples 4
```

#### 5. Explore Notebooks

```bash
jupyter notebook notebooks/
```

## Notebooks

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 1 | [Video Gen Overview](notebooks/01_video_gen_overview.ipynb) | Big picture: VAE + Text Encoder + Diffusion Model pipeline |
| 2 | [VAE & Latent Space](notebooks/02_vae_latent_space.ipynb) | Why compress video? Encode/decode demo. Latent visualization |
| 3 | [Building Blocks](notebooks/03_building_blocks.ipynb) | Every layer explained: RoPE, RMSNorm, AdaIN, attention, FFN |
| 4 | [DiT Architecture](notebooks/04_dit_architecture.ipynb) | Full model assembly. Data flow. Parameter counting |
| 5 | [Flow Matching](notebooks/05_flow_matching.ipynb) | DDPM vs flow matching. Noise schedule. Training objective |
| 6 | [Training](notebooks/06_training.ipynb) | Train on toy data. Loss curves. Intermediate generations |
| 7 | [Inference](notebooks/07_inference.ipynb) | Sampling loop. Denoising visualization. CFG explanation |

## Project Structure

```
nano-video-gen/
├── README.md
├── requirements.txt
├── setup_env.sh
├── nano_video_gen/
│   ├── model/
│   │   ├── components.py          # RMSNorm, RoPE, sinusoidal embedding, modulation
│   │   ├── attention.py           # Self-attention, cross-attention
│   │   ├── dit_block.py           # DiT block (self-attn + cross-attn + FFN + modulation)
│   │   ├── nano_dit.py            # Full NanoDiT model
│   │   └── nano_vae.py            # Simplified video VAE
│   ├── diffusion/
│   │   └── flow_match.py          # Flow matching scheduler + training loss
│   ├── data/
│   │   ├── dataset.py             # Video dataset loader + simple text encoder
│   │   └── generate_synthetic.py  # Synthetic moving-shapes data generator
│   └── visualization/
│       └── viz.py                 # Attention maps, RoPE, denoising, loss curves
├── notebooks/                     # 7 tutorial notebooks
├── scripts/
│   ├── train.py                   # CLI training script
│   ├── generate.py                # CLI inference script
│   └── download_data.sh           # Dataset download script
└── outputs/                       # Generated samples and checkpoints
```

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support (CPU works but is slow)
- ~2GB GPU memory (model is tiny)

## Pipeline Overview

```
Text Prompt ──→ [Text Encoder] ──→ text embeddings
                                        │
                                        ▼
Random Noise ──→ [DiT Denoising Loop] ──→ denoised latent
                  ↑ timestep embedding
                                        │
                                        ▼
                  [VAE Decoder] ──→ Generated Video
```

## Based On

This tutorial is based on the [Wan 2.1](https://github.com/Wan-Video/Wan2.1) video generation model architecture, as implemented in [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). No DiffSynth-Studio dependency is required.

## License

Educational use. Based on publicly available model architectures.
