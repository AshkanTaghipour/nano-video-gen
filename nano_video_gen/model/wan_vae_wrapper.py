"""
WanVAEWrapper: Wraps the pretrained Wan 2.1 VAE for use with NanoDiT.

The real Wan 2.1 VAE uses CausalConv3d layers, ResidualBlocks with RMSNorm,
AttentionBlocks in the middle, and produces 16-channel latents with 8x spatial
and 4x temporal compression.

This wrapper:
- Auto-downloads VAE weights from Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- Keeps the VAE on CPU (frozen, eval mode) to save GPU VRAM
- Provides encode()/decode() with per-channel normalization (latents_mean/latents_std)
- Handles CPU<->GPU device transfers transparently

Compression: [B, 3, 17, 128, 128] -> encode -> [B, 16, 5, 16, 16] -> decode -> [B, 3, 17, 128, 128]
  - Spatial: 8x (128 -> 16)
  - Temporal: (T-1)//4 + 1 (17 -> 5)
  - Channels: 3 -> 16
"""

import os
import torch
import torch.nn as nn


def _ensure_downloaded(model_path):
    """Download VAE weights from HuggingFace if not already present."""
    vae_dir = os.path.join(model_path, "vae")
    if not os.path.exists(vae_dir):
        from huggingface_hub import snapshot_download
        print(f"Downloading Wan 2.1 VAE weights to {model_path}...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            local_dir=model_path,
            allow_patterns=["vae/*"],
        )
        print("VAE download complete.")


class WanVAEWrapper(nn.Module):
    """
    Wrapper around the pretrained Wan 2.1 VAE (AutoencoderKLWan).

    The VAE is kept frozen on CPU to save GPU memory. Input tensors are
    moved to CPU for encoding/decoding, then results are moved back to
    the original device.

    Normalization uses per-channel latents_mean and latents_std from the
    VAE config (not a single scaling_factor like standard AutoencoderKL).

    Args:
        model_path: Directory containing (or to download to) the pretrained weights.
                    Defaults to ./pretrained_models/Wan2.1
        device: Device to keep the VAE on. Defaults to "cpu".
    """

    def __init__(self, model_path="./pretrained_models/Wan2.1", device="cpu"):
        super().__init__()
        _ensure_downloaded(model_path)

        from diffusers import AutoencoderKLWan

        self.device = torch.device(device)
        self.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
        self.vae = self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Per-channel normalization (shape: [1, z_dim, 1, 1, 1])
        mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        self.register_buffer("latents_mean", mean)
        self.register_buffer("latents_std", std)

    @property
    def latent_channels(self):
        return self.vae.config.z_dim

    @torch.no_grad()
    def encode(self, x):
        """
        Encode video to latent space.

        Args:
            x: Video tensor [B, 3, T, H, W] on any device, values in [-1, 1]

        Returns:
            Latent tensor [B, 16, T', H', W'] on the same device as input
            where T'=(T-1)//4+1, H'=H//8, W'=W//8
        """
        input_device = x.device
        x_cpu = x.to(self.device, dtype=torch.float32)
        latent = self.vae.encode(x_cpu).latent_dist.mode()
        # Per-channel normalization: (latent - mean) * (1/std)
        latent = (latent - self.latents_mean.to(latent.device)) / self.latents_std.to(latent.device)
        return latent.to(input_device)

    @torch.no_grad()
    def decode(self, z):
        """
        Decode latent back to video.

        Args:
            z: Latent tensor [B, 16, T', H', W'] on any device

        Returns:
            Video tensor [B, 3, T, H, W] on the same device as input,
            values in [-1, 1]
        """
        input_device = z.device
        z_cpu = z.to(self.device, dtype=torch.float32)
        # De-normalize: latent * std + mean
        z_cpu = z_cpu * self.latents_std.to(z_cpu.device) + self.latents_mean.to(z_cpu.device)
        video = self.vae.decode(z_cpu).sample
        return video.clamp(-1, 1).to(input_device)
