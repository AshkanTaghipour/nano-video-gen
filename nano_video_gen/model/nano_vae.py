"""
DummyVAE: A simplified Video Autoencoder for educational purposes.

The real Wan VAE uses:
- CausalConv3d layers for temporal causality
- ResidualBlocks with RMSNorm at each stage
- AttentionBlocks in the middle
- 4 downsampling stages (8x spatial, 4x temporal)
- 16 latent channels with mean/std normalization

Our DummyVAE preserves the dimensional concept:
- [B, 3, 16, 64, 64] → encode → [B, 4, 4, 16, 16] → decode → [B, 3, 16, 64, 64]
- Same compression ratios: 4x spatial, 4x temporal
- Simple Conv3d encoder/decoder (no causal convolutions)
- Latent normalization (mean/std like Wan)

This is intentionally simplified - the tutorial explains the real architecture side-by-side.

Reference: DiffSynth-Studio/diffsynth/models/wan_video_vae.py
"""

import torch
import torch.nn as nn


class DummyVAE(nn.Module):
    """
    Simplified Video VAE matching Wan's compression concept.

    Real Wan VAE architecture (for comparison):
    ┌─────────────────────────────────────────────────┐
    │ Encoder:                                         │
    │   CausalConv3d(3→96) → [ResBlock(96→96)×2       │
    │   + Downsample(spatial)]                         │
    │   → [ResBlock(96→192)×2 + Downsample(s+t)]      │
    │   → [ResBlock(192→384)×2 + Downsample(s+t)]     │
    │   → [ResBlock(384→384)×2]                        │
    │   → Middle(ResBlock + AttnBlock + ResBlock)       │
    │   → Head(RMSNorm + SiLU + CausalConv(384→32))   │
    │   → chunk → mean, log_var                        │
    │   → reparameterize → z (16 channels)             │
    └─────────────────────────────────────────────────┘

    Our DummyVAE:
    ┌─────────────────────────────────────────────────┐
    │ Encoder: Conv3d(3→32) → SiLU → Conv3d(32→64,    │
    │   stride=[4,4,4]) → SiLU → Conv3d(64→8)         │
    │   → chunk → mean, log_var → reparameterize       │
    │                                                   │
    │ Decoder: Conv3d(4→64) → SiLU → ConvTranspose3d   │
    │   (64→32, stride=[4,4,4]) → SiLU → Conv3d(32→3)  │
    │   → Tanh                                          │
    └─────────────────────────────────────────────────┘

    Compression: [B,3,16,64,64] → [B,4,4,16,16] (768x compression)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        spatial_factor: int = 4,
        temporal_factor: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor

        stride = [temporal_factor, spatial_factor, spatial_factor]
        kernel = [temporal_factor, spatial_factor, spatial_factor]

        # Encoder: video → latent (outputs 2x channels for mean + log_var)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=kernel, stride=stride),
            nn.SiLU(),
            nn.Conv3d(64, latent_channels * 2, kernel_size=3, padding=1),
        )

        # Decoder: latent → video
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=kernel, stride=stride),
            nn.SiLU(),
            nn.Conv3d(32, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Latent normalization statistics (like Wan's mean/std scaling)
        self.register_buffer('latent_mean', torch.zeros(1, latent_channels, 1, 1, 1))
        self.register_buffer('latent_std', torch.ones(1, latent_channels, 1, 1, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.

        Args:
            x: Video tensor, shape [B, 3, T, H, W] with values in [-1, 1]
               e.g., [B, 3, 16, 64, 64]

        Returns:
            Latent tensor, shape [B, latent_channels, T//tf, H//sf, W//sf]
            e.g., [B, 4, 4, 16, 16]
        """
        h = self.encoder(x)
        # Split into mean and log_variance (VAE reparameterization)
        mean, log_var = h.chunk(2, dim=1)

        if self.training:
            # Reparameterization trick: z = mean + std * epsilon
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean

        # Normalize latents (like Wan's mean/std scaling)
        z = (z - self.latent_mean) / (self.latent_std + 1e-6)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent back to video.

        Args:
            z: Latent tensor, shape [B, latent_channels, T', H', W']
               e.g., [B, 4, 4, 16, 16]

        Returns:
            Reconstructed video, shape [B, 3, T, H, W] with values in [-1, 1]
            e.g., [B, 3, 16, 64, 64]
        """
        # Denormalize latents
        z = z * (self.latent_std + 1e-6) + self.latent_mean
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Full encode-decode pass (for VAE training).

        Returns:
            (reconstruction, mean, log_var) for computing reconstruction + KL loss
        """
        h = self.encoder(x)
        mean, log_var = h.chunk(2, dim=1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + std * eps
        z_norm = (z - self.latent_mean) / (self.latent_std + 1e-6)
        z_denorm = z_norm * (self.latent_std + 1e-6) + self.latent_mean
        recon = self.decoder(z_denorm)
        return recon, mean, log_var
