"""
NanoDiT: A miniature Diffusion Transformer for video generation.

This is the full model that:
1. Patchifies the input latent video using 3D convolution
2. Adds sinusoidal timestep embedding
3. Projects text embeddings to model dimension
4. Runs through N DiT blocks (self-attn + cross-attn + FFN)
5. Applies final modulation and linear head
6. Unpatchifies back to latent video shape

Reference: DiffSynth-Studio/diffsynth/models/wan_video_dit.py lines 274-398
"""

import torch
import torch.nn as nn
import math
from typing import Tuple
from einops import rearrange

from .components import (
    sinusoidal_embedding_1d,
    precompute_freqs_cis_3d,
)
from .dit_block import DiTBlock


class Head(nn.Module):
    """
    Output head with timestep modulation.

    Projects from model dimension to (out_dim × patch_volume), then unpatchify
    recovers the original spatial dimensions.

    Has its own learnable modulation (2 params: shift + scale) separate from blocks.

    Reference: wan_video_dit.py lines 255-271
    """

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        # 2 modulation params: shift and scale for the output
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings, shape (B, S, dim)
            t: Timestep embedding (before projection to 6 params), shape (B, dim)

        Returns:
            Output tokens, shape (B, S, out_dim * prod(patch_size))
        """
        # modulation: (1, 2, dim), t: (B, dim) → unsqueeze to (B, 1, dim)
        # Result: (B, 2, dim) → chunk into shift (B, 1, dim) and scale (B, 1, dim)
        shift, scale = (
            self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1).expand(-1, 2, -1)
        ).chunk(2, dim=1)
        # shift/scale: (B, 1, dim) — broadcasts over sequence dimension S
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class NanoDiT(nn.Module):
    """
    Nano Diffusion Transformer for Video Generation.

    This is a miniature version of Wan 2.1's WanModel, preserving every
    architectural concept but at ~1.5M parameters instead of ~14B.

    Architecture comparison:
    ┌──────────────┬──────────┬──────────┐
    │ Component    │ Wan 14B  │ Nano     │
    ├──────────────┼──────────┼──────────┤
    │ dim          │ 5120     │ 128      │
    │ heads        │ 40       │ 4        │
    │ layers       │ 40       │ 2        │
    │ ffn_dim      │ 13824    │ 512      │
    │ in_dim       │ 16       │ 4        │
    │ text_dim     │ 4096     │ 64       │
    │ freq_dim     │ 256      │ 64       │
    │ patch_size   │ [1,2,2]  │ [1,2,2]  │
    │ ~params      │ ~14B     │ ~1.5M    │
    └──────────────┴──────────┴──────────┘

    Every layer type is preserved:
    - Sinusoidal time embedding → MLP
    - Text embedding projection (Linear → GELU → Linear)
    - Time projection to 6 modulation params per block
    - 3D RoPE (temporal + height + width)
    - Self-attention with Q/K RMSNorm
    - Cross-attention for text conditioning
    - AdaIN modulation (shift/scale/gate)
    - Feed-forward (Linear → GELU → Linear)
    - Learnable per-block modulation parameters
    - Output head with modulation + unpatchify

    Reference: wan_video_dit.py lines 274-397
    """

    def __init__(
        self,
        dim: int = 128,
        in_dim: int = 4,
        ffn_dim: int = 512,
        out_dim: int = 4,
        text_dim: int = 64,
        freq_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        # === Patchify: 3D Conv to embed video patches ===
        # Converts [B, in_dim, T, H, W] → [B, dim, T//pt, H//ph, W//pw]
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        # === Text embedding: project text features to model dimension ===
        # Wan uses a 2-layer MLP: Linear → GELU → Linear
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        # === Time embedding: sinusoidal → MLP ===
        # First sinusoidal_embedding_1d creates the frequency encoding,
        # then this MLP processes it: Linear → SiLU → Linear
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # === Time projection: project time embedding to 6 modulation params ===
        # SiLU → Linear(dim → dim*6)
        # The 6 params per block are: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # === Transformer blocks ===
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])

        # === Output head ===
        self.head = Head(dim, out_dim, patch_size, eps)

        # === Precompute 3D RoPE frequencies ===
        head_dim = dim // num_heads
        freqs = precompute_freqs_cis_3d(head_dim)
        # Register as buffers (not parameters, but saved with model)
        self.register_buffer('freqs_t', freqs[0], persistent=False)
        self.register_buffer('freqs_h', freqs[1], persistent=False)
        self.register_buffer('freqs_w', freqs[2], persistent=False)

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        Convert video tensor to patch sequence.

        [B, C, T, H, W] → Conv3d → [B, dim, f, h, w] → [B, f*h*w, dim]

        Returns:
            x: Patch embeddings, shape (B, num_patches, dim)
            grid_size: (f, h, w) for unpatchify
        """
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]  # (f, h, w)
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Convert patch sequence back to video tensor.

        [B, f*h*w, out_dim*pt*ph*pw] → [B, out_dim, T, H, W]
        """
        f, h, w = grid_size
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def build_rope_freqs(self, f: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Build 3D RoPE frequency tensor for the given grid size.

        Combines temporal, height, and width frequencies into a single tensor.

        Args:
            f, h, w: Grid dimensions (after patchify)
            device: Target device

        Returns:
            Combined frequencies, shape (f*h*w, 1, head_dim//2)
        """
        freqs = torch.cat([
            self.freqs_t[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(device)
        return freqs

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Nano DiT model.

        Args:
            x: Noisy latent video, shape (B, in_dim, T, H, W)
               e.g., (1, 4, 4, 16, 16) for nano
            timestep: Diffusion timestep, shape (B,)
            context: Text embeddings, shape (B, seq_len, text_dim)

        Returns:
            Predicted velocity/noise, shape (B, out_dim, T, H, W)
        """
        # 1. Time embedding: scalar timestep → vector
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )

        # 2. Project time embedding to 6 modulation params per block
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # 3. Project text embeddings to model dimension
        context = self.text_embedding(context)

        # 4. Patchify: video → patch sequence
        x, (f, h, w) = self.patchify(x)

        # 5. Build 3D RoPE frequencies for this grid size
        freqs = self.build_rope_freqs(f, h, w, x.device)

        # 6. Run through transformer blocks
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        # 7. Output head with modulation
        x = self.head(x, t)

        # 8. Unpatchify: patch sequence → video
        x = self.unpatchify(x, (f, h, w))

        return x
