"""
DiT Block: The core transformer block used in Wan's video diffusion model.

Each block contains:
1. Self-attention (with AdaIN modulation) - learns spatial-temporal relationships
2. Cross-attention - conditions on text embeddings
3. Feed-forward network (with AdaIN modulation) - non-linear transformation

The timestep controls behavior via 6 modulation parameters per block:
  [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]

Reference: DiffSynth-Studio/diffsynth/models/wan_video_dit.py lines 198-232
"""

import torch
import torch.nn as nn

from .components import modulate, GateModule
from .attention import SelfAttention, CrossAttention


class DiTBlock(nn.Module):
    """
    A single Diffusion Transformer block.

    Architecture per block:
        x_mod = AdaIN(LayerNorm(x), shift_msa, scale_msa)
        x = x + gate_msa * SelfAttention(x_mod, RoPE)
        x = x + CrossAttention(LayerNorm(x), text_context)
        x_mod = AdaIN(LayerNorm(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * FFN(x_mod)

    The 6 modulation params (shift/scale/gate for MSA and MLP) are computed as:
        learnable_per_block_param + timestep_projection

    This allows each block to have its own "personality" (via the learnable param)
    while also responding to the current noise level (via timestep projection).

    Wan 14B: dim=5120, ffn_dim=13824, 40 heads, 40 blocks
    Nano:    dim=128,  ffn_dim=512,   4 heads,  2 blocks

    Reference: wan_video_dit.py lines 198-232
    """

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        # Attention layers
        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        # Layer norms
        # norm1, norm2: no affine (modulation handles shift/scale)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # norm3: with affine (cross-attention has no modulation in Wan)
        self.norm3 = nn.LayerNorm(dim, eps=eps)

        # Feed-forward network: Linear → GELU → Linear
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # Learnable per-block modulation parameters (6 params: shift/scale/gate × 2)
        # Initialized with small random values (scaled by 1/sqrt(dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        # Gating module for residual connections
        self.gate = GateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Video patch embeddings, shape (B, S, dim)
            context: Text embeddings, shape (B, S_text, dim)
            t_mod: Timestep modulation, shape (B, 6, dim)
            freqs: 3D RoPE frequencies, shape (S, 1, head_dim//2)

        Returns:
            Updated patch embeddings, shape (B, S, dim)
        """
        # Compute 6 modulation parameters = learnable_base + timestep_signal
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        # 1. Self-attention with AdaIN modulation
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

        # 2. Cross-attention (text conditioning)
        x = x + self.cross_attn(self.norm3(x), context)

        # 3. Feed-forward with AdaIN modulation
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))

        return x
