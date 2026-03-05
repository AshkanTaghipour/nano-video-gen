"""
Attention modules for the Nano DiT model.

Implements self-attention (with RoPE) and cross-attention (for text conditioning).
These follow the Wan 2.1 architecture where Q and K are RMSNorm'd before attention.

Reference: DiffSynth-Studio/diffsynth/models/wan_video_dit.py lines 126-188
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .components import RMSNorm, rope_apply


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and Q/K normalization.

    In Wan's architecture, self-attention allows each video patch to attend
    to every other patch (including across frames), learning spatial and
    temporal relationships.

    Key design choices from Wan:
    1. Q and K are RMSNorm'd before attention (stabilizes training)
    2. RoPE is applied after normalization (encodes 3D position)
    3. Uses standard scaled dot-product attention

    Wan 14B: dim=5120, 40 heads → head_dim=128
    Nano:    dim=128,  4 heads  → head_dim=32

    Reference: wan_video_dit.py lines 126-149
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q, K, V projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        # Output projection
        self.o = nn.Linear(dim, dim)
        # RMSNorm on Q and K (following LLaMA/Wan convention)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, S, dim) where S = num_patches
            freqs: RoPE frequencies of shape (S, 1, head_dim//2)

        Returns:
            Output tensor of shape (B, S, dim)
        """
        # Project to Q, K, V
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Apply 3D RoPE to Q and K
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        # Multi-head attention using PyTorch's efficient implementation
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)

        return self.o(x)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for text conditioning.

    This is how text prompts guide video generation. The video patches (Q)
    attend to text embeddings (K, V), allowing each spatial-temporal location
    to pull in relevant semantic information from the prompt.

    Key design choices from Wan:
    1. Q comes from video features, K/V from text embeddings
    2. Q is RMSNorm'd (like self-attention)
    3. K is RMSNorm'd (stabilizes cross-attention)
    4. No RoPE (text has its own positional encoding)

    Wan 14B: dim=5120, text_dim=4096 (projected to dim)
    Nano:    dim=128,  text_dim=64   (projected to dim)

    Reference: wan_video_dit.py lines 152-188
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q from video, K/V from text
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # RMSNorm on Q and K
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video features of shape (B, S_video, dim)
            context: Text embeddings of shape (B, S_text, dim)
                     (already projected to model dim by text_embedding layer)

        Returns:
            Output tensor of shape (B, S_video, dim)
        """
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # Multi-head attention
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)

        return self.o(x)
