"""
Building blocks for the Nano DiT model.

These components mirror the architecture of Wan 2.1's DiT but at a much smaller scale.
Each component is annotated with its purpose in the video generation pipeline.

Reference: DiffSynth-Studio/diffsynth/models/wan_video_dit.py lines 55-113
"""

import torch
import torch.nn as nn
import math


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings for timestep conditioning.

    This encodes a scalar timestep into a high-dimensional vector using
    sin/cos functions at different frequencies, allowing the model to
    distinguish between different noise levels.

    Args:
        dim: Embedding dimension (must be even)
        position: Timestep values, shape (B,)

    Returns:
        Embeddings of shape (B, dim) with [cos, sin] concatenated
    """
    # Create frequency bands: 10000^(-i/(dim/2)) for i in [0, dim/2)
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)
        )
    )
    # Concatenate cos and sin components
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute 1D Rotary Position Embedding (RoPE) frequency table.

    RoPE encodes position information by rotating pairs of dimensions
    in the query/key vectors. This function creates the complex-valued
    rotation factors for each position.

    Args:
        dim: Dimension per head (will use dim//2 frequency pairs)
        end: Maximum sequence length
        theta: Base frequency (10000 in the original paper)

    Returns:
        Complex tensor of shape (end, dim//2) containing rotation factors
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*theta)
    return freqs_cis


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    Precompute 3D RoPE frequencies for video (temporal + height + width).

    Video tokens have 3 spatial dimensions. We split the head dimension
    into 3 parts and compute separate RoPE frequencies for each axis:
    - Temporal (frames): uses dim - 2*(dim//3) dimensions
    - Height: uses dim//3 dimensions
    - Width: uses dim//3 dimensions

    This allows the model to learn relative positions along each axis
    independently, which is crucial for video understanding.

    Args:
        dim: Head dimension to split across 3 axes
        end: Maximum positions per axis
        theta: Base frequency

    Returns:
        Tuple of (temporal_freqs, height_freqs, width_freqs)
    """
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings (RoPE) to query or key tensors.

    Rotates pairs of dimensions in Q/K by position-dependent angles,
    so that the dot product Q·K naturally encodes relative position.

    Args:
        x: Input tensor of shape (B, S, num_heads * head_dim)
        freqs: Complex rotation factors of shape (S, 1, head_dim//2)
        num_heads: Number of attention heads

    Returns:
        Rotated tensor with same shape as x
    """
    from einops import rearrange
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # View as complex pairs and multiply by rotation factors
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center the activations (no mean subtraction).
    It only normalizes by the RMS value, making it simpler and slightly faster.

    Used in Wan DiT for normalizing Q and K in attention (following LLaMA convention).

    Wan 14B: dim=5120, Nano: dim=128

    Reference: wan_video_dit.py lines 102-113
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Adaptive Instance Normalization (AdaIN) style modulation.

    This is how the timestep information controls each layer's behavior.
    The shift and scale are predicted from the timestep embedding, allowing
    the model to behave differently at different noise levels.

    Formula: output = x * (1 + scale) + shift

    When scale=0 and shift=0, output = x (identity).
    This allows the modulation to start as a no-op and learn to adjust.

    Reference: wan_video_dit.py line 65-66
    """
    return x * (1 + scale) + shift


class GateModule(nn.Module):
    """
    Gating mechanism for residual connections.

    Instead of a simple residual x + f(x), uses a learned gate:
        output = x + gate * f(x)

    The gate values are predicted from the timestep, allowing the model
    to control how much each layer contributes at each noise level.
    Early in denoising (high noise), gates might be small; later they
    might be large for fine detail adjustment.

    Reference: wan_video_dit.py lines 191-196
    """

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual
