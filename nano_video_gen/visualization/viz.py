"""
Visualization utilities for the Nano Video Generation tutorial.

Provides functions to visualize:
- Attention maps (self and cross)
- RoPE frequency patterns
- FFN activations
- Modulation effects
- Denoising process
- Latent space
- Noise schedule
- Data flow through the model
- Training curves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Dict


def visualize_attention_maps(
    attn_weights: torch.Tensor,
    title: str = "Attention Map",
    num_heads_to_show: int = 4,
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """
    Visualize attention weight matrices as heatmaps.

    Args:
        attn_weights: Attention weights, shape (B, num_heads, S, S) or (num_heads, S, S)
        title: Plot title
        num_heads_to_show: Max number of attention heads to display
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # Take first batch
    attn_weights = attn_weights.detach().cpu().float().numpy()

    n_heads = min(attn_weights.shape[0], num_heads_to_show)
    fig, axes = plt.subplots(1, n_heads, figsize=figsize)
    if n_heads == 1:
        axes = [axes]

    for i in range(n_heads):
        im = axes[i].imshow(attn_weights[i], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Head {i}')
        axes[i].set_xlabel('Key position')
        axes[i].set_ylabel('Query position')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def visualize_rope_frequencies(
    freqs: tuple,
    max_pos: int = 32,
    figsize: tuple = (15, 4),
) -> plt.Figure:
    """
    Visualize 3D RoPE frequency patterns for temporal, height, and width axes.

    Args:
        freqs: Tuple of (temporal_freqs, height_freqs, width_freqs)
        max_pos: Number of positions to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    labels = ['Temporal (frames)', 'Height', 'Width']
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, (freq, label) in enumerate(zip(freqs, labels)):
        # Extract rotation angles (not magnitude, which is always 1 for unit complex numbers)
        freq_np = freq[:max_pos].angle().cpu().float().numpy()
        im = axes[i].imshow(freq_np, cmap='plasma', aspect='auto')
        axes[i].set_title(f'RoPE: {label}')
        axes[i].set_xlabel('Frequency pair index')
        axes[i].set_ylabel('Position')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04, label='Angle (rad)')

    fig.suptitle('3D Rotary Position Embeddings', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_ffn_activations(
    activations: torch.Tensor,
    title: str = "FFN Activations",
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Visualize FFN activation distributions as histograms.

    Args:
        activations: FFN output tensor (any shape)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    acts = activations.detach().cpu().float().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(acts, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_title('Activation Distribution')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Statistics
    stats_text = f'Mean: {acts.mean():.4f}\nStd: {acts.std():.4f}\nMin: {acts.min():.4f}\nMax: {acts.max():.4f}'
    axes[1].text(0.5, 0.5, stats_text, transform=axes[1].transAxes,
                 fontsize=14, verticalalignment='center', horizontalalignment='center',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].set_title('Statistics')
    axes[1].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def visualize_modulation_effect(
    x_before: torch.Tensor,
    x_after: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Visualize the effect of AdaIN modulation (before vs after).

    Args:
        x_before: Input before modulation
        x_after: Output after modulation
        shift: Shift parameter
        scale: Scale parameter
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    before = x_before.detach().cpu().float().numpy().flatten()
    after = x_after.detach().cpu().float().numpy().flatten()
    shift_np = shift.detach().cpu().float().numpy().flatten()
    scale_np = scale.detach().cpu().float().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Before/After histograms
    axes[0].hist(before, bins=50, alpha=0.6, label='Before', color='blue')
    axes[0].hist(after, bins=50, alpha=0.6, label='After', color='red')
    axes[0].set_title('Distribution Shift')
    axes[0].legend()
    axes[0].set_xlabel('Value')

    # Shift values
    axes[1].bar(range(min(len(shift_np), 32)), shift_np[:32], color='orange', alpha=0.7)
    axes[1].set_title('Shift Parameters')
    axes[1].set_xlabel('Dimension')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Scale values
    axes[2].bar(range(min(len(scale_np), 32)), scale_np[:32], color='green', alpha=0.7)
    axes[2].set_title('Scale Parameters (1 + scale)')
    axes[2].set_xlabel('Dimension')
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)

    fig.suptitle('AdaIN Modulation Effect: x * (1 + scale) + shift', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_denoising_process(
    frames_at_each_step: List[torch.Tensor],
    frame_idx: int = 0,
    figsize: tuple = (16, 4),
) -> plt.Figure:
    """
    Visualize the denoising progression as a grid.

    Args:
        frames_at_each_step: List of video tensors at each denoising step,
                             each shape (B, C, T, H, W) or (C, T, H, W)
        frame_idx: Which temporal frame to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_steps = len(frames_at_each_step)
    # Show at most 10 steps evenly spaced
    max_show = min(n_steps, 10)
    indices = np.linspace(0, n_steps - 1, max_show, dtype=int)

    fig, axes = plt.subplots(1, max_show, figsize=figsize)
    if max_show == 1:
        axes = [axes]

    for ax_idx, step_idx in enumerate(indices):
        frame = frames_at_each_step[step_idx]
        if frame.dim() == 5:
            frame = frame[0]  # First batch
        # Get the specified temporal frame: (C, T, H, W) → (C, H, W)
        img = frame[:, min(frame_idx, frame.shape[1] - 1)]
        img = img.detach().cpu().float()
        # Normalize to [0, 1] for display
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()

        axes[ax_idx].imshow(img)
        axes[ax_idx].set_title(f'Step {step_idx}')
        axes[ax_idx].axis('off')

    fig.suptitle(f'Denoising Process (frame {frame_idx})', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_latent_space(
    latents: torch.Tensor,
    title: str = "Latent Space",
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """
    Visualize VAE latent channels.

    Args:
        latents: Latent tensor, shape (B, C, T, H, W) or (C, T, H, W)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if latents.dim() == 5:
        latents = latents[0]  # First batch
    latents = latents.detach().cpu().float()

    n_channels = latents.shape[0]
    # Use 2-row grid for many channels (e.g. 16), single row for few (e.g. 4)
    if n_channels > 8:
        ncols = (n_channels + 1) // 2
        nrows = 2
        if figsize == (12, 3):
            figsize = (max(12, ncols * 1.5), 6)
    else:
        ncols = n_channels
        nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for c in range(n_channels):
        img = latents[c, 0]  # First frame
        im = axes_flat[c].imshow(img.numpy(), cmap='coolwarm', aspect='equal')
        axes_flat[c].set_title(f'Ch {c}', fontsize=9)
        axes_flat[c].axis('off')

    # Hide unused axes
    for c in range(n_channels, len(axes_flat)):
        axes_flat[c].axis('off')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def visualize_noise_schedule(
    scheduler,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Visualize the noise schedule (sigma vs timestep).

    Args:
        scheduler: FlowMatchScheduler with timesteps set
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    timesteps = scheduler.timesteps.numpy()
    sigmas = scheduler.sigmas.numpy()

    # Sigma vs step index
    axes[0].plot(sigmas, 'b-o', markersize=3)
    axes[0].set_title('Sigma Schedule')
    axes[0].set_xlabel('Step index')
    axes[0].set_ylabel('Sigma (noise level)')
    axes[0].grid(True, alpha=0.3)

    # Timestep vs step index
    axes[1].plot(timesteps, 'r-o', markersize=3)
    axes[1].set_title('Timestep Schedule')
    axes[1].set_xlabel('Step index')
    axes[1].set_ylabel('Timestep (0-1000)')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Flow Matching Noise Schedule', fontsize=14)
    plt.tight_layout()
    return fig


def visualize_data_flow(
    model,
    sample_input: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Trace and visualize tensor shapes through each layer of the model.

    Args:
        model: NanoDiT model
        sample_input: Sample input tensor (B, C, T, H, W)
        timestep: Timestep tensor (B,)
        context: Text context tensor (B, seq_len, text_dim)
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    from ..model.components import sinusoidal_embedding_1d

    shapes = []
    labels = []

    with torch.no_grad():
        x = sample_input
        shapes.append(list(x.shape))
        labels.append(f'Input\n{list(x.shape)}')

        # Time embedding
        t_emb = model.time_embedding(
            sinusoidal_embedding_1d(model.freq_dim, timestep).to(x.dtype)
        )
        shapes.append(list(t_emb.shape))
        labels.append(f'Time Emb\n{list(t_emb.shape)}')

        # Time projection
        t_mod = model.time_projection(t_emb).unflatten(1, (6, model.dim))
        shapes.append(list(t_mod.shape))
        labels.append(f'Time Proj\n{list(t_mod.shape)}')

        # Text embedding
        ctx = model.text_embedding(context)
        shapes.append(list(ctx.shape))
        labels.append(f'Text Emb\n{list(ctx.shape)}')

        # Patchify
        x_patch, (f, h, w) = model.patchify(x)
        shapes.append(list(x_patch.shape))
        labels.append(f'Patchify\n{list(x_patch.shape)}')

        # Each block
        freqs = model.build_rope_freqs(f, h, w, x.device)
        x_block = x_patch
        for i, block in enumerate(model.blocks):
            x_block = block(x_block, ctx, t_mod, freqs)
            shapes.append(list(x_block.shape))
            labels.append(f'Block {i}\n{list(x_block.shape)}')

        # Head
        x_head = model.head(x_block, t_emb)
        shapes.append(list(x_head.shape))
        labels.append(f'Head\n{list(x_head.shape)}')

        # Unpatchify
        x_out = model.unpatchify(x_head, (f, h, w))
        shapes.append(list(x_out.shape))
        labels.append(f'Output\n{list(x_out.shape)}')

    # Create flow diagram
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n = len(labels)
    y = 0.5
    x_positions = np.linspace(0.05, 0.95, n)

    for i, (xp, label) in enumerate(zip(x_positions, labels)):
        color = 'lightblue'
        if 'Input' in label or 'Output' in label:
            color = 'lightyellow'
        elif 'Block' in label:
            color = 'lightgreen'
        elif 'Emb' in label or 'Proj' in label:
            color = 'lightsalmon'

        ax.annotate(
            label, (xp, y),
            fontsize=7, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='gray'),
        )
        if i < n - 1:
            ax.annotate(
                '', xy=(x_positions[i + 1] - 0.02, y),
                xytext=(xp + 0.02, y),
                arrowprops=dict(arrowstyle='->', color='gray'),
            )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('NanoDiT Data Flow', fontsize=14)
    plt.tight_layout()
    return fig


def save_video_grid(
    videos: List[torch.Tensor],
    path: str,
    nrow: int = 4,
    fps: int = 8,
):
    """
    Save multiple videos as a GIF grid.

    Args:
        videos: List of video tensors, each (C, T, H, W) in [-1, 1]
        path: Output GIF path
        nrow: Number of videos per row
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("imageio required for saving video grids")
        return

    if not videos:
        return

    n = len(videos)
    ncol = min(n, nrow)
    nrows = (n + ncol - 1) // ncol

    # Get dimensions from first video
    C, T, H, W = videos[0].shape

    gif_frames = []
    for t in range(T):
        # Create grid for this frame
        grid = np.zeros((nrows * H, ncol * W, 3), dtype=np.uint8)
        for idx, video in enumerate(videos):
            row = idx // ncol
            col = idx % ncol
            frame = video[:, t].detach().cpu().float()
            frame = ((frame + 1) / 2 * 255).clamp(0, 255).byte()
            frame = frame.permute(1, 2, 0).numpy()
            grid[row * H:(row + 1) * H, col * W:(col + 1) * W] = frame

        gif_frames.append(grid)

    imageio.mimsave(path, gif_frames, fps=fps, loop=0)
    print(f"Saved video grid to {path}")


def plot_training_curves(
    losses: List[float],
    title: str = "Training Loss",
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot training loss curve.

    Args:
        losses: List of loss values per step
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Raw loss
    axes[0].plot(losses, alpha=0.3, color='blue', label='Raw')
    # Smoothed loss (exponential moving average)
    if len(losses) > 10:
        alpha = 0.95
        smoothed = [losses[0]]
        for l in losses[1:]:
            smoothed.append(alpha * smoothed[-1] + (1 - alpha) * l)
        axes[0].plot(smoothed, color='red', linewidth=2, label='Smoothed')
    axes[0].set_title(title)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log scale
    axes[1].semilogy(losses, alpha=0.3, color='blue')
    if len(losses) > 10:
        axes[1].semilogy(smoothed, color='red', linewidth=2)
    axes[1].set_title(f'{title} (log scale)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss (log)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
