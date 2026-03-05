"""
Generate synthetic moving-shapes video dataset.

This is a fallback for when the ModelScope dataset is unavailable.
Creates simple videos of geometric shapes moving across frames,
with corresponding text prompts.

Usage:
    python -m nano_video_gen.data.generate_synthetic
"""

import os
import csv
import numpy as np
from PIL import Image, ImageDraw

import imageio


def generate_moving_circle(
    num_frames: int = 16,
    height: int = 64,
    width: int = 64,
    color: tuple = (255, 0, 0),
    radius: int = 8,
    direction: str = "right",
) -> np.ndarray:
    """Generate a video of a circle moving in the given direction."""
    frames = []
    for i in range(num_frames):
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        t = i / max(num_frames - 1, 1)

        if direction == "right":
            cx = int(radius + t * (width - 2 * radius))
            cy = height // 2
        elif direction == "left":
            cx = int(width - radius - t * (width - 2 * radius))
            cy = height // 2
        elif direction == "down":
            cx = width // 2
            cy = int(radius + t * (height - 2 * radius))
        elif direction == "up":
            cx = width // 2
            cy = int(height - radius - t * (height - 2 * radius))
        elif direction == "diagonal":
            cx = int(radius + t * (width - 2 * radius))
            cy = int(radius + t * (height - 2 * radius))
        else:
            cx, cy = width // 2, height // 2

        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=color,
        )
        frames.append(np.array(img))

    return np.stack(frames)


def generate_moving_square(
    num_frames: int = 16,
    height: int = 64,
    width: int = 64,
    color: tuple = (0, 0, 255),
    size: int = 12,
    direction: str = "right",
) -> np.ndarray:
    """Generate a video of a square moving in the given direction."""
    frames = []
    half = size // 2
    for i in range(num_frames):
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        t = i / max(num_frames - 1, 1)

        if direction == "right":
            cx = int(half + t * (width - size))
            cy = height // 2
        elif direction == "left":
            cx = int(width - half - t * (width - size))
            cy = height // 2
        elif direction == "down":
            cx = width // 2
            cy = int(half + t * (height - size))
        elif direction == "up":
            cx = width // 2
            cy = int(height - half - t * (height - size))
        else:
            cx = int(half + t * (width - size))
            cy = int(half + t * (height - size))

        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            fill=color,
        )
        frames.append(np.array(img))

    return np.stack(frames)


def generate_growing_circle(
    num_frames: int = 16,
    height: int = 64,
    width: int = 64,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Generate a video of a circle growing from center."""
    frames = []
    max_radius = min(height, width) // 3
    for i in range(num_frames):
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        t = i / max(num_frames - 1, 1)
        r = int(2 + t * (max_radius - 2))
        cx, cy = width // 2, height // 2

        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=color,
        )
        frames.append(np.array(img))

    return np.stack(frames)


GENERATORS = [
    (generate_moving_circle, {"color": (255, 0, 0), "direction": "right"},
     "a red circle moving to the right"),
    (generate_moving_circle, {"color": (255, 0, 0), "direction": "left"},
     "a red circle moving to the left"),
    (generate_moving_circle, {"color": (0, 255, 0), "direction": "down"},
     "a green circle moving downward"),
    (generate_moving_circle, {"color": (0, 255, 0), "direction": "up"},
     "a green circle moving upward"),
    (generate_moving_circle, {"color": (255, 255, 0), "direction": "diagonal"},
     "a yellow circle moving diagonally"),
    (generate_moving_square, {"color": (0, 0, 255), "direction": "right"},
     "a blue square moving to the right"),
    (generate_moving_square, {"color": (0, 0, 255), "direction": "left"},
     "a blue square moving to the left"),
    (generate_moving_square, {"color": (255, 0, 255), "direction": "down"},
     "a purple square moving downward"),
    (generate_moving_square, {"color": (255, 128, 0), "direction": "diagonal"},
     "an orange square moving diagonally"),
    (generate_growing_circle, {"color": (0, 255, 0)},
     "a green circle growing from the center"),
    (generate_growing_circle, {"color": (255, 0, 0)},
     "a red circle growing from the center"),
    (generate_growing_circle, {"color": (0, 0, 255)},
     "a blue circle growing from the center"),
]


def generate_dataset(
    output_dir: str = "./data/synthetic_dataset",
    num_frames: int = 16,
    height: int = 64,
    width: int = 64,
    num_repeats: int = 5,
):
    """
    Generate a synthetic video dataset.

    Creates moving shape videos with corresponding prompts,
    suitable for training the Nano DiT model.

    Args:
        output_dir: Where to save videos and metadata
        num_frames: Frames per video
        height: Frame height
        width: Frame width
        num_repeats: How many times to repeat each generator
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    video_idx = 0

    for repeat in range(num_repeats):
        for gen_fn, kwargs, prompt in GENERATORS:
            video = gen_fn(
                num_frames=num_frames,
                height=height,
                width=width,
                **kwargs,
            )

            filename = f"video_{video_idx:04d}.mp4"
            filepath = os.path.join(output_dir, filename)

            # Save as MP4
            writer = imageio.get_writer(filepath, fps=8, codec='libx264',
                                         pixelformat='yuv420p')
            for frame in video:
                writer.append_data(frame)
            writer.close()

            metadata.append({"video": filename, "prompt": prompt})
            video_idx += 1

    # Write metadata CSV
    csv_path = os.path.join(output_dir, "metadata.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Generated {video_idx} synthetic videos in {output_dir}")
    print(f"Metadata saved to {csv_path}")
    return output_dir


if __name__ == "__main__":
    generate_dataset()
