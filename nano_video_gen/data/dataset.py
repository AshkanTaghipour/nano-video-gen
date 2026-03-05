"""
Video dataset for training the Nano DiT model.

Loads MP4 videos from DiffSynth-Studio's example_video_dataset,
resizes to 64x64, samples 16 frames, and normalizes to [-1, 1].

The dataset also provides simple text embeddings by mapping unique
prompts to learned embedding indices.

Reference: DiffSynth-Studio/diffsynth/core/data/unified_dataset.py
"""

import os
import csv
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import imageio


def _is_valid_video(path):
    """Check that a file exists and is a real video (not a git-lfs pointer)."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1024:
        return False
    return True


class VideoDataset(Dataset):
    """
    Video dataset that loads MP4 files and their text prompts.

    Expected directory structure:
        base_path/
        ├── metadata.csv       # columns: video, prompt (or text)
        └── *.mp4 / subdir/*.mp4

    Scans all metadata*.csv and ltx2*.csv files to find video/prompt pairs.
    Falls back to scanning for mp4 files if no metadata CSVs are found.
    Uses encoding='utf-8-sig' to handle UTF-8 BOM in CSV files.

    Each item returns:
        {
            "video": tensor [3, num_frames, H, W] in [-1, 1],
            "prompt": string,
            "prompt_idx": int (index into unique prompts list)
        }
    """

    def __init__(
        self,
        base_path: str,
        metadata_path: str = None,
        height: int = 64,
        width: int = 64,
        num_frames: int = 16,
    ):
        super().__init__()
        self.base_path = base_path
        self.height = height
        self.width = width
        self.num_frames = num_frames

        self.videos = []
        self.prompts = []

        if metadata_path is not None:
            # Single metadata file specified
            self._load_metadata_csv(metadata_path)
        else:
            # Scan all metadata CSVs in the directory
            csv_patterns = [
                os.path.join(base_path, "metadata*.csv"),
                os.path.join(base_path, "ltx2*.csv"),
            ]
            csv_files = []
            for pattern in csv_patterns:
                csv_files.extend(sorted(glob.glob(pattern)))

            if csv_files:
                seen = set()
                for csv_file in csv_files:
                    self._load_metadata_csv(csv_file, seen=seen)
            else:
                # Fallback: find all mp4 files recursively
                for root, _, files in os.walk(base_path):
                    for fname in sorted(files):
                        if fname.endswith('.mp4'):
                            full_path = os.path.join(root, fname)
                            if _is_valid_video(full_path):
                                self.videos.append(full_path)
                                self.prompts.append("")

        # Build prompt-to-index mapping for simple text embeddings
        unique_prompts = sorted(set(self.prompts))
        self.prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
        self.num_prompts = len(unique_prompts)

    def _load_metadata_csv(self, csv_path, seen=None):
        """Load video/prompt pairs from a single metadata CSV file."""
        if not os.path.exists(csv_path):
            return
        if seen is None:
            seen = set()
        # Derive a subdirectory hint from the CSV filename
        # e.g. "ltx2_t2v.csv" -> try "ltx2/" subdirectory
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        subdir_hint = csv_name.split('_')[0] if '_' in csv_name else None

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row.get('video', row.get('file_name', ''))
                prompt = row.get('prompt', row.get('text', ''))
                if not video_path:
                    continue
                # Try base_path/video_path first, then base_path/subdir/video_path
                full_path = os.path.join(self.base_path, video_path)
                if not _is_valid_video(full_path) and subdir_hint:
                    full_path = os.path.join(self.base_path, subdir_hint, video_path)
                if full_path in seen:
                    continue
                if _is_valid_video(full_path):
                    self.videos.append(full_path)
                    self.prompts.append(prompt if prompt else '')
                    seen.add(full_path)

    def __len__(self):
        return len(self.videos)

    def _load_video(self, path: str) -> torch.Tensor:
        """Load video, resize, sample frames, normalize to [-1, 1]."""
        reader = imageio.get_reader(path)
        frames = np.stack([frame for frame in reader])
        reader.close()

        # frames shape: (T, H, W, C)
        total_frames = len(frames)

        # Sample num_frames uniformly
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.arange(self.num_frames) % total_frames

        sampled = frames[indices]  # (num_frames, H, W, C)

        # Resize each frame
        resized = []
        for frame in sampled:
            img = Image.fromarray(frame)
            img = img.resize((self.width, self.height), Image.BILINEAR)
            resized.append(np.array(img))

        video = np.stack(resized)  # (T, H, W, C)
        video = torch.from_numpy(video).float()

        # Rearrange to (C, T, H, W) and normalize to [-1, 1]
        video = video.permute(3, 0, 1, 2)  # (C, T, H, W)
        video = video / 127.5 - 1.0

        return video

    def __getitem__(self, idx):
        video = self._load_video(self.videos[idx])
        prompt = self.prompts[idx]
        prompt_idx = self.prompt_to_idx.get(prompt, 0)

        return {
            "video": video,
            "prompt": prompt,
            "prompt_idx": prompt_idx,
        }


class SimpleTextEncoder(torch.nn.Module):
    """
    Simple learned text encoder for the nano model.

    Instead of a full T5 or CLIP text encoder, we use a simple
    embedding table + small MLP. Each unique prompt gets an
    embedding vector that the model learns during training.

    Real Wan text encoder:
    - T5-style transformer with 24 layers, dim=4096, 256k vocab
    - Outputs [B, seq_len, 4096]

    Our simple version:
    - Embedding table mapping prompt_idx → vector
    - Projects to [B, seq_len, text_dim]

    Args:
        num_prompts: Number of unique prompts
        text_dim: Output embedding dimension (64 for nano)
        seq_len: Output sequence length
    """

    def __init__(self, num_prompts: int, text_dim: int = 64, seq_len: int = 8):
        super().__init__()
        self.text_dim = text_dim
        self.seq_len = seq_len
        self.embedding = torch.nn.Embedding(max(num_prompts, 1), text_dim * seq_len)
        self.norm = torch.nn.LayerNorm(text_dim)

    def forward(self, prompt_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompt_idx: Prompt indices, shape (B,)

        Returns:
            Text embeddings, shape (B, seq_len, text_dim)
        """
        x = self.embedding(prompt_idx)
        x = x.view(-1, self.seq_len, self.text_dim)
        x = self.norm(x)
        return x
