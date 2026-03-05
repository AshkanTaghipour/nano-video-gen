"""
T5 Text Encoder and Cached Embeddings for use with NanoDiT.

Two classes:
- T5TextEncoder: Loads the pretrained umt5-xxl text encoder from Wan 2.1,
  encodes text prompts to [B, seq_len, 4096] embeddings, then can be freed.
- CachedTextEmbeddings: Stores pre-computed embeddings as a buffer and serves
  them by index during training (no T5 model needed at train time).

Usage pattern:
    # 1. Encode all prompts once
    t5 = T5TextEncoder(model_path)
    embeddings = t5.encode(["prompt 1", "prompt 2", ...])  # [N, seq_len, 4096]
    t5.free_memory()  # free ~9 GB

    # 2. Use cached embeddings during training
    cached = CachedTextEmbeddings(embeddings).to(device)
    context = cached(prompt_indices)  # [B, seq_len, 4096]
"""

import os
import gc
import torch
import torch.nn as nn


def _ensure_downloaded(model_path):
    """Download text encoder + tokenizer weights from HuggingFace if not present."""
    tokenizer_dir = os.path.join(model_path, "tokenizer")
    encoder_dir = os.path.join(model_path, "text_encoder")
    if not os.path.exists(tokenizer_dir) or not os.path.exists(encoder_dir):
        from huggingface_hub import snapshot_download
        print(f"Downloading Wan 2.1 text encoder to {model_path}...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            local_dir=model_path,
            allow_patterns=["text_encoder/*", "tokenizer/*"],
        )
        print("Text encoder download complete.")


class T5TextEncoder:
    """
    Loads the pretrained umt5-xxl T5 text encoder from Wan 2.1.

    The model is loaded in float16 with low_cpu_mem_usage to minimize RAM.
    After encoding all prompts, call free_memory() to release ~9 GB.

    Args:
        model_path: Directory containing (or to download to) the pretrained weights.
        device: Device to run encoding on. Defaults to "cpu".
    """

    def __init__(self, model_path="./pretrained_models/Wan2.1", device="cpu"):
        _ensure_downloaded(model_path)

        from transformers import AutoTokenizer, UMT5EncoderModel

        self.device = torch.device(device)
        print("Loading T5 text encoder (this may take a moment)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        self.model = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        print("T5 text encoder loaded.")

    @torch.no_grad()
    def encode(self, prompts, max_length=226):
        """
        Encode text prompts to embeddings.

        Args:
            prompts: List of text strings.
            max_length: Maximum token length (Wan default: 226).

        Returns:
            Tensor of shape [len(prompts), max_length, 4096] in float32.
        """
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Convert to float32 for downstream use
        embeddings = outputs.last_hidden_state.float().cpu()
        return embeddings

    def free_memory(self):
        """Delete the T5 model and tokenizer to free ~9 GB of RAM."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("T5 text encoder freed from memory.")


class CachedTextEmbeddings(nn.Module):
    """
    Stores pre-computed T5 text embeddings and serves them by index.

    The embeddings are stored as a non-trainable buffer so they are
    saved/loaded with the model checkpoint.

    Args:
        embeddings: Tensor of shape [num_prompts, seq_len, 4096]
    """

    def __init__(self, embeddings):
        super().__init__()
        self.register_buffer("embeddings", embeddings)

    @property
    def num_prompts(self):
        return self.embeddings.shape[0]

    @property
    def text_dim(self):
        return self.embeddings.shape[-1]

    @property
    def seq_len(self):
        return self.embeddings.shape[1]

    def forward(self, prompt_idx):
        """
        Look up cached embeddings by prompt index.

        Args:
            prompt_idx: Tensor of shape [B] with integer prompt indices.

        Returns:
            Tensor of shape [B, seq_len, 4096]
        """
        return self.embeddings[prompt_idx]
