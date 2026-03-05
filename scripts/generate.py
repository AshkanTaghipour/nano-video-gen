"""
CLI inference script for the Nano DiT model.

Usage:
    # Educational mode (DummyVAE + SimpleTextEncoder):
    python scripts/generate.py --checkpoint outputs/checkpoint_final.pt --prompt_idx 0

    # Real model mode (Wan 2.1 VAE + cached T5 embeddings):
    python scripts/generate.py --checkpoint outputs/checkpoint_final.pt --prompt_idx 0 --use_real_models
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_video_gen.model.nano_dit import NanoDiT
from nano_video_gen.model.nano_vae import DummyVAE
from nano_video_gen.diffusion.flow_match import FlowMatchScheduler
from nano_video_gen.data.dataset import SimpleTextEncoder
from nano_video_gen.visualization.viz import save_video_grid, visualize_denoising_process


def parse_args():
    parser = argparse.ArgumentParser(description="Generate videos with trained Nano DiT")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs/generated",
                        help="Directory to save generated videos")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of videos to generate")
    parser.add_argument("--prompt_idx", type=int, default=0,
                        help="Prompt index to condition on")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Noise schedule shift factor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_denoising", action="store_true",
                        help="Save denoising process visualization")
    # Model architecture args (must match training)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--text_dim", type=int, default=64)
    parser.add_argument("--text_seq_len", type=int, default=8)
    parser.add_argument("--use_real_models", action="store_true",
                        help="Use pretrained Wan 2.1 VAE and cached T5 embeddings")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/Wan2.1",
                        help="Path for pretrained model weights (used with --use_real_models)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Auto-detect real models mode from checkpoint
    use_real = args.use_real_models or ckpt.get("use_real_models", False)

    if use_real:
        from nano_video_gen.model.wan_vae_wrapper import WanVAEWrapper
        from nano_video_gen.model.t5_text_encoder import CachedTextEmbeddings

        in_dim = 16
        out_dim = 16
        text_dim = 4096
        latent_shape = tuple(ckpt["latent_shape"])

        # Initialize NanoDiT with real model dimensions
        model = NanoDiT(
            dim=args.dim, in_dim=in_dim, ffn_dim=args.ffn_dim, out_dim=out_dim,
            text_dim=text_dim, freq_dim=64, num_heads=args.num_heads,
            num_layers=args.num_layers, patch_size=(1, 2, 2), eps=1e-6,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        # Load cached text embeddings from checkpoint
        cached_text = CachedTextEmbeddings(ckpt["text_embeddings"]).to(device)
        cached_text.eval()
        num_prompts = cached_text.num_prompts
        print(f"Loaded {num_prompts} cached text embeddings")

        # Load WanVAE on CPU for decoding
        vae = WanVAEWrapper(model_path=args.model_path, device="cpu")

        text_encoder = cached_text

    else:
        # Educational mode
        text_encoder_state = ckpt["text_encoder"]
        emb_weight = text_encoder_state["embedding.weight"]
        num_prompts = emb_weight.shape[0]
        latent_shape = (1, 4, 4, 16, 16)

        model = NanoDiT(
            dim=args.dim, in_dim=4, ffn_dim=args.ffn_dim, out_dim=4,
            text_dim=args.text_dim, freq_dim=64, num_heads=args.num_heads,
            num_layers=args.num_layers, patch_size=(1, 2, 2), eps=1e-6,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        vae = DummyVAE(in_channels=3, latent_channels=4).to(device)
        vae.load_state_dict(ckpt["vae"])
        vae.eval()

        text_encoder = SimpleTextEncoder(
            num_prompts=num_prompts,
            text_dim=args.text_dim,
            seq_len=args.text_seq_len,
        ).to(device)
        text_encoder.load_state_dict(ckpt["text_encoder"])
        text_encoder.eval()

    print(f"Model loaded (epoch {ckpt['epoch']})")
    print(f"Latent shape: {list(latent_shape)}")

    # Setup scheduler
    scheduler = FlowMatchScheduler()
    scheduler.set_timesteps(args.num_steps, shift=args.shift)

    # Generate
    torch.manual_seed(args.seed)
    all_videos = []

    for i in range(args.num_samples):
        print(f"Generating sample {i+1}/{args.num_samples}...")

        with torch.no_grad():
            # Start from noise
            x = torch.randn(latent_shape, device=device)

            # Text context
            prompt_idx = torch.tensor([args.prompt_idx], device=device)
            ctx = text_encoder(prompt_idx)

            # Track denoising for visualization
            denoising_frames = []

            # Denoising loop
            for step_idx, t in enumerate(scheduler.timesteps):
                timestep = t.unsqueeze(0).to(device)
                v_pred = model(x, timestep, ctx)
                x = scheduler.step(v_pred, t, x)

                if args.save_denoising and step_idx % max(1, len(scheduler.timesteps) // 10) == 0:
                    video_frame = vae.decode(x)
                    denoising_frames.append(video_frame.cpu())

            # Final decode
            video = vae.decode(x)
            all_videos.append(video[0].cpu())

            # Save individual video
            save_video_grid(
                [video[0].cpu()],
                os.path.join(args.output_dir, f"generated_{i:03d}.gif"),
            )

        # Save denoising visualization
        if args.save_denoising and denoising_frames:
            fig = visualize_denoising_process(denoising_frames)
            fig.savefig(
                os.path.join(args.output_dir, f"denoising_{i:03d}.png"),
                dpi=150, bbox_inches='tight',
            )
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Save grid of all generated videos
    if len(all_videos) > 1:
        save_video_grid(
            all_videos,
            os.path.join(args.output_dir, "generated_grid.gif"),
            nrow=min(4, len(all_videos)),
        )

    print(f"\nGenerated {args.num_samples} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
