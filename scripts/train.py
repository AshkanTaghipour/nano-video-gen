"""
CLI training script for the Nano DiT model.

Usage:
    # Educational mode (DummyVAE + SimpleTextEncoder, 64x64, 16 frames):
    python scripts/train.py --data_dir ./data/synthetic_dataset --epochs 50

    # Real model mode (Wan 2.1 VAE + T5, 128x128, 17 frames):
    python scripts/train.py --data_dir ./data/synthetic_dataset --epochs 50 --use_real_models
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_video_gen.model.nano_dit import NanoDiT
from nano_video_gen.model.nano_vae import DummyVAE
from nano_video_gen.diffusion.flow_match import FlowMatchScheduler
from nano_video_gen.data.dataset import VideoDataset, SimpleTextEncoder
from nano_video_gen.visualization.viz import plot_training_curves, save_video_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train Nano DiT video generation model")
    parser.add_argument("--data_dir", type=str, default="./data/synthetic_dataset",
                        help="Path to video dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save checkpoints and samples")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Generate samples every N epochs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detected if not set)")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--text_dim", type=int, default=64)
    parser.add_argument("--text_seq_len", type=int, default=8)
    parser.add_argument("--use_real_models", action="store_true",
                        help="Use pretrained Wan 2.1 VAE and T5 text encoder "
                             "(downloads ~9.5 GB to ./pretrained_models/)")
    parser.add_argument("--model_path", type=str, default="./pretrained_models/Wan2.1",
                        help="Path for pretrained model weights (used with --use_real_models)")
    return parser.parse_args()


def generate_samples(model, vae, text_encoder, scheduler, prompt_idx, device,
                     latent_shape, num_steps=20):
    """Generate a sample video for visualization."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(latent_shape, device=device)

        # Get text context
        ctx = text_encoder(prompt_idx.unsqueeze(0).to(device))

        # Set up scheduler
        scheduler.set_timesteps(num_steps, shift=5.0)

        # Denoising loop
        for t in scheduler.timesteps:
            timestep = t.unsqueeze(0).to(device)
            v_pred = model(x, timestep, ctx)
            x = scheduler.step(v_pred, t, x)

        # Decode latent to video
        video = vae.decode(x)
    model.train()
    return video


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Real models mode ---
    if args.use_real_models:
        from nano_video_gen.model.wan_vae_wrapper import WanVAEWrapper
        from nano_video_gen.model.t5_text_encoder import T5TextEncoder, CachedTextEmbeddings

        # Override dims for real models
        in_dim = 16
        out_dim = 16
        text_dim = 4096
        video_height = 128
        video_width = 128
        num_frames = 17

        # Dataset
        dataset = VideoDataset(args.data_dir, height=video_height, width=video_width,
                               num_frames=num_frames)
        if len(dataset) == 0:
            print(f"No videos found in {args.data_dir}")
            print("Generate synthetic data first: python -m nano_video_gen.data.generate_synthetic")
            return
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
        print(f"Dataset: {len(dataset)} videos, {dataset.num_prompts} unique prompts")

        # Collect unique prompts
        unique_prompts = sorted(set(dataset.prompts))
        prompt_to_idx = {p: i for i, p in enumerate(unique_prompts)}
        print(f"Unique prompts: {len(unique_prompts)}")

        # Pre-compute T5 embeddings then free T5
        t5 = T5TextEncoder(model_path=args.model_path, device="cpu")
        print("Encoding all prompts with T5...")
        all_embeddings = t5.encode(unique_prompts)
        print(f"Text embeddings shape: {list(all_embeddings.shape)}")
        t5.free_memory()

        cached_text = CachedTextEmbeddings(all_embeddings).to(device)
        text_seq_len = all_embeddings.shape[1]

        # Load WanVAE on CPU
        vae = WanVAEWrapper(model_path=args.model_path, device="cpu")
        print(f"WanVAE loaded (latent_channels={vae.latent_channels})")

        # Encode a sample to determine latent shape
        with torch.no_grad():
            sample_video = torch.randn(1, 3, num_frames, video_height, video_width)
            sample_latent = vae.encode(sample_video)
            latent_shape = (1,) + tuple(sample_latent.shape[1:])
            print(f"Latent shape: {list(latent_shape)}")

        # NanoDiT with real model dimensions
        model = NanoDiT(
            dim=args.dim, in_dim=in_dim, ffn_dim=args.ffn_dim, out_dim=out_dim,
            text_dim=text_dim, freq_dim=64, num_heads=args.num_heads,
            num_layers=args.num_layers, patch_size=(1, 2, 2), eps=1e-6,
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"NanoDiT parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

        # Only NanoDiT trains (VAE frozen on CPU, text embeddings are cached)
        all_params = list(model.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

        # Scheduler
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(50, shift=5.0)

        # Training loop
        losses = []
        print(f"\nStarting training for {args.epochs} epochs (real models)...")

        for epoch in range(args.epochs):
            epoch_losses = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                video = batch["video"].to(device)
                prompt_idx = batch["prompt_idx"].to(device)

                # Look up cached T5 embeddings
                context = cached_text(prompt_idx)

                # Compute flow matching loss (VAE encode happens inside on CPU)
                loss = scheduler.compute_loss(model, video, context, vae=vae)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                losses.append(loss_val)
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "text_embeddings": cached_text.embeddings.cpu(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                    "use_real_models": True,
                    "latent_shape": list(latent_shape),
                    "num_prompts": len(unique_prompts),
                }, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")

            # Generate samples
            if (epoch + 1) % args.sample_every == 0:
                sample = generate_samples(
                    model, vae, cached_text, scheduler,
                    torch.tensor(0), device,
                    latent_shape=latent_shape, num_steps=20,
                )
                sample_path = os.path.join(args.output_dir, f"sample_epoch{epoch+1}.gif")
                save_video_grid([sample[0].cpu()], sample_path)

        # Save final checkpoint
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save({
            "epoch": args.epochs,
            "model": model.state_dict(),
            "text_embeddings": cached_text.embeddings.cpu(),
            "losses": losses,
            "use_real_models": True,
            "latent_shape": list(latent_shape),
            "num_prompts": len(unique_prompts),
        }, final_path)
        print(f"\nFinal checkpoint saved to {final_path}")

    else:
        # --- Educational mode (DummyVAE + SimpleTextEncoder) ---
        dataset = VideoDataset(args.data_dir, height=64, width=64, num_frames=16)
        if len(dataset) == 0:
            print(f"No videos found in {args.data_dir}")
            print("Generate synthetic data first: python -m nano_video_gen.data.generate_synthetic")
            return
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, drop_last=True)
        print(f"Dataset: {len(dataset)} videos, {dataset.num_prompts} unique prompts")

        # Models
        model = NanoDiT(
            dim=args.dim, in_dim=4, ffn_dim=args.ffn_dim, out_dim=4,
            text_dim=args.text_dim, freq_dim=64, num_heads=args.num_heads,
            num_layers=args.num_layers, patch_size=(1, 2, 2), eps=1e-6,
        ).to(device)

        vae = DummyVAE(in_channels=3, latent_channels=4).to(device)
        text_encoder = SimpleTextEncoder(
            num_prompts=dataset.num_prompts,
            text_dim=args.text_dim,
            seq_len=args.text_seq_len,
        ).to(device)

        latent_shape = (1, 4, 4, 16, 16)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"NanoDiT parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

        # Optimizer
        all_params = list(model.parameters()) + list(text_encoder.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

        # Scheduler
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(50, shift=5.0)

        # Training loop
        losses = []
        print(f"\nStarting training for {args.epochs} epochs...")

        for epoch in range(args.epochs):
            epoch_losses = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                video = batch["video"].to(device)
                prompt_idx = batch["prompt_idx"].to(device)

                # Get text embeddings
                context = text_encoder(prompt_idx)

                # Compute flow matching loss
                loss = scheduler.compute_loss(model, video, context, vae=vae)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()

                loss_val = loss.item()
                epoch_losses.append(loss_val)
                losses.append(loss_val)
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "vae": vae.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "losses": losses,
                }, ckpt_path)
                print(f"  Saved checkpoint to {ckpt_path}")

            # Generate samples
            if (epoch + 1) % args.sample_every == 0:
                sample = generate_samples(
                    model, vae, text_encoder, scheduler,
                    torch.tensor(0), device,
                    latent_shape=latent_shape, num_steps=20,
                )
                sample_path = os.path.join(args.output_dir, f"sample_epoch{epoch+1}.gif")
                save_video_grid([sample[0]], sample_path)

        # Save final checkpoint and loss curve
        final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save({
            "epoch": args.epochs,
            "model": model.state_dict(),
            "vae": vae.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "losses": losses,
        }, final_path)
        print(f"\nFinal checkpoint saved to {final_path}")

    fig = plot_training_curves(losses)
    fig.savefig(os.path.join(args.output_dir, "training_loss.png"), dpi=150, bbox_inches='tight')
    print(f"Training loss curve saved to {args.output_dir}/training_loss.png")


if __name__ == "__main__":
    main()
