"""
CLI training script for the Nano DiT model.

Uses pretrained Wan 2.1 VAE (16 channels, on CPU) and T5 text encoder
(pre-computed embeddings, freed after encoding). Only NanoDiT trains.

Usage:
    python scripts/train.py --data_dir ./data/synthetic_dataset --epochs 50
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
from nano_video_gen.model.wan_vae_wrapper import WanVAEWrapper
from nano_video_gen.model.t5_text_encoder import T5TextEncoder, CachedTextEmbeddings
from nano_video_gen.diffusion.flow_match import FlowMatchScheduler
from nano_video_gen.data.dataset import VideoDataset
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
    parser.add_argument("--model_path", type=str, default="./pretrained_models/Wan2.1",
                        help="Path for pretrained Wan 2.1 weights")
    return parser.parse_args()


def generate_samples(model, vae, text_encoder, scheduler, prompt_idx, device,
                     latent_shape, num_steps=20):
    """Generate a sample video for visualization."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(latent_shape, device=device)
        ctx = text_encoder(prompt_idx.unsqueeze(0).to(device))

        scheduler.set_timesteps(num_steps, shift=5.0)
        for t in scheduler.timesteps:
            timestep = t.unsqueeze(0).to(device)
            v_pred = model(x, timestep, ctx)
            x = scheduler.step(v_pred, t, x)

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

    # Dataset (128x128, 17 frames for Wan VAE)
    dataset = VideoDataset(args.data_dir, height=128, width=128, num_frames=17)
    if len(dataset) == 0:
        print(f"No videos found in {args.data_dir}")
        print("Generate synthetic data first: python -m nano_video_gen.data.generate_synthetic")
        return
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, drop_last=True)
    print(f"Dataset: {len(dataset)} videos, {dataset.num_prompts} unique prompts")

    # Collect unique prompts and pre-compute T5 embeddings
    unique_prompts = sorted(set(dataset.prompts))
    print(f"Unique prompts: {len(unique_prompts)}")

    t5 = T5TextEncoder(model_path=args.model_path, device="cpu")
    print("Encoding all prompts with T5...")
    all_embeddings = t5.encode(unique_prompts)
    print(f"Text embeddings shape: {list(all_embeddings.shape)}")
    t5.free_memory()

    cached_text = CachedTextEmbeddings(all_embeddings).to(device)

    # Load WanVAE on CPU
    vae = WanVAEWrapper(model_path=args.model_path, device="cpu")
    print(f"WanVAE loaded (latent_channels={vae.latent_channels})")

    # Determine latent shape from a test encode
    with torch.no_grad():
        sample_video = torch.randn(1, 3, 17, 128, 128)
        sample_latent = vae.encode(sample_video)
        latent_shape = (1,) + tuple(sample_latent.shape[1:])
        print(f"Latent shape: {list(latent_shape)}")

    # NanoDiT (in_dim=16, text_dim=4096 for real VAE + T5)
    model = NanoDiT(
        dim=args.dim, in_dim=16, ffn_dim=args.ffn_dim, out_dim=16,
        text_dim=4096, freq_dim=64, num_heads=args.num_heads,
        num_layers=args.num_layers, patch_size=(1, 2, 2), eps=1e-6,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"NanoDiT parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Only NanoDiT trains (VAE frozen on CPU, text embeddings cached)
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

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

            context = cached_text(prompt_idx)
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
        "latent_shape": list(latent_shape),
        "num_prompts": len(unique_prompts),
    }, final_path)
    print(f"\nFinal checkpoint saved to {final_path}")

    fig = plot_training_curves(losses)
    fig.savefig(os.path.join(args.output_dir, "training_loss.png"), dpi=150, bbox_inches='tight')
    print(f"Training loss curve saved to {args.output_dir}/training_loss.png")


if __name__ == "__main__":
    main()
