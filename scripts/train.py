"""
CLI training script for the Nano DiT model.

Usage:
    python scripts/train.py --data_dir ./data/synthetic_dataset --epochs 50
    python scripts/train.py --data_dir ./data/example_video_dataset --epochs 100 --lr 3e-4
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
    return parser.parse_args()


def generate_samples(model, vae, text_encoder, scheduler, prompt_idx, device, num_steps=20):
    """Generate a sample video for visualization."""
    model.eval()
    with torch.no_grad():
        # Start from pure noise in latent space
        latent_shape = (1, 4, 4, 16, 16)  # (B, C, T, H, W) after VAE encoding
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

    # Dataset
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
                torch.tensor(0), device, num_steps=20,
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
