"""
Flow Matching Scheduler for training and inference.

Flow matching is the training paradigm used by Wan (and FLUX, SD3, etc.).
It's simpler and often faster-converging than DDPM/DDIM.

Key ideas:
- Forward process: x_t = (1-t)*x_0 + t*noise  (linear interpolation)
- Training target: velocity v = noise - x_0
- Denoising step: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)

The shift parameter controls the noise schedule distribution,
biasing compute toward more perceptually important timesteps.

Reference: DiffSynth-Studio/diffsynth/diffusion/flow_match.py
"""

import torch


class FlowMatchScheduler:
    """
    Flow Matching scheduler for video diffusion.

    Comparison with DDPM:
    ┌────────────────┬──────────────────────┬──────────────────────┐
    │ Aspect         │ DDPM                 │ Flow Matching        │
    ├────────────────┼──────────────────────┼──────────────────────┤
    │ Forward        │ x_t = √αt·x_0       │ x_t = (1-t)·x_0     │
    │                │   + √(1-αt)·noise    │   + t·noise          │
    │ Target         │ noise (epsilon)       │ velocity (v=noise-x) │
    │ Schedule       │ Beta schedule         │ Linear + shift       │
    │ Step           │ Complex formula       │ Euler step           │
    │ Convergence    │ Slower               │ Faster               │
    └────────────────┴──────────────────────┴──────────────────────┘

    Wan uses shift=5 to bias the schedule toward higher noise levels,
    spending more compute on the harder denoising steps.
    """

    def __init__(self):
        self.num_train_timesteps = 1000
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
    ):
        """
        Create the sigma (noise level) schedule for inference.

        The shift transformation biases the schedule:
            sigma' = shift * sigma / (1 + (shift - 1) * sigma)

        With shift=1: linear schedule (uniform timesteps)
        With shift=5: more steps at high noise levels (Wan default)

        Args:
            num_inference_steps: Number of denoising steps (default 50)
            denoising_strength: 1.0 for full generation, <1.0 for img2vid
            shift: Schedule shift factor (Wan default: 5)
        """
        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength

        # Linear spacing in sigma space
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]

        # Apply shift transformation
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # Convert to timestep space (0-1000)
        timesteps = sigmas * self.num_train_timesteps

        self.sigmas = sigmas
        self.timesteps = timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to clean samples (forward process).

        Flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise

        At t=0 (sigma=0): x_t = x_0 (clean)
        At t=1 (sigma=1): x_t = noise (pure noise)

        Args:
            original_samples: Clean data x_0
            noise: Random noise
            timestep: Current timestep value

        Returns:
            Noised samples x_t
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        One Euler denoising step.

        Formula: x_{t-1} = x_t + v * (sigma_{t-1} - sigma_t)

        Since sigma decreases over time, (sigma_{t-1} - sigma_t) < 0,
        so we're subtracting the predicted velocity scaled by the step size.

        Args:
            model_output: Predicted velocity v (output of DiT)
            timestep: Current timestep
            sample: Current noisy sample x_t

        Returns:
            Less noisy sample x_{t-1}
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]

        if timestep_id + 1 >= len(self.timesteps):
            sigma_next = 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]

        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the training target (velocity).

        v = noise - x_0

        The model learns to predict this velocity, which points from
        the clean data toward the noise. During inference, we follow
        this velocity in reverse to denoise.

        Args:
            sample: Clean data x_0
            noise: Random noise

        Returns:
            Velocity target v = noise - x_0
        """
        return noise - sample

    def compute_loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        context: torch.Tensor,
        vae=None,
    ) -> torch.Tensor:
        """
        Compute flow matching training loss.

        Steps:
        1. Optionally encode x0 through VAE
        2. Sample random timestep
        3. Add noise to get x_t
        4. Predict velocity with model
        5. MSE loss between predicted and true velocity

        Args:
            model: NanoDiT model
            x0: Clean video [B, C, T, H, W] (pixel space if vae given, else latent)
            context: Text embeddings [B, seq_len, text_dim]
            vae: Optional DummyVAE for encoding to latent space

        Returns:
            Scalar MSE loss
        """
        # Optionally encode to latent space
        if vae is not None:
            with torch.no_grad():
                x0 = vae.encode(x0)

        # Sample random timesteps uniformly
        batch_size = x0.shape[0]
        t = torch.rand(batch_size, device=x0.device)
        sigma = t  # Simple linear schedule for training

        # Add noise
        noise = torch.randn_like(x0)
        # Expand sigma for broadcasting: (B,) → (B, 1, 1, 1, 1)
        sigma_expand = sigma.view(-1, 1, 1, 1, 1)
        x_t = (1 - sigma_expand) * x0 + sigma_expand * noise

        # Convert to timestep values for the model
        timestep = t * self.num_train_timesteps

        # Predict velocity
        v_pred = model(x_t, timestep, context)

        # True velocity
        v_target = noise - x0

        # MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        return loss
