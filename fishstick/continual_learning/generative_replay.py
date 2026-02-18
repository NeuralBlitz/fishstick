"""
Generative Replay for Continual Learning.

Implements replay buffers with generative models (VAE, GAN) to
synthesize past experiences rather than storing them directly.

Classes:
- GenerativeReplayBuffer: Base generative replay interface
- VAEReplay: VAE-based replay buffer
- GANReplay: GAN-based replay buffer
- StableReplay: Stable diffusion-based replay
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import copy
from collections import deque


@dataclass
class GeneratedSample:
    """Synthesized sample from generative model."""

    state: Tensor
    label: Optional[Tensor] = None
    task_id: int = 0
    confidence: float = 1.0
    generation_method: str = "unknown"


class GenerativeReplayBuffer:
    """
    Base class for Generative Replay Buffers.

    Uses a generative model to synthesize past experiences
    instead of storing them explicitly.

    Args:
        generator: Generative model (VAE, GAN, etc.)
        buffer_size: Maximum number of real samples to store
        generator_steps: Number of generator updates per sample
        device: Device for computation
    """

    def __init__(
        self,
        generator: Optional[nn.Module] = None,
        buffer_size: int = 500,
        generator_steps: int = 10,
        device: str = "cpu",
    ):
        self.generator = generator
        self.buffer_size = buffer_size
        self.generator_steps = generator_steps
        self.device = device

        self.real_samples: Dict[int, List[Tensor]] = {}
        self.generated_samples: Dict[int, List[GeneratedSample]] = {}

    def add_real(
        self,
        states: Tensor,
        labels: Optional[Tensor] = None,
        task_id: int = 0,
    ) -> None:
        """Add real samples to buffer."""
        if task_id not in self.real_samples:
            self.real_samples[task_id] = []

        for i in range(len(states)):
            self.real_samples[task_id].append(states[i].detach().cpu())

            if len(self.real_samples[task_id]) > self.buffer_size:
                self.real_samples[task_id].pop(0)

    def generate(
        self,
        task_id: int,
        num_samples: int,
        labels: Optional[Tensor] = None,
    ) -> List[GeneratedSample]:
        """Generate synthetic samples for a task."""
        raise NotImplementedError("Subclass must implement generate()")

    def sample(
        self,
        task_id: int,
        batch_size: int,
        ratio: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample from real and generated data.

        Args:
            task_id: Task ID to sample from
            batch_size: Number of samples
            ratio: Ratio of real samples (1-ratio = generated)

        Returns:
            Tuple of (states, labels)
        """
        real_batch_size = int(batch_size * ratio)
        gen_batch_size = batch_size - real_batch_size

        states = []
        labels = []

        if task_id in self.real_samples and len(self.real_samples[task_id]) > 0:
            real_pool = self.real_samples[task_id]
            if len(real_pool) >= real_batch_size:
                indices = np.random.choice(
                    len(real_pool), real_batch_size, replace=False
                )
            else:
                indices = np.arange(len(real_pool))

            for idx in indices:
                states.append(real_pool[idx])
                labels.append(None)

        if (
            task_id in self.generated_samples
            and len(self.generated_samples[task_id]) > 0
        ):
            gen_pool = self.generated_samples[task_id]
            if len(gen_pool) >= gen_batch_size:
                indices = np.random.choice(len(gen_pool), gen_batch_size, replace=False)
            else:
                indices = np.arange(len(gen_pool))

            for idx in indices:
                states.append(gen_pool[idx].state)
                labels.append(gen_pool[idx].label)

        if len(states) == 0:
            raise ValueError("No samples available")

        return torch.stack(states), torch.tensor(
            [l.item() if l is not None else -1 for l in labels]
        )

    def update_generator(self, task_id: int, dataloader: DataLoader) -> None:
        """Update the generative model."""
        pass


class VAEReplay(GenerativeReplayBuffer):
    """
    VAE-based Generative Replay Buffer.

    Uses Variational Autoencoder to generate synthetic past experiences.

    Args:
        latent_dim: Dimension of latent space
        encoder: Encoder network
        decoder: Decoder network
        buffer_size: Maximum real samples to store
        device: Device for computation
    """

    def __init__(
        self,
        latent_dim: int = 128,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        buffer_size: int = 500,
        device: str = "cpu",
        input_shape: Tuple[int, ...] = (3, 32, 32),
    ):
        super().__init__(buffer_size=buffer_size, device=device)

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        if encoder is None:
            self.encoder = self._default_encoder(input_shape, latent_dim)
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = self._default_decoder(latent_dim, input_shape)
        else:
            self.decoder = decoder

        self.generator = nn.Sequential(self.encoder, self.decoder)

        self.latent_dist = nn.Linear(latent_dim, latent_dim * 2)

    def _default_encoder(
        self, input_shape: Tuple[int, ...], latent_dim: int
    ) -> nn.Module:
        """Create default encoder network."""
        c, h, w = input_shape
        return nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (h // 8) * (w // 8), latent_dim),
        )

    def _default_decoder(
        self, latent_dim: int, output_shape: Tuple[int, ...]
    ) -> nn.Module:
        """Create default decoder network."""
        c, h, w = output_shape
        hidden_dim = 128 * (h // 8) * (w // 8)
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Unflatten(1, (128, h // 8, w // 8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def generate(
        self,
        task_id: int,
        num_samples: int,
        labels: Optional[Tensor] = None,
    ) -> List[GeneratedSample]:
        """Generate samples using VAE."""
        self.generator.eval()

        generated = []

        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            recon = self.decoder(z)

            for i in range(num_samples):
                label = labels[i] if labels is not None and i < len(labels) else None
                sample = GeneratedSample(
                    state=recon[i].cpu(),
                    label=label,
                    task_id=task_id,
                    generation_method="vae",
                )
                generated.append(sample)

        if task_id not in self.generated_samples:
            self.generated_samples[task_id] = []

        self.generated_samples[task_id].extend(generated)

        return generated

    def update_generator(self, task_id: int, dataloader: DataLoader) -> None:
        """Train VAE on current task data."""
        if self.encoder is None or self.decoder is None:
            return

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3,
        )

        self.encoder.train()
        self.decoder.train()

        for epoch in range(5):
            for states, labels in dataloader:
                states = states.to(self.device)

                z_mean = self.encoder(states)
                z_logvar = torch.zeros_like(z_mean)
                z = z_mean + torch.randn_like(z_mean) * torch.exp(z_logvar * 0.5)

                recon = self.decoder(z)

                recon_loss = F.mse_loss(recon, states)
                kl_loss = -0.5 * torch.mean(
                    1 + z_logvar - z_mean.pow(2) - z_logvar.exp()
                )

                loss = recon_loss + 0.01 * kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to latent distribution."""
        self.encoder.eval()
        with torch.no_grad():
            z_mean = self.encoder(x)
            z_logvar = torch.zeros_like(z_mean)
            return z_mean, z_logvar

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to input space."""
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder(z)


class GANReplay(GenerativeReplayBuffer):
    """
    GAN-based Generative Replay Buffer.

    Uses Generative Adversarial Network to generate synthetic past experiences.

    Args:
        latent_dim: Dimension of generator input noise
        generator: Generator network
        discriminator: Discriminator network
        buffer_size: Maximum real samples to store
        device: Device for computation
    """

    def __init__(
        self,
        latent_dim: int = 100,
        generator: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        buffer_size: int = 500,
        device: str = "cpu",
        input_shape: Tuple[int, ...] = (3, 32, 32),
    ):
        super().__init__(buffer_size=buffer_size, device=device)

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        if generator is None:
            self.generator = self._default_generator(latent_dim, input_shape)
        else:
            self.generator = generator

        if discriminator is None:
            self.discriminator = self._default_discriminator(input_shape)
        else:
            self.discriminator = discriminator

    def _default_generator(
        self, latent_dim: int, output_shape: Tuple[int, ...]
    ) -> nn.Module:
        """Create default generator network."""
        c, h, w = output_shape
        return nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, c * h * w),
            nn.Tanh(),
        )

    def _default_discriminator(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Create default discriminator network."""
        c, h, w = input_shape
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * h * w, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def generate(
        self,
        task_id: int,
        num_samples: int,
        labels: Optional[Tensor] = None,
    ) -> List[GeneratedSample]:
        """Generate samples using GAN."""
        self.generator.eval()

        generated = []

        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            fake = self.generator(noise)

            fake = fake.view(num_samples, *self.input_shape)

            for i in range(num_samples):
                label = labels[i] if labels is not None and i < len(labels) else None
                sample = GeneratedSample(
                    state=fake[i].cpu(),
                    label=label,
                    task_id=task_id,
                    generation_method="gan",
                )
                generated.append(sample)

        if task_id not in self.generated_samples:
            self.generated_samples[task_id] = []

        self.generated_samples[task_id].extend(generated)

        return generated

    def update_generator(self, task_id: int, dataloader: DataLoader) -> None:
        """Train GAN on current task data."""
        g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999)
        )
        d_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)
        )

        self.generator.train()
        self.discriminator.train()

        for real_data, _ in dataloader:
            real_data = real_data.to(self.device)
            batch_size = real_data.size(0)

            d_opt.zero_grad()
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            real_output = self.discriminator(real_data)
            real_loss = F.binary_cross_entropy(real_output, real_labels)

            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data.detach())
            fake_loss = F.binary_cross_entropy(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_opt.step()

            g_opt.zero_grad()
            noise = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_data = self.generator(noise)
            output = self.discriminator(fake_data)
            g_loss = F.binary_cross_entropy(output, real_labels)

            g_loss.backward()
            g_opt.step()


class StableReplay(GenerativeReplayBuffer):
    """
    Stable Diffusion-based Replay Buffer.

    Uses pre-trained diffusion models for high-quality sample generation.

    Note: Requires external diffusion model (e.g., from diffusers library).

    Args:
        diffusion_model: Pre-trained diffusion model
        buffer_size: Maximum samples to store
        device: Device for computation
    """

    def __init__(
        self,
        diffusion_model: Optional[nn.Module] = None,
        buffer_size: int = 200,
        device: str = "cpu",
    ):
        super().__init__(buffer_size=buffer_size, device=device)

        self.diffusion_model = diffusion_model

    def generate(
        self,
        task_id: int,
        num_samples: int,
        labels: Optional[Tensor] = None,
    ) -> List[GeneratedSample]:
        """Generate samples using diffusion model."""
        if self.diffusion_model is None:
            return []

        self.diffusion_model.eval()

        generated = []

        with torch.no_grad():
            for i in range(num_samples):
                noise = torch.randn(1, 3, 64, 64, device=self.device)

                sample = self.diffusion_model(noise, num_inference_steps=50)

                label = labels[i] if labels is not None and i < len(labels) else None

                gen_sample = GeneratedSample(
                    state=sample[0].cpu(),
                    label=label,
                    task_id=task_id,
                    generation_method="diffusion",
                )
                generated.append(gen_sample)

        if task_id not in self.generated_samples:
            self.generated_samples[task_id] = []

        self.generated_samples[task_id].extend(generated)

        return generated
