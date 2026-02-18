"""
Sampling Methods for Energy-Based Models.

Implements various sampling algorithms for EBMs:
- Langevin dynamics
- Hamiltonian Monte Carlo (HMC)
- Stochastic Gradient Langevin Dynamics (SGLD)
"""

from typing import Optional, Tuple, List, Dict, Callable, Union
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class LangevinSampler:
    """
    Langevin dynamics sampler for EBMs.

    Performs iterative sampling using gradient-based noise injection.
    """

    def __init__(
        self,
        model: nn.Module,
        step_size: float = 0.01,
        noise_scale: float = 0.01,
    ):
        self.model = model
        self.step_size = step_size
        self.noise_scale = noise_scale

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 100,
        init_x: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Sample from EBM using Langevin dynamics.

        Args:
            shape: Sample shape
            num_steps: Number of Langevin steps
            init_x: Optional initialization

        Returns:
            Generated samples
        """
        if init_x is not None:
            x = init_x
        else:
            x = torch.randn(shape, device=next(self.model.parameters()).device)

        x.requires_grad_(True)

        for _ in range(num_steps):
            noise = torch.randn_like(x) * self.noise_scale

            energy = self.model.energy(x)

            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x,
                retain_graph=False,
            )[0]

            x = x - self.step_size * grad + noise

        x.requires_grad_(False)

        return x

    def sample_with_momentum(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 100,
        momentum: float = 0.9,
    ) -> Tensor:
        """
        Sample with momentum (Langevin with momentum).

        Args:
            shape: Sample shape
            num_steps: Number of steps
            momentum: Momentum coefficient

        Returns:
            Generated samples
        """
        x = torch.randn(shape, device=next(self.model.parameters()).device)
        v = torch.zeros_like(x)

        for _ in range(num_steps):
            noise = torch.randn_like(x) * self.noise_scale

            energy = self.model.energy(x)
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x,
                retain_graph=False,
            )[0]

            v = (
                momentum * v
                - self.step_size * grad
                + math.sqrt(2 * self.step_size) * noise
            )
            x = x + v

        return x


class HMCSampler:
    """
    Hamiltonian Monte Carlo (HMC) sampler.

    Uses Hamiltonian dynamics for efficient sampling from EBMs.
    """

    def __init__(
        self,
        model: nn.Module,
        step_size: float = 0.01,
        num_leapfrog: int = 10,
    ):
        self.model = model
        self.step_size = step_size
        self.num_leapfrog = num_leapfrog

    def potential_energy(self, x: Tensor) -> Tensor:
        """Compute potential energy (-log p(x))."""
        return self.model.energy(x)

    def kinetic_energy(self, v: Tensor) -> Tensor:
        """Compute kinetic energy (momentum)."""
        return 0.5 * (v**2).sum(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_samples: int = 1,
        burn_in: int = 100,
    ) -> Tensor:
        """
        Sample using HMC.

        Args:
            shape: Sample shape
            num_samples: Number of samples to return
            burn_in: Number of burn-in steps

        Returns:
            Generated samples
        """
        x = torch.randn(shape, device=next(self.model.parameters()).device)

        samples = []

        for _ in range(burn_in + num_samples):
            v = torch.randn_like(x)

            v = v * math.sqrt(self.step_size)

            x_new = x.clone()
            v_new = v.clone()

            for _ in range(self.num_leapfrog):
                energy = self.potential_energy(x_new)
                grad = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=x_new,
                    retain_graph=True,
                )[0]

                v_new = v_new - 0.5 * self.step_size * grad
                x_new = x_new + self.step_size * v_new

                energy = self.potential_energy(x_new)
                grad = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=x_new,
                    retain_graph=True,
                )[0]

                v_new = v_new - 0.5 * self.step_size * grad

            v_new = v_new * math.sqrt(self.step_size)

            current_energy = self.potential_energy(x) + self.kinetic_energy(v)
            proposed_energy = self.potential_energy(x_new) + self.kinetic_energy(v_new)

            accept_prob = torch.exp(current_energy - proposed_energy)

            uniform = torch.rand_like(accept_prob)
            x = torch.where(uniform < accept_prob, x_new, x)

            if _ >= burn_in:
                samples.append(x.clone())

        return torch.stack(samples)


class SGDLDSampler:
    """
    Stochastic Gradient Langevin Dynamics (SGLD) sampler.

    Uses stochastic gradients for efficient large-scale sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        step_size: float = 0.01,
        noise_scale: float = 0.01,
    ):
        self.model = model
        self.step_size = step_size
        self.noise_scale = noise_scale

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        num_steps: int = 1,
    ) -> Tensor:
        """
        Perform SGLD steps from current state.

        Args:
            x: Current state
            num_steps: Number of SGLD steps

        Returns:
            Updated samples
        """
        for _ in range(num_steps):
            noise = torch.randn_like(x) * self.noise_scale

            energy = self.model.energy(x)
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x,
                retain_graph=False,
            )[0]

            x = x - self.step_size * grad + noise

        return x


class EBMTrainer:
    """
    Trainer for Energy-Based Models.

    Implements contrastive divergence and related training methods.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
    ):
        self.model = model
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)

    def training_step(
        self,
        x_data: Tensor,
        k: int = 1,
        step_size: float = 0.01,
    ) -> Dict[str, Tensor]:
        """
        Single training step using contrastive divergence.

        Args:
            x_data: Real data samples
            k: Number of Gibbs sampling steps
            step_size: Step size for Langevin

        Returns:
            Dictionary with loss and energies
        """
        batch_size = x_data.shape[0]

        x_model = torch.randn_like(x_data).detach()
        x_model.requires_grad_(True)

        for _ in range(k):
            noise = torch.randn_like(x_model) * 0.1

            energy = self.model.energy(x_model)
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x_model,
                retain_graph=True,
            )[0]

            x_model = x_model - step_size * grad + noise

        x_model = x_model.detach()

        energy_pos = self.model.energy(x_data).mean()
        energy_neg = self.model.energy(x_model).mean()

        loss = energy_pos - energy_neg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss,
            "energy_pos": energy_pos,
            "energy_neg": energy_neg,
        }

    def training_step_alternating(
        self,
        x_data: Tensor,
        x_fake: Tensor,
        mcmc_steps: int = 10,
    ) -> Dict[str, Tensor]:
        """
        Training with alternating updates (as in Du & Mordatch 2019).

        Args:
            x_data: Real data
            x_fake: Fake data to update
            mcmc_steps: Number of MCMC steps

        Returns:
            Dictionary with losses
        """
        x_fake = x_fake.detach().requires_grad_(True)

        for _ in range(mcmc_steps):
            noise = torch.randn_like(x_fake) * 0.01

            energy = self.model.energy(x_fake)
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=x_fake,
                retain_graph=False,
            )[0]

            x_fake = x_fake - 0.01 * grad + noise

        x_fake = x_fake.detach()

        energy_pos = self.model.energy(x_data)
        energy_neg = self.model.energy(x_fake)

        loss = F.relu(energy_pos - energy_neg + 0.1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss,
            "x_fake": x_fake,
        }


class SwendsenWangSampler:
    """
    Swendsen-Wang algorithm for EBM sampling.

    Cluster-based sampling for faster mixing.
    """

    def __init__(
        self,
        model: nn.Module,
        coupling_prob: float = 0.5,
    ):
        self.model = model
        self.coupling_prob = coupling_prob

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        num_steps: int = 10,
    ) -> Tensor:
        """
        Swendsen-Wang sampling step.

        Args:
            x: Current state
            num_steps: Number of steps

        Returns:
            Updated samples
        """
        for _ in range(num_steps):
            energy = self.model.energy(x)

            probs = torch.sigmoid(-energy)

            couplings = (torch.rand_like(x) < probs).float()

            x = torch.where(couplings > 0.5, torch.randn_like(x), x)

        return x
