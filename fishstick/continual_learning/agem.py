"""
Average Gradient Episodic Memory (A-GEM) Implementation.

Memory-efficient version of GEM that maintains a single episodic memory buffer.

Classes:
- AverageGEM: A-GEM implementation
- EfficientGEM: Memory-efficient GEM variant
- ProjectedGEM: GEM with gradient projection
"""

from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import numpy as np


class AverageGEM:
    """
    Averaged Gradient Episodic Memory (A-GEM).

    Efficient GEM variant that uses a single averaged episodic memory buffer.

    Reference:
        Chaudhry et al., "Efficient Lifelong Learning with A-GEM", ICLR 2019

    Args:
        model: Neural network
        memory_size: Total memory buffer size
        device: Device for computation
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 1000,
        device: str = "cpu",
    ):
        self.model = model
        self.memory_size = memory_size
        self.device = device

        self.memory_x: List[Tensor] = []
        self.memory_y: List[Tensor] = []

    def store(self, x: Tensor, y: Tensor) -> None:
        """
        Store samples in episodic memory.

        Args:
            x: Input samples
            y: Target labels
        """
        self.memory_x.append(x.detach().cpu())
        self.memory_y.append(y.detach().cpu())

        if len(self.memory_x) > self.memory_size:
            self.memory_x = self.memory_x[-self.memory_size :]
            self.memory_y = self.memory_y[-self.memory_size :]

    def compute_reference_gradient(self) -> Optional[Dict[str, Tensor]]:
        """Compute gradient on reference memory."""
        if len(self.memory_x) == 0:
            return None

        x = torch.cat(self.memory_x).to(self.device)
        y = torch.cat(self.memory_y).to(self.device)

        self.model.zero_grad()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        return grads

    def project_gradient(
        self,
        current_grads: Dict[str, Tensor],
        reference_grads: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Project current gradients to avoid forgetting.

        Args:
            current_grads: Current gradients
            reference_grads: Reference gradients from memory

        Returns:
            Projected gradients
        """
        projected = {}

        for n, grad in current_grads.items():
            if n in reference_grads:
                ref = reference_grads[n]

                dot = (grad * ref).sum()

                if dot < 0:
                    ref_norm_sq = (ref**2).sum()

                    if ref_norm_sq > 1e-8:
                        projected[n] = grad - (dot / ref_norm_sq) * ref
                    else:
                        projected[n] = grad
                else:
                    projected[n] = grad
            else:
                projected[n] = grad

        return projected

    def apply_projection(self) -> None:
        """Apply gradient projection to model parameters."""
        ref_grads = self.compute_reference_gradient()

        if ref_grads is None:
            return

        current_grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        projected_grads = self.project_gradient(current_grads, ref_grads)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in projected_grads:
                p.grad.data = projected_grads[n]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "current_size": len(self.memory_x),
            "max_size": self.memory_size,
        }


class EfficientGEM(AverageGEM):
    """
    Efficient GEM with Simplified Projection.

    Uses simplified gradient matching for efficiency.

    Args:
        model: Neural network
        memory_size: Memory buffer size
        device: Device
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 1000,
        device: str = "cpu",
    ):
        super().__init__(model, memory_size, device)

    def compute_efficient_projection(
        self,
        current_grads: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute efficient projection using memory buffer."""
        if len(self.memory_x) < 2:
            return current_grads

        projected = {}

        x = torch.cat(self.memory_x[-50:]).to(self.device)
        y = torch.cat(self.memory_y[-50:]).to(self.device)

        self.model.zero_grad()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        ref_grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        return self.project_gradient(current_grads, ref_grads)


class ProjectedGEM:
    """
    GEM with Multiple Memory Banks.

    Maintains separate memory banks for different task types.

    Args:
        model: Neural network
        num_banks: Number of memory banks
        bank_size: Size per bank
        device: Device
    """

    def __init__(
        self,
        model: nn.Module,
        num_banks: int = 5,
        bank_size: int = 200,
        device: str = "cpu",
    ):
        self.model = model
        self.num_banks = num_banks
        self.bank_size = bank_size
        self.device = device

        self.banks: List[List[Tuple[Tensor, Tensor]]] = [[] for _ in range(num_banks)]

        self.bank_idx = 0

    def store(self, x: Tensor, y: Tensor, bank_id: Optional[int] = None) -> None:
        """Store in specified bank or rotate through banks."""
        if bank_id is None:
            bank_id = self.bank_idx
            self.bank_idx = (self.bank_idx + 1) % self.num_banks

        self.banks[bank_id].append((x.detach().cpu(), y.detach().cpu()))

        if len(self.banks[bank_id]) > self.bank_size:
            self.banks[bank_id] = self.banks[bank_id][-self.bank_size :]

    def compute_combined_reference(self) -> Optional[Dict[str, Tensor]]:
        """Compute reference gradient from all banks."""
        all_x = []
        all_y = []

        for bank in self.banks:
            for x, y in bank[-10:]:
                all_x.append(x)
                all_y.append(y)

        if len(all_x) == 0:
            return None

        x = torch.cat(all_x).to(self.device)
        y = torch.cat(all_y).to(self.device)

        self.model.zero_grad()

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        return {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    def apply_gradient_projection(self) -> None:
        """Apply gradient projection from combined memory."""
        ref_grads = self.compute_combined_reference()

        if ref_grads is None:
            return

        for n, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and n in ref_grads:
                grad = p.grad
                ref = ref_grads[n]

                dot = (grad * ref).sum()

                if dot < 0:
                    ref_norm = (ref**2).sum()

                    if ref_norm > 1e-8:
                        p.grad.data = grad - (dot / ref_norm) * ref


class StreamingGEM(ProjectedGEM):
    """
    Streaming GEM for Online Learning.

    Adapted for streaming data scenarios.

    Args:
        model: Neural network
        buffer_size: Buffer size
        device: Device
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
        device: str = "cpu",
    ):
        super().__init__(model, num_banks=1, bank_size=buffer_size, device=device)

    def update(self, x: Tensor, y: Tensor) -> None:
        """Update buffer with streaming data."""
        self.store(x, y, bank_id=0)
