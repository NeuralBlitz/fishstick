"""
Utility functions and helpers for Unified Intelligence Framework.
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import Tensor
import numpy as np


def compute_jacobian(f: torch.nn.Module, x: Tensor) -> Tensor:
    """
    Compute Jacobian matrix ∂f/∂x.

    Args:
        f: Function/module to differentiate
        x: Input tensor [batch, ...]

    Returns:
        Jacobian [batch, output_dim, input_dim]
    """
    x = x.requires_grad_(True)
    y = f(x)

    batch_size = x.shape[0]
    output_dim = y.shape[-1] if y.dim() > 1 else 1
    input_dim = x.shape[-1] if x.dim() > 1 else 1

    jacobian = torch.zeros(batch_size, output_dim, input_dim)

    for i in range(output_dim):
        grad = torch.zeros_like(y)
        grad[:, i] = 1
        y.backward(grad, retain_graph=True)
        jacobian[:, i, :] = x.grad.view(batch_size, -1)
        x.grad.zero_()

    return jacobian


def compute_hessian(f: torch.nn.Module, x: Tensor) -> Tensor:
    """
    Compute Hessian matrix ∂²f/∂x².
    """
    x = x.requires_grad_(True)
    y = f(x).sum()
    grad = torch.autograd.grad(y, x, create_graph=True)[0]

    input_dim = x.shape[-1]
    hessian = torch.zeros(input_dim, input_dim)

    for i in range(input_dim):
        grad_i = torch.autograd.grad(grad[0, i], x, retain_graph=True)[0]
        hessian[i, :] = grad_i.view(-1)

    return hessian


def spectral_norm(W: Tensor) -> float:
    """Compute spectral norm (largest singular value) of matrix."""
    if W.dim() == 1:
        return torch.abs(W).max().item()
    s = torch.linalg.svdvals(W)
    return s[0].item()


def condition_number(A: Tensor) -> float:
    """Compute condition number of matrix."""
    s = torch.linalg.svdvals(A)
    return (s[0] / s[-1]).item()


def log_det(A: Tensor) -> float:
    """Compute log determinant of positive definite matrix."""
    sign, logdet = torch.linalg.slogdet(A)
    return logdet.item()


def kl_divergence_gaussian(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor
) -> Tensor:
    """
    Compute KL divergence between two Gaussians.

    KL(N(μ1, Σ1) || N(μ2, Σ2))
    """
    if sigma1.dim() == 1:
        sigma1 = torch.diag(sigma1)
    if sigma2.dim() == 1:
        sigma2 = torch.diag(sigma2)

    d = mu1.shape[-1]

    sigma2_inv = torch.linalg.inv(sigma2)

    trace_term = torch.trace(sigma2_inv @ sigma1)
    mean_term = (mu2 - mu1).T @ sigma2_inv @ (mu2 - mu1)
    log_det_term = log_det(sigma2) - log_det(sigma1)

    return 0.5 * (trace_term + mean_term - d + log_det_term)


def wasserstein_distance_gaussian(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor
) -> Tensor:
    """
    Compute W_2 Wasserstein distance between two Gaussians.
    """
    mean_diff = ((mu1 - mu2) ** 2).sum()

    sigma1_sqrt = torch.linalg.matrix_power(sigma1, 0.5)
    temp = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    temp_sqrt = torch.linalg.matrix_power(temp, 0.5)

    trace_term = torch.trace(sigma1 + sigma2 - 2 * temp_sqrt)

    return torch.sqrt(mean_diff + trace_term)


def exponential_moving_average(data: List[float], alpha: float = 0.9) -> List[float]:
    """Compute exponential moving average."""
    ema = []
    current = data[0] if data else 0.0

    for x in data:
        current = alpha * current + (1 - alpha) * x
        ema.append(current)

    return ema


def topological_sort(edges: List[Tuple[str, str]]) -> List[str]:
    """Topological sort of nodes given edges."""
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    for src, dst in edges:
        graph[src].append(dst)
        in_degree[dst] += 1
        nodes.add(src)
        nodes.add(dst)

    queue = deque([n for n in nodes if in_degree[n] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


def format_tensor_stats(tensor: Tensor, name: str = "tensor") -> str:
    """Format tensor statistics for logging."""
    return (
        f"{name}: shape={list(tensor.shape)}, "
        f"mean={tensor.mean().item():.4f}, "
        f"std={tensor.std().item():.4f}, "
        f"min={tensor.min().item():.4f}, "
        f"max={tensor.max().item():.4f}"
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm for model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    return total_norm**0.5


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze all parameters in module."""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    """Unfreeze all parameters in module."""
    for p in module.parameters():
        p.requires_grad = True
