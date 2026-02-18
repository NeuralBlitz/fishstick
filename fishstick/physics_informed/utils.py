"""
Utilities for PINN Training
==========================

Helper functions for:
- Error computation
- Output formatting
- Training preparation
- Solution validation
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Callable, Union
import torch
from torch import Tensor, nn
import numpy as np


def compute_relative_error(
    u_pred: Tensor,
    u_exact: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute relative L2 error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution
        reduction: Reduction method

    Returns:
        Relative error
    """
    if u_exact is None:
        raise ValueError("Exact solution required for relative error")

    error = torch.norm(u_pred - u_exact, p=2, dim=-1)
    norm = torch.norm(u_exact, p=2, dim=-1)

    rel_error = error / (norm + 1e-8)

    if reduction == "mean":
        return rel_error.mean()
    elif reduction == "max":
        return rel_error.max()
    else:
        return rel_error


def compute_l2_error(
    u_pred: Tensor,
    u_exact: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute L2 error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution
        reduction: Reduction method

    Returns:
        L2 error
    """
    if u_exact is None:
        return None

    error = torch.norm(u_pred - u_exact, p=2, dim=-1)

    if reduction == "mean":
        return error.mean()
    elif reduction == "max":
        return error.max()
    else:
        return error


def compute_h1_error(
    u_pred: Tensor,
    u_exact: Tensor,
    x: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute H1 (Sobolev) error.

    H1 norm: ||u - v||_H1 = sqrt(||u-v||^2 + ||∇(u-v)||^2)

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution
        x: Spatial coordinates
        reduction: Reduction method

    Returns:
        H1 error
    """
    if u_exact is None:
        return None

    u_pred.requires_grad_(True)

    diff = u_pred - u_exact

    l2_term = torch.norm(diff, p=2, dim=-1) ** 2

    grad_diff = torch.autograd.grad(
        outputs=diff,
        inputs=x,
        grad_outputs=torch.ones_like(diff),
        create_graph=True,
        retain_graph=True,
    )[0]

    h1_term = torch.norm(grad_diff, p=2, dim=-1) ** 2

    h1_error = torch.sqrt(l2_term + h1_term)

    if reduction == "mean":
        return h1_error.mean()
    elif reduction == "max":
        return h1_error.max()
    else:
        return h1_error


def compute_max_error(
    u_pred: Tensor,
    u_exact: Tensor,
) -> Tensor:
    """
    Compute maximum (L-infinity) error.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Maximum error
    """
    if u_exact is None:
        return None

    return torch.max(torch.abs(u_pred - u_exact))


def compute_errors_dict(
    model: nn.Module,
    x_test: Tensor,
    t_test: Optional[Tensor],
    u_exact: Optional[Callable] = None,
) -> Dict[str, float]:
    """
    Compute all error metrics.

    Args:
        model: PINN model
        x_test: Test points
        t_test: Test times
        u_exact: Exact solution function

    Returns:
        Dictionary of errors
    """
    model.eval()

    with torch.no_grad():
        u_pred = model(x_test, t_test)

    errors = {}

    if u_exact is not None:
        u_exact_val = u_exact(x_test, t_test)

        errors["l2"] = compute_l2_error(u_pred, u_exact_val).item()
        errors["relative_l2"] = compute_relative_error(u_pred, u_exact_val).item()
        errors["max"] = compute_max_error(u_pred, u_exact_val).item()

        x_test.requires_grad_(True)
        errors["h1"] = compute_h1_error(u_pred, u_exact_val, x_test).item()

    return errors


def format_pinn_output(
    u: Tensor,
    format: str = "numpy",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Format PINN output for analysis.

    Args:
        u: Output tensor
        format: Output format ("tensor", "numpy")

    Returns:
        Formatted output
    """
    if format == "numpy":
        return u.detach().cpu().numpy()
    else:
        return u


def prepare_pinn_training(
    model: nn.Module,
    domain: List[Tuple[float, float]],
    n_collocation: int = 1000,
    n_boundary: int = 100,
    n_initial: int = 100,
    device: Optional[torch.device] = None,
) -> Dict[str, Tensor]:
    """
    Prepare training data for PINN.

    Args:
        model: PINN model
        domain: Spatial domain
        n_collocation: Number of collocation points
        n_boundary: Number of boundary points
        n_initial: Number of initial condition points
        device: Device

    Returns:
        Dictionary of training tensors
    """
    if device is None:
        device = next(model.parameters()).device

    n_dims = len(domain)

    x_collocation = torch.rand(n_collocation, n_dims, device=device)
    for i, (low, high) in enumerate(domain):
        x_collocation[:, i] = x_collocation[:, i] * (high - low) + low

    t_collocation = torch.rand(n_collocation, device=device)

    x_boundary = torch.rand(n_boundary, n_dims, device=device)
    for i, (low, high) in enumerate(domain):
        boundary_indices = torch.rand(n_boundary, device=device) > 0.5
        x_boundary[boundary_indices, i] = domain[i][1]
        x_boundary[~boundary_indices, i] = domain[i][0]

    t_boundary = torch.rand(n_boundary, device=device)

    x_initial = torch.rand(n_initial, n_dims, device=device)
    for i, (low, high) in enumerate(domain):
        x_initial[:, i] = x_initial[:, i] * (high - low) + low

    t_initial = torch.zeros(n_initial, device=device)

    return {
        "x_collocation": x_collocation,
        "t_collocation": t_collocation,
        "x_boundary": x_boundary,
        "t_boundary": t_boundary,
        "x_initial": x_initial,
        "t_initial": t_initial,
    }


def validate_pinn_solution(
    model: nn.Module,
    x_val: Tensor,
    t_val: Optional[Tensor],
    u_val_exact: Optional[Tensor] = None,
    tol: float = 0.01,
) -> Dict[str, bool]:
    """
    Validate PINN solution quality.

    Args:
        model: PINN model
        x_val: Validation points
        t_val: Validation times
        u_val_exact: Exact validation values
        tol: Tolerance threshold

    Returns:
        Validation results
    """
    model.eval()

    with torch.no_grad():
        u_pred = model(x_val, t_val)

    results = {"valid": True}

    if u_pred.isnan().any():
        results["valid"] = False
        results["has_nan"] = True

    if u_pred.isinf().any():
        results["valid"] = False
        results["has_inf"] = True

    if u_val_exact is not None:
        rel_error = compute_relative_error(u_pred, u_val_exact)
        results["relative_error"] = rel_error.item()

        if rel_error.item() > tol:
            results["valid"] = False
            results["error_too_high"] = True

    return results


def compute_pinn_gradient_norm(
    model: nn.Module,
    inputs: Dict[str, Tensor],
) -> float:
    """
    Compute gradient norm of PINN parameters.

    Args:
        model: PINN model
        inputs: Dictionary of inputs

    Returns:
        Gradient norm
    """
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm**0.5
    return total_norm


def get_pinn_summary(
    model: nn.Module,
) -> Dict[str, any]:
    """
    Get summary of PINN architecture.

    Args:
        model: PINN model

    Returns:
        Summary dictionary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "layers": [],
    }

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            summary["layers"].append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "params": sum(p.numel() for p in module.parameters()),
                }
            )

    return summary


class PINNTrainingLogger:
    """
    Logger for PINN training progress.
    """

    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def log(self, metrics: Dict[str, float]):
        """Log training metrics."""
        self.history.append(metrics)

    def get_history(self) -> List[Dict[str, float]]:
        """Get training history."""
        return self.history

    def get_best_loss(self) -> float:
        """Get best loss so far."""
        if not self.history:
            return float("inf")
        return min(h.get("total", float("inf")) for h in self.history)

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.history:
            return {}

        keys = self.history[0].keys()

        summary = {}
        for key in keys:
            values = [h[key] for h in self.history if key in h]
            if values:
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_final"] = values[-1]

        return summary


def plot_pinn_solution(
    model: nn.Module,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate solution grid for plotting.

    Args:
        model: PINN model
        x_range: X range
        y_range: Y range
        resolution: Grid resolution

    Returns:
        X, Y, U grids
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(-1)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(-1)
    xy = torch.cat([x_flat, y_flat], dim=-1)

    model.eval()
    with torch.no_grad():
        u_flat = model(xy, None)

    U = u_flat.reshape(resolution, resolution).cpu().numpy()

    return X, Y, U


def compute_solution_statistics(
    model: nn.Module,
    x: Tensor,
    t: Optional[Tensor] = None,
) -> Dict[str, float]:
    """
    Compute statistics of PINN solution.

    Args:
        model: PINN model
        x: Points
        t: Times

    Returns:
        Statistics dictionary
    """
    model.eval()

    with torch.no_grad():
        u = model(x, t)

    u_np = u.cpu().numpy()

    stats = {
        "mean": float(np.mean(u_np)),
        "std": float(np.std(u_np)),
        "min": float(np.min(u_np)),
        "max": float(np.max(u_np)),
        "range": float(np.max(u_np) - np.min(u_np)),
    }

    return stats
