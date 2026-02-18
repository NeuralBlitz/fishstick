"""
PINN Loss Functions
==================

Provides loss functions for training physics-informed neural networks:
- Physics loss
- Data loss
- Boundary and initial condition losses
- Combined loss formulations
- Advanced methods (penalty, augmented Lagrangian)
"""

from __future__ import annotations

from typing import Optional, Callable, Dict, List
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on PDE residual.

    Computes the mean squared residual of the governing PDE
    at collocation points.

    Args:
        pde_residual_fn: Function that computes PDE residual
        reduction: Reduction method ("mean", "sum", "none")
    """

    def __init__(
        self,
        pde_residual_fn: Optional[Callable] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.pde_residual_fn = pde_residual_fn
        self.reduction = reduction

    def forward(
        self,
        residual: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute physics loss.

        Args:
            residual: PDE residual [N, ...]
            weights: Optional weights [N]

        Returns:
            Loss scalar
        """
        loss = residual**2

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DataLoss(nn.Module):
    """
    Data fitting loss.

    Computes MSE between predictions and observed data.

    Args:
        reduction: Reduction method
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute data loss.

        Args:
            predictions: Model predictions [N, ...]
            targets: Target values [N, ...]
            weights: Optional weights [N]

        Returns:
            Loss scalar
        """
        loss = (predictions - targets) ** 2

        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BoundaryLoss(nn.Module):
    """
    Boundary condition loss.

    Args:
        bc_type: Type of BC ("dirichlet", "neumann", "robin")
        reduction: Reduction method
    """

    def __init__(self, bc_type: str = "dirichlet", reduction: str = "mean"):
        super().__init__()
        self.bc_type = bc_type
        self.reduction = reduction

    def forward(
        self,
        u_pred: Tensor,
        u_bc: Optional[Tensor] = None,
        du_pred: Optional[Tensor] = None,
        g_bc: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute boundary loss.

        Args:
            u_pred: Predicted solution at boundary
            u_bc: Dirichlet boundary value (if applicable)
            du_pred: Derivative at boundary (for Neumann)
            g_bc: Neumann boundary value (if applicable)

        Returns:
            Loss scalar
        """
        if self.bc_type == "dirichlet":
            if u_bc is None:
                return torch.tensor(0.0, device=u_pred.device)
            loss = (u_pred - u_bc) ** 2

        elif self.bc_type == "neumann":
            if du_pred is None or g_bc is None:
                return torch.tensor(0.0, device=u_pred.device)
            loss = (du_pred - g_bc) ** 2

        elif self.bc_type == "robin":
            if u_bc is None or du_pred is None or g_bc is None:
                return torch.tensor(0.0, device=u_pred.device)
            loss = (du_pred + g_bc * u_bc) ** 2

        else:
            raise ValueError(f"Unknown BC type: {self.bc_type}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class InitialLoss(nn.Module):
    """
    Initial condition loss.

    Args:
        reduction: Reduction method
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        u_pred: Tensor,
        u_ic: Tensor,
    ) -> Tensor:
        """
        Compute initial condition loss.

        Args:
            u_pred: Predicted solution at t=t0
            u_ic: Initial condition value

        Returns:
            Loss scalar
        """
        loss = (u_pred - u_ic) ** 2

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with weighted components.

    Args:
        lambda_physics: Weight for physics loss
        lambda_data: Weight for data loss
        lambda_bc: Weight for boundary loss
        lambda_ic: Weight for initial condition loss
    """

    def __init__(
        self,
        lambda_physics: float = 1.0,
        lambda_data: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
    ):
        super().__init__()

        self.lambda_physics = lambda_physics
        self.lambda_data = lambda_data
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

        self.physics_loss = PhysicsLoss()
        self.data_loss = DataLoss()
        self.bc_loss = BoundaryLoss()
        self.ic_loss = InitialLoss()

    def forward(
        self,
        physics_residual: Optional[Tensor] = None,
        u_pred: Optional[Tensor] = None,
        u_data: Optional[Tensor] = None,
        u_bc: Optional[Tensor] = None,
        du_bc: Optional[Tensor] = None,
        g_bc: Optional[Tensor] = None,
        u_ic: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.

        Returns:
            Dictionary with total loss and components
        """
        total_loss = torch.tensor(0.0)
        losses = {}

        if physics_residual is not None:
            l_physics = self.physics_loss(physics_residual)
            losses["physics"] = l_physics
            total_loss = total_loss + self.lambda_physics * l_physics

        if u_pred is not None and u_data is not None:
            l_data = self.data_loss(u_pred, u_data)
            losses["data"] = l_data
            total_loss = total_loss + self.lambda_data * l_data

        if u_bc is not None:
            l_bc = self.bc_loss(u_bc, du_pred=du_bc, g_bc=g_bc)
            losses["boundary"] = l_bc
            total_loss = total_loss + self.lambda_bc * l_bc

        if u_ic is not None:
            l_ic = self.ic_loss(u_ic[0], u_ic[1])
            losses["initial"] = l_ic
            total_loss = total_loss + self.lambda_ic * l_ic

        losses["total"] = total_loss
        return losses


class SoftBoundaryLoss(nn.Module):
    """
    Soft boundary condition enforcement.

    Uses a differentiable penalty that allows boundary conditions
    to be weakly enforced during training.
    """

    def __init__(
        self,
        bc_func: Callable,
        bc_type: str = "dirichlet",
    ):
        super().__init__()
        self.bc_func = bc_func
        self.bc_type = bc_type

    def forward(
        self,
        model: nn.Module,
        x_bc: Tensor,
        t_bc: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute soft boundary loss.

        Args:
            model: PINN model
            x_bc: Boundary points
            t_bc: Boundary times

        Returns:
            Loss scalar
        """
        x_bc.requires_grad_(True)

        u_pred = model(x_bc, t_bc)

        u_bc = self.bc_func(x_bc, t_bc)

        loss = F.mse_loss(u_pred, u_bc)

        return loss


class PenaltyMethodLoss(nn.Module):
    """
    Penalty method for constraint enforcement.

    Adds penalty terms to enforce constraints without modifying
    the network architecture.

    Args:
        penalty_param: Initial penalty parameter
        increase_factor: Factor to increase penalty
        max_penalty: Maximum penalty value
    """

    def __init__(
        self,
        penalty_param: float = 1.0,
        increase_factor: float = 10.0,
        max_penalty: float = 1e6,
    ):
        super().__init__()
        self.penalty_param = nn.Parameter(
            torch.tensor(penalty_param), requires_grad=False
        )
        self.increase_factor = increase_factor
        self.max_penalty = max_penalty
        self.prev_loss = None

    def forward(
        self,
        loss: Tensor,
        constraint_violation: Tensor,
    ) -> Tensor:
        """
        Compute penalized loss.

        Args:
            loss: Base loss
            constraint_violation: Constraint violation

        Returns:
            Penalized loss
        """
        penalty = self.penalty_param * (constraint_violation**2)

        total = loss + penalty

        return total

    def update_penalty(self, loss: Tensor):
        """Update penalty parameter based on convergence."""
        if self.prev_loss is not None:
            if loss.item() < self.prev_loss * 0.95:
                pass
            else:
                new_penalty = min(
                    self.penalty_param.item() * self.increase_factor, self.max_penalty
                )
                self.penalty_param.data = torch.tensor(new_penalty)

        self.prev_loss = loss.item()


class AugmentedLagrangianLoss(nn.Module):
    """
    Augmented Lagrangian method for constraint satisfaction.

    More efficient than penalty method for enforcing constraints.

    Args:
        penalty_param: Initial penalty parameter
        lambda_lag: Initial Lagrange multiplier estimates
    """

    def __init__(
        self,
        penalty_param: float = 1.0,
        lambda_lag: Optional[Dict[str, Tensor]] = None,
    ):
        super().__init__()

        self.penalty_param = nn.Parameter(
            torch.tensor(penalty_param), requires_grad=False
        )

        self.lambda_lag = nn.ParameterDict()
        if lambda_lag:
            for name, value in lambda_lag.items():
                self.lambda_lag[name] = nn.Parameter(
                    torch.tensor(value), requires_grad=False
                )

    def forward(
        self,
        loss: Tensor,
        constraints: Dict[str, Tensor],
    ) -> Tensor:
        """
        Compute augmented Lagrangian loss.

        Args:
            loss: Base loss
            constraints: Dictionary of constraint violations

        Returns:
            Augmented Lagrangian
        """
        aug_lagrangian = loss

        for name, constraint in constraints.items():
            if name not in self.lambda_lag:
                self.lambda_lag[name] = nn.Parameter(
                    torch.zeros_like(constraint.mean()), requires_grad=False
                )

            lam = self.lambda_lag[name]
            penalty = (self.penalty_param / 2) * (constraint**2)

            aug_lagrangian = aug_lagrangian + lam * constraint + penalty

        return aug_lagrangian

    def update_multipliers(
        self,
        constraints: Dict[str, Tensor],
    ):
        """Update Lagrange multipliers."""
        for name, constraint in constraints.items():
            if name in self.lambda_lag:
                new_lam = self.lambda_lag[name] + self.penalty_param * constraint.mean()
                self.lambda_lag[name].data = new_lam


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for scale-aware training.

    Applies different weights to different frequency components.
    """

    def __init__(
        self,
        n_scales: int = 4,
        scale_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if scale_weights is None:
            scale_weights = [1.0] * n_scales

        self.register_buffer("scale_weights", torch.tensor(scale_weights))

    def forward(
        self,
        residual: Tensor,
        x: Tensor,
    ) -> Tensor:
        """
        Compute multi-scale loss.

        Args:
            residual: PDE residual
            x: Spatial coordinates

        Returns:
            Weighted loss
        """
        loss = 0.0

        for i, w in enumerate(self.scale_weights):
            scale_mask = (torch.norm(x, dim=-1) > i).float()

            if scale_mask.sum() > 0:
                scale_loss = ((residual**2) * scale_mask).sum() / (
                    scale_mask.sum() + 1e-8
                )
                loss = loss + w * scale_loss

        return loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss weighting using gradient descent.

    Automatically learns optimal weights for different loss components.

    Reference: "A Gradient-Based Strategy for Automatic Loss Balancing"
    """

    def __init__(
        self,
        n_components: int,
        initial_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        if initial_weights is None:
            initial_weights = [1.0 / n_components] * n_components

        self.log_weights = nn.Parameter(
            torch.tensor([w + 1e-8 for w in initial_weights]).log()
        )

        self.n_components = n_components

    def forward(
        self,
        losses: List[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute adaptive weighted loss.

        Args:
            losses: List of loss components

        Returns:
            Tuple of (weighted loss, statistics)
        """
        weights = F.softmax(self.log_weights, dim=0)

        weighted_loss = sum(w * l for w, l in zip(weights, losses))

        stats = {
            "weighted_loss": weighted_loss,
            "weights": weights,
            "raw_losses": losses,
        }

        return weighted_loss, stats


class CurricularLoss(nn.Module):
    """
    Curricular loss for curriculum learning.

    Gradually increases the weight of "harder" samples during training.

    Args:
        n_epochs: Total number of epochs
        start_epoch: Epoch to start curriculum
    """

    def __init__(
        self,
        n_epochs: int,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.register_buffer("current_epoch", torch.tensor(0))

    def forward(
        self,
        residual: Tensor,
        epoch: int,
    ) -> Tensor:
        """
        Compute curricular loss.

        Args:
            residual: PDE residual
            epoch: Current epoch

        Returns:
            Weighted loss
        """
        if epoch < self.start_epoch:
            return residual.mean()

        curriculum_factor = (epoch - self.start_epoch) / (
            self.n_epochs - self.start_epoch
        )
        curriculum_factor = min(1.0, curriculum_factor)

        weights = torch.abs(residual)
        weights = weights / (weights.sum() + 1e-8)

        base_loss = residual**2

        curricular_loss = (
            base_loss * weights
        ).sum() * curriculum_factor + base_loss.mean() * (1 - curriculum_factor)

        return curricular_loss


class PINNLoss(nn.Module):
    """
    Main PINN loss module combining all components.

    This is the primary interface for computing PINN training losses.

    Args:
        pde: PDE specification
        use_soft_bc: Use soft boundary conditions
        use_adaptive: Use adaptive loss weighting
    """

    def __init__(
        self,
        pde: Optional[torch.nn.Module] = None,
        use_soft_bc: bool = True,
        use_adaptive: bool = False,
    ):
        super().__init__()

        self.pde = pde
        self.use_soft_bc = use_soft_bc
        self.use_adaptive = use_adaptive

        if use_adaptive:
            self.adaptive_loss = AdaptiveLoss(n_components=4)

        self.combined_loss = CombinedLoss()

    def forward(
        self,
        model: nn.Module,
        x_collocation: Tensor,
        t_collocation: Optional[Tensor] = None,
        x_boundary: Optional[Tensor] = None,
        t_boundary: Optional[Tensor] = None,
        u_boundary: Optional[Tensor] = None,
        x_initial: Optional[Tensor] = None,
        t_initial: Optional[Tensor] = None,
        u_initial: Optional[Tensor] = None,
        x_data: Optional[Tensor] = None,
        t_data: Optional[Tensor] = None,
        u_data: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute total PINN loss.

        Args:
            model: PINN model
            x_collocation: Collocation points
            t_collocation: Temporal collocation points
            x_boundary: Boundary points
            t_boundary: Boundary times
            u_boundary: Boundary values
            x_initial: Initial condition points
            t_initial: Initial time
            u_initial: Initial condition values
            x_data: Data points
            t_data: Data times
            u_data: Data values

        Returns:
            Dictionary of loss components
        """
        losses = {}

        if x_collocation is not None:
            x_collocation.requires_grad_(True)
            if t_collocation is not None:
                t_collocation.requires_grad_(True)

            residual = model.compute_pde_residual(x_collocation, t_collocation)
            l_physics = (residual**2).mean()
            losses["physics"] = l_physics

        if x_boundary is not None:
            x_boundary.requires_grad_(True)
            u_bc_pred = model(x_boundary, t_boundary)

            if u_boundary is not None:
                l_bc = F.mse_loss(u_bc_pred, u_boundary)
            else:
                l_bc = torch.tensor(0.0)
            losses["boundary"] = l_bc

        if x_initial is not None:
            x_initial.requires_grad_(True)
            u_ic_pred = model(x_initial, t_initial)

            if u_initial is not None:
                l_ic = F.mse_loss(u_ic_pred, u_initial)
            else:
                l_ic = torch.tensor(0.0)
            losses["initial"] = l_ic

        if x_data is not None:
            u_data_pred = model(x_data, t_data)
            l_data = F.mse_loss(u_data_pred, u_data)
            losses["data"] = l_data

        total_loss = sum(losses.values())
        losses["total"] = total_loss

        return losses
