"""
Advanced Training Techniques

SAM (Sharpness-Aware Minimization), AdaBelief, advanced LR schedulers, and regularization.
"""

from typing import Optional, Callable, Dict, Any, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from collections import defaultdict


class SAM(Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer.

    Seeks parameters that lie in neighborhoods having uniformly low loss,
    improving model generalization.

    Args:
        base_optimizer: Base optimizer (e.g., Adam, SGD)
        rho: Neighborhood size for SAM
        adaptive: Use adaptive SAM (ASAM)
    """

    def __init__(
        self, base_optimizer: Optimizer, rho: float = 0.05, adaptive: bool = False
    ):
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                if self.adaptive:
                    e_w = (torch.pow(p, 2) * torch.pow(p.grad, 2)).sum().sqrt() * scale
                else:
                    e_w = p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        raise NotImplementedError(
            "SAM requires calling first_step and second_step separately"
        )

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    (torch.abs(p) if self.adaptive else 1.0)
                    * p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


class AdaBelief(Optimizer):
    """AdaBelief optimizer: Adapting Step Size by the Belief in Observed Gradients.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Constant for numerical stability
        weight_decay: Weight decay (L2 penalty)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-16,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_var"] = torch.zeros_like(p)

                exp_avg, exp_avg_var = state["exp_avg"], state["exp_avg_var"]
                beta1, beta2 = group["betas"]

                if group["weight_decay"] != 0.0:
                    p = p.mul(1 - group["lr"] * group["weight_decay"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )
                step_size = group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                state["step"] += 1

        return loss


class LAMB(Optimizer):
    """LAMB (Layer-wise Adaptive Moments) optimizer for BERT training."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = exp_avg / (1 - beta1 ** state["step"])
                v_hat = exp_avg_sq / (1 - beta2 ** state["step"])

                adam_step = m_hat / (v_hat.sqrt() + group["eps"])

                if group["weight_decay"] != 0.0:
                    adam_step = adam_step + group["weight_decay"] * p

                r = adam_step / (adam_step.norm() + group["eps"])
                p_norm = p.norm()
                g_norm = adam_step.norm()

                trust_ratio = 1.0
                if p_norm > 0 and g_norm > 0:
                    trust_ratio = min(p_norm / g_norm, 1.0)

                p.add_(adam_step, alpha=-group["lr"] * trust_ratio)

                state["step"] += 1

        return loss


class WarmupCosineScheduler(_LRScheduler):
    """Cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of warmup epochs
        max_epochs: Total number of epochs
        warmup_start_lr: Starting learning rate for warmup
        eta_min: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """Linear learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            return [base_lr * (1 - progress) for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """One Cycle learning rate policy.

    The learning rate starts at a initial_lr, increases to max_lr, then
    decreases to min_lr in a triangular pattern.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        pct = step / self.total_steps
        if pct < self.pct_start:
            scale = pct / self.pct_start
        else:
            scale = (1 - pct) / (1 - self.pct_start)

        if self.anneal_strategy == "cos":
            scale = 0.5 * (1 + math.cos(math.pi * scale))

        return [self.max_lr * scale for _ in self.base_lrs]


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss.

    Args:
        smoothing: Smoothing factor (0.0 = no smoothing)
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Mixup(nn.Module):
    """Mixup data augmentation.

    Args:
        alpha: Beta distribution parameter
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMix(nn.Module):
    """CutMix data augmentation."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, float]:
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        _, _, H, W = x.shape
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = torch.randint(W, (batch_size,))
        cy = torch.randint(H, (batch_size,))

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)

        x_cut = x.clone()
        for i in range(batch_size):
            x_cut[i, :, bby1[i] : bby2[i], bbx1[i] : bbx2[i]] = x[
                index[i, :, bby1[i] : bby2[i], bbx1[i] : bbx2[i]]
            ]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        return x_cut, y_a, y_b, lam
