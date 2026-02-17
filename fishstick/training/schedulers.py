"""
Learning Rate Schedulers

Custom learning rate schedulers with warmup and advanced scheduling strategies.
"""

import math
from typing import Optional
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing with warm restarts."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch
        if self.T_cur >= self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class OneCycleLR(_LRScheduler):
    """One cycle learning rate policy."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        epochs: int,
        steps_per_epoch: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.total_steps = epochs * steps_per_epoch
        self.step_up_steps = int(self.total_steps * pct_start)
        self.step_down_steps = self.total_steps - self.step_up_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.step_up_steps:
            pct = step / self.step_up_steps
            if self.anneal_strategy == "cos":
                factor = (self.div_factor - 1) / self.div_factor + pct * (
                    1 / self.div_factor
                )
            else:
                factor = (self.div_factor - 1) / self.div_factor + pct * (
                    1 - 1 / self.div_factor
                )
        else:
            pct = (step - self.step_up_steps) / self.step_down_steps
            if self.anneal_strategy == "cos":
                factor = (1 / self.div_factor) * (
                    1 + pct * (self.final_div_factor - 1) / self.final_div_factor
                )
            else:
                factor = (1 / self.div_factor) * (
                    1 - pct * (1 - 1 / self.final_div_factor)
                )
        return [self.max_lr * factor for _ in self.base_lrs]


class WarmupScheduler(_LRScheduler):
    """Learning rate warmup scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        if self.base_scheduler:
            return self.base_scheduler.get_last_lr()
        return self.base_lrs

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CyclicCosineScheduler(_LRScheduler):
    """Cyclic cosine learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        cycle_epochs: int,
        lr_decay_epochs: int,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        last_epoch: int = -1,
    ):
        self.cycle_epochs = cycle_epochs
        self.lr_decay_epochs = lr_decay_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.lr_decay_epochs
        decay_factor = 0.5 * (1 + math.cos(math.pi * progress))

        cycle_progress = (self.last_epoch % self.cycle_epochs) / self.cycle_epochs
        lr = self.base_lr + (self.max_lr - self.base_lr) * 0.5 * (
            1 + math.cos(math.pi * cycle_progress)
        )

        return [lr * decay_factor for _ in self.base_lrs]


class PolynomialDecay(_LRScheduler):
    """Polynomial learning rate decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        power: float = 1.0,
        base_lr: float = 1e-3,
        end_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.base_lr = base_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.max_epochs:
            return [self.end_lr for _ in self.base_lrs]

        decay_ratio = (self.max_epochs - self.last_epoch) / self.max_epochs
        decay = decay_ratio**self.power
        return [
            self.end_lr + (base_lr - self.end_lr) * decay for base_lr in self.base_lrs
        ]


class ExponentialWarmup(_LRScheduler):
    """Exponential warmup followed by cosine decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = 0.1 ** (self.warmup_epochs - self.last_epoch - 1)
            return [base_lr * factor for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


class LinearWarmupCosineAnnealing(_LRScheduler):
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-7,
        eta_min: float = 1e-7,
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

        progress = (self.last_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]


class SGDRScheduler(_LRScheduler):
    """Stochastic Gradient Descent with Warm Restarts (SGDR)."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch
        if self.T_cur >= self.T_i:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
