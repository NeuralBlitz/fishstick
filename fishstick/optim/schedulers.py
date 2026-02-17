"""
Advanced Learning Rate Schedulers
"""

import math
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealing(_LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(
        self,
        optimizer,
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

        progress = (self.last_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_decay
            for base_lr in self.base_lrs
        ]


class LinearWarmup(_LRScheduler):
    """Linear warmup followed by constant or linear decay."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 1e-6,
        decay_lr: float = 1e-5,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.decay_lr = decay_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        return [
            base_lr - (base_lr - self.decay_lr) * progress for base_lr in self.base_lrs
        ]


class CyclicScheduler(_LRScheduler):
    """Cyclic learning rate scheduler."""

    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float,
        step_size: int,
        mode: str = "triangular",
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)

        scale = self.gamma**self.last_epoch

        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1 / (2 ** (cycle - 1))
        else:
            scale_factor = 0.9

        return [
            self.base_lr + (self.max_lr - self.base_lr) * scale_factor * x * scale
            for _ in self.base_lrs
        ]


class OneCycle(_LRScheduler):
    """One cycle learning rate policy."""

    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.total_steps * self.pct_start:
            pct = step / (self.total_steps * self.pct_start)
            factor = (self.div_factor - 1) / self.div_factor + pct / self.div_factor
        else:
            pct = (step - self.total_steps * self.pct_start) / (
                self.total_steps * (1 - self.pct_start)
            )
            factor = (1 / self.div_factor) * (
                1 + pct * (self.final_div_factor - 1) / self.final_div_factor
            )

        return [self.max_lr * factor for _ in self.base_lrs]


class PolynomialWarmup(_LRScheduler):
    """Polynomial decay with warmup."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        power: float = 1.0,
        warmup_start_lr: float = 1e-6,
        end_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.power = power
        self.warmup_start_lr = warmup_start_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]

        decay_ratio = (self.max_epochs - self.last_epoch) / (
            self.max_epochs - self.warmup_epochs
        )
        decay = decay_ratio**self.power

        return [
            self.end_lr + (base_lr - self.end_lr) * decay for base_lr in self.base_lrs
        ]
