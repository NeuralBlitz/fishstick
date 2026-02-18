import math
from typing import Optional, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingWarmRestarts as _CosineAnnealingWarmRestarts,
)


class CosineAnnealingWarmRestarts(_CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)


class OneCycleLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ):
        if total_steps is None and (epochs is None or steps_per_epoch is None):
            raise ValueError(
                "Either total_steps or both epochs and steps_per_epoch must be provided"
            )

        self.max_lr = max_lr
        self.total_steps = total_steps or (epochs * steps_per_epoch)
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        pct = step / self.total_steps

        if self.three_phase:
            if pct < self.pct_start / 2:
                phase_pct = pct / (self.pct_start / 2)
                return [
                    self.max_lr * (1 - phase_pct) / self.div_factor
                    + self.max_lr * phase_pct
                ]
            elif pct < self.pct_start:
                phase_pct = (pct - self.pct_start / 2) / (self.pct_start / 2)
                return [
                    self.max_lr * (1 - phase_pct)
                    + self.max_lr * phase_pct / self.div_factor
                ]
            else:
                phase_pct = (pct - self.pct_start) / (1 - self.pct_start)
                return [
                    self.max_lr / self.final_div_factor
                    + (self.max_lr / self.div_factor) * (1 - phase_pct)
                ]
        else:
            if pct < self.pct_start:
                phase_pct = pct / self.pct_start
                if self.anneal_strategy == "cos":
                    return [
                        self.max_lr / self.div_factor
                        + (self.max_lr - self.max_lr / self.div_factor)
                        * (1 + math.cos(math.pi * phase_pct))
                        / 2
                    ]
                else:
                    return [self.max_lr * (1 - phase_pct * (1 - 1 / self.div_factor))]
            else:
                phase_pct = (pct - self.pct_start) / (1 - self.pct_start)
                if self.anneal_strategy == "cos":
                    return [
                        self.max_lr / self.final_div_factor
                        + (
                            self.max_lr / self.div_factor
                            - self.max_lr / self.final_div_factor
                        )
                        * (1 + math.cos(math.pi * phase_pct))
                        / 2
                    ]
                else:
                    return [
                        self.max_lr
                        / self.div_factor
                        * (1 - phase_pct * (1 - 1 / self.final_div_factor))
                    ]


class CyclicLRWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Optional[callable] = None,
        scale_mode: str = "cycle",
        last_epoch: int = -1,
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_steps:
            warmup_factor = (step + 1) / self.warmup_steps
            return [self.base_lr + (self.max_lr - self.base_lr) * warmup_factor]

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)

        if self.scale_fn is not None:
            scale = self.scale_fn(
                cycle=1 + (step - self.warmup_steps) / self._cycle_length(),
                scale_mode=self.scale_mode,
            )
        elif self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (progress - 1))
        elif self.mode == "exp_range":
            scale = self.gamma**step
        else:
            scale = 1.0

        cycle = 1 + (step - self.warmup_steps) / self._cycle_length()
        x = 1.0 - progress
        return [
            self.base_lr
            + (self.max_lr - self.base_lr)
            * max(0, scale * (1 + math.cos(math.pi * x / 2)) / 2)
        ]

    def _cycle_length(self):
        return self.total_steps - self.warmup_steps


class PolynomialLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
        steps_per_epoch: int = 1,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = steps_per_epoch

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.warmup_epochs * self.steps_per_epoch:
            warmup_factor = (step + 1) / (self.warmup_epochs * self.steps_per_epoch)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        decay_step = step - self.warmup_epochs * self.steps_per_epoch
        total_decay_steps = (
            self.total_epochs - self.warmup_epochs
        ) * self.steps_per_epoch
        decay_factor = (1 - decay_step / total_decay_steps) ** self.power

        return [
            self.min_lr + (base_lr - self.min_lr) * decay_factor
            for base_lr in self.base_lrs
        ]
