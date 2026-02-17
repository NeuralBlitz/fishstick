"""
Advanced Optimizer Algorithms
"""

import math
from typing import Optional
import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer with weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                p.addcdiv_(exp_avg, denom, value=-step_size)

                state["step"] += 1

        return loss


class LAMB(Optimizer):
    """LAMB (Layer-wise Adaptive Moments) optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                r = exp_avg / (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group["eps"]
                )

                if group["weight_decay"] > 0:
                    r = r + group["weight_decay"] * p

                ratio = r.norm() / (p.norm() + group["eps"])
                ratio = (ratio + 1) / (ratio.abs() + 1)

                p.add_(exp_avg / bias_correction1, alpha=-group["lr"] * ratio)

                state["step"] += 1

        return loss


class NovoGrad(Optimizer):
    """NovoGrad optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = grad.pow(2).mean()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                r = (
                    exp_avg
                    / bias_correction1
                    / (exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group["eps"])
                )

                p.add_(r, alpha=-group["lr"])

                state["exp_avg_sq"] = (
                    beta2 * state["exp_avg_sq"] + (1 - beta2) * grad.pow(2).mean()
                )
                state["step"] += 1

        return loss


class Ranger(Optimizer):
    """RAdam + Lookahead optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

                state["step"] += 1

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

        return loss


class Lookahead(Optimizer):
    """Lookahead optimizer wrapper."""

    def __init__(self, base_optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_counter = 0

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            for p in self.base_optimizer.param_groups[0]["params"]:
                if p.grad is None:
                    continue
                state = self.base_optimizer.state[p]
                if "slow_weights" not in state:
                    state["slow_weights"] = p.data.clone()
                else:
                    state["slow_weights"].add_(
                        p.data - state["slow_weights"], alpha=self.alpha
                    )
                    p.data.copy_(state["slow_weights"])

        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()
