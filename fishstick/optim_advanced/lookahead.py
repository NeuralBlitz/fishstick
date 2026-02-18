import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, List, Optional, Callable
from collections import OrderedDict
import copy


class Lookahead(Optimizer):
    def __init__(self, base_optimizer: Optimizer, k: int = 6, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                if "slow_weights" not in state:
                    state["slow_weights"] = p.data.clone()
                    state["step_count"] = 0

                state["step_count"] += 1

                if state["step_count"] >= self.k:
                    slow_weights = state["slow_weights"]
                    slow_weights.add_(self.alpha * (p.data - slow_weights))
                    p.data.copy_(slow_weights)
                    state["step_count"] = 0

        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.state = self.base_optimizer.state

    def param_groups(self):
        return self.base_optimizer.param_groups


class GradientCentralization(Optimizer):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.ndim > 1:
                    grad = grad - grad.mean(
                        dim=tuple(range(1, grad.ndim)), keepdim=True
                    )
                    p.grad.data = grad

        return self.optimizer.step(closure)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def param_groups(self):
        return self.optimizer.param_groups


class SWA(Optimizer):
    def __init__(
        self, base_optimizer: Optimizer, swa_start: int = 10, swa_freq: int = 5
    ):
        self.base_optimizer = base_optimizer
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.swa_state = {}
        self.swa_started = False

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if not self.swa_started:
                    continue

                if p not in self.swa_state:
                    self.swa_state[p] = p.data.clone()
                else:
                    self.swa_state[p].mul_(0.99).add_(p.data, alpha=0.01)

        return loss

    def update_swa(self):
        self.swa_started = True
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.swa_state:
                    p.data.copy_(self.swa_state[p])

    def swap_swa(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.swa_state:
                    p.data, self.swa_state[p] = (
                        self.swa_state[p].clone(),
                        p.data.clone(),
                    )

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return {"base": self.base_optimizer.state_dict(), "swa": self.swa_state}

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict["base"])
        self.swa_state = state_dict["swa"]

    def param_groups(self):
        return self.base_optimizer.param_groups


class AveragedModel(nn.Module):
    def __init__(self, model: nn.Module, device=None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.device = device
        self.n_averaged = torch.tensor(0, dtype=torch.long)

    def update(self, model: nn.Module):
        self.module.load_state_dict(model.state_dict())

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class EMAModel(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        self.n_averaged = torch.tensor(0, dtype=torch.long)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.module.parameters(), model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
