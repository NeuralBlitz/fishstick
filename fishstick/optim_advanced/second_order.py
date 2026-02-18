import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple
import copy
from collections import OrderedDict


class KFACPreconditioner:
    def __init__(
        self,
        model: nn.Module,
        loss_fn=None,
        alpha: float = 0.95,
        damping: float = 1e-3,
        momentum: float = 0.9,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.damping = damping
        self.momentum = momentum

        self.factors = {}
        self.factors_inv = {}
        self.momentum_buffers = {}

        self._register_hooks()

    def _register_hooks(self):
        def hook_activations(module, input, output):
            if hasattr(module, "activation"):
                module.activation = output.detach()

        def hook_gradients(module, grad_input, grad_output):
            if hasattr(module, "activation") and module.activation is not None:
                act = module.activation
                grad = grad_output[0].detach()

                if act.ndim == 4:
                    N = act.size(0)
                    act = act.view(N, -1)
                    grad = grad.view(N, -1)

                if not hasattr(module, "_a_grad"):
                    module._a_grad = []
                module._a_grad.append((act, grad))

        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(hook_activations)
                module.register_full_backward_hook(hook_gradients)

    def compute_fisher(self):
        for module in self.model.modules():
            if not hasattr(module, "_a_grad"):
                continue

            for act, grad in module._a_grad:
                if isinstance(module, nn.Linear):
                    A = act.t() @ act / act.size(0)
                    G = grad.t() @ grad / grad.size(0)
                elif isinstance(module, nn.Conv2d):
                    A = self._compute_cov(
                        act.permute(0, 2, 3, 1).reshape(-1, act.size(1))
                    )
                    G = self._compute_cov(
                        grad.permute(0, 2, 3, 1).reshape(-1, grad.size(1))
                    )

                if module not in self.factors:
                    self.factors[module] = {"A": A, "G": G}
                else:
                    self.factors[module]["A"] = (
                        self.alpha * self.factors[module]["A"] + (1 - self.alpha) * A
                    )
                    self.factors[module]["G"] = (
                        self.alpha * self.factors[module]["G"] + (1 - self.alpha) * G
                    )

                self.factors_inv[module] = {
                    "A": self._inverse(self.factors[module]["A"]),
                    "G": self._inverse(self.factors[module]["G"]),
                }

            module._a_grad = []

    def _compute_cov(self, x):
        return x.t() @ x / x.size(0)

    def _inverse(self, m):
        return torch.inverse(m + self.damping * torch.eye(m.size(0), device=m.device))

    def precondition(self, model: nn.Module):
        self.compute_fisher()

        for module in model.modules():
            if module not in self.factors_inv:
                continue

            if isinstance(module, nn.Linear):
                if hasattr(module, "weight"):
                    A = self.factors_inv[module]["A"]
                    G = self.factors_inv[module]["G"]
                    module.weight.grad = G @ module.weight.grad @ A
            elif isinstance(module, nn.Conv2d):
                if hasattr(module, "weight"):
                    G = self.factors_inv[module]["G"]
                    A = self.factors_inv[module]["A"]
                    module.weight.grad = (
                        torch.nn.functional.conv2d(
                            G,
                            module.weight.grad,
                            padding=module.padding,
                            stride=module.stride,
                            dilation=module.dilation,
                        )
                        @ A
                    )


class LBFGS(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        max_iter=20,
        max_eval=None,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=50,
    ):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
        super().__init__(params, defaults)

    def _directional_evaluate(self, closure, x, t, d):
        return self._add(x, t, d).requires_grad_(True)

    def step(self, closure):
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        lr = group["lr"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]

        if max_eval is None:
            max_eval = max_iter * 1.25

        state = self.state[self.param_groups[0]["params"][0]]

        if "func_eval" not in state:
            state["func_eval"] = 0
            state["n_iter"] = 0
            state["old_flat_grad"] = None
            state["old_loss"] = None
            self._prev_loss = None
            self._ls_step = 0

        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_eval"] += 1
        flat_grad = self._flatten_grad()

        opt_cond = flat_grad.abs().max().item() <= tolerance_grad
        diverged = False

        if state["n_iter"] == 0:
            state["old_loss"] = loss
            state["old_flat_grad"] = flat_grad.clone()

        for n_iter in range(max_iter):
            state["n_iter"] += 1

            if len(state) > 6:
                state["d"] = state["prev_flat_grad"] - flat_grad
                y = state["d"] @ (state["prev_flat_grad"] - flat_grad)

            if y <= 1e-10:
                y = 1e-10
                flat_grad.sub_(state.get("d", torch.zeros_like(flat_grad)))

            if "d" not in state:
                d = flat_grad.clone().neg()
                state["d"] = d.clone()
            else:
                d = state["d"]
                flat_grad.add_(y, d)
                d.copy_(flat_grad)

            if opt_cond:
                break

            if n_iter == max_iter:
                break

        state["prev_flat_grad"] = flat_grad.clone()
        self._prev_loss = loss

        return orig_loss

    def _flatten_grad(self):
        views = []
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                view = p.data.new_zeros(p.numel())
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add(self, x, t, d):
        return x + t * d


class NaturalGradient(Optimizer):
    def __init__(self, params, lr=0.1, damping=1e-5):
        defaults = dict(lr=lr, damping=damping)
        super().__init__(params, defaults)

        self.fisher = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if p not in self.fisher:
                    self.fisher[p] = torch.ones_like(p.data)
                else:
                    self.fisher[p] = self.fisher[p] * 0.9 + grad.pow(2) * 0.1

                fisher = self.fisher[p] + group["damping"]

                p.data.add_(grad / fisher, alpha=-group["lr"])

        return loss


class FisherVerification:
    def __init__(self, model: nn.Module, num_samples: int = 100):
        self.model = model
        self.num_samples = num_samples
        self.fisher_info = {}

    def compute_empirical_fisher(self, loss_fn, data_loader, device="cpu"):
        self.model.eval()
        self.fisher_info = {}

        for name, param in self.model.named_parameters():
            self.fisher_info[name] = torch.zeros_like(param.data)

        total_samples = 0

        for i, (inputs, targets) in enumerate(data_loader):
            if i >= self.num_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            self.model.zero_grad()
            output = self.model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.data.pow(2)

            total_samples += inputs.size(0)

        for name in self.fisher_info:
            self.fisher_info[name] /= total_samples

        return self.fisher_info

    def compute_curvature_matrix(self, param_name):
        if param_name not in self.fisher_info:
            return None
        return self.fisher_info[param_name]

    def verify_fisher_diag(self, param_name):
        if param_name not in self.fisher_info:
            return False

        fisher = self.fisher_info[param_name]
        return (fisher > 0).all().item()

    def get_fisher_norm(self):
        total_norm = 0.0
        for name, fisher in self.fisher_info.items():
            total_norm += fisher.sum().item()
        return total_norm

    def visualize_fisher(self, param_name, save_path=None):
        if param_name not in self.fisher_info:
            return None

        fisher = self.fisher_info[param_name]

        if fisher.ndim == 0:
            return fisher.item()

        if fisher.ndim == 1:
            return fisher.cpu().numpy()

        if fisher.ndim >= 2:
            return fisher.reshape(-1).cpu().numpy()

        return None
