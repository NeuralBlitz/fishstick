import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from scipy import stats


class RandomizedSmoothingCertifier:
    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.25,
        num_samples: int = 100,
        num_classes: int = 10,
        alpha: float = 0.05,
    ):
        self.model = model
        self.sigma = sigma
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.alpha = alpha

    def certify(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()

        batch_size = x.shape[0]

        noise = (
            torch.randn(batch_size, self.num_samples, *x.shape[1:], device=x.device)
            * self.sigma
        )
        x_noisy = (x.unsqueeze(1) + noise).view(-1, *x.shape[1:])
        x_noisy = torch.clamp(x_noisy, 0, 1)

        with torch.no_grad():
            outputs = self.model(x_noisy)
            preds = outputs.argmax(dim=1).view(batch_size, self.num_samples)

        pred_counts = F.one_hot(preds, self.num_classes).sum(dim=1).float()
        pred_probs = pred_counts / self.num_samples

        predictions = pred_probs.argmax(dim=1)
        top2_probs = pred_probs.topk(2, dim=1).values

        radius = self._compute_radius(top2_probs[:, 0], top2_probs[:, 1])

        return predictions, radius

    def _compute_radius(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        diff = p_a - p_b
        radius = self.sigma * torch.sqrt(2 * diff.clamp(min=1e-10))
        return radius

    def batch_certify(
        self, x: torch.Tensor, batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_predictions = []
        all_radii = []

        for i in range(0, x.shape[0], batch_size):
            batch_x = x[i : i + batch_size]
            preds, radii = self.certify(batch_x)
            all_predictions.append(preds)
            all_radii.append(radii)

        return torch.cat(all_predictions), torch.cat(all_radii)


class IBP:
    def __init__(self, model: nn.Module, epsilon: float = 0.1, num_classes: int = 10):
        self.model = model
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_ub = x + self.epsilon
        x_lb = x - self.epsilon
        x_ub = torch.clamp(x_ub, 0, 1)
        x_lb = torch.clamp(x_lb, 0, 1)

        return x_lb, x_ub

    def compute_ibp_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        step_size: float = 0.01,
        num_steps: int = 5,
    ) -> torch.Tensor:
        x_lb, x_ub = self.forward_bounds(x)

        for _ in range(num_steps):
            x_ub.requires_grad = True
            x_lb.requires_grad = True

            out_ub = self.model(x_ub)
            out_lb = self.model(x_lb)

            loss_ub = F.cross_entropy(out_ub, y)
            loss_lb = F.cross_entropy(out_lb, y)

            loss = (loss_ub + loss_lb) / 2

            if x_ub.grad is not None:
                with torch.no_grad():
                    x_ub = x_ub + step_size * x_ub.grad.sign()
                    x_ub = torch.clamp(x_ub, x_lb.detach(), 1)

            if x_lb.grad is not None:
                with torch.no_grad():
                    x_lb = x_lb - step_size * x_lb.grad.sign()
                    x_lb = torch.clamp(x_lb, 0, x_ub.detach())

        final_out = (self.model(x_ub) + self.model(x_lb)) / 2
        return F.cross_entropy(final_out, y)

    def certify(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_lb, x_ub = self.forward_bounds(x)

        with torch.no_grad():
            out_lb = self.model(x_lb)
            out_ub = self.model(x_ub)

        predictions = out_ub.argmax(dim=1)

        worst_case_probs = torch.minimum(out_lb, out_ub)
        certified_radius = (
            worst_case_probs.max(dim=1)[0] - worst_case_probs.median(dim=1)[0]
        )

        return predictions, certified_radius * self.epsilon


class CROWN:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        num_classes: int = 10,
        bound_type: bool = True,
    ):
        self.model = model
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.bound_type = bound_type

    def compute_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_lb = torch.clamp(x - self.epsilon, 0, 1)
        x_ub = torch.clamp(x + self.epsilon, 0, 1)

        def compute_upper_bound(module, input, output):
            return output.clamp(0, 1)

        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                hooks.append(module.register_forward_hook(compute_upper_bound))

        with torch.no_grad():
            _ = self.model(x_ub)
            _ = self.model(x_lb)

        for hook in hooks:
            hook.remove()

        return x_lb, x_ub

    def linear_bound(
        self, W: torch.Tensor, b: torch.Tensor, x_lb: torch.Tensor, x_ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        W_pos = W.clamp(min=0)
        W_neg = W.clamp(max=0)

        center = (x_ub + x_lb) / 2
        width = (x_ub - x_lb) / 2

        lower = W @ center - (W_pos.abs() + W_neg.abs()) @ width + b
        upper = W @ center + (W_pos.abs() + W_neg.abs()) @ width + b

        return lower, upper

    def ibp_crown_bound(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_lb, x_ub = self.compute_bounds(x)

        features = x_lb
        bounds = [(x_lb, x_ub)]

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight
                b = module.bias

                lb, ub = self.linear_bound(W, b, bounds[-1][0], bounds[-1][1])
                bounds.append((lb, ub))
            elif isinstance(module, nn.Conv2d):
                W = module.weight
                b = module.bias if module.bias is not None else torch.zeros(W.shape[0])

                lb, ub = self.linear_bound(
                    W.view(W.shape[0], -1),
                    b,
                    bounds[-1][0].flatten(1),
                    bounds[-1][1].flatten(1),
                )
                bounds.append(
                    (lb.view(-1, *module.kernel_size), ub.view(-1, *module.kernel_size))
                )
            elif isinstance(module, nn.ReLU):
                lb_prev, ub_prev = bounds[-1]
                lb_new = F.relu(lb_prev)
                ub_new = F.relu(ub_prev)
                bounds.append((lb_new, ub_new))

        return bounds[-1]

    def compute_robustness_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lb, ub = self.ibp_crown_bound(x)

        target_lb = lb[torch.arange(x.shape[0]), y]
        other_ub = ub.scatter(1, y.unsqueeze(1), -float("inf")).max(dim=1)[0]

        margin = target_lb - other_ub

        return -margin.mean()


class CROWN_IBP:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        num_classes: int = 10,
        alpha: float = 0.5,
    ):
        self.model = model
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.alpha = alpha
        self.ibp = IBP(model, epsilon, num_classes)
        self.crown = CROWN(model, epsilon, num_classes)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ibp_loss = self.ibp.compute_ibp_loss(x, y)
        crown_loss = self.crown.compute_robustness_loss(x, y)

        return self.alpha * ibp_loss + (1 - self.alpha) * crown_loss

    def certify(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ibp_preds, ibp_radii = self.ibp.certify(x)
        crown_lb, crown_ub = self.crown.ibp_crown_bound(x)

        crown_radii = (crown_ub.max(dim=1)[0] - crown_lb.min(dim=1)[0]) / 2

        combined_radii = torch.minimum(ibp_radii, crown_radii)

        predictions = ibp_preds

        return predictions, combined_radii


class FastCertify:
    def __init__(
        self,
        model: nn.Module,
        sigma: float = 0.25,
        num_classes: int = 10,
        n0: int = 100,
        n: int = 100000,
        alpha: float = 0.001,
    ):
        self.model = model
        self.sigma = sigma
        self.num_classes = num_classes
        self.n0 = n0
        self.n = n
        self.alpha = alpha

    def certify(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()

        batch_size = x.shape[0]

        noise = (
            torch.randn(batch_size, self.n0, *x.shape[1:], device=x.device) * self.sigma
        )
        x_noisy = (x.unsqueeze(1) + noise).view(-1, self.n0 * batch_size, *x.shape[1:])
        x_noisy = torch.clamp(x_noisy, 0, 1)

        with torch.no_grad():
            outputs = self.model(x_noisy)
            preds = outputs.argmax(dim=1).view(batch_size, self.n0)

        pred_counts = F.one_hot(preds, self.num_classes).sum(dim=1).float()
        pred_probs = pred_counts / self.n0

        predictions = pred_probs.argmax(dim=1)

        pA = pred_probs.max(dim=1)[0]

        pB_all = []
        for c in range(self.num_classes):
            mask = preds != c
            pB_c = (preds == c).float().sum(dim=1) / self.n0
            pB_all.append(pB_c)

        pB = torch.stack(pB_all, dim=1).max(dim=1)[0]

        n = self.n
        alpha = self.alpha

        def compute_radius(pA_val, pB_val):
            if pA_val < pB_val:
                return torch.tensor(0.0)

            delta = pA_val - pB_val

            try:
                from scipy.stats import norm

                n0, n = self.n0, self.n

                s = torch.sqrt(-torch.log(1 - alpha / (2 * self.num_classes)))

                radius = self.sigma / 2 * (1 - s / torch.sqrt(torch.tensor(n0)))

                return radius.clamp(min=0)
            except:
                return torch.tensor(0.0)

        radii = torch.stack(
            [compute_radius(pA[i], pB[i]) for i in range(batch_size)]
        ).to(x.device)

        is_certified = (pA - pB > 0).float()

        return predictions, radii, is_certified


class IntervalBoundedNetwork:
    def __init__(self, model: nn.Module):
        self.model = model
        self.lower_bounds = {}
        self.upper_bounds = {}

    def forward_with_bounds(
        self, x: torch.Tensor, epsilon: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_lb = torch.clamp(x - epsilon, 0, 1)
        x_ub = torch.clamp(x + epsilon, 0, 1)

        out_lb, out_ub = self._propagate_bounds(x_lb, x_ub)

        return out_lb, out_ub, (out_lb + out_ub) / 2

    def _propagate_bounds(
        self, x_lb: torch.Tensor, x_ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = (x_lb, x_ub)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight
                b = (
                    module.bias
                    if module.bias is not None
                    else torch.zeros(W.shape[0], device=W.device)
                )

                lb, ub = self._linear_bounds(W, b, bounds[0], bounds[1])
                bounds = (lb, ub)

            elif isinstance(module, nn.Conv2d):
                W = module.weight
                b = (
                    module.bias
                    if module.bias is not None
                    else torch.zeros(W.shape[0], device=W.device)
                )

                lb, ub = self._conv_bounds(W, b, bounds[0], bounds[1], module.padding)
                bounds = (lb, ub)

            elif isinstance(module, nn.ReLU):
                lb, ub = bounds
                new_lb = F.relu(lb)
                new_ub = F.relu(ub)
                bounds = (new_lb, new_ub)

            elif isinstance(module, nn.Flatten):
                lb, ub = bounds
                bounds = (lb.flatten(1), ub.flatten(1))

        return bounds

    def _linear_bounds(
        self, W: torch.Tensor, b: torch.Tensor, x_lb: torch.Tensor, x_ub: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        W_pos = W.clamp(min=0)
        W_neg = W.clamp(max=0)

        center = (x_ub + x_lb) / 2
        radius = (x_ub - x_lb) / 2

        lb = W_pos @ center + W_neg @ center - (W_pos.abs() + W_neg.abs()) @ radius + b
        ub = W_pos @ center + W_neg @ center + (W_pos.abs() + W_neg.abs()) @ radius + b

        return lb, ub

    def _conv_bounds(
        self,
        W: torch.Tensor,
        b: torch.Tensor,
        x_lb: torch.Tensor,
        x_ub: torch.Tensor,
        padding: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        W_flat = W.view(W.shape[0], -1)
        x_lb_flat = F.pad(x_lb, [padding] * 4).flatten(1)
        x_ub_flat = F.pad(x_ub, [padding] * 4).flatten(1)

        return self._linear_bounds(W_flat, b, x_lb_flat, x_ub_flat)
