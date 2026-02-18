import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


def clip_gradients(
    model: nn.Module, max_norm: float, norm_type: float = 2.0
) -> torch.Tensor:
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
    return total_norm


def add_noise(
    parameters: Iterator[nn.Parameter],
    noise_multiplier: float,
    max_norm: float,
    device: torch.device,
) -> None:
    for param in parameters:
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * noise_multiplier * max_norm
            param.grad.add_(noise.to(device))


def compute_rdp_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
    orders: Optional[List[float]] = None,
) -> Tuple[float, List[float]]:
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    rdp = compute_rdp(sample_rate, noise_multiplier, orders)
    cumulative_rdp = rdp * steps

    epsilon = compute_epsilon(cumulative_rdp, delta)
    return epsilon, orders


def compute_rdp(q: float, sigma: float, orders: List[float]) -> List[float]:
    rdp_values = []
    for alpha in orders:
        if alpha == 1:
            rdp_values.append(float("inf"))
        else:
            cpp = cpp_alpha(alpha, q, sigma)
            rdp_values.append(cpp)
    return rdp_values


def cpp_alpha(alpha: float, q: float, sigma: float) -> float:
    if q == 0:
        return 0.0

    if alpha == 1:
        return float("inf")

    a = 1.0 / (alpha - 1)
    val = 0.5 * ((alpha - 1) * (1 + q * (math.exp(a * sigma**2) - 1)) / sigma**2)
    return max(0, val)


def compute_epsilon(
    rdp: List[float], delta: float, target_delta: Optional[float] = None
) -> float:
    if target_delta is None:
        target_delta = delta

    min_epsilon = float("inf")
    for i, rdp_val in enumerate(rdp):
        if rdp_val < float("inf"):
            epsilon = rdp_val - math.log(target_delta) / (i + 1)
            min_epsilon = min(min_epsilon, epsilon)

    return max(0, min_epsilon)


class PrivacyEngine:
    def __init__(
        self,
        model: nn.Module,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        target_delta: float = 1e-5,
        secure_rng: bool = False,
    ):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.secure_rng = secure_rng

        self.steps = 0
        self.accumulated_epsilon = 0.0

        self._rng = torch.Generator()

    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        self.steps += 1

        loss.backward()

        total_norm = clip_gradients(self.model, self.max_grad_norm)

        add_noise(
            self.model.parameters(),
            self.noise_multiplier,
            self.max_grad_norm,
            self.model.parameters().__next__().device,
        )

        epsilon, orders = compute_rdp_epsilon(
            self.noise_multiplier, 1.0, self.steps, self.target_delta
        )
        self.accumulated_epsilon = epsilon

        return {
            "epsilon": epsilon,
            "delta": self.target_delta,
            "max_grad_norm": total_norm.item()
            if isinstance(total_norm, torch.Tensor)
            else total_norm,
            "noise_multiplier": self.noise_multiplier,
            "steps": self.steps,
        }

    def get_privacy_spent(self) -> Tuple[float, float]:
        return self.accumulated_epsilon, self.target_delta

    def attach(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

        original_step = optimizer.step

        def dp_step():
            for param_group in optimizer.param_groups:
                for p in param_group["params"]:
                    if p.grad is None:
                        continue
                    p.grad = p.grad
            return original_step()

        optimizer.step = dp_step

    def disable(self) -> None:
        self._disabled = True

    def enable(self) -> None:
        self._disabled = False


class DPSGDOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        minibatch_size: int = 32,
        generator: Optional[torch.Generator] = None,
    ):
        defaults = dict(
            lr=lr,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            minibatch_size=minibatch_size,
        )
        super().__init__(params, defaults)
        self.generator = generator or torch.Generator()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                grad_norm = torch.norm(grad, p=2)
                clip_factor = group["max_grad_norm"] / (grad_norm + 1e-6)
                clip_factor = min(clip_factor, 1.0)
                clipped_grad = grad * clip_factor

                noise = (
                    torch.randn_like(grad)
                    * group["noise_multiplier"]
                    * group["max_grad_norm"]
                )
                noised_grad = clipped_grad + noise

                p.data.add_(noised_grad, alpha=-group["lr"])

        return loss


def apply_dp_sgd(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    secure_rng: bool = False,
) -> PrivacyEngine:
    privacy_engine = PrivacyEngine(
        model=model,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        secure_rng=secure_rng,
    )
    privacy_engine.attach(optimizer)
    return privacy_engine


def compute_sigma_from_epsilon(
    target_epsilon: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
    tolerance: float = 1e-3,
) -> float:
    low, high = 0.1, 100.0

    for _ in range(50):
        mid = (low + high) / 2
        epsilon, _ = compute_rdp_epsilon(mid, sample_rate, steps, delta)

        if abs(epsilon - target_epsilon) < tolerance:
            return mid
        elif epsilon < target_epsilon:
            low = mid
        else:
            high = mid

    return (low + high) / 2
