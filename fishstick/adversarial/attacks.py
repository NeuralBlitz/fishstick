import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, List
import numpy as np


class FGSM:
    def __init__(self, model: nn.Module, epsilon: float = 0.03):
        self.model = model
        self.epsilon = epsilon

    def attack(
        self, x: torch.Tensor, y: torch.Tensor, loss_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        x_adv = x.detach().clone().requires_grad_(True)
        outputs = self.model(x_adv)
        loss = loss_fn(outputs, y)
        loss.backward()

        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + self.epsilon * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv


class PGD:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.001,
        steps: int = 10,
        random_start: bool = True,
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(
        self, x: torch.Tensor, y: torch.Tensor, loss_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        x_adv = x.detach().clone()

        if self.random_start:
            x_adv = x_adv + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        x_adv = x_adv.requires_grad_(True)

        for _ in range(self.steps):
            outputs = self.model(x_adv)
            loss = loss_fn(outputs, y)
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv + self.alpha * x_adv.grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
                x_adv.requires_grad_(True)

        return x_adv


class CW:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        max_iter: int = 1000,
        initial_const: float = 0.01,
        targeted: bool = False,
    ):
        self.model = model
        self.lr = lr
        self.max_iter = max_iter
        self.initial_const = initial_const
        self.targeted = targeted

    def attack(
        self, x: torch.Tensor, y: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.model.eval()

        batch_size = x.shape[0]
        const = torch.full((batch_size,), self.initial_const, device=x.device)

        delta = torch.zeros_like(x).requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        x_adv = x + delta

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            x_adv = torch.clamp(x + delta, 0, 1)
            outputs = self.model(x_adv)

            if self.targeted and target is not None:
                loss = -F.cross_entropy(outputs, target)
            else:
                loss = F.cross_entropy(outputs, y)

            _, predicted = outputs.max(1)
            correct = predicted.eq(y).float()

            dist = torch.sum((x_adv - x).view(batch_size, -1) ** 2, dim=1)
            loss = loss + const * dist

            loss.sum().backward()
            optimizer.step()

            const = const + 0.5 * (1 - correct).detach()

        return torch.clamp(x + delta, 0, 1)


class DeepFool:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        max_iter: int = 50,
        overshoot: float = 0.02,
    ):
        self.model = model
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.overshoot = overshoot

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.eval()

        x_adv = x.detach().clone().requires_grad_(True)
        batch_size = x.shape[0]
        perturbations = torch.zeros_like(x)

        for i in range(batch_size):
            x_i = x_adv[i : i + 1].clone().detach().requires_grad_(True)
            current_pred = y[i].item()

            for _ in range(self.max_iter):
                outputs = self.model(x_i)
                pred = outputs.argmax(dim=1).item()

                if pred != current_pred:
                    break

                gradients = []
                for c in range(self.num_classes):
                    if c == current_pred:
                        continue
                    self.model.zero_grad()
                    loss = outputs[0, c]
                    loss.backward(retain_graph=True)
                    gradients.append(x_i.grad.data.clone())

                if not gradients:
                    break

                combined_grad = gradients[0]
                for g in gradients[1:]:
                    combined_grad = combined_grad + g

                norm = torch.norm(combined_grad.view(-1))
                if norm > 0:
                    perturbation = (
                        combined_grad / norm * (x[i : i + 1] - x_i).abs().min()
                    )
                    x_i = x_i + (1 + self.overshoot) * perturbation
                    x_i = torch.clamp(x_i, 0, 1)
                    x_i.requires_grad_(True)

            perturbations[i] = x_i - x[i]

        x_adv = x + perturbations
        return torch.clamp(x_adv, 0, 1)


class AutoAttack:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        versions: Optional[List[str]] = None,
    ):
        self.model = model
        self.epsilon = epsilon
        self.versions = versions or ["fab", "square", "apgd-ce", "apgd-dlr"]
        self.attacks = {}

        self._setup_attacks()

    def _setup_attacks(self):
        self.attacks["fgsm"] = FGSM(self.model, self.epsilon)
        self.attacks["pgd"] = PGD(self.model, self.epsilon, steps=10)

        if "square" in self.versions:
            self.attacks["square"] = SquareAttack(self.model, self.epsilon)

        if "apgd-ce" in self.versions or "apgd-dlr" in self.versions:
            self.attacks["apgd"] = APGD(self.model, self.epsilon)

    def attack(
        self, x: torch.Tensor, y: torch.Tensor, version: Optional[str] = None
    ) -> torch.Tensor:
        if version is not None:
            if version == "fgsm":
                return self.attacks["fgsm"].attack(x, y)
            elif version == "pgd":
                return self.attacks["pgd"].attack(x, y)
            elif version == "square" and "square" in self.attacks:
                return self.attacks["square"].attack(x, y)
            elif version in ["apgd-ce", "apgd-dlr"] and "apgd" in self.attacks:
                return self.attacks["apgd"].attack(x, y, version)

        x_adv = x.clone()
        for v in self.versions:
            if v == "fgsm":
                x_adv = self.attacks["fgsm"].attack(x_adv, y)
            elif v == "pgd":
                x_adv = self.attacks["pgd"].attack(x_adv, y)
            elif v == "square" and "square" in self.attacks:
                x_adv = self.attacks["square"].attack(x_adv, y)
            elif v in ["apgd-ce", "apgd-dlr"] and "apgd" in self.attacks:
                x_adv = self.attacks["apgd"].attack(x_adv, y, v)

        return x_adv


class SquareAttack:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        n_restarts: int = 1,
        max_iter: int = 5000,
    ):
        self.model = model
        self.epsilon = epsilon
        self.n_restarts = n_restarts
        self.max_iter = max_iter

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_adv = x.clone()

        for i in range(batch_size):
            x_i = x[i : i + 1]
            best_adv = x_i.clone()
            best_loss = -float("inf")

            for _ in range(self.max_iter):
                delta = torch.rand_like(x_i) * 2 * self.epsilon - self.epsilon
                x_candidate = torch.clamp(x_i + delta, 0, 1)

                with torch.no_grad():
                    outputs = self.model(x_candidate)
                    loss = F.cross_entropy(outputs, y[i : i + 1])

                if loss > best_loss:
                    best_loss = loss
                    best_adv = x_candidate.clone()

                if outputs.argmax(dim=1) != y[i]:
                    best_adv = x_candidate
                    break

            x_adv[i] = best_adv

        return x_adv


class APGD:
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        steps: int = 100,
        target: bool = False,
    ):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.target = target

    def attack(
        self, x: torch.Tensor, y: torch.Tensor, version: str = "apgd-ce"
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        x_adv = x.clone().requires_grad_(True)

        delta = torch.zeros_like(x)

        for step in range(self.steps):
            outputs = self.model(x_adv)

            if version == "apgd-ce":
                loss = F.cross_entropy(outputs, y)
            else:
                probs = F.softmax(outputs, dim=1)
                loss = -probs.max(dim=1)[0]

            loss.backward()

            with torch.no_grad():
                delta = delta + x_adv.grad.sign() * (self.epsilon / self.steps)
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
                x_adv.requires_grad_(True)

        return x_adv


class HotFlip:
    def __init__(self, model: nn.Module, n_flips: int = 1):
        self.model = model
        self.n_flips = n_flips

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.detach().clone().requires_grad_(True)

        for _ in range(self.n_flips):
            outputs = self.model(x_adv)
            loss = F.cross_entropy(outputs, y)
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad
                flip_dir = grad.sign()
                x_adv = x_adv + flip_dir * 0.1
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv.requires_grad_(True)

        return x_adv
