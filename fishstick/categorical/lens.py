"""
Lens-based bidirectional learning.

Lenses model bidirectional transformations:
- get (forward pass): S → A
- put (backward pass): S × A' → S'

This naturally captures backpropagation as lens composition.
"""

from typing import TypeVar, Callable, Optional, Tuple, Generic
from dataclasses import dataclass
import torch
from torch import Tensor, nn

S = TypeVar("S")
T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")


@dataclass
class Lens(Generic[S, A]):
    """
    Lens (get, put) for bidirectional transformation.

    In neural networks:
    - get: forward pass f_θ(x)
    - put: backward update θ ← θ - η∇_θ L

    Lens composition follows the chain rule:
        L₂ ∘ L₁ = (get₂ ∘ get₁, put₂ ∘ put₁)
    """

    get: Callable[[S], A]
    put: Callable[[S, A], S]

    def __call__(self, s: S) -> A:
        return self.get(s)

    def compose(self, other: "Lens[A, B]") -> "Lens[S, B]":
        """
        Compose two lenses following the chain rule.

        (L₂ ∘ L₁).get(s) = L₂.get(L₁.get(s))
        (L₂ ∘ L₁).put(s, b) = L₁.put(s, L₂.put(L₁.get(s), b))
        """

        def new_get(s: S) -> B:
            a = self.get(s)
            return other.get(a)

        def new_put(s: S, b: B) -> S:
            a = self.get(s)
            a_new = other.put(a, b)
            return self.put(s, a_new)

        return Lens(get=new_get, put=new_put)

    @staticmethod
    def identity() -> "Lens[S, S]":
        """Identity lens."""
        return Lens(get=lambda s: s, put=lambda s, a: a)


class NeuralLens(nn.Module):
    """
    Neural network as a lens.

    Implements forward pass (get) and parameter update (put).
    """

    def __init__(self, module: nn.Module, lr: float = 0.01):
        super().__init__()
        self.module = module
        self.lr = lr
        self._last_input = None
        self._last_output = None

    def get(self, x: Tensor) -> Tensor:
        """Forward pass."""
        self._last_input = x.detach().clone()
        self._last_output = self.module(x)
        return self._last_output

    def forward(self, x: Tensor) -> Tensor:
        return self.get(x)

    def put(self, grad_output: Tensor) -> None:
        """
        Backward pass with parameter update.

        Equivalent to: θ ← θ - η ∇_θ L
        """
        if self._last_output is not None and self._last_input is not None:
            self._last_output.backward(grad_output)
            with torch.no_grad():
                for p in self.module.parameters():
                    if p.grad is not None:
                        p -= self.lr * p.grad
                        p.grad = None


class BidirectionalLearner(nn.Module):
    """
    Learner based on lens composition.

    Models the full learning process as bidirectional dataflow:
    - Forward: input → representation → output
    - Backward: gradient → parameter update → state change
    """

    def __init__(self, modules: list, lr: float = 0.01):
        super().__init__()
        self.lenses = nn.ModuleList(
            [NeuralLens(m, lr) if not isinstance(m, NeuralLens) else m for m in modules]
        )
        self._composed_lens = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through composed lenses."""
        for lens in self.lenses:
            x = lens(x)
        return x

    def backward(self, loss: Tensor) -> None:
        """
        Backward pass through lenses in reverse order.

        This is the categorical dual of forward composition.
        """
        grad = torch.autograd.grad(
            loss, self.lenses[-1]._last_output, retain_graph=True
        )[0]

        for lens in reversed(self.lenses):
            lens.put(grad)
            if lens._last_input is not None and lens._last_input.requires_grad:
                grad = torch.autograd.grad(
                    lens._last_output,
                    lens._last_input,
                    grad_outputs=grad,
                    retain_graph=True,
                )[0]


class Optic(Generic[S, A]):
    """
    Optic: generalized lens with bidirectional state.

    An optic is a pair (res, core) where:
    - res: S → M (resolve to middle state)
    - core: (M, A) → M (core update)
    - reindex: M → S (reindex back)

    Optics subsume lenses and are more general for modeling
    differentiable programming patterns.
    """

    def __init__(
        self,
        resolve: Callable[[S], Tuple],
        core: Callable[[Tuple, A], Tuple],
        reindex: Callable[[Tuple], S],
    ):
        self.resolve = resolve
        self.core = core
        self.reindex = reindex

    def get(self, s: S) -> A:
        m = self.resolve(s)
        return m[0] if isinstance(m, tuple) else m

    def put(self, s: S, a: A) -> S:
        m = self.resolve(s)
        m_new = self.core(m, a)
        return self.reindex(m_new)


def lens_chain(*lenses: Lens) -> Lens:
    """Compose multiple lenses into a chain."""
    result = Lens.identity()
    for lens in lenses:
        result = result.compose(lens)
    return result
