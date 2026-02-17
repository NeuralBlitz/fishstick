"""Categorical structures for compositional learning systems."""

from typing import TypeVar, Generic, Callable, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
from torch import Tensor

from ..core.types import Morphism, T, S, A, B

Obj = TypeVar("Obj")
Morph = TypeVar("Morph")
F = TypeVar("F")
G = TypeVar("G")


@dataclass
class Object:
    """Object in a category."""

    name: str
    shape: Optional[Tuple[int, ...]] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


class Category(ABC):
    """Abstract category with objects and morphisms."""

    @abstractmethod
    def objects(self) -> List[Object]:
        """Return list of objects in category."""
        pass

    @abstractmethod
    def morphisms(self, source: Object, target: Object) -> List[Morphism]:
        """Return morphisms from source to target."""
        pass

    @abstractmethod
    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Compose two morphisms: g ∘ f."""
        pass

    @abstractmethod
    def identity(self, obj: Object) -> Morphism:
        """Return identity morphism for object."""
        pass


class MonoidalCategory(Category):
    """
    Monoidal Category (C, ⊗, I) with tensor product structure.

    Used to model parallel composition of neural network layers
    and data flow through computational graphs.
    """

    def __init__(self, name: str = "Monoidal"):
        self.name = name
        self._objects: Dict[str, Object] = {}
        self._morphisms: Dict[Tuple[str, str], List[Morphism]] = {}

    def add_object(self, obj: Object) -> None:
        self._objects[obj.name] = obj

    def add_morphism(self, source: str, target: str, morph: Morphism) -> None:
        key = (source, target)
        if key not in self._morphisms:
            self._morphisms[key] = []
        self._morphisms[key].append(morph)

    def objects(self) -> List[Object]:
        return list(self._objects.values())

    def morphisms(self, source: Object, target: Object) -> List[Morphism]:
        return self._morphisms.get((source.name, target.name), [])

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        def composed(x):
            return g(f(x))

        return composed

    def identity(self, obj: Object) -> Morphism:
        return lambda x: x

    def tensor_product(self, obj1: Object, obj2: Object) -> Object:
        """
        Tensor product ⊗: C × C → C

        For neural networks, this represents parallel/batch composition.
        """
        new_shape = None
        if obj1.shape is not None and obj2.shape is not None:
            new_shape = obj1.shape + obj2.shape

        return Object(
            name=f"{obj1.name}⊗{obj2.name}",
            shape=new_shape,
            constraints={**obj1.constraints, **obj2.constraints},
        )

    def unit_object(self) -> Object:
        """Return the monoidal unit I."""
        return Object(name="I", shape=(1,))


class TracedMonoidalCategory(MonoidalCategory):
    """
    Traced Symmetric Monoidal Category with trace operator.

    The trace operator models feedback loops and recurrence:
        Tr^U_{A,B}(f: A⊗U → B⊗U) : A → B

    This enables modeling recurrent networks, attention rollouts,
    and memory updates.
    """

    def __init__(self, name: str = "TracedMonoidal"):
        super().__init__(name)

    def trace(self, f: Morphism, dim: int = 0) -> Morphism:
        """
        Apply trace operator to create feedback loop.

        For tensors, this contracts along specified dimension.
        """

        def traced(x: Tensor) -> Tensor:
            if isinstance(x, Tensor):
                result = f(x)
                while result.shape[dim] > 1:
                    result = result.sum(dim=dim, keepdim=True)
                return result.squeeze(dim)
            return f(x)

        return traced


class Functor(Generic[A, B]):
    """
    Functor F: C → D between categories.

    Maps objects to objects and morphisms to morphisms,
    preserving composition and identity:
        F(id) = id
        F(g ∘ f) = F(g) ∘ F(f)

    In ML, a functor represents a learning algorithm mapping
    data category to model category.
    """

    def __init__(
        self,
        on_objects: Callable[[A], B],
        on_morphisms: Callable[[Morphism], Morphism],
        source_cat: Optional[Category] = None,
        target_cat: Optional[Category] = None,
    ):
        self._on_objects = on_objects
        self._on_morphisms = on_morphisms
        self.source = source_cat
        self.target = target_cat

    def __call__(self, x: A) -> B:
        return self._on_objects(x)

    def map_morphism(self, f: Morphism) -> Morphism:
        return self._on_morphisms(f)

    def compose(self, other: "Functor") -> "Functor":
        """Compose two functors: (F ∘ G)(x) = F(G(x))."""
        return Functor(
            on_objects=lambda x: self(other(x)),
            on_morphisms=lambda f: self.map_morphism(other.map_morphism(f)),
        )


class NaturalTransformation(Generic[F, G]):
    """
    Natural transformation α: F ⇒ G between functors.

    A family of morphisms α_A: F(A) → G(A) for each object A,
    satisfying the naturality square:
        G(f) ∘ α_A = α_B ∘ F(f)

    In ML, natural transformations represent training dynamics:
    parameter updates that commute with data transformations.
    """

    def __init__(
        self,
        components: Dict[str, Morphism],
        source_functor: Functor,
        target_functor: Functor,
    ):
        self.components = components
        self.source = source_functor
        self.target = target_functor

    def __getitem__(self, obj_name: str) -> Morphism:
        return self.components.get(obj_name, lambda x: x)

    def verify_naturality(
        self, obj_a: str, obj_b: str, morphism: Morphism, test_input: Tensor
    ) -> bool:
        """
        Verify naturality condition: G(f) ∘ α_A = α_B ∘ F(f)
        """
        alpha_a = self[obj_a]
        alpha_b = self[obj_b]

        f = morphism
        F_f = self.source.map_morphism(f)
        G_f = self.target.map_morphism(f)

        path1 = G_f(alpha_a(test_input))
        path2 = alpha_b(F_f(test_input))

        return torch.allclose(path1, path2, atol=1e-6)


class MonoidalFunctor(Functor):
    """
    Lax monoidal functor between monoidal categories.

    Preserves tensor product up to coherent morphisms:
        F(A ⊗ B) → F(A) ⊗ F(B)
        I_D → F(I_C)

    Enables modeling batch independence and pipeline structure.
    """

    def __init__(
        self,
        on_objects: Callable[[A], B],
        on_morphisms: Callable[[Morphism], Morphism],
        tensor_map: Optional[Callable] = None,
        unit_map: Optional[Callable] = None,
    ):
        super().__init__(on_objects, on_morphisms)
        self.tensor_map = tensor_map or (lambda x, y: (x, y))
        self.unit_map = unit_map or (lambda: None)


class DaggerCategory(MonoidalCategory):
    """
    Dagger compact closed category.

    Has a dagger operation †: C^op → C giving adjoints for backpropagation.

    This structure encodes:
    - Compositionality (functoriality)
    - Reversibility (dagger for backpropagation)
    - Resource constraints (compact closure → no-cloning)
    """

    def __init__(self, name: str = "Dagger"):
        super().__init__(name)

    def dagger(self, f: Morphism) -> Morphism:
        """
        Compute adjoint of morphism.

        For neural networks, this corresponds to backward pass:
        F† is adjoint w.r.t. Fisher inner product.
        """

        def adjoint(x):
            if isinstance(x, Tensor) and x.requires_grad:
                return x.grad if x.grad is not None else torch.zeros_like(x)
            return x

        return adjoint

    def dual(self, obj: Object) -> Object:
        """Compute dual object A*."""
        return Object(name=f"{obj.name}*", shape=obj.shape)

    def cap(self, obj: Object) -> Morphism:
        """Cap morphism: I → A* ⊗ A"""

        def cap_fn(x):
            if isinstance(x, Tensor):
                d = x.shape[-1] if x.dim() > 0 else 1
                return torch.eye(d).flatten()
            return x

        return cap_fn

    def cup(self, obj: Object) -> Morphism:
        """Cup morphism: A ⊗ A* → I"""

        def cup_fn(x):
            if isinstance(x, Tensor):
                d = int(torch.sqrt(torch.tensor(x.shape[-1])))
                return x.view(d, d).trace()
            return x

        return cup_fn


def compose_string_diagram(*morphisms: Morphism) -> Morphism:
    """
    Compose morphisms in string diagram order (left-to-right = top-to-bottom).

    String diagrams provide diagrammatic reasoning for neural architectures.
    """

    def composed(x):
        result = x
        for m in morphisms:
            result = m(result)
        return result

    return composed
