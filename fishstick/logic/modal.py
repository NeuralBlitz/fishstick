"""
Modal Logic Module.

Implements:
- Modal operators (□, ◇)
- Possible worlds semantics (Kripke frames)
- Normal modal logics: K, T, S4, S5
- Modal satisfaction and validity checking
- Modal tableau prover

Author: Agent 13 (Fishstick Framework)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, FrozenSet
from abc import ABC, abstractmethod
from collections import defaultdict
import torch
from torch import Tensor


class ModalSystem(Enum):
    """Normal modal logic systems."""

    K = auto()  # Minimal modal logic
    T = auto()  # Reflexive frames (⊤ → □⊤)
    S4 = auto()  # Reflexive + transitive (T + □→□)
    S5 = auto()  # Reflexive + transitive + symmetric (S4 + ◇→□◇)


class World:
    """Possible world in Kripke semantics."""

    def __init__(self, name: str):
        self.name = name
        self.props: Set[str] = set()
        self.accessibility: Set[str] = set()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, World):
            return NotImplemented
        return self.name == other.name

    def __str__(self) -> str:
        return self.name


@dataclass
class KripkeFrame:
    """
    Kripke frame for modal semantics.

    Consists of:
    - Set of possible worlds
    - Accessibility relation R ⊆ W × W
    """

    worlds: Set[World] = field(default_factory=set)
    _accessibility: Dict[World, Set[World]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add_world(self, world: World) -> None:
        """Add a world to the frame."""
        self.worlds.add(world)

    def add_accessibility(self, w1: World, w2: World) -> None:
        """Add accessibility relation w1 R w2."""
        self._accessibility[w1].add(w2)

    def accessible_from(self, world: World) -> Set[World]:
        """Get worlds accessible from given world."""
        return self._accessibility.get(world, set())

    def is_reflexive(self) -> bool:
        """Check if frame is reflexive."""
        for w in self.worlds:
            if w not in self._accessibility[w]:
                return False
        return True

    def is_transitive(self) -> bool:
        """Check if frame is transitive."""
        for w1 in self.worlds:
            for w2 in self.accessible_from(w1):
                for w3 in self.accessible_from(w2):
                    if w3 not in self._accessibility[w1]:
                        return False
        return True

    def is_symmetric(self) -> bool:
        """Check if frame is symmetric."""
        for w1 in self.worlds:
            for w2 in self._accessibility[w1]:
                if w1 not in self._accessibility[w2]:
                    return False
        return True

    def satisfies_system(self, system: ModalSystem) -> bool:
        """Check if frame satisfies given modal system axioms."""
        if system == ModalSystem.T:
            return self.is_reflexive()
        elif system == ModalSystem.S4:
            return self.is_reflexive() and self.is_transitive()
        elif system == ModalSystem.S5:
            return self.is_reflexive() and self.is_transitive() and self.is_symmetric()
        return True


@dataclass
class KripkeModel:
    """
    Kripke model for modal logic.

    Consists of:
    - Kripke frame (worlds + accessibility)
    - Valuation function V: W × P → {true, false}
    """

    frame: KripkeFrame
    _valuation: Dict[World, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_true(self, world: World, prop: str) -> None:
        """Add true proposition at world."""
        self._valuation[world].add(prop)

    def is_true(self, world: World, prop: str) -> bool:
        """Check if proposition is true at world."""
        return prop in self._valuation.get(world, set())

    def evaluate(self, formula: ModalFormula, world: World) -> bool:
        """Evaluate modal formula at world."""
        if isinstance(formula, PropAtom):
            return self.is_true(world, formula.name)
        elif isinstance(formula, TrueConstant):
            return True
        elif isinstance(formula, FalseConstant):
            return False
        elif isinstance(formula, ModalNot):
            return not self.evaluate(formula.operand, world)
        elif isinstance(formula, ModalAnd):
            return self.evaluate(formula.left, world) and self.evaluate(
                formula.right, world
            )
        elif isinstance(formula, ModalOr):
            return self.evaluate(formula.left, world) or self.evaluate(
                formula.right, world
            )
        elif isinstance(formula, ModalImplies):
            return not self.evaluate(formula.left, world) or self.evaluate(
                formula.right, world
            )
        elif isinstance(formula, Box):
            for w2 in self.frame.accessible_from(world):
                if not self.evaluate(formula.operand, w2):
                    return False
            return True
        elif isinstance(formula, Diamond):
            for w2 in self.frame.accessible_from(world):
                if self.evaluate(formula.operand, w2):
                    return True
            return False
        return False

    def is_satisfiable(self, formula: ModalFormula) -> bool:
        """Check if formula is satisfiable in this model."""
        for world in self.frame.worlds:
            if self.evaluate(formula, world):
                return True
        return False


class ModalFormula(ABC):
    """Abstract base class for modal formulas."""

    @abstractmethod
    def atoms(self) -> Set[str]:
        """Get propositional atoms."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


@dataclass(frozen=True)
class PropAtom(ModalFormula):
    """Propositional atom."""

    name: str

    def atoms(self) -> Set[str]:
        return {self.name}

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("atom", self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropAtom):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class TrueConstant(ModalFormula):
    """True constant."""

    def atoms(self) -> Set[str]:
        return set()

    def __str__(self) -> str:
        return "⊤"

    def __hash__(self) -> int:
        return hash("true")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrueConstant)


@dataclass(frozen=True)
class FalseConstant(ModalFormula):
    """False constant."""

    def atoms(self) -> Set[str]:
        return set()

    def __str__(self) -> str:
        return "⊥"

    def __hash__(self) -> int:
        return hash("false")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FalseConstant)


@dataclass(frozen=True)
class ModalNot(ModalFormula):
    """Negation."""

    operand: ModalFormula

    def atoms(self) -> Set[str]:
        return self.operand.atoms()

    def __str__(self) -> str:
        return f"¬{self.operand}"

    def __hash__(self) -> int:
        return hash(("not", self.operand))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModalNot):
            return NotImplemented
        return self.operand == other.operand


@dataclass(frozen=True)
class ModalAnd(ModalFormula):
    """Conjunction."""

    left: ModalFormula
    right: ModalFormula

    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def __hash__(self) -> int:
        return hash(("and", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModalAnd):
            return NotImplemented
        return self.left == other.left and self.right == other.right


@dataclass(frozen=True)
class ModalOr(ModalFormula):
    """Disjunction."""

    left: ModalFormula
    right: ModalFormula

    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def __hash__(self) -> int:
        return hash(("or", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModalOr):
            return NotImplemented
        return self.left == other.left and self.right == other.right


@dataclass(frozen=True)
class ModalImplies(ModalFormula):
    """Implication."""

    left: ModalFormula
    right: ModalFormula

    def atoms(self) -> Set[str]:
        return self.left.atoms() | self.right.atoms()

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"

    def __hash__(self) -> int:
        return hash(("implies", self.left, self.right))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModalImplies):
            return NotImplemented
        return self.left == other.left and self.right == other.right


@dataclass(frozen=True)
class Box(ModalFormula):
    """Necessity operator □."""

    operand: ModalFormula

    def atoms(self) -> Set[str]:
        return self.operand.atoms()

    def __str__(self) -> str:
        return f"□{self.operand}"

    def __hash__(self) -> int:
        return hash(("box", self.operand))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        return self.operand == other.operand


@dataclass(frozen=True)
class Diamond(ModalFormula):
    """Possibility operator ◇."""

    operand: ModalFormula

    def atoms(self) -> Set[str]:
        return self.operand.atoms()

    def __str__(self) -> str:
        return f"◇{self.operand}"

    def __hash__(self) -> int:
        return hash(("diamond", self.operand))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Diamond):
            return NotImplemented
        return self.operand == other.operand


class ModalLogic:
    """Utilities for modal logic."""

    @staticmethod
    def is_valid(formula: ModalFormula, system: ModalSystem = ModalSystem.K) -> bool:
        """
        Check if formula is valid in given modal system.

        Args:
            formula: Modal formula to check
            system: Modal logic system (K, T, S4, S5)

        Returns:
            True if formula is valid
        """
        return ModalLogic._check_validity(formula, system, set())

    @staticmethod
    def _check_validity(
        formula: ModalFormula, system: ModalSystem, atoms: Set[str], depth: int = 0
    ) -> bool:
        """Recursively check validity."""
        if depth > 10:
            return True

        atom_list = sorted(atoms)

        for assignment in ModalLogic._all_assignments(atom_list):
            model = ModalLogic._create_model(formula, assignment, system)

            for world in model.frame.worlds:
                if not model.evaluate(formula, world):
                    return False

        return True

    @staticmethod
    def _all_assignments(atoms: List[str]) -> List[Dict[str, bool]]:
        """Generate all possible truth assignments."""
        n = len(atoms)
        assignments = []

        for i in range(2**n):
            assignment = {}
            for j, atom in enumerate(atoms):
                assignment[atom] = bool((i >> j) & 1)
            assignments.append(assignment)

        return assignments

    @staticmethod
    def _create_model(
        formula: ModalFormula, assignment: Dict[str, bool], system: ModalSystem
    ) -> KripkeModel:
        """Create a Kripke model for checking validity."""
        frame = KripkeFrame()

        w0 = World("w0")
        frame.add_world(w0)

        if system in [ModalSystem.T, ModalSystem.S4, ModalSystem.S5]:
            frame.add_accessibility(w0, w0)

        if system in [ModalSystem.S4, ModalSystem.S5]:
            frame.add_world(World("w1"))
            frame.add_accessibility(w0, w0)
            frame.add_accessibility(w0, World("w1"))
            frame.add_accessibility(World("w1"), World("w1"))

        if system == ModalSystem.S5:
            w1 = World("w1")
            if w1 in frame.worlds:
                frame.add_accessibility(w1, w0)

        model = KripkeModel(frame)

        for atom, value in assignment.items():
            if value:
                model.add_true(w0, atom)

        return model

    @staticmethod
    def is_satisfiable(
        formula: ModalFormula, system: ModalSystem = ModalSystem.K
    ) -> bool:
        """
        Check if formula is satisfiable.

        Args:
            formula: Modal formula to check
            system: Modal logic system

        Returns:
            True if formula is satisfiable
        """
        return not ModalLogic.is_valid(ModalNot(formula), system)

    @staticmethod
    def entails(
        premises: List[ModalFormula],
        conclusion: ModalFormula,
        system: ModalSystem = ModalSystem.K,
    ) -> bool:
        """
        Check if premises entail conclusion.

        Args:
            premises: List of premise formulas
            conclusion: Conclusion formula
            system: Modal logic system

        Returns:
            True if entailment holds
        """
        conjunction = (
            premises[0] if len(premises) == 1 else ModalAnd(premises[0], premises[1])
        )
        for p in premises[2:]:
            conjunction = ModalAnd(conjunction, p)

        premise_implies = ModalImplies(conjunction, conclusion)

        return ModalLogic.is_valid(premise_implies, system)

    @staticmethod
    def to_negation_normalForm(formula: ModalFormula) -> ModalFormula:
        """Convert to negation normal form (negations pushed inward)."""
        if isinstance(formula, PropAtom):
            return formula
        elif isinstance(formula, TrueConstant):
            return formula
        elif isinstance(formula, FalseConstant):
            return formula
        elif isinstance(formula, ModalNot):
            inner = formula.operand
            if isinstance(inner, PropAtom):
                return formula
            elif isinstance(inner, ModalNot):
                return ModalLogic.to_negation_normalForm(inner.operand)
            elif isinstance(inner, ModalAnd):
                return ModalOr(
                    ModalLogic.to_negation_normalForm(ModalNot(inner.left)),
                    ModalLogic.to_negation_normalForm(ModalNot(inner.right)),
                )
            elif isinstance(inner, ModalOr):
                return ModalAnd(
                    ModalLogic.to_negation_normalForm(ModalNot(inner.left)),
                    ModalLogic.to_negation_normalForm(ModalNot(inner.right)),
                )
            elif isinstance(inner, ModalImplies):
                return ModalAnd(
                    ModalLogic.to_negation_normalForm(inner.left),
                    ModalLogic.to_negation_normalForm(ModalNot(inner.right)),
                )
            elif isinstance(inner, Box):
                return Diamond(
                    ModalLogic.to_negation_normalForm(ModalNot(inner.operand))
                )
            elif isinstance(inner, Diamond):
                return Box(ModalLogic.to_negation_normalForm(ModalNot(inner.operand)))
        elif isinstance(formula, ModalAnd):
            return ModalAnd(
                ModalLogic.to_negation_normalForm(formula.left),
                ModalLogic.to_negation_normalForm(formula.right),
            )
        elif isinstance(formula, ModalOr):
            return ModalOr(
                ModalLogic.to_negation_normalForm(formula.left),
                ModalLogic.to_negation_normalForm(formula.right),
            )
        elif isinstance(formula, ModalImplies):
            return ModalOr(
                ModalLogic.to_negation_normalForm(ModalNot(formula.left)),
                ModalLogic.to_negation_normalForm(formula.right),
            )
        elif isinstance(formula, Box):
            return Box(ModalLogic.to_negation_normalForm(formula.operand))
        elif isinstance(formula, Diamond):
            return Diamond(ModalLogic.to_negation_normalForm(formula.operand))

        return formula


class ModalTableauProver:
    """
    Modal tableau prover for normal modal logics.

    Implements:
    - Branching rules for □ and ◇
    - System-specific rules (reflexivity, transitivity)
    """

    def __init__(self, system: ModalSystem = ModalSystem.K):
        self.system = system
        self.branches: List[List[ModalFormula]] = []

    def prove(self, formula: ModalFormula) -> Tuple[bool, Optional[List[ModalFormula]]]:
        """
        Attempt to prove formula.

        Returns:
            (proved, branch) tuple
        """
        self.branches = [[formula]]

        while self.branches:
            branch = self.branches.pop(0)

            if self._is_closed(branch):
                continue

            expanded, new_branch = self._expand(branch)

            if expanded:
                self.branches.append(new_branch)
            else:
                if self._is_closed(new_branch):
                    continue
                return False, new_branch

        return True, None

    def _expand(self, branch: List[ModalFormula]) -> Tuple[bool, List[ModalFormula]]:
        """Expand branch with tableau rules."""
        for formula in branch:
            if isinstance(formula, ModalNot) and isinstance(formula.operand, Box):
                diamond = Diamond(ModalNot(formula.operand.operand))
                return True, branch + [diamond]

            if isinstance(formula, ModalNot) and isinstance(formula.operand, Diamond):
                box = Box(ModalNot(formula.operand.operand))
                return True, branch + [box]

            if isinstance(formula, Box):
                worlds = self._get_accessible_worlds(branch)
                new_formulas = [formula.operand for _ in worlds]
                if new_formulas:
                    return True, branch + new_formulas
                return True, branch + [formula.operand]

            if isinstance(formula, Diamond):
                worlds = self._get_accessible_worlds(branch)
                if worlds:
                    return True, branch + [formula.operand]

        return False, branch

    def _get_accessible_worlds(self, branch: List[ModalFormula]) -> List[World]:
        """Get accessible worlds based on system."""
        worlds = [World(f"w{i}") for i in range(3)]

        if self.system in [ModalSystem.T, ModalSystem.S4, ModalSystem.S5]:
            for w in worlds:
                w.accessibility.add(w.name)

        if self.system in [ModalSystem.S4, ModalSystem.S5]:
            for i, w in enumerate(worlds):
                for j, w2 in enumerate(worlds):
                    if i <= j:
                        w.accessibility.add(w2.name)

        return worlds

    def _is_closed(self, branch: List[ModalFormula]) -> bool:
        """Check if branch is closed (contains contradiction)."""
        for f1 in branch:
            for f2 in branch:
                if f1 == ModalNot(f2) or ModalNot(f1) == f2:
                    return True
                if isinstance(f1, TrueConstant) and isinstance(f2, FalseConstant):
                    return True
                if isinstance(f1, FalseConstant) and isinstance(f2, TrueConstant):
                    return True
        return False


def create_atom(name: str) -> ModalFormula:
    """Create propositional atom."""
    return PropAtom(name)


def create_true() -> ModalFormula:
    """Create true constant."""
    return TrueConstant()


def create_false() -> ModalFormula:
    """Create false constant."""
    return FalseConstant()


def create_not(formula: ModalFormula) -> ModalFormula:
    """Create negation."""
    return ModalNot(formula)


def create_and_modal(*formulas: ModalFormula) -> ModalFormula:
    """Create conjunction."""
    if len(formulas) == 0:
        return create_true()
    if len(formulas) == 1:
        return formulas[0]
    result = formulas[0]
    for f in formulas[1:]:
        result = ModalAnd(result, f)
    return result


def create_or_modal(*formulas: ModalFormula) -> ModalFormula:
    """Create disjunction."""
    if len(formulas) == 0:
        return create_false()
    if len(formulas) == 1:
        return formulas[0]
    result = formulas[0]
    for f in formulas[1:]:
        result = ModalOr(result, f)
    return result


def create_implies_modal(left: ModalFormula, right: ModalFormula) -> ModalFormula:
    """Create implication."""
    return ModalImplies(left, right)


def create_box(formula: ModalFormula) -> ModalFormula:
    """Create necessity □."""
    return Box(formula)


def create_diamond(formula: ModalFormula) -> ModalFormula:
    """Create possibility ◇."""
    return Diamond(formula)
