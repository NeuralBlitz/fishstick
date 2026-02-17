"""
Propositional Logic Module.

Implements:
- Propositional atoms and formulas
- Truth table evaluation
- SAT checking (brute-force, Davis-Putnam algorithm)
- Logical entailment and equivalence checking

Author: Agent 13 (Fishstick Framework)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class ConnectiveType(Enum):
    """Propositional connectives."""

    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()


class FormulaType(Enum):
    """Formula node types in AST."""

    ATOM = auto()
    CONSTANT = auto()
    UNARY = auto()
    BINARY = auto()


@dataclass(frozen=True)
class Atom:
    """Propositional atom (boolean variable)."""

    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Atom({self.name!r})"


@dataclass(frozen=True)
class Constant:
    """Boolean constant (True/False)."""

    value: bool

    def __str__(self) -> str:
        return "⊤" if self.value else "⊥"

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Formula(ABC):
    """Abstract base class for propositional formulas."""

    @abstractmethod
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """Evaluate formula under given truth assignment."""
        pass

    @abstractmethod
    def atoms(self) -> Set[Atom]:
        """Get all atoms in the formula."""
        pass

    @abstractmethod
    def subformulas(self) -> List[Formula]:
        """Get all subformulas."""
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
class AtomicFormula(Formula):
    """Atomic propositional formula (atom or constant)."""

    content: Union[Atom, Constant]

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        if isinstance(self.content, Atom):
            if self.content.name not in assignment:
                raise ValueError(f"Atom {self.content.name} not assigned")
            return assignment[self.content.name]
        return self.content.value

    def atoms(self) -> Set[Atom]:
        if isinstance(self.content, Atom):
            return {self.content}
        return set()

    def subformulas(self) -> List[Formula]:
        return [self]

    def __str__(self) -> str:
        return str(self.content)

    def __hash__(self) -> int:
        return hash((self.content, "atomic"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicFormula):
            return NotImplemented
        return self.content == other.content


@dataclass(frozen=True)
class UnaryFormula(Formula):
    """Unary formula (NOT)."""

    connective: ConnectiveType
    operand: Formula

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        if self.connective != ConnectiveType.NOT:
            raise ValueError(f"Expected NOT, got {self.connective}")
        return not self.operand.evaluate(assignment)

    def atoms(self) -> Set[Atom]:
        return self.operand.atoms()

    def subformulas(self) -> List[Formula]:
        return [self] + self.operand.subformulas()

    def __str__(self) -> str:
        return f"¬{self.operand}"

    def __hash__(self) -> int:
        return hash(("unary", self.connective, self.operand))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnaryFormula):
            return NotImplemented
        return self.connective == other.connective and self.operand == other.operand


@dataclass(frozen=True)
class BinaryFormula(Formula):
    """Binary formula (AND, OR, IMPLIES, IFF)."""

    connective: ConnectiveType
    left: Formula
    right: Formula

    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        lv = self.left.evaluate(assignment)
        rv = self.right.evaluate(assignment)

        if self.connective == ConnectiveType.AND:
            return lv and rv
        elif self.connective == ConnectiveType.OR:
            return lv or rv
        elif self.connective == ConnectiveType.IMPLIES:
            return not lv or rv
        elif self.connective == ConnectiveType.IFF:
            return lv == rv
        else:
            raise ValueError(f"Unknown connective: {self.connective}")

    def atoms(self) -> Set[Atom]:
        return self.left.atoms() | self.right.atoms()

    def subformulas(self) -> List[Formula]:
        return [self] + self.left.subformulas() + self.right.subformulas()

    def __str__(self) -> str:
        left_str = str(self.left)
        right_str = str(self.right)

        if self.connective == ConnectiveType.AND:
            return f"({left_str} ∧ {right_str})"
        elif self.connective == ConnectiveType.OR:
            return f"({left_str} ∨ {right_str})"
        elif self.connective == ConnectiveType.IMPLIES:
            return f"({left_str} → {right_str})"
        elif self.connective == ConnectiveType.IFF:
            return f"({left_str} ↔ {right_str})"
        return f"({left_str} {self.connective} {right_str})"

    def __hash__(self) -> int:
        return hash(("binary", self.connective, self.left, self.right))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryFormula):
            return NotImplemented
        return (
            self.connective == other.connective
            and self.left == other.left
            and self.right == other.right
        )


class PropositionalLogic:
    """Utilities for propositional logic."""

    @staticmethod
    def truth_table(formula: Formula) -> List[Tuple[Dict[str, bool], bool]]:
        """
        Generate truth table for a formula.

        Args:
            formula: Propositional formula

        Returns:
            List of (assignment, truth_value) tuples
        """
        atoms = sorted(formula.atoms(), key=lambda a: a.name)
        n = len(atoms)
        results = []

        for i in range(2**n):
            assignment = {}
            for j, atom in enumerate(atoms):
                assignment[atom.name] = bool((i >> j) & 1)

            truth_value = formula.evaluate(assignment)
            results.append((assignment, truth_value))

        return results

    @staticmethod
    def is_tautology(formula: Formula) -> bool:
        """Check if formula is a tautology (always true)."""
        for _, truth_value in PropositionalLogic.truth_table(formula):
            if not truth_value:
                return False
        return True

    @staticmethod
    def is_contradiction(formula: Formula) -> bool:
        """Check if formula is a contradiction (always false)."""
        for _, truth_value in PropositionalLogic.truth_table(formula):
            if truth_value:
                return False
        return True

    @staticmethod
    def is_satisfiable(formula: Formula) -> bool:
        """Check if formula is satisfiable (true under some assignment)."""
        for _, truth_value in PropositionalLogic.truth_table(formula):
            if truth_value:
                return True
        return False

    @staticmethod
    def is_valid(formula: Formula) -> bool:
        """Check if formula is valid (tautology)."""
        return PropositionalLogic.is_tautology(formula)

    @staticmethod
    def entails(premises: List[Formula], conclusion: Formula) -> bool:
        """
        Check if premises entail conclusion.

        Args:
            premises: List of premise formulas
            conclusion: Conclusion formula

        Returns:
            True if premises entail conclusion
        """
        all_atoms = set()
        for p in premises:
            all_atoms |= p.atoms()
        all_atoms |= conclusion.atoms()
        atoms = sorted(all_atoms, key=lambda a: a.name)
        n = len(atoms)

        for i in range(2**n):
            assignment = {}
            for j, atom in enumerate(atoms):
                assignment[atom.name] = bool((i >> j) & 1)

            all_true = all(p.evaluate(assignment) for p in premises)
            if all_true and not conclusion.evaluate(assignment):
                return False

        return True

    @staticmethod
    def are_equivalent(f1: Formula, f2: Formula) -> bool:
        """Check if two formulas are logically equivalent."""
        all_atoms = f1.atoms() | f2.atoms()
        atoms = sorted(all_atoms, key=lambda a: a.name)
        n = len(atoms)

        for i in range(2**n):
            assignment = {}
            for j, atom in enumerate(atoms):
                assignment[atom.name] = bool((i >> j) & 1)

            if f1.evaluate(assignment) != f2.evaluate(assignment):
                return False

        return True

    @staticmethod
    def to_cnf(formula: Formula) -> Formula:
        """
        Convert formula to Conjunctive Normal Form.

        Args:
            formula: Propositional formula

        Returns:
            Formula in CNF
        """
        formula = PropositionalLogic._remove_implications(formula)
        formula = PropositionalLogic._push_negation_inward(formula)
        formula = PropositionalLogic._distribute_or_over_and(formula)
        return formula

    @staticmethod
    def _remove_implications(formula: Formula) -> Formula:
        """Remove implication and biconditional."""
        if isinstance(formula, AtomicFormula):
            return formula
        elif isinstance(formula, UnaryFormula):
            return UnaryFormula(
                formula.connective,
                PropositionalLogic._remove_implications(formula.operand),
            )
        elif isinstance(formula, BinaryFormula):
            new_left = PropositionalLogic._remove_implications(formula.left)
            new_right = PropositionalLogic._remove_implications(formula.right)

            if formula.connective == ConnectiveType.IMPLIES:
                return BinaryFormula(
                    ConnectiveType.OR,
                    UnaryFormula(ConnectiveType.NOT, new_left),
                    new_right,
                )
            elif formula.connective == ConnectiveType.IFF:
                return BinaryFormula(
                    ConnectiveType.AND,
                    BinaryFormula(
                        ConnectiveType.OR,
                        UnaryFormula(ConnectiveType.NOT, new_left),
                        new_right,
                    ),
                    BinaryFormula(
                        ConnectiveType.OR,
                        new_left,
                        UnaryFormula(ConnectiveType.NOT, new_right),
                    ),
                )
            return BinaryFormula(formula.connective, new_left, new_right)

        return formula

    @staticmethod
    def _push_negation_inward(formula: Formula) -> Formula:
        """Push negation using De Morgan's laws."""
        if isinstance(formula, AtomicFormula):
            return formula
        elif isinstance(formula, UnaryFormula):
            if formula.connective == ConnectiveType.NOT:
                inner = formula.operand
                if isinstance(inner, AtomicFormula):
                    return formula
                elif isinstance(inner, UnaryFormula):
                    return PropositionalLogic._push_negation_inward(inner.operand)
                elif isinstance(inner, BinaryFormula):
                    if inner.connective == ConnectiveType.AND:
                        return BinaryFormula(
                            ConnectiveType.OR,
                            PropositionalLogic._push_negation_inward(
                                UnaryFormula(ConnectiveType.NOT, inner.left)
                            ),
                            PropositionalLogic._push_negation_inward(
                                UnaryFormula(ConnectiveType.NOT, inner.right)
                            ),
                        )
                    elif inner.connective == ConnectiveType.OR:
                        return BinaryFormula(
                            ConnectiveType.AND,
                            PropositionalLogic._push_negation_inward(
                                UnaryFormula(ConnectiveType.NOT, inner.left)
                            ),
                            PropositionalLogic._push_negation_inward(
                                UnaryFormula(ConnectiveType.NOT, inner.right)
                            ),
                        )
            return UnaryFormula(
                formula.connective,
                PropositionalLogic._push_negation_inward(formula.operand),
            )
        elif isinstance(formula, BinaryFormula):
            return BinaryFormula(
                formula.connective,
                PropositionalLogic._push_negation_inward(formula.left),
                PropositionalLogic._push_negation_inward(formula.right),
            )
        return formula

    @staticmethod
    def _distribute_or_over_and(formula: Formula) -> Formula:
        """Distribute OR over AND to get CNF."""
        if isinstance(formula, AtomicFormula):
            return formula
        elif isinstance(formula, UnaryFormula):
            return UnaryFormula(
                formula.connective,
                PropositionalLogic._distribute_or_over_and(formula.operand),
            )
        elif isinstance(formula, BinaryFormula):
            new_left = PropositionalLogic._distribute_or_over_and(formula.left)
            new_right = PropositionalLogic._distribute_or_over_and(formula.right)

            if formula.connective == ConnectiveType.OR:
                if (
                    isinstance(new_left, BinaryFormula)
                    and new_left.connective == ConnectiveType.AND
                ):
                    return BinaryFormula(
                        ConnectiveType.AND,
                        BinaryFormula(ConnectiveType.OR, new_left.left, new_right),
                        BinaryFormula(ConnectiveType.OR, new_left.right, new_right),
                    )
                elif (
                    isinstance(new_right, BinaryFormula)
                    and new_right.connective == ConnectiveType.AND
                ):
                    return BinaryFormula(
                        ConnectiveType.AND,
                        BinaryFormula(ConnectiveType.OR, new_left, new_right.left),
                        BinaryFormula(ConnectiveType.OR, new_left, new_right.right),
                    )

            return BinaryFormula(formula.connective, new_left, new_right)

        return formula


class DPLLSolver:
    """
    Davis-Putnam-Logemann-Loveland (DPLL) SAT solver.

    Implements:
    - Unit propagation
    - Pure literal elimination
    - Backtracking search
    """

    def __init__(self):
        self.clauses: List[List[int]] = []
        self.variables: Set[int] = set()
        self.assignment: Dict[int, bool] = {}
        self.unit_clauses: List[int] = []
        self.pure_literals: Set[int] = set()

    def add_clause(self, clause: List[int]) -> None:
        """Add a clause (list of literals)."""
        self.clauses.append(clause)
        for lit in clause:
            var = abs(lit)
            self.variables.add(var)
            if len(clause) == 1:
                self.unit_clauses.append(clause[0])

    def solve(self) -> Tuple[bool, Optional[Dict[int, bool]]]:
        """
        Run DPLL algorithm.

        Returns:
            (satisfiable, assignment) tuple
        """
        self.assignment = {}

        while True:
            self.unit_clauses = []
            self._find_unit_clauses()
            self._find_pure_literals()

            if self._has_conflict():
                if not self.assignment:
                    return False, None
                self._backtrack()
                continue

            if not self.unit_clauses and not self.pure_literals:
                break

            for lit in self.unit_clauses:
                var = abs(lit)
                value = lit > 0
                self.assignment[var] = value

            for lit in self.pure_literals:
                var = abs(lit)
                if var not in self.assignment:
                    self.assignment[var] = lit > 0

        if self._all_variables_assigned():
            return True, self.assignment.copy()

        var = self._choose_unassigned_var()

        for value in [True, False]:
            self.assignment[var] = value
            if self._propagate(var, value):
                result, assign = self.solve()
                if result:
                    return True, assign
            del self.assignment[var]

        if self.assignment:
            self._backtrack()

        return False, None

    def _find_unit_clauses(self) -> None:
        """Find unit clauses (clauses with one unassigned literal)."""
        self.unit_clauses = []
        for clause in self.clauses:
            unassigned = []
            for lit in clause:
                var = abs(lit)
                if var not in self.assignment:
                    unassigned.append(lit)
                elif self.assignment[var] != (lit < 0):
                    break
            else:
                if unassigned:
                    self.unit_clauses.append(unassigned[0])

    def _find_pure_literals(self) -> None:
        """Find pure literals (appear with only one polarity)."""
        self.pure_literals = set()
        positive = set()
        negative = set()

        for clause in self.clauses:
            for lit in clause:
                var = abs(lit)
                if var not in self.assignment:
                    if lit > 0:
                        positive.add(var)
                    else:
                        negative.add(var)

        self.pure_literals = (positive - negative) | (negative - positive)

    def _has_conflict(self) -> bool:
        """Check for conflict (empty clause)."""
        for clause in self.clauses:
            if not clause:
                return True
            all_false = True
            for lit in clause:
                var = abs(lit)
                if var not in self.assignment:
                    all_false = False
                    break
                if self.assignment[var] != (lit < 0):
                    all_false = False
                    break
            if all_false:
                return True
        return False

    def _all_variables_assigned(self) -> bool:
        """Check if all variables are assigned."""
        return len(self.assignment) == len(self.variables)

    def _choose_unassigned_var(self) -> int:
        """Choose an unassigned variable."""
        for var in self.variables:
            if var not in self.assignment:
                return var
        return 0

    def _propagate(self, var: int, value: bool) -> bool:
        """Check if assignment is consistent."""
        for clause in self.clauses:
            satisfied = False
            for lit in clause:
                v = abs(lit)
                if v == var:
                    if (lit > 0) == value:
                        satisfied = True
                        break
                elif v in self.assignment:
                    if (lit > 0) == self.assignment[v]:
                        satisfied = True
                        break

            if not satisfied:
                for lit in clause:
                    v = abs(lit)
                    if v not in self.assignment:
                        break
                else:
                    return False

        return True

    def _backtrack(self) -> None:
        """Backtrack to previous decision level."""
        if self.assignment:
            var = max(self.assignment.keys())
            del self.assignment[var]


class SATProblem:
    """
    SAT Problem wrapper for propositional formulas.

    Convert formula to CNF and solve using DPLL.
    """

    def __init__(self, formula: Formula):
        self.formula = formula
        self.solver = DPLLSolver()

    def is_satisfiable(self) -> bool:
        """Check if formula is satisfiable."""
        if PropositionalLogic.is_satisfiable(self.formula):
            return True

        cnf = PropositionalLogic.to_cnf(self.formula)
        self._add_cnf_to_solver(cnf)
        result, _ = self.solver.solve()
        return result

    def _add_cnf_to_solver(self, cnf: Formula) -> None:
        """Add CNF formula to solver as clauses."""
        if isinstance(cnf, AtomicFormula):
            return

        if isinstance(cnf, BinaryFormula):
            if cnf.connective == ConnectiveType.AND:
                self._add_cnf_to_solver(cnf.left)
                self._add_cnf_to_solver(cnf.right)
            elif cnf.connective == ConnectiveType.OR:
                clause = self._extract_disjunction(cnf)
                self.solver.add_clause(clause)

    def _extract_disjunction(self, formula: Formula) -> List[int]:
        """Extract literals from disjunction."""
        literals = []

        def _extract(f: Formula):
            if isinstance(f, AtomicFormula):
                if isinstance(f.content, Atom):
                    return [1 if f.content.name.startswith("_") else 1]
                return []
            elif isinstance(f, UnaryFormula):
                if f.connective == ConnectiveType.NOT:
                    if isinstance(f.operand, AtomicFormula):
                        if isinstance(f.operand.content, Atom):
                            return [-1]
            elif isinstance(f, BinaryFormula):
                if f.connective == ConnectiveType.OR:
                    _extract(f.left)
                    _extract(f.right)

        _extract(formula)
        return literals


def create_var(name: str) -> Formula:
    """Create a propositional variable."""
    return AtomicFormula(Atom(name))


def create_true() -> Formula:
    """Create true constant."""
    return AtomicFormula(Constant(True))


def create_false() -> Formula:
    """Create false constant."""
    return AtomicFormula(Constant(False))


def create_and(*args: Formula) -> Formula:
    """Create conjunction of formulas."""
    if len(args) == 0:
        return create_true()
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = BinaryFormula(ConnectiveType.AND, result, arg)
    return result


def create_or(*args: Formula) -> Formula:
    """Create disjunction of formulas."""
    if len(args) == 0:
        return create_false()
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = BinaryFormula(ConnectiveType.OR, result, arg)
    return result


def create_not(formula: Formula) -> Formula:
    """Create negation of formula."""
    return UnaryFormula(ConnectiveType.NOT, formula)


def create_implies(left: Formula, right: Formula) -> Formula:
    """Create implication."""
    return BinaryFormula(ConnectiveType.IMPLIES, left, right)


def create_if(left: Formula, right: Formula) -> Formula:
    """Create biconditional (iff)."""
    return BinaryFormula(ConnectiveType.IFF, left, right)
