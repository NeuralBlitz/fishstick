"""
First-Order Logic Module.

Implements:
- Terms, predicates, and quantifiers
- First-order formulas (FOL)
- Unification algorithm (MGU)
- Skolemization for quantifier elimination
- CNF conversion for FOL

Author: Agent 13 (Fishstick Framework)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, FrozenSet
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class TermType(Enum):
    """Types of terms in FOL."""

    VARIABLE = auto()
    CONSTANT = auto()
    FUNCTION = auto()


@dataclass(frozen=True)
class Variable:
    """First-order variable."""

    name: str

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(("var", self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class Constant:
    """First-order constant symbol."""

    name: str

    def __str__(self) -> str:
        return self.name.lower()

    def __hash__(self) -> int:
        return hash(("const", self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constant):
            return NotImplemented
        return self.name == other.name


@dataclass(frozen=True)
class Function:
    """First-order function symbol applied to terms."""

    name: str
    args: Tuple[Term, ...]

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

    def __hash__(self) -> int:
        return hash(("func", self.name, self.args))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Function):
            return NotImplemented
        return self.name == other.name and self.args == other.args


Term = Union[Variable, Constant, Function]


class QuantifierType(Enum):
    """Quantifier types."""

    FORALL = auto()
    EXISTS = auto()


class FormulaTypeFOL(Enum):
    """Formula types in FOL."""

    ATOMIC = auto()
    NEGATED = auto()
    BINARY = auto()
    QUANTIFIED = auto


class AtomicFormula(ABC):
    """Base class for atomic formulas."""

    @abstractmethod
    def get_variables(self) -> Set[Variable]:
        """Get all free variables in formula."""
        pass

    @abstractmethod
    def get_constants(self) -> Set[Constant]:
        """Get all constants in formula."""
        pass

    @abstractmethod
    def substitute(self, substitution: Dict[Variable, Term]) -> AtomicFormula:
        """Apply substitution to formula."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


@dataclass(frozen=True)
class Predicate(AtomicFormula):
    """First-order predicate."""

    name: str
    args: Tuple[Term, ...]

    def get_variables(self) -> Set[Variable]:
        variables = set()
        for arg in self.args:
            if isinstance(arg, Variable):
                variables.add(arg)
            elif isinstance(arg, Function):
                variables |= self._get_vars_from_term(arg)
        return variables

    def _get_vars_from_term(self, term: Term) -> Set[Variable]:
        if isinstance(term, Variable):
            return {term}
        elif isinstance(term, Constant):
            return set()
        elif isinstance(term, Function):
            result = set()
            for arg in term.args:
                result |= self._get_vars_from_term(arg)
            return result
        return set()

    def get_constants(self) -> Set[Constant]:
        constants = set()
        for arg in self.args:
            if isinstance(arg, Constant):
                constants.add(arg)
            elif isinstance(arg, Function):
                constants |= self._get_consts_from_term(arg)
        return constants

    def _get_consts_from_term(self, term: Term) -> Set[Constant]:
        if isinstance(term, Constant):
            return {term}
        elif isinstance(term, Variable):
            return set()
        elif isinstance(term, Function):
            result = set()
            for arg in term.args:
                result |= self._get_consts_from_term(arg)
            return result
        return set()

    def substitute(self, substitution: Dict[Variable, Term]) -> Predicate:
        new_args = tuple(
            substitution.get(arg, arg) if isinstance(arg, Variable) else arg
            for arg in self.args
        )
        new_args = tuple(self._substitute_term(arg, substitution) for arg in self.args)
        return Predicate(self.name, new_args)

    def _substitute_term(self, term: Term, substitution: Dict[Variable, Term]) -> Term:
        if isinstance(term, Variable):
            return substitution.get(term, term)
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Function):
            new_args = tuple(self._substitute_term(a, substitution) for a in term.args)
            return Function(term.name, new_args)
        return term

    def __str__(self) -> str:
        if not self.args:
            return self.name
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"

    def __hash__(self) -> int:
        return hash(("pred", self.name, self.args))


@dataclass(frozen=True)
class Equality(AtomicFormula):
    """Equality predicate (term equality)."""

    left: Term
    right: Term

    def get_variables(self) -> Set[Variable]:
        variables = set()
        variables |= self._get_vars_from_term(self.left)
        variables |= self._get_vars_from_term(self.right)
        return variables

    def _get_vars_from_term(self, term: Term) -> Set[Variable]:
        if isinstance(term, Variable):
            return {term}
        elif isinstance(term, Constant):
            return set()
        elif isinstance(term, Function):
            result = set()
            for arg in term.args:
                result |= self._get_vars_from_term(arg)
            return result
        return set()

    def get_constants(self) -> Set[Constant]:
        constants = set()
        constants |= self._get_consts_from_term(self.left)
        constants |= self._get_consts_from_term(self.right)
        return constants

    def _get_consts_from_term(self, term: Term) -> Set[Constant]:
        if isinstance(term, Constant):
            return {term}
        elif isinstance(term, Variable):
            return set()
        elif isinstance(term, Function):
            result = set()
            for arg in term.args:
                result |= self._get_consts_from_term(arg)
            return result
        return set()

    def substitute(self, substitution: Dict[Variable, Term]) -> Equality:
        new_left = self._substitute_term(self.left, substitution)
        new_right = self._substitute_term(self.right, substitution)
        return Equality(new_left, new_right)

    def _substitute_term(self, term: Term, substitution: Dict[Variable, Term]) -> Term:
        if isinstance(term, Variable):
            return substitution.get(term, term)
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Function):
            new_args = tuple(self._substitute_term(a, substitution) for a in term.args)
            return Function(term.name, new_args)
        return term

    def __str__(self) -> str:
        return f"{self.left} = {self.right}"

    def __hash__(self) -> int:
        return hash(("eq", self.left, self.right))


FOLFormula = Union[
    AtomicFormula, "NegatedFormula", "BinaryFormulaFOL", "QuantifiedFormula"
]


@dataclass(frozen=True)
class NegatedFormula:
    """Negated formula."""

    formula: FOLFormula

    def get_variables(self) -> Set[Variable]:
        return self.formula.get_variables()

    def get_constants(self) -> Set[Constant]:
        return self.formula.get_constants()

    def __str__(self) -> str:
        return f"¬{self.formula}"

    def __hash__(self) -> int:
        return hash(("not", self.formula))


@dataclass(frozen=True)
class BinaryFormulaFOL:
    """Binary connective formula."""

    connective: str  # AND, OR, IMPLIES
    left: FOLFormula
    right: FOLFormula

    def get_variables(self) -> Set[Variable]:
        return self.left.get_variables() | self.right.get_variables()

    def get_constants(self) -> Set[Constant]:
        return self.left.get_constants() | self.right.get_constants()

    def __str__(self) -> str:
        sym = {"AND": "∧", "OR": "∨", "IMPLIES": "→"}.get(
            self.connective, self.connective
        )
        return f"({self.left} {sym} {self.right})"

    def __hash__(self) -> int:
        return hash((self.connective, self.left, self.right))


@dataclass(frozen=True)
class QuantifiedFormula:
    """Quantified formula (FORALL or EXISTS)."""

    quantifier: QuantifierType
    variable: Variable
    formula: FOLFormula

    def get_variables(self) -> Set[Variable]:
        inner_vars = self.formula.get_variables()
        inner_vars.discard(self.variable)
        return inner_vars

    def get_constants(self) -> Set[Constant]:
        return self.formula.get_constants()

    def __str__(self) -> str:
        quant_sym = "∀" if self.quantifier == QuantifierType.FORALL else "∃"
        return f"{quant_sym}{self.variable}{self.formula}"

    def __hash__(self) -> int:
        return hash((self.quantifier, self.variable, self.formula))


class UnificationResult(Enum):
    """Result of unification attempt."""

    SUCCESS = auto()
    FAILURE = auto()
    OCCURS_CHECK_FAILURE = auto()


@dataclass
class Unifier:
    """Most General Unifier (MGU)."""

    substitution: Dict[Variable, Term]

    def __init__(self):
        self.substitution = {}

    def apply(self, term: Term) -> Term:
        """Apply substitution to term."""
        if isinstance(term, Variable):
            return self.substitution.get(term, term)
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Function):
            new_args = tuple(self.apply(arg) for arg in term.args)
            return Function(term.name, new_args)
        return term

    def compose(self, other: Dict[Variable, Term]) -> None:
        """Compose with another substitution."""
        for var in list(self.substitution.keys()):
            self.substitution[var] = other.get(var, self.substitution[var])
        for var, term in other.items():
            if var not in self.substitution:
                self.substitution[var] = term


class UnifierFOL:
    """
    First-order unification algorithm.

    Implements:
    - Occurs check
    - Most General Unifier (MGU) computation
    """

    def __init__(self):
        self.substitution: Dict[Variable, Term] = {}

    def unify(
        self, t1: Term, t2: Term
    ) -> Tuple[UnificationResult, Optional[Dict[Variable, Term]]]:
        """
        Unify two terms.

        Returns:
            (result, substitution) tuple
        """
        self.substitution = {}

        if self._unify(t1, t2):
            return UnificationResult.SUCCESS, self.substitution.copy()
        return UnificationResult.FAILURE, None

    def _unify(self, t1: Term, t2: Term) -> bool:
        """Recursive unification."""
        t1 = self._apply_substitution(t1)
        t2 = self._apply_substitution(t2)

        if t1 == t2:
            return True

        if isinstance(t1, Variable):
            return self._unify_variable(t1, t2)

        if isinstance(t2, Variable):
            return self._unify_variable(t2, t1)

        if isinstance(t1, Constant) and isinstance(t2, Constant):
            return t1.name == t2.name

        if isinstance(t1, Function) and isinstance(t2, Function):
            if t1.name != t2.name or len(t1.args) != len(t2.args):
                return False
            for arg1, arg2 in zip(t1.args, t2.args):
                if not self._unify(arg1, arg2):
                    return False
            return True

        return False

    def _unify_variable(self, var: Variable, term: Term) -> bool:
        """Unify variable with term."""
        if self._occurs_check(var, term):
            return False

        self.substitution[var] = term
        return True

    def _occurs_check(self, var: Variable, term: Term) -> bool:
        """Check if variable occurs in term."""
        if isinstance(term, Variable):
            return var == term
        elif isinstance(term, Constant):
            return False
        elif isinstance(term, Function):
            return any(self._occurs_check(var, arg) for arg in term.args)
        return False

    def _apply_substitution(self, term: Term) -> Term:
        """Apply current substitution to term."""
        if isinstance(term, Variable):
            if term in self.substitution:
                return self._apply_substitution(self.substitution[term])
            return term
        elif isinstance(term, Constant):
            return term
        elif isinstance(term, Function):
            new_args = tuple(self._apply_substitution(arg) for arg in term.args)
            return Function(term.name, new_args)
        return term


class Skolemizer:
    """
    Skolemization for quantifier elimination.

    Replaces existential quantifiers with Skolem functions/constants.
    """

    def __init__(self):
        self.skolem_counter = 0
        self.skolem_functions: Dict[Variable, Function] = {}
        self.universal_vars: List[Variable] = []

    def skolemize(self, formula: FOLFormula) -> FOLFormula:
        """
        Skolemize a formula.

        Args:
            formula: First-order formula

        Returns:
            Formula in skolemized form (existential quantifiers removed)
        """
        self.skolem_counter = 0
        self.skolem_functions = {}

        formula = self._push_quantifiers(formula)

        return self._remove_quantifiers(formula)

    def _push_quantifiers(self, formula: FOLFormula) -> QuantifiedFormula:
        """Move all quantifiers to the front."""
        quantifiers = []

        def _extract_quantifiers(
            f: FOLFormula,
        ) -> Tuple[List[Tuple[QuantifierType, Variable]], FOLFormula]:
            if isinstance(f, QuantifiedFormula):
                inner_quantifiers, inner_formula = _extract_quantifiers(f.formula)
                return [(f.quantifier, f.variable)] + inner_quantifiers, inner_formula
            return [], f

        quantifiers, matrix = _extract_quantifiers(formula)

        if not quantifiers:
            return QuantifiedFormula(QuantifierType.FORALL, Variable("_"), formula)

        return QuantifiedFormula(QuantifierType.FORALL, Variable("_"), matrix)

    def _remove_quantifiers(self, formula: FOLFormula) -> FOLFormula:
        """Remove quantifiers and add Skolem terms."""
        if isinstance(formula, QuantifiedFormula):
            if formula.quantifier == QuantifierType.EXISTS:
                skolem_term = self._create_skolem_term(formula.variable)
                self.skolem_functions[formula.variable] = skolem_term
                return self._remove_quantifiers(formula.formula)
            else:
                self.universal_vars.append(formula.variable)
                return self._remove_quantifiers(formula.formula)
        elif isinstance(formula, NegatedFormula):
            return NegatedFormula(self._remove_quantifiers(formula.formula))
        elif isinstance(formula, BinaryFormulaFOL):
            return BinaryFormulaFOL(
                formula.connective,
                self._remove_quantifiers(formula.left),
                self._remove_quantifiers(formula.right),
            )
        elif isinstance(formula, AtomicFormula):
            return self._substitute_skolem(formula)

        return formula

    def _create_skolem_term(self, var: Variable) -> Term:
        """Create Skolem term for existential variable."""
        self.skolem_counter += 1
        if self.universal_vars:
            args = tuple(self._create_skolem_term(v) for v in self.universal_vars)
            return Function(f"sk{self.skolem_counter}", args)
        else:
            return Constant(f"sk{self.skolem_counter}")

    def _substitute_skolem(self, formula: AtomicFormula) -> AtomicFormula:
        """Substitute Skolem terms in formula."""
        if isinstance(formula, Predicate):
            new_args = tuple(
                self.skolem_functions.get(arg, arg)
                if isinstance(arg, Variable)
                else arg
                for arg in formula.args
            )
            return Predicate(formula.name, new_args)
        elif isinstance(formula, Equality):
            new_left = (
                self.skolem_functions.get(formula.left, formula.left)
                if isinstance(formula.left, Variable)
                else formula.left
            )
            new_right = (
                self.skolem_functions.get(formula.right, formula.right)
                if isinstance(formula.right, Variable)
                else formula.right
            )
            return Equality(new_left, new_right)
        return formula


class FOLToCNF:
    """
    Convert first-order formula to Conjunctive Normal Form.

    Steps:
    1. Remove implications and biconditionals
    2. Move negations inward (push NOT through quantifiers)
    3. Standardize variables (rename to avoid clashes)
    4. Skolemize (remove existential quantifiers)
    5. Drop universal quantifiers
    6. Distribute OR over AND
    """

    def __init__(self):
        self.skolemizer = Skolemizer()
        self.variable_counter = 0

    def to_cnf(self, formula: FOLFormula) -> FOLFormula:
        """Convert formula to CNF."""
        formula = self._remove_implications(formula)
        formula = self._nnf(formula)
        formula = self.skolemizer.skolemize(formula)
        formula = self._drop_universal_quantifiers(formula)
        formula = self._distribute_or_over_and(formula)
        return formula

    def _remove_implications(self, formula: FOLFormula) -> FOLFormula:
        """Remove implications."""
        if isinstance(formula, AtomicFormula):
            return formula
        elif isinstance(formula, NegatedFormula):
            inner = formula.formula
            if isinstance(inner, BinaryFormulaFOL) and inner.connective == "IMPLIES":
                return BinaryFormulaFOL(
                    "OR",
                    self._remove_implications(NegatedFormula(inner.left)),
                    self._remove_implications(inner.right),
                )
            return NegatedFormula(self._remove_implications(inner))
        elif isinstance(formula, BinaryFormulaFOL):
            if formula.connective == "IMPLIES":
                return BinaryFormulaFOL(
                    "OR",
                    self._remove_implications(NegatedFormula(formula.left)),
                    self._remove_implications(formula.right),
                )
            return BinaryFormulaFOL(
                formula.connective,
                self._remove_implications(formula.left),
                self._remove_implications(formula.right),
            )
        elif isinstance(formula, QuantifiedFormula):
            return QuantifiedFormula(
                formula.quantifier,
                formula.variable,
                self._remove_implications(formula.formula),
            )
        return formula

    def _nnf(self, formula: FOLFormula) -> FOLFormula:
        """Convert to Negation Normal Form."""
        if isinstance(formula, AtomicFormula):
            return formula
        elif isinstance(formula, NegatedFormula):
            inner = formula.formula
            if isinstance(inner, AtomicFormula):
                return formula
            elif isinstance(inner, NegatedFormula):
                return self._nnf(inner.formula)
            elif isinstance(inner, BinaryFormulaFOL):
                if inner.connective == "AND":
                    return BinaryFormulaFOL(
                        "OR",
                        self._nnf(NegatedFormula(inner.left)),
                        self._nnf(NegatedFormula(inner.right)),
                    )
                elif inner.connective == "OR":
                    return BinaryFormulaFOL(
                        "AND",
                        self._nnf(NegatedFormula(inner.left)),
                        self._nnf(NegatedFormula(inner.right)),
                    )
            elif isinstance(inner, QuantifiedFormula):
                new_quantifier = (
                    QuantifierType.EXISTS
                    if inner.quantifier == QuantifierType.FORALL
                    else QuantifierType.FORALL
                )
                return QuantifiedFormula(
                    new_quantifier,
                    inner.variable,
                    self._nnf(NegatedFormula(inner.formula)),
                )
        elif isinstance(formula, BinaryFormulaFOL):
            return BinaryFormulaFOL(
                formula.connective, self._nnf(formula.left), self._nnf(formula.right)
            )
        elif isinstance(formula, QuantifiedFormula):
            return QuantifiedFormula(
                formula.quantifier, formula.variable, self._nnf(formula.formula)
            )
        return formula

    def _drop_universal_quantifiers(self, formula: FOLFormula) -> FOLFormula:
        """Drop universal quantifiers (assuming closed world)."""
        if isinstance(formula, QuantifiedFormula):
            if formula.quantifier == QuantifierType.FORALL:
                return self._drop_universal_quantifiers(formula.formula)
            return self._drop_universal_quantifiers(formula.formula)
        elif isinstance(formula, NegatedFormula):
            return NegatedFormula(self._drop_universal_quantifiers(formula.formula))
        elif isinstance(formula, BinaryFormulaFOL):
            return BinaryFormulaFOL(
                formula.connective,
                self._drop_universal_quantifiers(formula.left),
                self._drop_universal_quantifiers(formula.right),
            )
        return formula

    def _distribute_or_over_and(self, formula: FOLFormula) -> FOLFormula:
        """Distribute OR over AND."""
        if isinstance(formula, AtomicFormula):
            return formula
        elif isinstance(formula, NegatedFormula):
            return NegatedFormula(self._distribute_or_over_and(formula.formula))
        elif isinstance(formula, BinaryFormulaFOL):
            if formula.connective == "OR":
                left = self._distribute_or_over_and(formula.left)
                right = self._distribute_or_over_and(formula.right)

                if isinstance(left, BinaryFormulaFOL) and left.connective == "AND":
                    return BinaryFormulaFOL(
                        "AND",
                        BinaryFormulaFOL("OR", left.left, right),
                        BinaryFormulaFOL("OR", left.right, right),
                    )
                elif isinstance(right, BinaryFormulaFOL) and right.connective == "AND":
                    return BinaryFormulaFOL(
                        "AND",
                        BinaryFormulaFOL("OR", left, right.left),
                        BinaryFormulaFOL("OR", left, right.right),
                    )
                return BinaryFormulaFOL("OR", left, right)
            return BinaryFormulaFOL(
                formula.connective,
                self._distribute_or_over_and(formula.left),
                self._distribute_or_over_and(formula.right),
            )
        elif isinstance(formula, QuantifiedFormula):
            return QuantifiedFormula(
                formula.quantifier,
                formula.variable,
                self._distribute_or_over_and(formula.formula),
            )
        return formula


def create_predicate(name: str, *args: Term) -> AtomicFormula:
    """Create a predicate."""
    return Predicate(name, tuple(args))


def create_equal(t1: Term, t2: Term) -> AtomicFormula:
    """Create equality formula."""
    return Equality(t1, t2)


def create_not(formula: FOLFormula) -> FOLFormula:
    """Create negation."""
    return NegatedFormula(formula)


def create_and_fol(*formulas: FOLFormula) -> FOLFormula:
    """Create conjunction."""
    if len(formulas) == 0:
        return Predicate("true", ())
    if len(formulas) == 1:
        return formulas[0]
    result = formulas[0]
    for f in formulas[1:]:
        result = BinaryFormulaFOL("AND", result, f)
    return result


def create_or_fol(*formulas: FOLFormula) -> FOLFormula:
    """Create disjunction."""
    if len(formulas) == 0:
        return Predicate("false", ())
    if len(formulas) == 1:
        return formulas[0]
    result = formulas[0]
    for f in formulas[1:]:
        result = BinaryFormulaFOL("OR", result, f)
    return result


def create_implies_fol(left: FOLFormula, right: FOLFormula) -> FOLFormula:
    """Create implication."""
    return BinaryFormulaFOL("IMPLIES", left, right)


def create_forall(var: Variable, formula: FOLFormula) -> FOLFormula:
    """Create universal quantification."""
    return QuantifiedFormula(QuantifierType.FORALL, var, formula)


def create_exists(var: Variable, formula: FOLFormula) -> FOLFormula:
    """Create existential quantification."""
    return QuantifiedFormula(QuantifierType.EXISTS, var, formula)
