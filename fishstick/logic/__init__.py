"""
Fishstick Formal Logic & Reasoning Systems.

This package implements comprehensive formal logic tools:

1. Propositional Logic (propositional.py)
   - Atoms, connectives, formulas
   - Truth table evaluation
   - SAT checking (brute-force, DPLL)
   - Logical entailment and equivalence

2. First-Order Logic (first_order.py)
   - Terms, predicates, quantifiers
   - Unification algorithm (MGU)
   - Skolemization for quantifier elimination
   - CNF conversion

3. Modal Logic (modal.py)
   - Kripke semantics (possible worlds)
   - Modal systems: K, T, S4, S5
   - Modal satisfaction checking
   - Tableau prover

4. Description Logic (description.py)
   - ALC description logic
   - TBox (terminological) reasoning
   - ABox (assertional) reasoning
   - Structural and tableau reasoning

5. Automated Reasoning (reasoning.py)
   - CDCL SAT solver
   - First-order resolution
   - Natural deduction prover

Author: Agent 13 (Fishstick Framework)
"""

from .propositional import (
    Atom,
    Constant,
    Formula as PropFormula,
    AtomicFormula,
    UnaryFormula,
    BinaryFormula,
    ConnectiveType,
    PropositionalLogic,
    DPLLSolver,
    SATProblem,
    create_var,
    create_true,
    create_false,
    create_and,
    create_or,
    create_not,
    create_implies,
    create_if,
)

from .first_order import (
    Variable,
    Constant as FOConstant,
    Function,
    Term,
    Predicate,
    Equality,
    QuantifierType,
    FOLFormula,
    NegatedFormula,
    BinaryFormulaFOL,
    QuantifiedFormula,
    UnificationResult,
    Unifier,
    UnifierFOL,
    Skolemizer,
    FOLToCNF,
    create_predicate,
    create_equal,
    create_not as create_not_fol,
    create_and_fol,
    create_or_fol,
    create_implies_fol,
    create_forall,
    create_exists,
)

from .modal import (
    ModalSystem,
    World,
    KripkeFrame,
    KripkeModel,
    ModalFormula,
    PropAtom,
    TrueConstant,
    FalseConstant,
    ModalNot,
    ModalAnd,
    ModalOr,
    ModalImplies,
    Box,
    Diamond,
    ModalLogic,
    ModalTableauProver,
    create_atom as create_atom_modal,
    create_true as create_true_modal,
    create_false as create_false_modal,
    create_not as create_not_modal,
    create_and_modal,
    create_or_modal,
    create_implies_modal,
    create_box,
    create_diamond,
)

from .description import (
    ConceptType,
    Concept,
    Role,
    TBox,
    ABox,
    StructuralReasoner,
    TableauAlgorithm,
    CompletionTree,
    CompletionNode,
    create_atomic_concept,
    create_top,
    create_bottom,
    create_not as create_not_dl,
    create_and_concept,
    create_or_concept,
    create_exists,
    create_forall as create_forall_dl,
    create_role,
)

from .reasoning import (
    Literal,
    Clause,
    CDCLSolver,
    ResolutionProver,
    FOTerm,
    FOVariable,
    FOConstant,
    FOFunction,
    FOLiteral,
    FirstOrderClause,
    ResolutionProof,
    NaturalDeductionProver,
    Formula,
    AtomicFormula as NDAtomicFormula,
    BinaryFormula as NDBinaryFormula,
    ProofStep,
    create_literal,
    create_clause,
    create_fol_literal,
    create_fol_clause,
)

__all__ = [
    "Atom",
    "Constant",
    "PropFormula",
    "AtomicFormula",
    "UnaryFormula",
    "BinaryFormula",
    "ConnectiveType",
    "PropositionalLogic",
    "DPLLSolver",
    "SATProblem",
    "Variable",
    "FOConstant",
    "Function",
    "Term",
    "Predicate",
    "Equality",
    "QuantifierType",
    "FOLFormula",
    "NegatedFormula",
    "BinaryFormulaFOL",
    "QuantifiedFormula",
    "UnificationResult",
    "Unifier",
    "UnifierFOL",
    "Skolemizer",
    "FOLToCNF",
    "ModalSystem",
    "World",
    "KripkeFrame",
    "KripkeModel",
    "ModalFormula",
    "PropAtom",
    "TrueConstant",
    "FalseConstant",
    "ModalNot",
    "ModalAnd",
    "ModalOr",
    "ModalImplies",
    "Box",
    "Diamond",
    "ModalLogic",
    "ModalTableauProver",
    "ConceptType",
    "Concept",
    "Role",
    "TBox",
    "ABox",
    "StructuralReasoner",
    "TableauAlgorithm",
    "CompletionTree",
    "CompletionNode",
    "Literal",
    "Clause",
    "CDCLSolver",
    "ResolutionProver",
    "FOTerm",
    "FOVariable",
    "FOConstant",
    "FOFunction",
    "FOLiteral",
    "FirstOrderClause",
    "ResolutionProof",
    "NaturalDeductionProver",
    "ProofStep",
]
