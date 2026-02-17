"""
Neuro-Symbolic Computation and Program Synthesis.

This module provides tools for combining neural networks with symbolic reasoning:

- Differentiable Logic Layers (differentiable_logic.py)
  - Soft logic gates with continuous relaxations
  - Logic Tensor Networks (LTN)
  - Differentiable SAT solvers
  - Logical regularization

- Rule Learning (rule_learning.py)
  - Association rule mining (Apriori)
  - Inductive Logic Programming (ILP)
  - Rule extraction from neural networks
  - Neural rule reasoning

- Symbolic Expression Trees (symbolic_expression_trees.py)
  - Expression tree representations
  - Differentiable expression evaluation
  - Symbolic differentiation
  - Expression simplification

- Program Synthesis (program_synthesis.py)
  - Program AST representations
  - Neural program interpreter
  - Differentiable program search
  - Evolutionary program synthesis

- Neural-Symbolic Reasoning (neurosymbolic_reasoning.py)
  - Knowledge base with embeddings
  - ComplEx knowledge graph embeddings
  - Relational reasoning
  - Logical constraint satisfaction

- Theorem Proving (theorem_proving.py)
  - Proof tree representations
  - Neural unification
  - Backward/forward chaining
  - Natural deduction system

- Hybrid Architectures (hybrid_architectures.py)
  - Lifted neural networks
  - Relational neural networks
  - Differentiable ILP
  - Neural Logic Machines

Example usage:
    from fishstick.neurosymbolic import (
        DifferentiableFormula,
        LogicTensorNetwork,
        SymbolicExpression,
        ProgramSynthesis,
    )
"""

from typing import Optional, List, Dict, Tuple, Set, Any

from .differentiable_logic import (
    GumbelSoftmax,
    SoftLogicGate,
    DifferentiableClause,
    DifferentiableFormula,
    LogicTensorNetwork,
    DifferentiableSATSolver,
    LogicalRegularizer,
    NeuralPredicate,
)

from .rule_learning import (
    LogicalRule,
    AssociationRuleMiner,
    InductiveLogicProgramming,
    RuleExtractor,
    NeuralRuleReasoner,
    RuleAttention,
)

from .symbolic_expression_trees import (
    OpType,
    ExpressionNode,
    DifferentiableExpression,
    simplify_expression,
    expression_to_function,
    parse_expression,
)

from .program_synthesis import (
    ProgramOp,
    ProgramNode,
    Program,
    ProgramSpace,
    NeuralProgramInterpreter,
    DifferentiableProgramSearch,
    ProgramSketch,
    ProgramEvaluator,
    ProgramSynthesis,
)

from .neurosymbolic_reasoning import (
    KnowledgeBase,
    NeuralKnowledgeBase,
    ComplEx,
    RelationalReasoning,
    LogicalConstraintSatisfaction,
    NeuroSymbolicLayer,
    EntailmentGrounder,
    SemanticLoss,
    KnowledgeGraphReasoner,
)

from .theorem_proving import (
    FormulaType,
    Formula,
    Substitution,
    Unifier,
    NeuralUnifier,
    ProofStep,
    ProofTree,
    NaturalDeduction,
    NeuralTheoremProver,
    BackwardChainer,
    ForwardChainer,
    TPTPReader,
)

from .hybrid_architectures import (
    LiftedNeuralNetwork,
    RelationalNeuralNetwork,
    DifferentiableILP,
    NeuralLogicMachine,
    LogicGraphNetwork,
    SemanticParsingNetwork,
    NeuroSymbolicIntegration,
)

__all__ = [
    # Differentiable Logic
    "GumbelSoftmax",
    "SoftLogicGate",
    "DifferentiableClause",
    "DifferentiableFormula",
    "LogicTensorNetwork",
    "DifferentiableSATSolver",
    "LogicalRegularizer",
    "NeuralPredicate",
    # Rule Learning
    "LogicalRule",
    "AssociationRuleMiner",
    "InductiveLogicProgramming",
    "RuleExtractor",
    "NeuralRuleReasoner",
    "RuleAttention",
    # Symbolic Expression Trees
    "OpType",
    "ExpressionNode",
    "DifferentiableExpression",
    "simplify_expression",
    "expression_to_function",
    "parse_expression",
    # Program Synthesis
    "ProgramOp",
    "ProgramNode",
    "Program",
    "ProgramSpace",
    "NeuralProgramInterpreter",
    "DifferentiableProgramSearch",
    "ProgramSketch",
    "ProgramEvaluator",
    "ProgramSynthesis",
    # Neural-Symbolic Reasoning
    "KnowledgeBase",
    "NeuralKnowledgeBase",
    "ComplEx",
    "RelationalReasoning",
    "LogicalConstraintSatisfaction",
    "NeuroSymbolicLayer",
    "EntailmentGrounder",
    "SemanticLoss",
    "KnowledgeGraphReasoner",
    # Theorem Proving
    "FormulaType",
    "Formula",
    "Substitution",
    "Unifier",
    "NeuralUnifier",
    "ProofStep",
    "ProofTree",
    "NaturalDeduction",
    "NeuralTheoremProver",
    "BackwardChainer",
    "ForwardChainer",
    "TPTPReader",
    # Hybrid Architectures
    "LiftedNeuralNetwork",
    "RelationalNeuralNetwork",
    "DifferentiableILP",
    "NeuralLogicMachine",
    "LogicGraphNetwork",
    "SemanticParsingNetwork",
    "NeuroSymbolicIntegration",
]
