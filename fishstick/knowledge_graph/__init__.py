"""
Knowledge Graph Module

A comprehensive module for knowledge graph and relational learning in the fishstick AI framework.

This module provides:
- Core data structures (kg_core.py)
- Knowledge graph embeddings (kg_embeddings.py)
- Relation reasoning (reasoning.py)
- Schema validation (schema.py)
- Entity resolution (entity_resolution.py)
- Inference rule engine (inference_engine.py)

Example usage:
    >>> from fishstick.knowledge_graph import KnowledgeGraph, Entity, Relation, TransE
    >>> kg = KnowledgeGraph(name="my_graph")
    >>> kg.add_entity(Entity("e1", "Alice", "person"))
    >>> kg.add_entity(Entity("e2", "Bob", "person"))
    >>> kg.add_relation(Relation("r1", "knows", "e1", "e2"))
    >>> print(kg.get_stats())
"""

from .kg_core import (
    Entity,
    Relation,
    RelationType,
    KnowledgeGraph,
    create_graph_from_triplets,
)

from .kg_embeddings import (
    NegativeSampler,
    KGEmbeddingModel,
    TransE,
    DistMult,
    ComplEx,
    RotatE,
    margin_ranking_loss,
    get_score_function,
)

from .reasoning import (
    HornRule,
    QueryPattern,
    RuleMining,
    PathRanking,
    LogicalInference,
    QueryEmbedding,
)

from .schema import (
    SchemaConstraint,
    ConstraintType,
    Schema,
    ValidationResult,
    SchemaValidator,
    SchemaInference,
    CycleDetector,
)

from .entity_resolution import (
    EntityMention,
    EntityCandidate,
    BlockingStrategy,
    CharacterNGramBlocking,
    TypeBlocking,
    EntityLinker,
    EmbeddingMatcher,
    GraphDisambiguator,
    EntityFusion,
)

from .inference_engine import (
    InferenceRule,
    Fact,
    InferenceDirection,
    MaterializedView,
    InferenceEngine,
    DatalogEngine,
    create_transitive_closure_rule,
    create_inverse_rule,
    create_symmetric_rule,
)

__all__ = [
    # Core
    "Entity",
    "Relation",
    "RelationType",
    "KnowledgeGraph",
    "create_graph_from_triplets",
    # Embeddings
    "NegativeSampler",
    "KGEmbeddingModel",
    "TransE",
    "DistMult",
    "ComplEx",
    "RotatE",
    "margin_ranking_loss",
    "get_score_function",
    # Reasoning
    "HornRule",
    "QueryPattern",
    "RuleMining",
    "PathRanking",
    "LogicalInference",
    "QueryEmbedding",
    # Schema
    "SchemaConstraint",
    "ConstraintType",
    "Schema",
    "ValidationResult",
    "SchemaValidator",
    "SchemaInference",
    "CycleDetector",
    # Entity Resolution
    "EntityMention",
    "EntityCandidate",
    "BlockingStrategy",
    "CharacterNGramBlocking",
    "TypeBlocking",
    "EntityLinker",
    "EmbeddingMatcher",
    "GraphDisambiguator",
    "EntityFusion",
    # Inference
    "InferenceRule",
    "Fact",
    "InferenceDirection",
    "MaterializedView",
    "InferenceEngine",
    "DatalogEngine",
    "create_transitive_closure_rule",
    "create_inverse_rule",
    "create_symmetric_rule",
]
