"""
Knowledge Graph Core Module

Provides fundamental data structures and operations for knowledge graphs,
including entities, relations, and graph manipulation utilities.

This module implements:
- Entity and Relation dataclasses
- KnowledgeGraph container with CRUD operations
- Graph traversal and query operations
- Serialization/deserialization utilities
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from enum import Enum, auto
from collections import defaultdict
import torch
from torch import Tensor
import json
import hashlib


class RelationType(Enum):
    """Types of relations in knowledge graphs."""

    MANY_TO_MANY = auto()
    ONE_TO_MANY = auto()
    MANY_TO_ONE = auto()
    ONE_TO_ONE = auto()
    SYMMETRIC = auto()
    ANTI_SYMMETRIC = auto()
    REFLEXIVE = auto()
    IRREFLEXIVE = auto()
    TRANSITIVE = auto()


@dataclass
class Entity:
    """
    Represents an entity in a knowledge graph.

    Attributes:
        id: Unique identifier for the entity
        name: Human-readable name
        entity_type: Type/category of the entity
        attributes: Dictionary of entity attributes
        embeddings: Optional pre-computed embeddings
        metadata: Additional metadata
    """

    id: str
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class Relation:
    """
    Represents a relation between entities in a knowledge graph.

    Attributes:
        id: Unique identifier for the relation
        name: Human-readable name (e.g., "father_of", "works_at")
        source: Source entity ID
        target: Target entity ID
        relation_type: Type of relation
        attributes: Dictionary of relation attributes
        confidence: Confidence score [0, 1]
        weight: Edge weight for weighted graphs
    """

    id: str
    name: str
    source: str
    target: str
    relation_type: RelationType = RelationType.MANY_TO_MANY
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    weight: float = 1.0

    def __hash__(self):
        return hash((self.source, self.target, self.name))

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (
            self.source == other.source
            and self.target == other.target
            and self.name == other.name
        )


@dataclass
class KnowledgeGraph:
    """
    Container for knowledge graph data with CRUD operations.

    Provides efficient storage and querying of entities and relations
    with support for batch operations and graph traversal.

    Attributes:
        name: Optional name for the graph
        directed: Whether edges are directed (default: True)
        entities: Dictionary of entity_id -> Entity
        relations: Dictionary of (source, target, rel_name) -> Relation
        entity_index: Reverse index for entity lookups
        relation_index: Index for relation lookups by name

    Example:
        >>> kg = KnowledgeGraph(name="family_graph")
        >>> kg.add_entity(Entity("e1", "John", "person"))
        >>> kg.add_entity(Entity("e2", "Jane", "person"))
        >>> kg.add_relation(Relation("r1", "married_to", "e1", "e2"))
        >>> neighbors = kg.get_neighbors("e1")
    """

    name: str = "knowledge_graph"
    directed: bool = True
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[Tuple[str, str, str], Relation] = field(default_factory=dict)
    entity_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    relation_index: Dict[str, Set[Tuple[str, str, str]]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def __post_init__(self):
        """Initialize indexes after construction."""
        if not isinstance(self.entity_index, defaultdict):
            self.entity_index = defaultdict(set)
        if not isinstance(self.relation_index, defaultdict):
            self.relation_index = defaultdict(set)

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.

        Args:
            entity: Entity object to add

        Raises:
            ValueError: If entity with same ID already exists
        """
        if entity.id in self.entities:
            raise ValueError(f"Entity with id '{entity.id}' already exists")
        self.entities[entity.id] = entity
        self.entity_index[entity.entity_type].add(entity.id)

    def add_relation(self, relation: Relation) -> None:
        """
        Add a relation to the graph.

        Args:
            relation: Relation object to add

        Raises:
            ValueError: If source or target entity doesn't exist
        """
        if relation.source not in self.entities:
            raise ValueError(f"Source entity '{relation.source}' not found")
        if relation.target not in self.entities:
            raise ValueError(f"Target entity '{relation.target}' not found")

        key = (relation.source, relation.target, relation.name)
        self.relations[key] = relation
        self.relation_index[relation.name].add(key)

    def remove_entity(self, entity_id: str) -> None:
        """
        Remove an entity and all its relations.

        Args:
            entity_id: ID of entity to remove
        """
        if entity_id not in self.entities:
            return

        entity = self.entities[entity_id]

        relations_to_remove = [
            key
            for key in self.relations.keys()
            if key[0] == entity_id or key[1] == entity_id
        ]

        for key in relations_to_remove:
            rel = self.relations.pop(key, None)
            if rel:
                self.relation_index[rel.name].discard(key)

        self.entity_index[entity.entity_type].discard(entity_id)
        del self.entities[entity_id]

    def remove_relation(self, source: str, target: str, rel_name: str) -> None:
        """
        Remove a relation from the graph.

        Args:
            source: Source entity ID
            target: Target entity ID
            rel_name: Relation name
        """
        key = (source, target, rel_name)
        if key in self.relations:
            rel = self.relations.pop(key)
            self.relation_index[rel.name].discard(key)

    def get_neighbors(
        self,
        entity_id: str,
        relation_name: Optional[str] = None,
        include_reverse: bool = True,
    ) -> List[Tuple[str, str, Entity]]:
        """
        Get all neighboring
        Args:
            entity_id: ID of the entity entities.

            relation_name: Optional filter by relation name
            include_reverse: Include incoming edges

        Returns:
            List of tuples (neighbor_id, relation_name, entity)
        """
        if entity_id not in self.entities:
            return []

        neighbors = []

        for (src, tgt, rel_name), rel in self.relations.items():
            if relation_name is not None and rel_name != relation_name:
                continue

            if src == entity_id:
                if tgt in self.entities:
                    neighbors.append((tgt, rel_name, self.entities[tgt]))
            elif include_reverse and tgt == entity_id:
                if src in self.entities:
                    neighbors.append((src, f"inv_{rel_name}", self.entities[src]))

        return neighbors

    def get_relations(
        self, entity_id: str, as_source: bool = True, as_target: bool = True
    ) -> List[Relation]:
        """
        Get all relations for an entity.

        Args:
            entity_id: ID of the entity
            as_source: Include relations where entity is source
            as_target: Include relations where entity is target

        Returns:
            List of Relation objects
        """
        result = []

        for (src, tgt, rel_name), rel in self.relations.items():
            if as_source and src == entity_id:
                result.append(rel)
            if as_target and tgt == entity_id:
                result.append(rel)

        return result

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Type of entities to retrieve

        Returns:
            List of Entity objects
        """
        entity_ids = self.entity_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def find_relations_by_name(self, rel_name: str) -> List[Relation]:
        """
        Find all relations with a specific name.

        Args:
            rel_name: Name of the relation

        Returns:
            List of Relation objects
        """
        keys = self.relation_index.get(rel_name, set())
        return [self.relations[key] for key in keys if key in self.relations]

    def get_subgraph(
        self, entity_ids: Set[str], include_interior_relations: bool = True
    ) -> "KnowledgeGraph":
        """
        Extract a subgraph containing specified entities.

        Args:
            entity_ids: Set of entity IDs to include
            include_interior_relations: Include relations between included entities

        Returns:
            New KnowledgeGraph containing the subgraph
        """
        subgraph = KnowledgeGraph(name=f"{self.name}_subgraph", directed=self.directed)

        for eid in entity_ids:
            if eid in self.entities:
                subgraph.add_entity(self.entities[eid])

        if include_interior_relations:
            for key, rel in self.relations.items():
                src, tgt, _ = key
                if src in entity_ids and tgt in entity_ids:
                    subgraph.add_relation(rel)

        return subgraph

    def to_adjacency_list(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """
        Convert graph.

        Returns to adjacency list format:
            Dictionary mapping entity_id to list of (target, relation, weight)
        """
        adj = defaultdict(list)

        for (src, tgt, rel_name), rel in self.relations.items():
            adj[src].append((tgt, rel_name, rel.weight))

            if not self.directed:
                adj[tgt].append((src, rel_name, rel.weight))

        return dict(adj)

    def to_edge_list(self) -> List[Tuple[str, str, str, float]]:
        """
        Convert graph to edge list format.

        Returns:
            List of (source, target, relation, weight) tuples
        """
        return [
            (src, tgt, rel_name, rel.weight)
            for (src, tgt, rel_name), rel in self.relations.items()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Returns:
            Dictionary with graph statistics
        """
        entity_types = defaultdict(int)
        for entity in self.entities.values():
            entity_types[entity.entity_type] += 1

        relation_names = defaultdict(int)
        for rel in self.relations.values():
            relation_names[rel.name] += 1

        num_directed = sum(1 for _ in self.relations.keys())
        num_undirected = num_directed // 2 if not self.directed else num_directed

        return {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "entity_types": dict(entity_types),
            "relation_names": dict(relation_names),
            "avg_degree": (2 * len(self.relations) / len(self.entities))
            if self.entities
            else 0,
            "is_directed": self.directed,
        }

    def save(self, filepath: str) -> None:
        """
        Save knowledge graph to JSON file.

        Args:
            filepath: Path to save the graph
        """
        data = {
            "name": self.name,
            "directed": self.directed,
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.entity_type,
                    "attributes": e.attributes,
                    "metadata": e.metadata,
                }
                for e in self.entities.values()
            ],
            "relations": [
                {
                    "id": r.id,
                    "name": r.name,
                    "source": r.source,
                    "target": r.target,
                    "type": r.relation_type.name,
                    "attributes": r.attributes,
                    "confidence": r.confidence,
                    "weight": r.weight,
                }
                for r in self.relations.values()
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """
        Load knowledge graph from JSON file.

        Args:
            filepath: Path to load the graph from

        Returns:
            Loaded KnowledgeGraph
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        kg = cls(name=data["name"], directed=data["directed"])

        for e_data in data["entities"]:
            entity = Entity(
                id=e_data["id"],
                name=e_data["name"],
                entity_type=e_data["type"],
                attributes=e_data.get("attributes", {}),
                metadata=e_data.get("metadata", {}),
            )
            kg.add_entity(entity)

        for r_data in data["relations"]:
            relation = Relation(
                id=r_data["id"],
                name=r_data["name"],
                source=r_data["source"],
                target=r_data["target"],
                relation_type=RelationType[r_data.get("type", "MANY_TO_MANY")],
                attributes=r_data.get("attributes", {}),
                confidence=r_data.get("confidence", 1.0),
                weight=r_data.get("weight", 1.0),
            )
            kg.add_relation(relation)

        return kg


def create_graph_from_triplets(
    triplets: List[Tuple[str, str, str]],
    entity_types: Optional[Dict[str, str]] = None,
    relation_types: Optional[Dict[str, RelationType]] = None,
    directed: bool = True,
) -> KnowledgeGraph:
    """
    Create a knowledge graph from a list of triplets.

    Args:
        triplets: List of (subject, predicate, object) tuples
        entity_types: Optional mapping of entity -> type
        relation_types: Optional mapping of relation -> type
        directed: Whether graph is directed

    Returns:
        KnowledgeGraph constructed from triplets
    """
    kg = KnowledgeGraph(directed=directed)
    entity_types = entity_types or {}
    relation_types = relation_types or {}

    entities_seen = set()

    for subj, pred, obj in triplets:
        if subj not in entities_seen:
            entity = Entity(
                id=subj,
                name=subj,
                entity_type=entity_types.get(subj, "entity"),
            )
            kg.add_entity(entity)
            entities_seen.add(subj)

        if obj not in entities_seen:
            entity = Entity(
                id=obj,
                name=obj,
                entity_type=entity_types.get(obj, "entity"),
            )
            kg.add_entity(entity)
            entities_seen.add(obj)

        relation = Relation(
            id=f"{subj}_{pred}_{obj}",
            name=pred,
            source=subj,
            target=obj,
            relation_type=relation_types.get(pred, RelationType.MANY_TO_MANY),
        )
        kg.add_relation(relation)

    return kg
