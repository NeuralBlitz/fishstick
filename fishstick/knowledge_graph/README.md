# Knowledge Graph

Knowledge graph representations and embeddings.

## Installation

```bash
pip install fishstick[knowledge_graph]
```

## Overview

The `knowledge_graph` module provides knowledge graph data structures and embedding methods.

## Usage

```python
from fishstick.knowledge_graph import KnowledgeGraph, Entity, Relation

# Create knowledge graph
kg = KnowledgeGraph()

# Add entities and relations
kg.add_entity("Alice", {"type": "person", "age": 30})
kg.add_entity("Bob", {"type": "person", "age": 25})
kg.add_relation("Alice", "knows", "Bob")

# Query
results = kg.query("Alice", "knows", ?)
```

## Classes

| Class | Description |
|-------|-------------|
| `Entity` | Entity representation |
| `Relation` | Relation representation |
| `KnowledgeGraph` | Knowledge graph container |

## Methods

- Graph traversal and querying
- CRUD operations (add, update, delete)
- Serialization (JSON, RDF)

## Examples

See `examples/knowledge_graph/` for complete examples.
