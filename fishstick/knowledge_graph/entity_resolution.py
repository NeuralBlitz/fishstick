"""
Entity Resolution Module

Provides entity linking, matching, and disambiguation for knowledge graphs.

This module provides:
- EntityLinker: Link mentions to entities
- Embedding-based similarity matching
- Blocking strategies for efficient matching
- Graph-based disambiguation
- Fusion of multiple mentions
"""

from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.spatial.distance import cosine
import numpy as np


@dataclass
class EntityMention:
    """
    Represents a mention of an entity in text.

    Attributes:
        text: The mention text
        start_pos: Start position in text
        end_pos: End position in text
        context: Surrounding context
        confidence: Mention confidence score
    """

    text: str
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""
    confidence: float = 1.0


@dataclass
class EntityCandidate:
    """
    Represents a candidate entity for linking.

    Attributes:
        entity_id: Entity identifier
        entity_name: Entity name
        entity_type: Entity type
        score: Linking score
        features: Additional features
    """

    entity_id: str
    entity_name: str
    entity_type: str
    score: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)


class BlockingStrategy:
    """
    Base class for blocking strategies to reduce comparison space.
    """

    def __init__(self):
        pass

    def create_blocks(
        self, mentions: List[EntityMention], entities: Dict[str, str]
    ) -> Dict[int, List[str]]:
        """
        Create blocking keys for mentions and entities.

        Returns:
            Dict mapping mention index to list of entity indices
        """
        raise NotImplementedError


class CharacterNGramBlocking(BlockingStrategy):
    """
    Character n-gram based blocking.
    """

    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n

    def create_blocks(
        self, mentions: List[EntityMention], entities: Dict[str, Any]
    ) -> Dict[int, List[int]]:
        entity_list = []

        for eid, value in entities.items():
            if isinstance(value, tuple):
                ename = value[0]
            else:
                ename = str(value)
            entity_list.append((eid, ename))

        entity_ngrams: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

        for idx, (eid, ename) in enumerate(entity_list):
            ngrams = self._get_ngrams(ename.lower())
            for ng in ngrams:
                entity_ngrams[ng].append(idx)

        blocks: Dict[int, List[int]] = defaultdict(list)

        for m_idx, mention in enumerate(mentions):
            mention_ngrams = self._get_ngrams(mention.text.lower())

            candidate_indices = set()
            for ng in mention_ngrams:
                candidate_indices.update(entity_ngrams.get(ng, []))

            blocks[m_idx] = list(candidate_indices)

        return dict(blocks)

    def _get_ngrams(self, text: str) -> Set[Tuple[str, ...]]:
        ngrams = set()
        for i in range(len(text) - self.n + 1):
            ngrams.add(tuple(text[i : i + self.n]))
        return ngrams


class TypeBlocking(BlockingStrategy):
    """
    Type-based blocking using entity types.
    """

    def __init__(self):
        super().__init__()

    def create_blocks(
        self,
        mentions: List[EntityMention],
        entities: Dict[str, Tuple[str, str]],
    ) -> Dict[int, List[int]]:
        entity_list = list(entities.items())

        type_to_entities: Dict[str, List[int]] = defaultdict(list)

        for idx, (eid, (ename, etype)) in enumerate(entity_list):
            type_to_entities[etype].append(idx)

        blocks = {}

        for m_idx, mention in enumerate(mentions):
            candidate_types = self._infer_types(mention)

            candidates = set()
            for etype in candidate_types:
                candidates.update(type_to_entities.get(etype, []))

            blocks[m_idx] = list(candidates)

        return blocks

    def _infer_types(self, mention: EntityMention) -> Set[str]:
        inferred = {"entity"}
        text_lower = mention.text.lower()

        if any(w in text_lower for w in ["person", "man", "woman", "people"]):
            inferred.add("person")
        if any(w in text_lower for w in ["organization", "company", "corp", "inc"]):
            inferred.add("organization")
        if any(w in text_lower for w in ["place", "city", "country", "location"]):
            inferred.add("location")

        return inferred


class EntityLinker:
    """
    Links entity mentions to knowledge graph entities.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        blocking_strategy: Optional[BlockingStrategy] = None,
    ):
        self.embedding_dim = embedding_dim
        self.blocking_strategy = blocking_strategy or CharacterNGramBlocking()

        self.entity_embeddings: Dict[str, Tensor] = {}
        self.entity_names: Dict[str, str] = {}
        self.entity_types: Dict[str, str] = {}

        self.mention_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def index_entities(
        self,
        entities: Dict[str, Tuple[str, str, Optional[Tensor]]],
    ) -> None:
        """
        Index entities for fast lookup.

        Args:
            entities: Dict mapping entity_id to (name, type, embedding)
        """
        for eid, (name, etype, embed) in entities.items():
            self.entity_names[eid] = name
            self.entity_types[eid] = etype

            if embed is not None:
                self.entity_embeddings[eid] = embed
            else:
                self.entity_embeddings[eid] = self._compute_default_embedding(name)

    def _compute_default_embedding(self, name: str) -> Tensor:
        chars = torch.randn(len(name), self.embedding_dim)
        chars = F.normalize(chars, dim=-1)
        return chars.mean(dim=0, keepdim=True)

    def link_mentions(
        self,
        mentions: List[EntityMention],
        top_k: int = 5,
    ) -> List[List[EntityCandidate]]:
        """
        Link mentions to entities.

        Args:
            mentions: List of entity mentions
            top_k: Number of candidates to return

        Returns:
            List of candidate lists for each mention
        """
        if not self.entity_embeddings:
            return [[] for _ in mentions]

        entity_items = {
            eid: (self.entity_names[eid], self.entity_types[eid])
            for eid in self.entity_names.keys()
        }

        blocks = self.blocking_strategy.create_blocks(mentions, entity_items)

        results = []

        for m_idx, mention in enumerate(mentions):
            mention_embed = self._encode_mention(mention)

            candidate_indices = blocks.get(
                m_idx, list(range(len(self.entity_embeddings)))
            )

            if not candidate_indices:
                results.append([])
                continue

            candidates = []
            entity_ids = list(self.entity_embeddings.keys())

            for idx in candidate_indices:
                if idx < len(entity_ids):
                    eid = entity_ids[idx]
                    entity_embed = self.entity_embeddings[eid]

                    score = F.cosine_similarity(
                        mention_embed.unsqueeze(0),
                        entity_embed.unsqueeze(0),
                    ).item()

                    candidates.append(
                        EntityCandidate(
                            entity_id=eid,
                            entity_name=self.entity_names.get(eid, ""),
                            entity_type=self.entity_types.get(eid, "entity"),
                            score=score,
                        )
                    )

            candidates.sort(key=lambda x: x.score, reverse=True)
            results.append(candidates[:top_k])

        return results

    def _encode_mention(self, mention: EntityMention) -> Tensor:
        char_embed = torch.randn(len(mention.text), self.embedding_dim)
        char_embed = F.normalize(char_embed, dim=-1)

        output, _ = self.mention_encoder(char_embed.unsqueeze(0))

        return output.mean(dim=1).squeeze(0)


class EmbeddingMatcher:
    """
    Embedding-based entity matching.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        similarity_threshold: float = 0.8,
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

    def compute_similarity(
        self,
        embedding1: Tensor,
        embedding2: Tensor,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        sim = F.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0),
        )
        return sim.item()

    def find_matches(
        self,
        source_embeddings: Dict[str, Tensor],
        target_embeddings: Dict[str, Tensor],
        top_k: int = 1,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find matches between source and target entities.

        Args:
            source_embeddings: Dict of entity_id -> embedding
            target_embeddings: Dict of entity_id -> embedding

        Returns:
            Dict mapping source_id to list of (target_id, score) tuples
        """
        matches = {}

        source_ids = list(source_embeddings.keys())
        target_ids = list(target_embeddings.keys())

        target_embeds = torch.stack([target_embeddings[tid] for tid in target_ids])

        for sid in source_ids:
            source_embed = source_embeddings[sid]

            similarities = F.cosine_similarity(
                source_embed.unsqueeze(0),
                target_embeds,
            )

            top_scores, top_indices = similarities.topk(min(top_k, len(target_ids)))

            matches[sid] = [
                (target_ids[idx], score.item())
                for score, idx in zip(top_scores, top_indices)
            ]

        return matches


class GraphDisambiguator:
    """
    Graph-based entity disambiguation using PageRank.
    """

    def __init__(self, damping: float = 0.85, max_iterations: int = 100):
        self.damping = damping
        self.max_iterations = max_iterations

    def disambiguate(
        self,
        candidates: List[EntityCandidate],
        graph: Dict[str, List[str]],
    ) -> EntityCandidate:
        """
        Disambiguate candidates using graph structure.

        Args:
            candidates: List of candidate entities
            graph: Adjacency list of entity connections

        Returns:
            Best disambiguated candidate
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        candidate_ids = [c.entity_id for c in candidates]

        pagerank_scores = self._compute_pagerank(candidate_ids, graph)

        for candidate in candidates:
            candidate.score += pagerank_scores.get(candidate.entity_id, 0.0) * 0.5

        return max(candidates, key=lambda x: x.score)

    def _compute_pagerank(
        self,
        entity_ids: List[str],
        graph: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Compute PageRank scores for entities."""

        subgraph_nodes = set(entity_ids)

        for eid in entity_ids:
            if eid in graph:
                for neighbor in graph[eid]:
                    if neighbor in graph:
                        subgraph_nodes.add(neighbor)

        nodes = list(subgraph_nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        n = len(nodes)

        if n == 0:
            return {}

        transition_matrix = torch.zeros(n, n)

        for i, node in enumerate(nodes):
            neighbors = graph.get(node, [])

            if neighbors:
                for neighbor in neighbors:
                    if neighbor in node_to_idx:
                        transition_matrix[node_to_idx[neighbor], i] = 1.0 / len(
                            neighbors
                        )
            else:
                transition_matrix[:, i] = 1.0 / n

        pr = torch.ones(n) / n

        for _ in range(self.max_iterations):
            new_pr = (1 - self.damping) / n + self.damping * transition_matrix @ pr

            if torch.norm(new_pr - pr, p=1) < 1e-6:
                break

            pr = new_pr

        return {nodes[i]: pr[i].item() for i in range(n)}


class EntityFusion:
    """
    Fuses multiple entity mentions into unified representation.
    """

    def __init__(self):
        self.fusion_strategy = "weighted_average"

    def fuse_entities(
        self,
        entities: List[Tuple[str, Tensor, float]],
    ) -> Tuple[str, Tensor]:
        """
        Fuse multiple entity representations.

        Args:
            entities: List of (entity_id, embedding, weight) tuples

        Returns:
            Tuple of (representative_id, fused_embedding)
        """
        if not entities:
            return None, None

        if len(entities) == 1:
            return entities[0][0], entities[0][1]

        weights = torch.tensor([w for _, _, w in entities])
        weights = weights / weights.sum()

        embeddings = torch.stack([e for _, e, _ in entities])

        fused = torch.sum(embeddings * weights.unsqueeze(-1), dim=0)

        representative = max(entities, key=lambda x: x[2])[0]

        return representative, fused

    def merge_entity_properties(
        self,
        entity_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge properties from multiple entity mentions.

        Args:
            entity_data: List of entity property dictionaries

        Returns:
            Merged properties
        """
        if not entity_data:
            return {}

        if len(entity_data) == 1:
            return entity_data[0]

        merged = {}

        all_keys = set()
        for data in entity_data:
            all_keys.update(data.keys())

        for key in all_keys:
            values = [d.get(key) for d in entity_data if key in d]
            values = [v for v in values if v is not None]

            if not values:
                continue

            if isinstance(values[0], (int, float)):
                merged[key] = np.mean(values)
            elif isinstance(values[0], str):
                merged[key] = max(set(values), key=values.count)
            elif isinstance(values[0], list):
                merged[key] = (
                    list(set(values[0]) | set(values[1:]))
                    if len(values) > 1
                    else values[0]
                )
            else:
                merged[key] = values[0]

        return merged
