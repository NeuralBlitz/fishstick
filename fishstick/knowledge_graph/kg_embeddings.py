"""
Knowledge Graph Embeddings Module

Implements state-of-the-art knowledge graph embedding models for link prediction
and entity relationship learning.

This module provides:
- TransE: Translation-based embeddings
- DistMult: Bilinear diagonal embeddings
- ComplEx: Complex-valued embeddings
- RotatE: Rotation-based embeddings
- Negative sampling strategies
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, Embedding, Dropout


class NegativeSampler:
    """
    Negative sampler for knowledge graph training.

    Generates corrupted triplets by randomly replacing subject or object entities.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        corrupt_prob: float = 0.5,
        uniform_ratio: float = 1.0,
    ):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.corrupt_prob = corrupt_prob
        self.uniform_ratio = uniform_ratio

    def sample(
        self,
        positive_triplets: Tensor,
        num_negatives: int,
        entity_frequency: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = positive_triplets.shape[0]
        device = positive_triplets.device

        negatives = positive_triplets.repeat(num_negatives, 1)

        should_corrupt_subject = (
            torch.rand(num_negatives * batch_size, device=device) < self.corrupt_prob
        )

        if entity_frequency is not None and self.uniform_ratio < 1.0:
            probs = entity_frequency / entity_frequency.sum()
            random_entities = torch.multinomial(
                probs, num_negatives * batch_size, replacement=True
            )
        else:
            random_entities = torch.randint(
                0, self.num_entities, (num_negatives * batch_size,), device=device
            )

        negatives[should_corrupt_subject, 0] = random_entities[should_corrupt_subject]
        negatives[~should_corrupt_subject, 2] = random_entities[~should_corrupt_subject]

        return negatives


class KGEmbeddingModel(nn.Module):
    """Base class for knowledge graph embedding models."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

    def get_embeddings(
        self, entities: Tensor, relations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError

    def score_triplets(self, triplets: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, positive: Tensor, negative: Tensor) -> Tuple[Tensor, Tensor]:
        pos_scores = self.score_triplets(positive)
        neg_scores = self.score_triplets(negative)
        return pos_scores, neg_scores


class TransE(KGEmbeddingModel):
    """
    TransE: Translational Embedding Model.

    Based on Translating Embeddings for Modeling Multi-relational Data (Bordes et al., NIPS 2013).
    Represents relations as translation vectors: h + r approximately equals t.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        p_norm: float = 2.0,
    ):
        super().__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.p_norm = p_norm

        self.entity_embeddings = Embedding(num_entities, embedding_dim)
        self.relation_embeddings = Embedding(num_relations, embedding_dim)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.uniform_(
            self.entity_embeddings.weight,
            -6.0 / math.sqrt(self.embedding_dim),
            6.0 / math.sqrt(self.embedding_dim),
        )
        nn.init.uniform_(
            self.relation_embeddings.weight,
            -6.0 / math.sqrt(self.embedding_dim),
            6.0 / math.sqrt(self.embedding_dim),
        )

    def get_embeddings(
        self, entities: Tensor, relations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        entity_embeds = self.entity_embeddings(entities)
        if relations is not None:
            relation_embeds = self.relation_embeddings(relations)
            return entity_embeds, relation_embeds
        return entity_embeds, None

    def score_triplets(self, triplets: Tensor) -> Tensor:
        subjects, relations, objects = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        subject_embeds = self.entity_embeddings(subjects)
        relation_embeds = self.relation_embeddings(relations)
        object_embeds = self.entity_embeddings(objects)

        translations = subject_embeds + relation_embeds - object_embeds

        scores = -torch.norm(translations, p=self.p_norm, dim=-1)

        return scores

    def predict(self, subject: int, relation: int) -> Tensor:
        subject_embed = self.entity_embeddings(
            torch.tensor([subject], device=self.entity_embeddings.weight.device)
        )
        relation_embed = self.relation_embeddings(
            torch.tensor([relation], device=self.relation_embeddings.weight.device)
        )

        translation = subject_embed + relation_embed

        all_objects = self.entity_embeddings.weight

        scores = -torch.norm(translation - all_objects, p=self.p_norm, dim=-1)

        return scores


class DistMult(KGEmbeddingModel):
    """
    DistMult: Bilinear Diagonal Embedding Model.

    Based on Embedding Entities and Relations for Learning and Inference in Knowledge Bases (Yang et al., ICLR 2015).
    Represents relations as diagonal matrices.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
    ):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embeddings = Embedding(num_entities, embedding_dim)
        self.relation_embeddings = Embedding(num_relations, embedding_dim)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def get_embeddings(
        self, entities: Tensor, relations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        entity_embeds = self.entity_embeddings(entities)
        if relations is not None:
            relation_embeds = self.relation_embeddings(relations)
            return entity_embeds, relation_embeds
        return entity_embeds, None

    def score_triplets(self, triplets: Tensor) -> Tensor:
        subjects, relations, objects = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        subject_embeds = self.entity_embeddings(subjects)
        relation_embeds = self.relation_embeddings(relations)
        object_embeds = self.entity_embeddings(objects)

        scores = torch.sum(subject_embeds * relation_embeds * object_embeds, dim=-1)

        return scores

    def predict(self, subject: int, relation: int) -> Tensor:
        subject_embed = self.entity_embeddings(
            torch.tensor([subject], device=self.entity_embeddings.weight.device)
        )
        relation_embed = self.relation_embeddings(
            torch.tensor([relation], device=self.relation_embeddings.weight.device)
        )

        all_objects = self.entity_embeddings.weight

        scores = torch.sum(subject_embed * relation_embed * all_objects, dim=-1)

        return scores


class ComplEx(KGEmbeddingModel):
    """
    ComplEx: Complex-valued Embedding Model.

    Based on Complex Embeddings for Simple Link Prediction (Trouillon et al., ICML 2016).
    Uses complex-valued embeddings to model asymmetric relations.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
    ):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embeddings_real = Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = Embedding(num_relations, embedding_dim)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)

    def get_complex_embeddings(self, entities: Tensor) -> Tuple[Tensor, Tensor]:
        real = self.entity_embeddings_real(entities)
        imag = self.entity_embeddings_imag(entities)
        return real, imag

    def get_embeddings(
        self, entities: Tensor, relations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        real, imag = self.get_complex_embeddings(entities)
        entity_embeds = torch.complex(real, imag)

        if relations is not None:
            rel_real = self.relation_embeddings_real(relations)
            rel_imag = self.relation_embeddings_imag(relations)
            relation_embeds = torch.complex(rel_real, rel_imag)
            return entity_embeds, relation_embeds
        return entity_embeds, None

    def score_triplets(self, triplets: Tensor) -> Tensor:
        subjects, relations, objects = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        subject_real = self.entity_embeddings_real(subjects)
        subject_imag = self.entity_embeddings_imag(subjects)
        rel_real = self.relation_embeddings_real(relations)
        rel_imag = self.relation_embeddings_imag(relations)
        object_real = self.entity_embeddings_real(objects)
        object_imag = self.entity_embeddings_imag(objects)

        score_real = subject_real * object_real + subject_imag * object_imag + rel_real
        score_imag = subject_real * object_imag - subject_imag * object_real + rel_imag

        scores = torch.sum(score_real + score_imag, dim=-1)

        return scores


class RotatE(KGEmbeddingModel):
    """
    RotatE: Rotation-based Embedding Model.

    Based on RotatE: Knowledge Graph Completion with Relation Rotation (Sun et al., 2019).
    Represents relations as rotations in complex plane.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 12.0,
    ):
        super().__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin

        self.entity_embeddings = Embedding(num_entities, embedding_dim)
        self.relation_embeddings = Embedding(num_relations, embedding_dim // 2)

        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_embeddings.weight, -3.14159, 3.14159)

    def get_embeddings(
        self, entities: Tensor, relations: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        entity_embeds = self.entity_embeddings(entities)

        if relations is not None:
            relation_phases = self.relation_embeddings(relations)
            return entity_embeds, relation_phases
        return entity_embeds, None

    def score_triplets(self, triplets: Tensor) -> Tensor:
        subjects, relations, objects = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        subject_embeds = self.entity_embeddings(subjects)
        object_embeds = self.entity_embeddings(objects)
        relation_phases = self.relation_embeddings(relations)

        phase = relation_phases / (self.embedding_dim / 2)

        cos_r = torch.cos(phase)
        sin_r = torch.sin(phase)

        re_head = subject_embeds[..., : self.embedding_dim // 2]
        im_head = subject_embeds[..., self.embedding_dim // 2 :]

        re_transformed = re_head * cos_r - im_head * sin_r
        im_transformed = re_head * sin_r + im_head * cos_r

        re_tail = object_embeds[..., : self.embedding_dim // 2]
        im_tail = object_embeds[..., self.embedding_dim // 2 :]

        distance = torch.sqrt(
            (re_transformed - re_tail) ** 2 + (im_transformed - im_tail) ** 2
        )

        scores = -torch.sum(distance, dim=-1)

        return scores


def margin_ranking_loss(
    positive_scores: Tensor,
    negative_scores: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """
    Compute margin ranking loss for knowledge graph training.

    Args:
        positive_scores: Scores for positive triplets
        negative_scores: Scores for negative triplets
        margin: Margin for ranking loss

    Returns:
        Loss value
    """
    loss = torch.clamp(margin - positive_scores.unsqueeze(1) + negative_scores, min=0)
    return loss.mean()


def get_score_function(model_name: str):
    """
    Get scoring function for a given model.

    Args:
        model_name: Name of the model (transe, distmult, complex, rotate)

    Returns:
        Scoring function
    """
    score_functions = {
        "transe": lambda m, t: m.score_triplets(t),
        "distmult": lambda m, t: m.score_triplets(t),
        "complex": lambda m, t: m.score_triplets(t),
        "rotate": lambda m, t: m.score_triplets(t),
    }
    return score_functions.get(model_name.lower())
