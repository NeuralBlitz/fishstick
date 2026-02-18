import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BaseEmbedding(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

    def forward(self, head, relation, tail):
        raise NotImplementedError

    def get_entity_embeddings(self, entities: torch.Tensor) -> torch.Tensor:
        return self.entity_embeddings(entities)

    def get_relation_embeddings(self, relations: torch.Tensor) -> torch.Tensor:
        return self.relation_embeddings(relations)


class TransE(BaseEmbedding):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 1.0,
        norm: int = 1,
    ):
        super().__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.norm = norm

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        score = h + r - t
        return -torch.norm(score, p=self.norm, dim=-1)

    def score_head_prediction(
        self, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        return -(self.entity_embeddings.weight + r.unsqueeze(1) - t.unsqueeze(0)).norm(
            p=self.norm, dim=-1
        )

    def score_tail_prediction(
        self, head: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        return -(h + r.unsqueeze(1) - self.entity_embeddings.weight.unsqueeze(0)).norm(
            p=self.norm, dim=-1
        )


class TransR(BaseEmbedding):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int,
        relation_dim: int,
        margin: float = 1.0,
    ):
        super().__init__(num_entities, num_relations, entity_dim)
        self.relation_dim = relation_dim
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, entity_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        self.transfer_matrices = nn.Embedding(num_relations, entity_dim * relation_dim)

        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.transfer_matrices.weight, -0.1, 0.1)

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        M = self.transfer_matrices(relation).view(
            -1, self.relation_dim, self.embedding_dim
        )

        h_trans = torch.bmm(h.unsqueeze(1), M.transpose(1, 2)).squeeze(1)
        t_trans = torch.bmm(t.unsqueeze(1), M.transpose(1, 2)).squeeze(1)

        score = h_trans + r - t_trans
        return -torch.norm(score, p=2, dim=-1)


class RotatE(BaseEmbedding):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 12.0,
    ):
        super().__init__(num_entities, num_relations, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        assert embedding_dim % 2 == 0, "RotatE embedding_dim must be even"

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim // 2)

        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        h_real, h_imag = h.chunk(2, dim=-1)
        t_real, t_imag = t.chunk(2, dim=-1)

        r_pi = r * math.pi

        r_real = torch.cos(r_pi)
        r_imag = torch.sin(r_pi)

        h_real_ = h_real * r_real - h_imag * r_imag
        h_imag_ = h_real * r_imag + h_imag * r_real

        score = (h_real_ - t_real) ** 2 + (h_imag_ - t_imag) ** 2
        return -score.sum(dim=-1)

    def score_tail_prediction(
        self, head: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)

        h_real, h_imag = h.chunk(2, dim=-1)
        r_pi = r * math.pi
        r_real = torch.cos(r_pi)
        r_imag = torch.sin(r_pi)

        t_real = h_real * r_real - h_imag * r_imag
        t_imag = h_real * r_imag + h_imag * r_real

        t_all = torch.cat([t_real, t_imag], dim=-1)

        entity_emb = self.entity_embeddings.weight
        scores = -((entity_emb.unsqueeze(0) - t_all.unsqueeze(1)) ** 2).sum(dim=-1)
        return scores


class ComplEx(BaseEmbedding):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(num_relations, embedding_dim)

        nn.init.uniform_(self.entity_embeddings_real.weight, -0.1, 0.1)
        nn.init.uniform_(self.entity_embeddings_imag.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings_real.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings_imag.weight, -0.1, 0.1)

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        h_real = self.entity_embeddings_real(head)
        h_imag = self.entity_embeddings_imag(head)
        r_real = self.relation_embeddings_real(relation)
        r_imag = self.relation_embeddings_imag(relation)
        t_real = self.entity_embeddings_real(tail)
        t_imag = self.entity_embeddings_imag(tail)

        score = h_real * t_real * r_real + h_imag * t_imag * r_real
        score += h_real * t_imag * r_imag - h_imag * t_real * r_imag
        return score.sum(dim=-1)

    def score_head_prediction(
        self, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        r_real = self.relation_embeddings_real(relation)
        r_imag = self.relation_embeddings_imag(relation)
        t_real = self.entity_embeddings_real(tail)
        t_imag = self.entity_embeddings_imag(tail)

        h_real = self.entity_embeddings_real.weight
        h_imag = self.entity_embeddings_imag.weight

        score = h_real * t_real * r_real + h_imag * t_imag * r_real
        score += h_real * t_imag * r_imag - h_imag * t_real * r_imag
        return score.sum(dim=-1)


class DistMult(BaseEmbedding):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__(num_entities, num_relations, embedding_dim)

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        score = (h * r * t).sum(dim=-1)
        return score


class QuatE(BaseEmbedding):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__(num_entities, num_relations, embedding_dim)
        assert embedding_dim % 4 == 0, "QuatE embedding_dim must be divisible by 4"
        self.quat_dim = embedding_dim // 4

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.relation_embeddings.weight, -0.1, 0.1)

    def _quaternion_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_i, a_j, a_k, a_l = a.chunk(4, dim=-1)
        b_i, b_j, b_k, b_l = b.chunk(4, dim=-1)

        o_i = a_i * b_i - a_j * b_j - a_k * b_k - a_l * b_l
        o_j = a_i * b_j + a_j * b_i + a_k * b_l - a_l * b_k
        o_k = a_i * b_k - a_j * b_l + a_k * b_i + a_l * b_j
        o_l = a_i * b_l + a_j * b_k - a_k * b_j + a_l * b_i

        return torch.cat([o_i, o_j, o_k, o_l], dim=-1)

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)

        hr = self._quaternion_mul(h, r)
        score = (hr * t).sum(dim=-1)
        return score

    def score_tail_prediction(
        self, head: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)

        hr = self._quaternion_mul(h, r)
        scores = (hr.unsqueeze(1) * self.entity_embeddings.weight.unsqueeze(0)).sum(
            dim=-1
        )
        return scores


class KGEmbeddingModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        model_type: str = "TransE",
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.model_type = model_type

        model_classes = {
            "TransE": TransE,
            "TransR": TransR,
            "RotatE": RotatE,
            "ComplEx": ComplEx,
            "DistMult": DistMult,
            "QuatE": QuatE,
        }

        if model_type not in model_classes:
            raise ValueError(
                f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}"
            )

        if model_type == "TransR":
            self.model = model_classes[model_type](
                num_entities, num_relations, embedding_dim, embedding_dim
            )
        else:
            self.model = model_classes[model_type](
                num_entities, num_relations, embedding_dim
            )

    def forward(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        return self.model(head, relation, tail)

    def score_tail_prediction(
        self, head: torch.Tensor, relation: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.model, "score_tail_prediction"):
            return self.model.score_tail_prediction(head, relation)

        h = self.model.entity_embeddings(head)
        r = self.model.relation_embeddings(relation)

        scores = self.model.score_tail_prediction(head, relation)
        return scores

    def score_head_prediction(
        self, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.model, "score_head_prediction"):
            return self.model.score_head_prediction(relation, tail)

        t = self.model.entity_embeddings(tail)
        r = self.model.relation_embeddings(relation)

        return self.model.score_head_prediction(relation, tail)
