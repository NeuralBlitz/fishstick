import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Query:
    anchor: torch.Tensor
    path: List[torch.Tensor]
    target: Optional[torch.Tensor] = None


class QueryEmbedding(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        query_type: str = "2hop",
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.query_type = query_type

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.path_projection = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, query: Query) -> torch.Tensor:
        anchor_emb = self.entity_embeddings(query.anchor)

        if len(query.path) == 0:
            return anchor_emb

        path_embs = []
        for rel in query.path:
            rel_emb = self.relation_embeddings(rel)
            path_embs.append(rel_emb)

        combined = torch.cat([anchor_emb] + path_embs, dim=-1)
        projected = F.relu(self.path_projection(combined))

        return projected

    def embed_query(self, anchor: int, relations: List[int]) -> torch.Tensor:
        anchor_emb = self.entity_embeddings(
            torch.tensor([anchor], device=self.entity_embeddings.weight.device)
        )

        if len(relations) == 0:
            return anchor_emb.squeeze(0)

        path_embs = [
            self.entity_embeddings(
                torch.tensor([anchor], device=self.entity_embeddings.weight.device)
            )
        ]

        current = path_embs[0]
        for rel in relations:
            rel_emb = self.relation_embeddings(
                torch.tensor([rel], device=self.relation_embeddings.weight.device)
            )
            current = current + rel_emb.squeeze(0)
            path_embs.append(current)

        return path_embs[-1]


class MultiHopReasoning(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        num_hops: int = 2,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_hops = num_hops
        self.aggregation = aggregation

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.query_encoder = QueryEmbedding(num_entities, num_relations, embedding_dim)

        self.hop_projs = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(num_hops)]
        )

    def forward(
        self,
        anchor: torch.Tensor,
        relations: List[torch.Tensor],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        current = self.entity_embeddings(anchor)

        for i, rel in enumerate(relations[: self.num_hops]):
            rel_emb = self.relation_embeddings(rel)

            current = current + rel_emb

            current = self.hop_projs[i](current)
            current = F.relu(current)

        return current

    def reason(
        self,
        start_entity: int,
        relation_path: List[int],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        num_candidates: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index

        candidates = dst[src == start_entity]
        rels = edge_type[src == start_entity]

        if len(candidates) == 0:
            return torch.tensor([]), torch.tensor([])

        candidate_embs = self.entity_embeddings(candidates)

        current = self.entity_embeddings(
            torch.tensor([start_entity], device=edge_index.device)
        )

        for rel in relation_path:
            rel_emb = self.relation_embeddings(
                torch.tensor([rel], device=edge_index.device)
            )
            current = current + rel_emb

        scores = (candidate_embs * current).sum(dim=-1)

        top_scores, top_idx = torch.topk(scores, min(num_candidates, len(scores)))

        return candidates[top_idx], top_scores

    def multi_hop_search(
        self,
        anchor: int,
        max_hops: int,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_constraints: Optional[List[int]] = None,
    ) -> Dict[int, List[Tuple[int, int, float]]]:
        src, dst = edge_index

        visited = {anchor}
        reachable = {anchor: [(anchor, [], 1.0)]}

        current_emb = self.entity_embeddings(
            torch.tensor([anchor], device=edge_index.device)
        )

        for hop in range(max_hops):
            next_reachable = {}

            for entity in reachable:
                entity_emb = self.entity_embeddings(
                    torch.tensor([entity], device=edge_index.device)
                )

                mask = src == entity
                neighbors = dst[mask]
                rels = edge_type[mask]

                for neighbor, rel in zip(neighbors, rels):
                    if neighbor.item() not in visited:
                        rel_emb = self.relation_embeddings(rel)
                        score = (entity_emb * rel_emb).item()

                        path_key = neighbor.item()
                        if path_key not in next_reachable:
                            next_reachable[path_key] = []

                        next_reachable[path_key].append((entity, rel.item(), score))

            for e in next_reachable:
                visited.add(e)

            reachable = next_reachable

        return reachable


class LogicalRuleLearner(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        num_rules: int = 32,
        rule_length: int = 2,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_rules = num_rules
        self.rule_length = rule_length

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.rule_weights = nn.Parameter(
            torch.randn(num_rules, rule_length, num_relations)
        )
        self.rule_confidence = nn.Parameter(torch.zeros(num_rules))

        nn.init.xavier_uniform_(self.rule_weights)

    def get_rules(self) -> List[Tuple[List[int], float]]:
        rules = []
        weights = F.softmax(self.rule_weights, dim=-1)

        for i in range(self.num_rules):
            rule_relations = weights[i].argmax(dim=-1).tolist()
            confidence = torch.sigmoid(self.rule_confidence[i]).item()
            rules.append((rule_relations, confidence))

        return rules

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        head_neighbors = dst[src == head]
        head_rel = edge_type[src == head]

        candidate_scores = torch.zeros(self.num_entities, device=head.device)

        for rule_idx in range(self.num_rules):
            rule_weight = torch.sigmoid(self.rule_confidence[rule_idx])

            rule_relations = self.rule_weights[rule_idx].argmax(dim=-1)

            if len(rule_relations) > 0:
                first_rel = rule_relations[0]

                mask = head_rel == first_rel
                intermediate = head_neighbors[mask]

                if len(intermediate) > 0:
                    int_emb = self.entity_embeddings(intermediate)
                    target_emb = self.entity_embeddings.weight

                    scores = (int_emb @ target_emb.T).max(dim=0)[0]

                    candidate_scores += rule_weight * scores

        return candidate_scores

    def mine_rules(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        min_support: int = 10,
        min_confidence: float = 0.1,
    ) -> List[Tuple[Tuple[int, int, int], float, float]]:
        src, dst = edge_index

        triples = {}
        for s, r, o in zip(src.tolist(), edge_type.tolist(), dst.tolist()):
            key = (s, r, o)
            triples[key] = triples.get(key, 0) + 1

        rules = []

        for r1 in range(self.num_relations):
            for r2 in range(self.num_relations):
                if r1 == r2:
                    continue

                head_body = {}
                body_head = {}

                for (s, r, o), count in triples.items():
                    if r == r1:
                        if (s, o) not in head_body:
                            head_body[(s, o)] = []
                        head_body[(s, o)].append(count)

                        if s not in body_head:
                            body_head[s] = {}
                        if o not in body_head[s]:
                            body_head[s][o] = 0
                        body_head[s][o] += count

                support = 0
                for key, counts in head_body.items():
                    support += sum(counts)

                if support >= min_support:
                    correct = 0
                    for (s, o), intermediates in head_body.items():
                        if s in body_head and o in body_head[s]:
                            correct += min(intermediates)

                    confidence = correct / max(support, 1)

                    if confidence >= min_confidence:
                        rules.append(((r1, r2), support, confidence))

        return rules


class KGReasoner(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        num_hops: int = 2,
        use_rules: bool = True,
        num_rules: int = 32,
    ):
        super().__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.query_embedder = QueryEmbedding(num_entities, num_relations, embedding_dim)
        self.multi_hop_reasoner = MultiHopReasoning(
            num_entities, num_relations, embedding_dim, num_hops
        )

        if use_rules:
            self.rule_learner = LogicalRuleLearner(
                num_entities, num_relations, embedding_dim, num_rules
            )
        else:
            self.rule_learner = None

        self.score_proj = nn.Linear(embedding_dim, 1)

    def forward(
        self, queries: List[Query], edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        query_embs = []
        for query in queries:
            q_emb = self.query_embedder(query)
            query_embs.append(q_emb)

        query_embs = torch.stack(query_embs)

        entity_embs = self.entity_embeddings.weight

        scores = query_embs @ entity_embs.T

        return scores

    def answer_query(
        self,
        anchor: int,
        relation_path: List[int],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        use_reasoning: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_reasoning and self.rule_learner is not None:
            rule_scores = self.rule_learner(
                torch.tensor([anchor]),
                torch.tensor([relation_path[0]] if relation_path else 0),
                edge_index,
                edge_type,
            )
        else:
            rule_scores = torch.zeros(self.num_entities)

        hop_scores = self.multi_hop_reasoner(
            torch.tensor([anchor]),
            [torch.tensor([r]) for r in relation_path],
            edge_index,
            edge_type,
        )

        combined_scores = rule_scores + hop_scores

        return combined_scores

    @property
    def num_entities(self) -> int:
        return self.entity_embeddings.num_embeddings

    @property
    def num_relations(self) -> int:
        return self.relation_embeddings.num_embeddings
