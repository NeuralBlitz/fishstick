import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class MatchingNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        fce: bool = False,
        lstm_layers: int = 1,
        lstm_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.fce = fce

        if fce:
            self.lstm = nn.LSTM(
                input_size=encoder.out_features,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.attention = nn.Linear(lstm_hidden * 2, 1)

    @property
    def out_features(self):
        return self.encoder.out_features if not self.fce else 128

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)
        n_query = query.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, n_query, -1)

        if self.fce:
            support_emb, query_emb = self._fce(support_emb, query_emb)

        logits = self._cosine_similarity(query_emb, support_emb)

        if support_labels is not None:
            return self._compute_loss(logits, support_labels, n_query, n_support)

        return logits

    def _fce(
        self, support: torch.Tensor, query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_way = support.size(0)
        n_support = support.size(1)
        n_query = query.size(1)

        combined = torch.cat([support, query], dim=1)
        combined, _ = self.lstm(combined)

        support = combined[:, :n_support]
        query = combined[:, n_support:]

        return support, query

    def _cosine_similarity(
        self, query: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        query = F.normalize(query, p=2, dim=-1)
        support = F.normalize(support, p=2, dim=-1)

        similarities = torch.einsum("bqk,bsk->bqsk", query, support)
        similarities = similarities.mean(-1)

        return similarities

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_query: int,
        n_support: int,
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)

        query_labels = labels[:, n_support:]
        predictions = logits[:, n_support:]

        loss = F.cross_entropy(
            predictions.view(-1, self.n_classes), query_labels.view(-1)
        )
        return loss


class RelationNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        relation_dim: int = 8,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes

        encoder_out = encoder.out_features
        self.relation_module = nn.Sequential(
            nn.Linear(encoder_out * 2, relation_dim),
            nn.ReLU(),
            nn.Linear(relation_dim, 1),
            nn.Sigmoid(),
        )

    @property
    def out_features(self):
        return self.encoder.out_features

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)
        n_query = query.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, n_query, -1)

        class_embeddings = self._aggregate_support(support_emb)

        relations = self._compute_relations(query_emb, class_embeddings)

        if support_labels is not None:
            return self._compute_loss(relations, support_labels, n_query)

        return relations

    def _aggregate_support(self, support: torch.Tensor) -> torch.Tensor:
        return support.mean(dim=1)

    def _compute_relations(
        self, query: torch.Tensor, class_emb: torch.Tensor
    ) -> torch.Tensor:
        n_way = class_emb.size(0)
        n_query = query.size(1)

        query_expanded = query.unsqueeze(2).expand(-1, -1, n_way, -1)
        class_expanded = class_emb.unsqueeze(1).expand(-1, n_query, -1, -1)

        combined = torch.cat([query_expanded, class_expanded], dim=-1)
        relations = self.relation_module(combined)
        relations = relations.squeeze(-1)

        return relations

    def _compute_loss(
        self, relations: torch.Tensor, labels: torch.Tensor, n_query: int
    ) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(relations.size(0), -1)

        target_relations = torch.zeros_like(relations)
        for i in range(self.n_classes):
            mask = labels[:, n_query:] == i
            target_relations[mask.unsqueeze(-1).expand_as(target_relations)] = 1.0

        loss = F.mse_loss(relations[:, n_query:], target_relations[:, n_query:])
        return loss


class MAML(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

        self.classifier = nn.Linear(encoder.out_features, n_classes)

    @property
    def out_features(self):
        return self.encoder.out_features

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_support = support.size(1)
        n_query = query.size(1)

        support_emb = self.encoder(support.view(-1, *support.shape[2:]))
        query_emb = self.encoder(query.view(-1, *query.shape[2:]))

        support_emb = support_emb.view(-2, n_support, -1)
        query_emb = query_emb.view(-2, n_query, -1)

        adapted_classifier = self._inner_update(support_emb, support_labels)

        logits = adapted_classifier(query_emb)

        if support_labels is not None:
            return self._compute_loss(logits, support_labels)

        return logits

    def _inner_update(
        self,
        support_emb: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> "MAMLClassifier":
        fast_weights = {
            "weight": self.classifier.weight.clone(),
            "bias": self.classifier.bias.clone()
            if self.classifier.bias is not None
            else None,
        }

        for _ in range(self.inner_steps):
            logits = F.linear(support_emb, fast_weights["weight"], fast_weights["bias"])
            loss = F.cross_entropy(logits, support_labels)

            grads = torch.autograd.grad(
                loss, fast_weights.values(), create_graph=True, allow_unused=True
            )
            grads = [
                g if g is not None else torch.zeros_like(w)
                for g, w in zip(grads, fast_weights.values())
            ]

            fast_weights["weight"] = fast_weights["weight"] - self.inner_lr * grads[0]
            if fast_weights["bias"] is not None:
                fast_weights["bias"] = fast_weights["bias"] - self.inner_lr * grads[1]

        return MAMLClassifier(fast_weights["weight"], fast_weights["bias"])

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.unsqueeze(0).expand(logits.size(0), -1)
        query_labels = labels[:, logits.size(1) :]
        return F.cross_entropy(logits.view(-1, self.n_classes), query_labels.view(-1))


class MAMLClassifier(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MetaLearner(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        method: str = "matching",
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.method = method

        if method == "matching":
            self.network = MatchingNetwork(encoder, n_classes, **kwargs)
        elif method == "relation":
            self.network = RelationNetwork(encoder, n_classes, **kwargs)
        elif method == "maml":
            self.network = MAML(encoder, n_classes, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(
        self,
        support: torch.Tensor,
        query: torch.Tensor,
        support_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.network(support, query, support_labels)
