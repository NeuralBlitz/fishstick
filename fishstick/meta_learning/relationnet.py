import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class RelationModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim * 2

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        class_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query_embeddings.size(0)
        num_classes = class_embeddings.size(0)

        query_expanded = query_embeddings.unsqueeze(1).expand(-1, num_classes, -1)
        class_expanded = class_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        combined = torch.cat([query_expanded, class_expanded], dim=-1)

        relations = self.net(combined)
        return relations.squeeze(-1)


class RelationNetwork(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        relation_module: Optional[nn.Module] = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = encoder

        if relation_module is None:
            self.relation_module = RelationModule(
                input_dim=self._get_embedding_dim(),
                hidden_dim=hidden_dim,
            )
        else:
            self.relation_module = relation_module

    def _get_embedding_dim(self) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 84, 84)
            dummy_output = self.encoder(dummy_input)
            return dummy_output.size(-1)

    def forward(
        self,
        query_x: torch.Tensor,
        class_x: torch.Tensor,
    ) -> torch.Tensor:
        query_embeddings = self.encoder(query_x)
        class_embeddings = self.encoder(class_x)

        relations = self.relation_module(query_embeddings, class_embeddings)

        return relations

    def predict(
        self,
        query_x: torch.Tensor,
        class_x: torch.Tensor,
    ) -> torch.Tensor:
        relations = self.forward(query_x, class_x)
        predictions = relations.argmax(dim=-1)
        return predictions

    def episode_train_step(
        self,
        episode: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        support_x = episode["support_x"]
        support_y = episode["support_y"]
        query_x = episode["query_x"]
        query_y = episode["query_y"]

        classes = torch.unique(support_y)

        class_embeddings = []
        for c in classes:
            class_mask = support_y == c
            class_samples = support_x[class_mask]
            class_emb = self.encoder(class_samples)
            class_emb = class_emb.mean(dim=0, keepdim=True)
            class_embeddings.append(class_emb)

        class_embeddings = torch.cat(class_embeddings, dim=0)

        query_embeddings = self.encoder(query_x)

        relations = self.relation_module(query_embeddings, class_embeddings)

        relations = F.sigmoid(relations)

        query_y_one_hot = F.one_hot(query_y, num_classes=len(classes)).float()

        loss = F.mse_loss(relations, query_y_one_hot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predictions = relations.argmax(dim=-1)
            accuracy = (predictions == query_y).float().mean().item()

        return {"loss": loss.item(), "accuracy": accuracy}

    def meta_train_step(
        self,
        tasks: List[Dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        total_loss = 0.0
        total_accuracy = 0.0

        for episode in tasks:
            support_x = episode["support_x"]
            support_y = episode["support_y"]
            query_x = episode["query_x"]
            query_y = episode["query_y"]

            classes = torch.unique(support_y)

            class_embeddings = []
            for c in classes:
                class_mask = support_y == c
                class_samples = support_x[class_mask]
                class_emb = self.encoder(class_samples)
                class_emb = class_emb.mean(dim=0, keepdim=True)
                class_embeddings.append(class_emb)

            class_embeddings = torch.cat(class_embeddings, dim=0)

            query_embeddings = self.encoder(query_x)

            relations = self.relation_module(query_embeddings, class_embeddings)
            relations = F.sigmoid(relations)

            query_y_one_hot = F.one_hot(query_y, num_classes=len(classes)).float()

            loss = F.mse_loss(relations, query_y_one_hot)
            total_loss += loss.item()

            with torch.no_grad():
                predictions = relations.argmax(dim=-1)
                accuracy = (predictions == query_y).float().mean().item()
                total_accuracy += accuracy

        avg_loss = total_loss / len(tasks)
        avg_accuracy = total_accuracy / len(tasks)

        optimizer.zero_grad()
        torch.autograd.grad(
            avg_loss,
            self.parameters(),
            retain_graph=True,
        )
        optimizer.step()

        return {"loss": avg_loss, "accuracy": avg_accuracy}


class RelationNetworkFewShot(RelationNetwork):
    def __init__(
        self,
        encoder: nn.Module,
        n_way: int,
        n_support: int,
        n_query: int,
        hidden_dim: int = 128,
    ):
        super().__init__(encoder, hidden_dim=hidden_dim)
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query


class AttentionRelationModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()

        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        class_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        Q = self.query_proj(query_embeddings)
        K = self.key_proj(class_embeddings)
        V = self.value_proj(class_embeddings)

        Q = Q.unsqueeze(1)
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

        relations, _ = self.attention(Q, K, V)

        relations = self.fc(relations.squeeze(1))

        return relations.squeeze(-1)


class GraphRelationModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.graph_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        class_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query_embeddings.size(0)
        num_classes = class_embeddings.size(0)

        query_expanded = query_embeddings.unsqueeze(1).expand(-1, num_classes, -1)
        class_expanded = class_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        node_features = torch.cat([query_expanded, class_expanded], dim=-1)

        node_features = self.node_mlp(node_features)

        for mlp in self.graph_mlps:
            node_features = mlp(node_features)

        relations = self.classifier(node_features)

        return relations.squeeze(-1)


class RelationNetworkMultiQuery(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        use_attention: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_attention = use_attention

        if use_attention:
            self.relation_module = AttentionRelationModule(
                input_dim=self._get_embedding_dim(),
                hidden_dim=hidden_dim,
            )
        else:
            self.relation_module = RelationModule(
                input_dim=self._get_embedding_dim(),
                hidden_dim=hidden_dim,
            )

    def _get_embedding_dim(self) -> int:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 84, 84)
            dummy_output = self.encoder(dummy_input)
            return dummy_output.size(-1)

    def forward(
        self,
        query_x: torch.Tensor,
        class_x: torch.Tensor,
    ) -> torch.Tensor:
        query_embeddings = self.encoder(query_x)
        class_embeddings = self.encoder(class_x)

        relations = self.relation_module(query_embeddings, class_embeddings)

        return F.sigmoid(relations)


class RelationEpisodicTrainer:
    def __init__(
        self,
        model: RelationNetwork,
        optimizer: torch.optim.Optimizer,
        n_way: int,
        n_support: int,
        n_query: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

    def create_episode(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        classes = torch.unique(y)
        selected_classes = classes[torch.randperm(len(classes))[: self.n_way]]

        support_indices = []
        query_indices = []

        for c in selected_classes:
            class_mask = y == c
            class_indices = torch.where(class_mask)[0]

            perm = torch.randperm(len(class_indices))
            support_indices.append(class_indices[perm[: self.n_support]])
            query_indices.append(
                class_indices[perm[self.n_support : self.n_support + self.n_query]]
            )

        support_indices = torch.cat(support_indices)
        query_indices = torch.cat(query_indices)

        relabel_map = {c.item(): i for i, c in enumerate(selected_classes)}

        support_x = x[support_indices]
        support_y = torch.tensor(
            [relabel_map[y[i].item()] for i in support_indices],
            dtype=torch.long,
            device=x.device,
        )

        query_x = x[query_indices]
        query_y = torch.tensor(
            [relabel_map[y[i].item()] for i in query_indices],
            dtype=torch.long,
            device=x.device,
        )

        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y,
        }

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        episode = self.create_episode(x, y)
        return self.model.episode_train_step(episode, self.optimizer)


def pairwise_relation(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    relation_module: nn.Module,
) -> torch.Tensor:
    return relation_module(embeddings1, embeddings2)


def relation_classification_loss(
    relations: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
    return F.mse_loss(relations, one_hot_labels)
