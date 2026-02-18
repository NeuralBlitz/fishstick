import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MatrixFactorization(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        user_bias: bool = True,
        item_bias: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_bias_flag = user_bias
        self.item_bias_flag = item_bias

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        if user_bias:
            self.user_bias = nn.Embedding(num_users, 1)
        if item_bias:
            self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        if self.user_bias_flag:
            nn.init.zeros_(self.user_bias.weight)
        if self.item_bias_flag:
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        dot_product = (user_emb * item_emb).sum(dim=-1)

        result = dot_product + self.global_bias

        if self.user_bias_flag:
            result = result + self.user_bias(user_ids).squeeze(-1)
        if self.item_bias_flag:
            result = result + self.item_bias(item_ids).squeeze(-1)

        return result

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_embeddings(user_ids)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embeddings(item_ids)

    def recommend(
        self,
        user_id: int,
        item_ids: Optional[torch.Tensor] = None,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embeddings(
            torch.tensor([user_id], device=self.user_embeddings.weight.device)
        )

        if item_ids is None:
            scores = (user_emb * self.item_embeddings.weight).sum(dim=-1)
            if self.item_bias_flag:
                scores = scores + self.item_bias.weight.squeeze(-1)
        else:
            item_emb = self.item_embeddings(item_ids)
            scores = (user_emb * item_emb).sum(dim=-1)
            if self.item_bias_flag:
                scores = scores + self.item_bias(item_ids).squeeze(-1)

        scores = scores + self.global_bias
        if self.user_bias_flag:
            scores = scores + self.user_bias(
                torch.tensor([user_id], device=self.user_embeddings.weight.device)
            ).squeeze(-1)

        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
        return top_scores, top_indices


class BPR(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embeddings(user_ids)
        pos_item_emb = self.item_embeddings(pos_item_ids)
        neg_item_emb = self.item_embeddings(neg_item_ids)

        pos_score = (user_emb * pos_item_emb).sum(dim=-1)
        neg_score = (user_emb * neg_item_emb).sum(dim=-1)

        bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()

        user_item_concat = torch.cat([user_emb, pos_item_emb], dim=-1)
        mlp_score = self.mlp(user_item_concat).squeeze(-1)

        return bpr_loss, mlp_score

    def get_scores(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        return (user_emb * item_emb).sum(dim=-1)


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_dim: int = 32,
        mlp_dim: int = 64,
        mlp_layers: list = [64, 32, 16],
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.mlp_dim = mlp_dim

        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)

        self.mlp_user_embedding = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_dim)

        mlp_modules = []
        input_dim = mlp_dim * 2
        for hidden_dim in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_modules)

        self.output = nn.Linear(mlp_layers[-1] + mf_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        mf_user_emb = self.mf_user_embedding(user_ids)
        mf_item_emb = self.mf_item_embedding(item_ids)
        mf_output = mf_user_emb * mf_item_emb

        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)

        concat = torch.cat([mf_output, mlp_output], dim=-1)
        score = self.output(concat).squeeze(-1)

        return score

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        user_ids = torch.full(
            (len(item_ids),), user_id, dtype=torch.long, device=item_ids.device
        )
        return self.forward(user_ids, item_ids)


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def get_adjacency_matrix(self, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = self.num_users + self.num_items
        row, col = edge_index
        row = row + self.num_users

        values = torch.ones(len(row), device=edge_index.device)
        adj = torch.sparse_coo_tensor(
            torch.stack([row, col]),
            values,
            (num_nodes, num_nodes),
        )

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        row_indices, col_indices = adj.indices()
        deg_inv_sqrt_row = deg_inv_sqrt[row_indices]
        deg_inv_sqrt_col = deg_inv_sqrt[col_indices]

        normalized_values = values * deg_inv_sqrt_row * deg_inv_sqrt_col

        adj_normalized = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            normalized_values,
            (num_nodes, num_nodes),
        )

        return adj_normalized

    def graph_propagation(
        self, adj: torch.sparse.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )

        user_embeddings_list = []
        item_embeddings_list = []

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            user_emb = all_embeddings[: self.num_users]
            item_emb = all_embeddings[self.num_users :]
            user_embeddings_list.append(user_emb)
            item_embeddings_list.append(item_emb)

        user_embeddings = torch.stack(user_embeddings_list).mean(dim=0)
        item_embeddings = torch.stack(item_embeddings_list).mean(dim=0)

        return user_embeddings, item_embeddings

    def forward(
        self,
        edge_index: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        adj = self.get_adjacency_matrix(edge_index)
        user_embeddings, item_embeddings = self.graph_propagation(adj)

        if user_ids is not None and item_ids is not None:
            user_emb = user_embeddings[user_ids]
            item_emb = item_embeddings[item_ids]
            return (user_emb * item_emb).sum(dim=-1)

        return user_embeddings, item_embeddings

    def recommend(
        self,
        user_id: int,
        edge_index: torch.Tensor,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adj = self.get_adjacency_matrix(edge_index)
        user_embeddings, item_embeddings = self.graph_propagation(adj)

        user_emb = user_embeddings[user_id : user_id + 1]
        scores = (user_emb * item_embeddings).sum(dim=-1)
        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        return top_scores, top_indices
