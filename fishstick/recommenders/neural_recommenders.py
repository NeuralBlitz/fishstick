"""
Neural Recommender Models.

Implements deep learning-based recommendation models including
NeuMF, DeepFM, Wide&Deep, and AutoRec.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from .base import InteractionMatrix


class EmbeddingLayer(nn.Module):
    """Embedding layer for user and item embeddings.

    Args:
        num_embeddings: Number of entities (users/items)
        embedding_dim: Dimension of embeddings
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, indices: Tensor) -> Tensor:
        return self.embedding(indices)


class FactorizationMachine(nn.Module):
    """Factorization Machine layer for 2nd-order feature interactions.

    Args:
        embedding_dim: Dimension of embedding vectors
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        square_of_sum = torch.sum(x, dim=1, keepdim=True) ** 2
        sum_of_square = torch.sum(x**2, dim=1, keepdim=True)
        interaction = (square_of_sum - sum_of_square) * 0.5
        return interaction.squeeze(1)


class DeepComponent(nn.Module):
    """Deep neural network component for recommendations.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        activation: Activation function
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        return x


class WideComponent(nn.Module):
    """Wide (linear) component for Wide&Deep model.

    Args:
        input_dim: Input dimension
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze(-1)


class NeuMF:
    """Neural Matrix Factorization (NeuMF).

    Combines generalized matrix factorization (GMF) and multi-layer
    perceptron (MLP) for learning user-item interactions.

    Attributes:
        n_users: Number of users
        n_items: Number of items
        gmf_dim: Dimension for GMF component
        mlp_dims: List of hidden dimensions for MLP
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 32,
        mlp_dims: List[int] = [64, 32, 16],
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.gmf_dim = gmf_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.device = device

        self.user_embedding_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, gmf_dim)

        mlp_input_dim = gmf_dim * 2
        self.mlp_layers = nn.ModuleList()
        for dim in mlp_dims:
            self.mlp_layers.append(nn.Linear(mlp_input_dim, dim))
            mlp_input_dim = dim

        self.mlp_dropout = nn.Dropout(dropout)

        self.output = nn.Linear(gmf_dim + mlp_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.item_embedding_gmf.weight)

        for layer in self.mlp_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _forward_mlp(self, user_emb: Tensor, item_emb: Tensor) -> Tensor:
        x = torch.cat([user_emb, item_emb], dim=-1)

        for layer in self.mlp_layers:
            x = F.relu(layer(x))
            x = self.mlp_dropout(x)

        return x

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        user_emb_gmf = self.user_embedding_gmf(users)
        item_emb_gmf = self.item_embedding_gmf(items)

        gmf_output = user_emb_gmf * item_emb_gmf

        mlp_output = self._forward_mlp(user_emb_gmf, item_emb_gmf)

        combined = torch.cat([gmf_output, mlp_output], dim=-1)

        output = self.output(combined)

        return output.squeeze(-1)

    def fit(
        self,
        interactions: InteractionMatrix,
        n_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train NeuMF model.

        Args:
            interactions: Training interactions
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            verbose: Print training progress

        Returns:
            Training history
        """
        coo = interactions.ratings.tocoo()
        users = torch.LongTensor(coo.row).to(self.device)
        items = torch.LongTensor(coo.col).to(self.device)
        ratings = torch.FloatTensor(coo.data).to(self.device)

        dataset = TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        history = {"loss": []}

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()

                predictions = self(batch_users, batch_items)
                loss = F.mse_loss(predictions, batch_ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating for user-item pair."""
        self.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx]).to(self.device)
            item = torch.LongTensor([item_idx]).to(self.device)
            pred = self(user, item).item()
        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
        known_items: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        self.eval()

        if exclude_known and known_items is None:
            known_items = []

        with torch.no_grad():
            users = torch.LongTensor([user_idx] * self.n_items).to(self.device)
            items = torch.LongTensor(list(range(self.n_items))).to(self.device)

            scores = self(users, items).cpu().numpy()

        predictions = [
            (i, float(scores[i]))
            for i in range(self.n_items)
            if not exclude_known or i not in known_items
        ]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class DeepFM:
    """Deep Factorization Machine (DeepFM).

    Combines FM's ability to learn 2nd-order feature interactions
    with deep neural network's ability to learn higher-order interactions.

    Attributes:
        n_users: Number of users
        n_items: Number of items
        n_fields: Number of fields (typically 2: user + item)
        embedding_dim: Dimension of embeddings
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_fields: int = 2,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.fm = FactorizationMachine(embedding_dim)

        total_dim = embedding_dim * n_fields
        self.deep = DeepComponent(total_dim, hidden_dims, dropout)

        self.output = nn.Linear(hidden_dims[-1] + 1, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        concat_emb = torch.stack([user_emb, item_emb], dim=1)

        fm_output = self.fm(concat_emb)

        deep_input = concat_emb.view(concat_emb.size(0), -1)
        deep_output = self.deep(deep_input)

        combined = torch.cat([fm_output.unsqueeze(1), deep_output], dim=1)

        output = self.output(combined)

        return output.squeeze(-1)

    def fit(
        self,
        interactions: InteractionMatrix,
        n_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train DeepFM model."""
        coo = interactions.ratings.tocoo()
        users = torch.LongTensor(coo.row).to(self.device)
        items = torch.LongTensor(coo.col).to(self.device)
        ratings = torch.FloatTensor(coo.data).to(self.device)

        dataset = TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        history = {"loss": []}

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()

                predictions = self(batch_users, batch_items)
                loss = F.mse_loss(predictions, batch_ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        self.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx]).to(self.device)
            item = torch.LongTensor([item_idx]).to(self.device)
            pred = self(user, item).item()
        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
        known_items: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        self.eval()

        if exclude_known and known_items is None:
            known_items = []

        with torch.no_grad():
            users = torch.LongTensor([user_idx] * self.n_items).to(self.device)
            items = torch.LongTensor(list(range(self.n_items))).to(self.device)

            scores = self(users, items).cpu().numpy()

        predictions = [
            (i, float(scores[i]))
            for i in range(self.n_items)
            if not exclude_known or i not in known_items
        ]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class WideAndDeep:
    """Wide & Deep Learning for Recommendations.

    Combines a wide (linear) component that memorizes feature interactions
    with a deep component that generalizes to unseen interactions.

    Attributes:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Dimension of embeddings
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.wide = WideComponent(2)

        deep_input_dim = embedding_dim * 2
        self.deep = DeepComponent(deep_input_dim, hidden_dims, dropout)

        self.output = nn.Linear(hidden_dims[-1] + 1, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        concat_emb = torch.cat([user_emb, item_emb], dim=-1)

        wide_output = self.wide(concat_emb)

        deep_output = self.deep(concat_emb)

        combined = torch.cat([wide_output.unsqueeze(1), deep_output], dim=1)

        output = self.output(combined)

        return output.squeeze(-1)

    def fit(
        self,
        interactions: InteractionMatrix,
        n_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train Wide&Deep model."""
        coo = interactions.ratings.tocoo()
        users = torch.LongTensor(coo.row).to(self.device)
        items = torch.LongTensor(coo.col).to(self.device)
        ratings = torch.FloatTensor(coo.data).to(self.device)

        dataset = TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        history = {"loss": []}

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()

                predictions = self(batch_users, batch_items)
                loss = F.mse_loss(predictions, batch_ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        self.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx]).to(self.device)
            item = torch.LongTensor([item_idx]).to(self.device)
            pred = self(user, item).item()
        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
        known_items: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        self.eval()

        if exclude_known and known_items is None:
            known_items = []

        with torch.no_grad():
            users = torch.LongTensor([user_idx] * self.n_items).to(self.device)
            items = torch.LongTensor(list(range(self.n_items))).to(self.device)

            scores = self(users, items).cpu().numpy()

        predictions = [
            (i, float(scores[i]))
            for i in range(self.n_items)
            if not exclude_known or i not in known_items
        ]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class DeepFactorizationMachine(nn.Module):
    """Deep Factorization Machine for Feature-rich CTR Prediction.

    An extension of FM that uses deep neural networks for learning
    higher-order feature interactions.

    Attributes:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Dimension of embeddings
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 16,
        hidden_dims: List[int] = [128, 64],
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.linear = nn.Linear(2, 1)

        self.fm = FactorizationMachine(embedding_dim)

        total_input = embedding_dim * 2
        self.deep = DeepComponent(total_input, hidden_dims)

        self.output = nn.Linear(hidden_dims[-1] + 1, 1)

    def forward(self, users: Tensor, items: Tensor) -> Tensor:
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        linear_input = torch.stack([user_emb, item_emb], dim=1).sum(dim=1)
        linear_output = self.linear(linear_input.unsqueeze(1)).squeeze(-1)

        concat_emb = torch.stack([user_emb, item_emb], dim=1)
        fm_output = self.fm(concat_emb)

        deep_input = concat_emb.view(concat_emb.size(0), -1)
        deep_output = self.deep(deep_input)

        combined = torch.cat([fm_output.unsqueeze(1), deep_output], dim=1)

        output = self.output(combined)

        return output.squeeze(-1)


class AutoRec:
    """AutoRec: Autoencoder-based Collaborative Filtering.

    Uses autoencoders to learn latent representations of user/item
    rating vectors for recommendations.

    Attributes:
        n_users: Number of users (for item-based AutoRec)
        n_items: Number of items
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_items: int,
        n_users: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_items = n_items
        self.n_users = n_users
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.device = device

        encoder_dims = [n_items] + hidden_dims
        decoder_dims = hidden_dims[::-1] + [n_items]

        self.encoder = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            self.encoder.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))

        self.decoder = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))

        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for layer in self.encoder:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in self.decoder:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _encode(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        return x

    def _decode(self, z: Tensor) -> Tensor:
        for i, layer in enumerate(self.decoder):
            z = layer(z)
            if i < len(self.decoder) - 1:
                z = F.relu(z)
                z = self.dropout_layer(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        z = self._encode(x)
        output = self._decode(z)
        return output

    def fit(
        self,
        interactions: InteractionMatrix,
        n_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train AutoRec model."""
        R = interactions.ratings.toarray()

        mask = R > 0
        R_masked = R.copy()
        R_masked[~mask] = 0

        R_tensor = torch.FloatTensor(R_masked).to(self.device)

        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        history = {"loss": []}

        n_samples = interactions.n_users

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples).to(self.device)

            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                batch_data = R_tensor[batch_idx]

                optimizer.zero_grad()

                output = self(batch_data)

                mask_batch = mask[batch_idx.cpu().numpy()]
                if mask_batch.sum() > 0:
                    loss = F.mse_loss(output[mask_batch], batch_data[mask_batch])
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        self.eval()

        R = (
            self.interactions.ratings.toarray()
            if hasattr(self, "interactions")
            else None
        )
        if R is None:
            raise RuntimeError("Model not fitted with interactions.")

        user_ratings = R[user_idx].copy()

        with torch.no_grad():
            input_vec = torch.FloatTensor(user_ratings).unsqueeze(0).to(self.device)
            output = self(input_vec)

        return output[0, item_idx].item()

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if not hasattr(self, "interactions"):
            raise RuntimeError("Model not fitted.")

        self.eval()

        R = self.interactions.ratings.toarray()

        with torch.no_grad():
            input_vec = torch.FloatTensor(R[user_idx]).unsqueeze(0).to(self.device)
            scores = self(input_vec).squeeze(0).cpu().numpy()

        known_items = (
            set()
            if not exclude_known
            else set(self.interactions.get_positive_items(user_idx))
        )

        predictions = [
            (i, float(scores[i])) for i in range(self.n_items) if i not in known_items
        ]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]

    def set_interactions(self, interactions: InteractionMatrix):
        """Set interactions for prediction."""
        self.interactions = interactions


class DeepNeuralCollaborativeFiltering(nn.Module):
    """Deep Neural Collaborative Filtering (DNCF).

    Uses graph neural network-inspired message passing for
    learning complex user-item interactions.

    Attributes:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Dimension of embeddings
        n_layers: Number of GCN layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.gc_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gc_layers.append(nn.Linear(embedding_dim, embedding_dim))

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(embedding_dim * (n_layers + 1), 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def _graph_conv(self, x: Tensor, adj: Tensor) -> Tensor:
        return torch.sparse.mm(adj, x)

    def forward(
        self, users: Tensor, items: Tensor, adj: Optional[Tensor] = None
    ) -> Tensor:
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        embeddings = [torch.cat([user_emb, item_emb], dim=0)]

        x = torch.cat([user_emb, item_emb], dim=0)

        for layer in self.gc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            embeddings.append(x)

        combined = torch.cat(embeddings, dim=-1)

        user_combined = combined[: len(users)]
        item_combined = combined[len(users) :]

        output = user_combined * item_combined

        return self.output(output).squeeze(-1)

    def fit(
        self,
        interactions: InteractionMatrix,
        n_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train DNCF model."""
        coo = interactions.ratings.tocoo()
        users = torch.LongTensor(coo.row).to(self.device)
        items = torch.LongTensor(coo.col).to(self.device)
        ratings = torch.FloatTensor(coo.data).to(self.device)

        dataset = TensorDataset(users, items, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.to(self.device)
        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        history = {"loss": []}

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()

                predictions = self(batch_users, batch_items)
                loss = F.mse_loss(predictions, batch_ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict rating."""
        self.eval()
        with torch.no_grad():
            user = torch.LongTensor([user_idx]).to(self.device)
            item = torch.LongTensor([item_idx]).to(self.device)
            pred = self(user, item).item()
        return pred

    def recommend(
        self,
        user_idx: int,
        n_items: int,
        exclude_known: bool = True,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        self.eval()

        with torch.no_grad():
            users = torch.LongTensor([user_idx] * self.n_items).to(self.device)
            items = torch.LongTensor(list(range(self.n_items))).to(self.device)

            scores = self(users, items).cpu().numpy()

        predictions = [(i, float(scores[i])) for i in range(self.n_items)]

        predictions.sort(key=lambda x: x[1], reverse=True)

        if exclude_known:
            known_items = set(self.interactions.get_positive_items(user_idx))
            predictions = [(i, s) for i, s in predictions if i not in known_items]

        return predictions[:n_items]

    def set_interactions(self, interactions: InteractionMatrix):
        """Set interactions for prediction."""
        self.interactions = interactions
