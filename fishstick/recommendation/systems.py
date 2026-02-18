"""
Comprehensive Recommendation Systems Module for Fishstick.

This module provides state-of-the-art recommendation algorithms including:
- Collaborative Filtering (SVD, NCF, AutoRec)
- Content-Based Filtering (TF-IDF, Deep Content)
- Deep Learning Models (WideAndDeep, DeepFM, etc.)
- Sequential Recommendations (GRU4Rec, BERT4Rec, etc.)
- Graph-Based Methods (NGCF, LightGCN, etc.)
- Multi-Task Learning (MMOE, PLE, etc.)
- Training Utilities and Deployment Tools

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Literal, Optional,
    Protocol, Sequence, Tuple, TypeVar, Union, runtime_checkable
)
import warnings
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Type definitions
T = TypeVar('T')
Tensor = torch.Tensor
SparseMatrix = sparse.csr_matrix


# =============================================================================
# Base Classes and Protocols
# =============================================================================

@runtime_checkable
class Recommender(Protocol):
    """Protocol for all recommender systems."""
    
    def fit(self, interactions: Any) -> 'Recommender':
        """Train the recommender on interaction data."""
        ...
    
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict scores for user-item pairs."""
        ...
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate recommendations for a user."""
        ...


class BaseRecommender(ABC):
    """Abstract base class for recommender systems."""
    
    def __init__(self):
        self.is_fitted = False
        self.n_users: Optional[int] = None
        self.n_items: Optional[int] = None
    
    @abstractmethod
    def fit(self, interactions: Any) -> 'BaseRecommender':
        """Train the recommender."""
        pass
    
    @abstractmethod
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict scores."""
        pass
    
    def recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making recommendations")
        
        all_items = np.arange(self.n_items)
        user_ids = np.full(len(all_items), user_id)
        scores = self.predict(user_ids, all_items)
        
        if exclude_seen and hasattr(self, '_user_seen_items'):
            seen_items = self._user_seen_items.get(user_id, set())
            mask = np.array([i not in seen_items for i in all_items])
            scores = scores[mask]
            all_items = all_items[mask]
        
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [
            (int(all_items[i]), float(scores[i]))
            for i in top_indices
        ]
        
        return recommendations


# =============================================================================
# Section 1: Collaborative Filtering
# =============================================================================

class MatrixFactorization(BaseRecommender):
    """
    Matrix Factorization using SVD or NMF.
    
    Decomposes user-item interaction matrix into latent factor matrices.
    
    Args:
        n_factors: Number of latent factors
        method: Factorization method ('svd' or 'nmf')
        max_iter: Maximum iterations for NMF
        random_state: Random seed for reproducibility
    
    Example:
        >>> mf = MatrixFactorization(n_factors=50, method='svd')
        >>> mf.fit(interactions)
        >>> recs = mf.recommend(user_id=0, n_recommendations=10)
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        method: Literal['svd', 'nmf'] = 'svd',
        max_iter: int = 500,
        random_state: int = 42
    ):
        super().__init__()
        self.n_factors = n_factors
        self.method = method
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
    
    def fit(self, interactions: Union[np.ndarray, sparse.csr_matrix]) -> 'MatrixFactorization':
        """
        Fit matrix factorization model.
        
        Args:
            interactions: User-item interaction matrix (users x items)
        
        Returns:
            self
        """
        if isinstance(interactions, np.ndarray):
            interactions = csr_matrix(interactions)
        
        self.n_users, self.n_items = interactions.shape
        self._user_seen_items = defaultdict(set)
        
        # Track seen items
        rows, cols = interactions.nonzero()
        for u, i in zip(rows, cols):
            self._user_seen_items[int(u)].add(int(i))
        
        # Compute biases
        self.global_bias = interactions.data.mean() if len(interactions.data) > 0 else 0
        
        # Convert to dense for bias computation
        dense_matrix = interactions.toarray()
        self.user_bias = np.mean(dense_matrix, axis=1) - self.global_bias
        self.item_bias = np.mean(dense_matrix, axis=0) - self.global_bias
        
        # Center the matrix
        centered = dense_matrix - self.global_bias
        centered = centered - self.user_bias[:, np.newaxis]
        centered = centered - self.item_bias[np.newaxis, :]
        centered = np.clip(centered, -10, 10)  # Prevent extreme values
        
        # Fit factorization model
        if self.method == 'svd':
            self.model = TruncatedSVD(
                n_components=self.n_factors,
                random_state=self.random_state
            )
        else:
            self.model = NMF(
                n_components=self.n_factors,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
        
        self.user_factors = self.model.fit_transform(centered)
        self.item_factors = self.model.components_.T
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """
        Predict ratings for user-item pairs.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
        
        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        scores = np.sum(
            self.user_factors[user_ids] * self.item_factors[item_ids],
            axis=1
        )
        scores += self.global_bias
        scores += self.user_bias[user_ids]
        scores += self.item_bias[item_ids]
        
        return scores


class SVDpp(BaseRecommender):
    """
    SVD++: Matrix Factorization with implicit feedback.
    
    Extends SVD by incorporating implicit feedback signals from
    items users have interacted with, even without explicit ratings.
    
    Args:
        n_factors: Number of latent factors
        n_epochs: Training epochs
        lr: Learning rate
        reg: Regularization parameter
        implicit_weight: Weight for implicit feedback
        random_state: Random seed
    
    Example:
        >>> svdpp = SVDpp(n_factors=50, n_epochs=20)
        >>> svdpp.fit(ratings_matrix)
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        implicit_weight: float = 0.1,
        random_state: int = 42
    ):
        super().__init__()
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.implicit_weight = implicit_weight
        self.random_state = random_state
        
        self.user_factors = None
        self.item_factors = None
        self.user_implicit_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        self.implicit_items = None
    
    def fit(self, interactions: Union[np.ndarray, sparse.csr_matrix]) -> 'SVDpp':
        """Fit SVD++ model."""
        np.random.seed(self.random_state)
        
        if isinstance(interactions, np.ndarray):
            interactions = csr_matrix(interactions)
        
        self.n_users, self.n_items = interactions.shape
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.01, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.01, (self.n_items, self.n_factors))
        self.user_implicit_factors = np.random.normal(0, 0.01, (self.n_users, self.n_factors))
        
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = interactions.data.mean() if len(interactions.data) > 0 else 0
        
        # Store implicit items for each user
        self.implicit_items = defaultdict(list)
        rows, cols = interactions.nonzero()
        for u, i in zip(rows, cols):
            self.implicit_items[u].append(i)
        
        # Training
        for epoch in range(self.n_epochs):
            for u in range(self.n_users):
                items = self.implicit_items[u]
                if len(items) == 0:
                    continue
                
                # Compute implicit feedback vector
                implicit_sum = np.sum(
                    self.item_factors[items],
                    axis=0
                ) / np.sqrt(len(items))
                
                for i in items:
                    r_ui = 1.0  # Implicit feedback treated as positive
                    
                    # Prediction with implicit feedback
                    pred = (
                        self.global_bias +
                        self.user_bias[u] +
                        self.item_bias[i] +
                        np.dot(
                            self.user_factors[u] + implicit_sum * self.implicit_weight,
                            self.item_factors[i]
                        )
                    )
                    
                    err = r_ui - pred
                    
                    # Update biases
                    self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                    self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])
                    
                    # Update factors
                    self.user_factors[u] += self.lr * (
                        err * self.item_factors[i] - self.reg * self.user_factors[u]
                    )
                    self.item_factors[i] += self.lr * (
                        err * (self.user_factors[u] + implicit_sum * self.implicit_weight) -
                        self.reg * self.item_factors[i]
                    )
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict ratings."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        scores = np.zeros(len(user_ids))
        
        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            items = self.implicit_items.get(u, [])
            if len(items) > 0:
                implicit_sum = np.sum(self.item_factors[items], axis=0) / np.sqrt(len(items))
            else:
                implicit_sum = np.zeros(self.n_factors)
            
            scores[idx] = (
                self.global_bias +
                self.user_bias[u] +
                self.item_bias[i] +
                np.dot(
                    self.user_factors[u] + implicit_sum * self.implicit_weight,
                    self.item_factors[i]
                )
            )
        
        return scores


class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF) for Neural Collaborative Filtering.
    
    Learns user-item interactions through element-wise product of embeddings.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_factors: Embedding dimension
    
    Example:
        >>> gmf = GMF(n_users=1000, n_items=500, n_factors=64)
        >>> scores = gmf(user_ids, item_ids)
    """
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.output = nn.Linear(n_factors, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()
    
    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: User indices [batch_size]
            item_ids: Item indices [batch_size]
        
        Returns:
            Predicted scores [batch_size, 1]
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Element-wise product (generalized interaction)
        interaction = user_emb * item_emb
        
        output = self.output(interaction)
        return torch.sigmoid(output)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Neural Collaborative Filtering.
    
    Uses concatenated embeddings fed through multiple dense layers.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_factors: Embedding dimension
        layers: List of layer dimensions (default: [64, 32, 16, 8])
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 64,
        layers: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        if layers is None:
            layers = [64, 32, 16, 8]
        self.layers = layers
        
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # Build MLP layers
        mlp_layers = []
        input_dim = n_factors * 2
        
        for layer_dim in layers:
            mlp_layers.append(nn.Linear(input_dim, layer_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.output = nn.Linear(layers[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()
    
    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Forward pass."""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        vector = torch.cat([user_emb, item_emb], dim=-1)
        
        # MLP layers
        hidden = self.mlp(vector)
        output = self.output(hidden)
        
        return torch.sigmoid(output)


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF).
    
    Combines GMF and MLP by concatenating their outputs.
    GMF captures linear interactions, MLP captures non-linear patterns.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_factors: GMF embedding dimension
        layers: MLP layer dimensions
        dropout: Dropout probability
    
    Example:
        >>> neumf = NeuMF(n_users=1000, n_items=500)
        >>> scores = neumf(user_ids, item_ids)
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 64,
        layers: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        if layers is None:
            layers = [64, 32, 16, 8]
        self.layers = layers
        
        # GMF component
        self.gmf_user_embedding = nn.Embedding(n_users, n_factors)
        self.gmf_item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP component
        self.mlp_user_embedding = nn.Embedding(n_users, layers[0] // 2)
        self.mlp_item_embedding = nn.Embedding(n_items, layers[0] // 2)
        
        # MLP layers
        mlp_layers = []
        input_dim = layers[0]
        for layer_dim in layers[1:]:
            mlp_layers.append(nn.Linear(input_dim, layer_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        final_dim = n_factors + layers[-1]
        self.output = nn.Linear(final_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()
    
    def forward(self, user_ids: Tensor, item_ids: Tensor) -> Tensor:
        """Forward pass combining GMF and MLP."""
        # GMF path
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_vector = gmf_user * gmf_item
        
        # MLP path
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_vector = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_vector = self.mlp(mlp_vector)
        
        # Concatenate both paths
        concat_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        output = self.output(concat_vector)
        
        return torch.sigmoid(output)


class NCF(BaseRecommender):
    """
    Neural Collaborative Filtering (NCF) wrapper.
    
    Wrapper class that combines training loop for NCF models.
    
    Args:
        model_type: Type of model ('gmf', 'mlp', 'neumf')
        n_factors: Embedding dimension
        layers: MLP layer dimensions
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Computation device
    """
    
    def __init__(
        self,
        model_type: Literal['gmf', 'mlp', 'neumf'] = 'neumf',
        n_factors: int = 64,
        layers: Optional[List[int]] = None,
        n_epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
        device: str = 'cpu'
    ):
        super().__init__()
        self.model_type = model_type
        self.n_factors = n_factors
        self.layers = layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
    
    def fit(
        self,
        interactions: Union[np.ndarray, sparse.csr_matrix],
        val_split: float = 0.1
    ) -> 'NCF':
        """Fit NCF model."""
        if isinstance(interactions, np.ndarray):
            interactions = csr_matrix(interactions)
        
        self.n_users, self.n_items = interactions.shape
        
        # Create model
        if self.model_type == 'gmf':
            self.model = GMF(self.n_users, self.n_items, self.n_factors)
        elif self.model_type == 'mlp':
            self.model = MLP(self.n_users, self.n_items, self.n_factors, self.layers)
        else:
            self.model = NeuMF(self.n_users, self.n_items, self.n_factors, self.layers)
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Prepare training data
        train_data = self._prepare_data(interactions)
        
        # Training loop
        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in self._get_batches(train_data):
                users, items, labels = batch
                users = users.to(self.device)
                items = items.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(users, items)
                loss = self.criterion(predictions.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {total_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _prepare_data(
        self,
        interactions: sparse.csr_matrix
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare training data with negative sampling."""
        users, items, labels = [], [], []
        
        rows, cols = interactions.nonzero()
        for u, i in zip(rows, cols):
            users.append(u)
            items.append(i)
            labels.append(1.0)
            
            # Add negative samples
            neg_item = np.random.randint(0, self.n_items)
            while interactions[u, neg_item] > 0:
                neg_item = np.random.randint(0, self.n_items)
            users.append(u)
            items.append(neg_item)
            labels.append(0.0)
        
        return (
            torch.LongTensor(users),
            torch.LongTensor(items),
            torch.FloatTensor(labels)
        )
    
    def _get_batches(
        self,
        data: Tuple[Tensor, Tensor, Tensor]
    ) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        """Generate batches."""
        users, items, labels = data
        n_samples = len(users)
        indices = torch.randperm(n_samples)
        
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_idx = indices[start:end]
            yield users[batch_idx], items[batch_idx], labels[batch_idx]
    
    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """Predict ratings."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        self.model.eval()
        with torch.no_grad():
            users = torch.LongTensor(user_ids).to(self.device)
            items = torch.LongTensor(item_ids).to(self.device)
            predictions = self.model(users, items)
        
        return predictions.cpu().numpy().squeeze()


class AutoRec(nn.Module):
    """
    Autoencoder Recommender.
    
    Uses autoencoder to reconstruct user-item interaction vectors.
    Can be user-based or item-based.
    
    Args:
        n_input: Input dimension (n_items for user-based, n_users for item-based)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        mode: 'user' or 'item' based autoencoder
    
    Example:
        >>> autorec = AutoRec(n_input=1000, hidden_dims=[500, 250])
        >>> reconstructed = autorec(ratings_vector)
    """
    
    def __init__(
        self,
        n_input: int,
        hidden_dims: List[int],
        dropout: float = 0.5,
        mode: Literal['user', 'item'] = 'user'
    ):
        super().__init__()
        self.n_input = n_input
        self.hidden_dims = hidden_dims
        self.mode = mode
        
        # Encoder
        encoder_layers = []
        input_dim = n_input
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.Sigmoid())
            encoder_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i-1]))
            decoder_layers.append(nn.Sigmoid())
            decoder_layers.append(nn.Dropout(dropout))
        decoder_layers.append(nn.Linear(hidden_dims[0], n_input))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input vector [batch_size, n_input]
            mask: Mask for observed ratings [batch_size, n_input]
        
        Returns:
            Reconstructed vector [batch_size, n_input]
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Apply mask to only compute loss on observed entries
        if mask is not None:
            decoded = decoded * mask
        
        return decoded
