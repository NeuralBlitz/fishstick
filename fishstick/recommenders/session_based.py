"""
Session-Based Recommendation Models.

Implements sequential and session-based recommendation models including
GRU4Rec and SASRec for predicting the next item in a sequence.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import math

from .base import InteractionMatrix


class SessionDataset(torch.utils.data.Dataset):
    """Dataset for session-based recommendations.

    Args:
        sessions: List of sessions (each session is a list of item IDs)
        n_items: Total number of items
        max_seq_len: Maximum sequence length
    """

    def __init__(self, sessions: List[List[int]], n_items: int, max_seq_len: int = 50):
        self.sessions = sessions
        self.n_items = n_items
        self.max_seq_len = max_seq_len

        self.data = []
        for session in sessions:
            if len(session) < 2:
                continue

            for i in range(1, len(session)):
                input_seq = session[:i]
                target = session[i]

                self.data.append((input_seq, target))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        input_seq, target = self.data[idx]

        seq_len = min(len(input_seq), self.max_seq_len)

        padded_seq = [0] * (self.max_seq_len - seq_len) + input_seq[-seq_len:]

        return (torch.LongTensor(padded_seq), torch.LongTensor([target]).squeeze(0))


class SessionEncoder(nn.Module):
    """Base session encoder.

    Args:
        n_items: Number of items
        embedding_dim: Dimension of item embeddings
    """

    def __init__(self, n_items: int, embedding_dim: int):
        super().__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

    def forward(self, sequences: Tensor) -> Tensor:
        raise NotImplementedError


class GRU4Rec(nn.Module):
    """GRU4Rec: Session-based Recommendations with GRU.

    Uses GRU (Gated Recurrent Unit) to model user session sequences
    for next-item prediction.

    Attributes:
        n_items: Number of items
        embedding_dim: Dimension of item embeddings
        hidden_dim: Dimension of hidden state
        num_layers: Number of GRU layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 100,
        hidden_dim: int = 100,
        num_layers: int = 1,
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(hidden_dim, n_items)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequences: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        embedded = self.embedding(sequences)

        embedded = self.dropout(embedded)

        if hidden is None:
            batch_size = sequences.size(0)
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim, device=sequences.device
            )

        output, hidden = self.gru(embedded, hidden)

        output = output[:, -1, :]

        output = self.dropout(output)

        logits = self.output(output)

        return logits

    def predict(self, sequence: List[int]) -> np.ndarray:
        """Predict next item scores for a sequence.

        Args:
            sequence: List of item IDs in the session

        Returns:
            Array of scores for each item
        """
        self.eval()

        seq_len = min(len(sequence), 50)
        padded_seq = [0] * (50 - seq_len) + sequence[-seq_len:]

        with torch.no_grad():
            seq_tensor = torch.LongTensor([padded_seq]).to(self.device)
            logits = self.forward(seq_tensor)
            scores = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return scores

    def recommend(
        self,
        sequence: List[int],
        n_items: int,
        exclude_known: bool = False,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a session.

        Args:
            sequence: Session sequence of item IDs
            n_items: Number of items to recommend
            exclude_known: Whether to exclude items in sequence

        Returns:
            List of (item_idx, score) tuples
        """
        scores = self.predict(sequence)

        if exclude_known:
            known = set(sequence)
            predictions = [
                (i, float(scores[i])) for i in range(self.n_items) if i not in known
            ]
        else:
            predictions = [(i, float(scores[i])) for i in range(self.n_items)]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class GRU4RecTrainer:
    """Trainer for GRU4Rec model.

    Args:
        model: GRU4Rec model
        device: Device to train on
    """

    def __init__(self, model: GRU4Rec, device: str = "cuda"):
        self.model = model
        self.device = device

    def fit(
        self,
        sessions: List[List[int]],
        n_items: int,
        n_epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        max_seq_len: int = 50,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train GRU4Rec model.

        Args:
            sessions: List of user sessions
            n_items: Number of items
            n_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_seq_len: Maximum sequence length
            verbose: Print training progress

        Returns:
            Training history
        """
        dataset = SessionDataset(sessions, n_items, max_seq_len)

        if len(dataset) == 0:
            raise ValueError("No valid training samples in sessions")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        history = {"loss": []}

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                logits = self.model(sequences)
                loss = F.cross_entropy(logits, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history


class SelfAttentiveSequential(nn.Module):
    """SASRec: Self-Attentive Sequential Recommendation.

    Uses self-attention to capture sequential patterns in user behavior.

    Attributes:
        n_items: Number of items
        embedding_dim: Dimension of item embeddings
        hidden_dim: Dimension of hidden layer
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)

        self.positional_embedding = nn.Embedding(200, embedding_dim)

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embedding_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embedding_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embedding_dim) for _ in range(num_layers * 2)]
        )

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(embedding_dim, n_items)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequences: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len = sequences.size()

        embedded = self.embedding(sequences)

        positions = (
            torch.arange(seq_len, device=sequences.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_embedded = self.positional_embedding(positions)

        x = embedded + pos_embedded
        x = self.dropout(x)

        key_padding_mask = sequences == 0

        for i in range(self.num_layers):
            attn_layer = self.attention_layers[i]
            ffn_layer = self.ffn_layers[i]

            attn_out, _ = attn_layer(x, x, x, key_padding_mask=key_padding_mask)
            x = self.layer_norms[i * 2](x + attn_out)

            ffn_out = ffn_layer(x)
            x = self.layer_norms[i * 2 + 1](x + ffn_out)

        output = x[:, -1, :]

        logits = self.output(output)

        return logits

    def predict(self, sequence: List[int]) -> np.ndarray:
        """Predict next item scores."""
        self.eval()

        seq_len = min(len(sequence), 200)
        padded_seq = [0] * (200 - seq_len) + sequence[-seq_len:]

        with torch.no_grad():
            seq_tensor = torch.LongTensor([padded_seq]).to(self.device)
            logits = self.forward(seq_tensor)
            scores = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return scores

    def recommend(
        self,
        sequence: List[int],
        n_items: int,
        exclude_known: bool = False,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        scores = self.predict(sequence)

        if exclude_known:
            known = set(sequence)
            predictions = [
                (i, float(scores[i])) for i in range(self.n_items) if i not in known
            ]
        else:
            predictions = [(i, float(scores[i])) for i in range(self.n_items)]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class SASRec:
    """Wrapper for SASRec model with training functionality.

    Attributes:
        n_items: Number of items
        embedding_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden layer
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_items = n_items
        self.device = device

        self.model = SelfAttentiveSequential(
            n_items=n_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            device=device,
        )

    def fit(
        self,
        sessions: List[List[int]],
        n_epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        max_seq_len: int = 200,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train SASRec model."""
        dataset = SessionDataset(sessions, self.n_items, max_seq_len)

        if len(dataset) == 0:
            raise ValueError("No valid training samples in sessions")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        history = {"loss": []}

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                logits = self.model(sequences)
                loss = F.cross_entropy(logits, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        return history

    def recommend(
        self,
        sequence: List[int],
        n_items: int,
        exclude_known: bool = False,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        return self.model.recommend(sequence, n_items, exclude_known)


class NextItemPredictor:
    """General next item prediction framework.

    A generic wrapper that can use any underlying model for
    session-based recommendations.

    Attributes:
        model: Underlying prediction model
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def predict(self, sequence: List[int]) -> np.ndarray:
        """Predict next item scores."""
        if hasattr(self.model, "predict"):
            return self.model.predict(sequence)

        self.model.eval()

        max_len = 50
        seq_len = min(len(sequence), max_len)
        padded_seq = [0] * (max_len - seq_len) + sequence[-seq_len:]

        with torch.no_grad():
            seq_tensor = torch.LongTensor([padded_seq]).to(self.device)

            if hasattr(self.model, "forward"):
                logits = self.model(seq_tensor)
            else:
                logits = seq_tensor.float()

            if logits.dim() > 1:
                scores = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            else:
                scores = logits.cpu().numpy()

        return scores

    def recommend(
        self,
        sequence: List[int],
        n_items: int,
        exclude_known: bool = False,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        scores = self.predict(sequence)

        n_output_items = scores.shape[0] if scores.ndim > 0 else 0

        if exclude_known:
            known = set(sequence)
            predictions = [
                (i, float(scores[i])) for i in range(n_output_items) if i not in known
            ]
        else:
            predictions = [(i, float(scores[i])) for i in range(n_output_items)]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


class SessionGRU:
    """Session-GRU with optional memory.

    GRU-based session model with optional memory augmentation
    for long-term user preferences.

    Attributes:
        n_items: Number of items
        embedding_dim: Dimension of embeddings
        hidden_dim: Hidden state dimension
        memory_size: Size of memory buffer
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 100,
        hidden_dim: int = 100,
        memory_size: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.device = device

        self.embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.memory_key = nn.Linear(hidden_dim, embedding_dim)
        self.memory_value = nn.Linear(hidden_dim, embedding_dim)

        self.output = nn.Linear(hidden_dim + embedding_dim, n_items)

        self._init_weights()

        self.user_memory: Dict[int, List[int]] = {}

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, sequences: Tensor) -> Tensor:
        embedded = self.embedding(sequences)

        output, hidden = self.gru(embedded)

        final_hidden = hidden[-1]

        memory_emb = torch.zeros(
            sequences.size(0), self.embedding_dim, device=sequences.device
        )

        return self.output(final_hidden)

    def predict(self, sequence: List[int]) -> np.ndarray:
        """Predict next item scores."""
        self.eval()

        max_len = 50
        seq_len = min(len(sequence), max_len)
        padded_seq = [0] * (max_len - seq_len) + sequence[-seq_len:]

        with torch.no_grad():
            seq_tensor = torch.LongTensor([padded_seq]).to(self.device)
            logits = self.forward(seq_tensor)
            scores = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return scores

    def recommend(
        self,
        sequence: List[int],
        n_items: int,
        exclude_known: bool = False,
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations."""
        scores = self.predict(sequence)

        if exclude_known:
            known = set(sequence)
            predictions = [
                (i, float(scores[i])) for i in range(self.n_items) if i not in known
            ]
        else:
            predictions = [(i, float(scores[i])) for i in range(self.n_items)]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n_items]


def sessions_from_interactions(
    interactions: InteractionMatrix,
    min_session_length: int = 2,
    max_session_length: int = 50,
) -> List[List[int]]:
    """Extract sessions from interaction matrix.

    Simple heuristic: group consecutive interactions within
    a time window or by user into sessions.

    Args:
        interactions: User-item interaction matrix
        min_session_length: Minimum session length
        max_session_length: Maximum session length

    Returns:
        List of sessions (each session is a list of item IDs)
    """
    sessions = []

    for user_idx in range(interactions.n_users):
        row = interactions.ratings.getrow(user_idx)

        items = row.indices

        if len(items) >= min_session_length:
            session = items[:max_session_length].tolist()
            sessions.append(session)

    return sessions


def create_session_data(
    interactions: InteractionMatrix,
    n_train_sessions: int = 1000,
    session_length: int = 5,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Create training and test session data.

    Args:
        interactions: Interaction matrix
        n_train_sessions: Number of training sessions to generate
        session_length: Length of each session

    Returns:
        Tuple of (train_sessions, test_sessions)
    """
    sessions = sessions_from_interactions(interactions)

    if len(sessions) == 0:
        raise ValueError("No valid sessions found in interactions")

    n_train = min(n_train_sessions, len(sessions) - len(sessions) // 5)

    train_sessions = sessions[:n_train]
    test_sessions = sessions[n_train:]

    return train_sessions, test_sessions
