"""
Content-Based Memory Addressing.

Implements various content-addressing mechanisms:
- Cosine similarity addressing
- Euclidean distance addressing
- Attention-based addressing
- Multi-head content addressing
- Learned similarity metrics

Based on:
- Neural Turing Machines (Graves et al., 2014)
- Differentiable Neural Computers (Graves et al., 2016)
- Memory Networks (Weston et al., 2014)
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AddressingResult:
    """Result of a memory addressing operation."""

    weights: Tensor
    read_vector: Tensor
    similarity: Optional[Tensor] = None
    debug_info: Dict[str, Any] = None


class CosineSimilarityAddressing(nn.Module):
    """
    Content addressing using cosine similarity.

    Similarity = (k · M) / (||k|| * ||M||)
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.temperature = temperature

        self.key_projection = nn.Linear(item_size, item_size)
        nn.init.xavier_uniform_(self.key_projection.weight)

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Cosine similarity content addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Query key [batch, item_size]
            beta: Sharpness parameter [batch]

        Returns:
            AddressingResult with weights and read vector
        """
        batch_size = memory.size(0)

        projected_key = torch.tanh(self.key_projection(key))

        key_norm = projected_key / (projected_key.norm(dim=-1, keepdim=True) + 1e-8)
        mem_norm = memory / (memory.norm(dim=-1, keepdim=True) + 1e-8)

        similarity = torch.matmul(key_norm, mem_norm.transpose(-2, -1))

        if beta is None:
            beta = torch.ones(batch_size, device=memory.device)

        weights = F.softmax(beta.unsqueeze(-1) * similarity / self.temperature, dim=-1)

        read_vector = torch.einsum("bm,bmi->bi", weights, memory)

        return AddressingResult(
            weights=weights,
            read_vector=read_vector,
            similarity=similarity,
            debug_info={
                "projected_key": projected_key,
                "key_norm": key_norm,
                "mem_norm": mem_norm,
            },
        )


class EuclideanDistanceAddressing(nn.Module):
    """
    Content addressing using negative Euclidean distance.

    Similarity = -||key - memory||^2
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.temperature = temperature

        self.key_projection = nn.Linear(item_size, item_size)
        nn.init.xavier_uniform_(self.key_projection.weight)

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Euclidean distance content addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Query key [batch, item_size]
            beta: Sharpness parameter [batch]

        Returns:
            AddressingResult with weights and read vector
        """
        batch_size = memory.size(0)

        projected_key = torch.tanh(self.key_projection(key))

        key_expanded = projected_key.unsqueeze(1).expand(-1, self.memory_size, -1)

        diff = key_expanded - memory
        distances = (diff**2).sum(dim=-1)

        similarity = -distances

        if beta is None:
            beta = torch.ones(batch_size, device=memory.device)

        weights = F.softmax(beta.unsqueeze(-1) * similarity / self.temperature, dim=-1)

        read_vector = torch.einsum("bm,bmi->bi", weights, memory)

        return AddressingResult(
            weights=weights,
            read_vector=read_vector,
            similarity=-distances,
            debug_info={
                "projected_key": projected_key,
                "distances": distances,
            },
        )


class DotProductAddressing(nn.Module):
    """
    Simple dot product content addressing.

    Similarity = key · memory
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        temperature: float = 1.0,
        use_key_projection: bool = True,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.temperature = temperature

        if use_key_projection:
            self.key_projection = nn.Linear(item_size, item_size)
            nn.init.xavier_uniform_(self.key_projection.weight)
        else:
            self.key_projection = None

        self.scale = item_size**-0.5

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Dot product content addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Query key [batch, item_size]
            beta: Sharpness parameter [batch]

        Returns:
            AddressingResult with weights and read vector
        """
        batch_size = memory.size(0)

        if self.key_projection is not None:
            projected_key = torch.tanh(self.key_projection(key))
        else:
            projected_key = key

        similarity = torch.matmul(projected_key, memory.transpose(-2, -1))

        similarity = similarity * self.scale

        if beta is None:
            beta = torch.ones(batch_size, device=memory.device)

        weights = F.softmax(beta.unsqueeze(-1) * similarity / self.temperature, dim=-1)

        read_vector = torch.einsum("bm,bmi->bi", weights, memory)

        return AddressingResult(
            weights=weights,
            read_vector=read_vector,
            similarity=similarity,
            debug_info={"projected_key": projected_key},
        )


class LearnedSimilarityAddressing(nn.Module):
    """
    Content addressing with learned similarity metric.

    Uses a neural network to compute similarity between keys and memory.
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.temperature = temperature

        self.similarity_net = nn.Sequential(
            nn.Linear(item_size * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Learned similarity content addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Query key [batch, item_size]
            beta: Sharpness parameter [batch]

        Returns:
            AddressingResult with weights and read vector
        """
        batch_size = memory.size(0)

        key_expanded = key.unsqueeze(1).expand(-1, self.memory_size, -1)

        combined = torch.cat([key_expanded, memory], dim=-1)

        similarity = self.similarity_net(combined).squeeze(-1)

        if beta is None:
            beta = torch.ones(batch_size, device=memory.device)

        weights = F.softmax(beta.unsqueeze(-1) * similarity / self.temperature, dim=-1)

        read_vector = torch.einsum("bm,bmi->bi", weights, memory)

        return AddressingResult(
            weights=weights,
            read_vector=read_vector,
            similarity=similarity,
            debug_info={"similarity_raw": similarity},
        )


class MultiHeadContentAddressing(nn.Module):
    """
    Multi-head content addressing for parallel memory lookup.

    Multiple attention heads attending to different aspects of memory.
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        num_heads: int = 4,
        temperature: float = 1.0,
        addressing_type: str = "cosine",
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.num_heads = num_heads
        self.temperature = temperature

        self.head_dim = item_size // num_heads

        self.key_projection = nn.Linear(item_size, item_size)

        for i in range(num_heads):
            if addressing_type == "cosine":
                addressing = CosineSimilarityAddressing(
                    memory_size, self.head_dim, temperature
                )
            elif addressing_type == "euclidean":
                addressing = EuclideanDistanceAddressing(
                    memory_size, self.head_dim, temperature
                )
            elif addressing_type == "dot":
                addressing = DotProductAddressing(
                    memory_size, self.head_dim, temperature
                )
            else:
                addressing = LearnedSimilarityAddressing(
                    memory_size, self.head_dim, temperature=temperature
                )

            setattr(self, f"head_{i}", addressing)

        self.output_projection = nn.Linear(item_size, item_size)

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Multi-head content addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Query key [batch, item_size]
            beta: Sharpness parameter [batch] or [batch, num_heads]

        Returns:
            AddressingResult with combined weights and read vector
        """
        batch_size = memory.size(0)

        projected_key = torch.tanh(self.key_projection(key))
        projected_key = projected_key.view(batch_size, self.num_heads, self.head_dim)

        memory_chunks = memory.view(
            batch_size, self.memory_size, self.num_heads, self.head_dim
        )

        head_reads = []
        head_weights = []

        for i in range(self.num_heads):
            head_key = projected_key[:, i, :]
            head_mem = memory_chunks[:, :, i, :]

            if beta is not None and beta.dim() > 1:
                head_beta = beta[:, i]
            else:
                head_beta = beta

            head = getattr(self, f"head_{i}")
            result = head(head_mem, head_key, head_beta)

            head_reads.append(result.read_vector)
            head_weights.append(result.weights)

        combined_read = torch.cat(head_reads, dim=-1)
        output = self.output_projection(combined_read)

        combined_weights = torch.stack(head_weights, dim=1)

        return AddressingResult(
            weights=combined_weights,
            read_vector=output,
            debug_info={
                "head_weights": head_weights,
                "head_reads": head_reads,
            },
        )


class HybridAddressing(nn.Module):
    """
    Hybrid addressing combining multiple addressing schemes.

    Interpolates between content-based and location-based addressing.
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.temperature = temperature

        self.content_addressing = CosineSimilarityAddressing(
            memory_size, item_size, temperature
        )

        self.interpolation_gate = nn.Linear(item_size, 1)
        self.location_shift = nn.Linear(item_size, 3)
        self.location_gamma = nn.Linear(item_size, 1)

    def forward(
        self,
        memory: Tensor,
        key: Tensor,
        prev_weights: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Hybrid content + location addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            key: Query key [batch, item_size]
            prev_weights: Previous attention weights [batch, memory_size]
            beta: Sharpness parameter [batch]

        Returns:
            AddressingResult with combined weights
        """
        content_result = self.content_addressing(memory, key, beta)

        gate = torch.sigmoid(self.interpolation_gate(key))

        if prev_weights is not None:
            shift = F.softmax(self.location_shift(key), dim=-1)
            gamma = F.softplus(self.location_gamma(key)) + 1e-3

            location_weights = self._shift_weights(prev_weights, shift)
            location_weights = location_weights**gamma
            location_weights = location_weights / location_weights.sum(
                dim=-1, keepdim=True
            )
        else:
            location_weights = content_result.weights

        weights = gate * content_result.weights + (1 - gate) * location_weights

        read_vector = torch.einsum("bm,bmi->bi", weights, memory)

        return AddressingResult(
            weights=weights,
            read_vector=read_vector,
            similarity=content_result.similarity,
            debug_info={
                "content_weights": content_result.weights,
                "location_weights": location_weights,
                "gate": gate,
            },
        )

    def _shift_weights(self, weights: Tensor, shift: Tensor) -> Tensor:
        """Apply circular shift to attention weights."""
        shifted = torch.zeros_like(weights)

        shifted[:, :-2] = weights[:, 2:]
        shifted[:, -2:] = 0

        return shifted


class AttentionBasedAddressing(nn.Module):
    """
    Standard attention mechanism for memory access.

    Implements query-key-value attention for memory lookup.
    """

    def __init__(
        self,
        memory_size: int,
        item_size: int,
        num_heads: int = 4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.num_heads = num_heads
        self.temperature = temperature

        self.head_dim = item_size // num_heads

        self.q_proj = nn.Linear(item_size, item_size)
        self.k_proj = nn.Linear(item_size, item_size)
        self.v_proj = nn.Linear(item_size, item_size)

        self.out_proj = nn.Linear(item_size, item_size)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        memory: Tensor,
        query: Tensor,
        beta: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Attention-based memory addressing.

        Args:
            memory: Memory matrix [batch, memory_size, item_size]
            query: Query [batch, item_size]
            beta: Sharpness parameter (overrides scale if provided)

        Returns:
            AddressingResult with weights and read vector
        """
        batch_size = memory.size(0)

        q = self.q_proj(query)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.memory_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.memory_size, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.scale
        if beta is not None:
            scale = beta.unsqueeze(-1).unsqueeze(-1) * scale

        energy = torch.matmul(q, k.transpose(-2, -1)) * scale

        weights = F.softmax(energy, dim=-1)

        attended = torch.matmul(weights, v)

        output = attended.transpose(1, 2).contiguous().view(batch_size, -1)
        output = self.out_proj(output)

        return AddressingResult(
            weights=weights, read_vector=output, debug_info={"energy": energy}
        )


class MemoryBank(nn.Module):
    """
    Complete memory bank with multiple addressing mechanisms.

    Supports read/write operations with various addressing schemes.
    """

    def __init__(
        self,
        memory_size: int = 128,
        item_size: int = 64,
        addressing_type: str = "cosine",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.item_size = item_size
        self.addressing_type = addressing_type
        self.temperature = temperature

        self.memory = nn.Parameter(torch.randn(memory_size, item_size) * 0.01)

        if addressing_type == "cosine":
            self.addressing = CosineSimilarityAddressing(
                memory_size, item_size, temperature
            )
        elif addressing_type == "euclidean":
            self.addressing = EuclideanDistanceAddressing(
                memory_size, item_size, temperature
            )
        elif addressing_type == "dot":
            self.addressing = DotProductAddressing(memory_size, item_size, temperature)
        elif addressing_type == "attention":
            self.addressing = AttentionBasedAddressing(
                memory_size, item_size, temperature=temperature
            )
        elif addressing_type == "hybrid":
            self.addressing = HybridAddressing(memory_size, item_size, temperature)
        else:
            self.addressing = LearnedSimilarityAddressing(
                memory_size, item_size, temperature=temperature
            )

        self.write_key_net = nn.Linear(item_size, item_size)
        self.write_value_net = nn.Linear(item_size, item_size)
        self.erase_net = nn.Linear(item_size, item_size)

    def read(
        self,
        key: Tensor,
        beta: Optional[Tensor] = None,
        prev_weights: Optional[Tensor] = None,
    ) -> AddressingResult:
        """
        Read from memory.

        Args:
            key: Query key [batch, item_size]
            beta: Sharpness parameter [batch]
            prev_weights: Previous weights for location addressing

        Returns:
            AddressingResult
        """
        memory = self.memory.unsqueeze(0).expand(key.size(0), -1, -1)

        if isinstance(self.addressing, HybridAddressing):
            result = self.addressing(memory, key, prev_weights, beta)
        else:
            result = self.addressing(memory, key, beta)

        return result

    def write(
        self,
        key: Tensor,
        value: Tensor,
        weights: Optional[Tensor] = None,
    ) -> None:
        """
        Write to memory.

        Args:
            key: Content key [batch, item_size]
            value: Value to write [batch, item_size]
            weights: Attention weights for write location
        """
        batch_size = key.size(0)

        if weights is None:
            read_result = self.read(key)
            weights = read_result.weights

        write_key = torch.tanh(self.write_key_net(key))
        write_value = torch.tanh(self.write_value_net(value))
        erase = torch.sigmoid(self.erase_net(key))

        weights = weights.unsqueeze(-1)

        erase_matrix = weights * erase.unsqueeze(1)
        add_matrix = weights * write_value.unsqueeze(1)

        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        new_memory = memory * (1 - erase_matrix) + add_matrix

        self.memory.data = new_memory.mean(dim=0).detach()

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset memory to initial state."""
        self.memory.data = torch.randn_like(self.memory) * 0.01


class AssociativeMemory(nn.Module):
    """
    Associative memory for key-value pair storage and retrieval.

    Uses content addressing to store and retrieve arbitrary associations.
    """

    def __init__(
        self,
        capacity: int = 128,
        key_size: int = 64,
        value_size: int = 64,
    ):
        super().__init__()
        self.capacity = capacity
        self.key_size = key_size
        self.value_size = value_size

        self.keys = nn.Parameter(torch.randn(capacity, key_size) * 0.01)
        self.values = nn.Parameter(torch.randn(capacity, value_size) * 0.01)

        self.usage = nn.Parameter(torch.zeros(capacity))

        self.key_proj = nn.Linear(key_size, key_size)
        self.value_proj = nn.Linear(value_size, value_size)

    def store(self, keys: Tensor, values: Tensor) -> None:
        """
        Store key-value associations.

        Args:
            keys: Keys to store [batch, key_size]
            values: Values to store [batch, value_size]
        """
        batch_size = keys.size(0)

        keys_proj = torch.tanh(self.key_proj(keys))

        similarity = torch.matmul(keys_proj, self.keys.T)

        match_weights = F.softmax(similarity, dim=-1)

        top_match = match_weights.max(dim=-1).indices

        for i in range(batch_size):
            idx = top_match[i].item()

            self.keys.data[idx] = keys_proj[i].detach()
            self.values.data[idx] = torch.tanh(self.value_proj(values[i])).detach()

            self.usage.data[idx] = 0.0

        self.usage.data += 1.0 / self.capacity

    def retrieve(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Retrieve values for query.

        Args:
            query: Query key [batch, key_size]

        Returns:
            retrieved_values: Retrieved values [batch, value_size]
            attention: Attention weights [batch, capacity]
        """
        batch_size = query.size(0)

        query_proj = torch.tanh(self.key_proj(query))

        similarity = torch.matmul(query_proj, self.keys.T)

        attention = F.softmax(similarity, dim=-1)

        retrieved_values = torch.matmul(attention, self.values)

        return retrieved_values, attention

    def forward(
        self, keys: Tensor, values: Tensor, query: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Store then retrieve.

        Args:
            keys: Keys to store
            values: Values to store
            query: Query for retrieval

        Returns:
            Retrieved values
        """
        self.store(keys, query)
        return self.retrieve(query)
