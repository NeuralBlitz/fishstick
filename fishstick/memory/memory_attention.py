"""
Memory Attention Mechanisms.

Implements attention mechanisms designed for memory access:
- Hopfield memory attention
- Sparse memory attention
- Memory-augmented attention
- Key-value memory attention
- Associative attention

Based on:
- Hopfield Networks (Hopfield, 1982)
- Modern Hopfield Networks (Krotov & Hopfield, 2016)
- Memory Networks (Weston et al., 2014)
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class HopfieldAttention(nn.Module):
    """
    Hopfield network-based attention for memory retrieval.

    Uses continuous Hopfield network for associative memory lookup.
    Supports fixed and learnable patterns.

    Based on: "Dense Associative Memory for Pattern Recognition" (Krotov & Hopfield, 2016)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        memory_size: int = 128,
        beta: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.beta = beta

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.memory_keys = nn.Parameter(torch.randn(memory_size, embed_dim) * 0.01)
        self.memory_values = nn.Parameter(torch.randn(memory_size, embed_dim) * 0.01)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: Tensor,
        value: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Hopfield attention forward pass.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            value: Optional value override (uses memory if None)
            memory_mask: Optional mask for memory locations

        Returns:
            output: Attended output [batch, seq_len, embed_dim]
            info: Dictionary with attention weights and debug info
        """
        batch_size, seq_len, _ = query.shape

        q = self.q_proj(query)

        if value is None:
            k = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
            v = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            k = self.k_proj(value)
            v = self.v_proj(value)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        similarity = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        energy = self.beta * similarity

        if memory_mask is not None:
            energy = energy.masked_fill(memory_mask == 0, float("-inf"))

        attn_weights = F.softmax(energy, dim=-1)

        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        info = {
            "attention_weights": attn_weights,
            "similarity": similarity,
            "q": q,
            "k": k,
            "v": v,
        }

        return output, info


class SparseMemoryAttention(nn.Module):
    """
    Sparse attention for large memory systems.

    Uses local attention windows.
    Computationally efficient for large memory sizes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        memory_size: int = 1024,
        window_size: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size
        self.window_size = window_size

        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Sparse attention with local windows.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            output: Attended output [batch, seq_len, embed_dim]
            info: Debug information
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, device=x.device
        )

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)

            local_q = q[:, :, i : i + 1, :]
            local_k = k[:, :, start:end, :]
            local_v = v[:, :, start:end, :]

            local_energy = torch.matmul(local_q, local_k.transpose(-2, -1)) * self.scale

            attn_weights[:, :, i : i + 1, start:end] = F.softmax(local_energy, dim=-1)

        local_output = torch.matmul(attn_weights, v)

        output = local_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.out_proj(output)

        info = {
            "attention_weights": attn_weights,
            "local_output": local_output,
        }

        return output, info


class KeyValueMemoryAttention(nn.Module):
    """
    Key-Value memory attention mechanism.

    Separate key and value memories with content-based addressing.
    """

    def __init__(
        self,
        key_size: int,
        value_size: int,
        memory_size: int = 128,
        num_slots: int = 4,
    ):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.memory_size = memory_size
        self.num_slots = num_slots

        self.key_memory = nn.Parameter(torch.randn(memory_size, key_size) * 0.01)
        self.value_memory = nn.Parameter(torch.randn(memory_size, value_size) * 0.01)

        self.slot_keys = nn.Parameter(torch.randn(num_slots, key_size) * 0.01)
        self.slot_values = nn.Parameter(torch.randn(num_slots, value_size) * 0.01)

        self.key_proj = nn.Linear(key_size, key_size)
        self.query_net = nn.Sequential(
            nn.Linear(key_size, key_size),
            nn.Tanh(),
            nn.Linear(key_size, num_slots),
        )

        self.value_proj = nn.Linear(value_size, value_size)

    def forward(
        self,
        query: Tensor,
        read_memory: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Key-value memory attention.

        Args:
            query: Query tensor [batch, key_size]
            read_memory: Whether to read from external memory

        Returns:
            output: Retrieved values [batch, value_size]
            info: Debug information
        """
        batch_size = query.size(0)

        if read_memory:
            key_memory = self.key_memory.unsqueeze(0).expand(batch_size, -1, -1)
            value_memory = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1)

            query_key = torch.tanh(self.key_proj(query))

            similarity = torch.matmul(query_key, key_memory.transpose(-2, -1))

            attn_weights = F.softmax(similarity, dim=-1)

            retrieved_value = torch.matmul(attn_weights, value_memory)
        else:
            retrieved_value = torch.zeros(
                batch_size, self.value_size, device=query.device
            )

        slot_attn = self.query_net(query)
        slot_weights = F.softmax(slot_attn, dim=-1)

        slot_output = torch.matmul(slot_weights, self.slot_values)

        output = self.value_proj(retrieved_value + slot_output)

        info = {
            "query_key": query_key if read_memory else None,
            "memory_attention": attn_weights if read_memory else None,
            "slot_weights": slot_weights,
            "retrieved_value": retrieved_value,
            "slot_output": slot_output,
        }

        return output, info

    def write(
        self,
        keys: Tensor,
        values: Tensor,
    ) -> None:
        """
        Write to external memory.

        Args:
            keys: Keys to write [batch, key_size]
            values: Values to write [batch, value_size]
        """
        batch_size = keys.size(0)

        similarity = torch.matmul(keys, self.key_memory.unsqueeze(0).transpose(-2, -1))

        write_weights = F.softmax(similarity, dim=-1)

        update = torch.matmul(write_weights.unsqueeze(-1), values.unsqueeze(1))
        update = update.mean(dim=0)

        self.value_memory.data = self.value_memory * 0.9 + update * 0.1


class AssociativeAttention(nn.Module):
    """
    Associative attention for relational memory.

    Learns relationships between key-value pairs.
    """

    def __init__(
        self,
        embed_dim: int,
        num_relations: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_relations = num_relations
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.relation_keys = nn.Parameter(torch.randn(num_relations, embed_dim) * 0.01)
        self.relation_values = nn.Parameter(
            torch.randn(num_relations, embed_dim) * 0.01
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Associative attention forward.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            output: Attended output [batch, seq_len, embed_dim]
            info: Debug information
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn_weights = F.softmax(energy, dim=-1)

        attended = torch.matmul(attn_weights, v)

        rel_q = attended.mean(dim=2)

        rel_similarity = torch.matmul(rel_q, self.relation_keys.transpose(-2, -1))
        rel_attn = F.softmax(rel_similarity, dim=-1)

        rel_output = torch.matmul(rel_attn, self.relation_values)

        output = attended + rel_output.unsqueeze(2)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        info = {
            "attention_weights": attn_weights,
            "relation_attention": rel_attn,
        }

        return output, info


class MemoryAugmentedAttention(nn.Module):
    """
    Standard attention augmented with external memory.

    Combines self-attention with read/write memory operations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        memory_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.memory_size = memory_size

        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.memory_keys = nn.Parameter(torch.randn(memory_size, embed_dim) * 0.01)
        self.memory_values = nn.Parameter(torch.randn(memory_size, embed_dim) * 0.01)

        self.mem_q_proj = nn.Linear(embed_dim, embed_dim)
        self.mem_k_proj = nn.Linear(embed_dim, embed_dim)
        self.mem_v_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(
        self,
        x: Tensor,
        memory_write: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Memory-augmented attention.

        Args:
            x: Input [batch, seq_len, embed_dim]
            memory_write: Optional values to write to memory

        Returns:
            output: Attended output [batch, seq_len, embed_dim]
            info: Debug information
        """
        batch_size, seq_len, _ = x.shape

        x_norm = self.norm1(x)

        self_attn_out, self_attn_weights = self.self_attn(x_norm, x_norm, x_norm)

        x = x + self_attn_out

        if memory_write is not None:
            mem_k = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1)
            mem_v = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1)

            q = self.mem_q_proj(x)

            mem_similarity = torch.matmul(q, mem_k.transpose(-2, -1)) * (
                self.embed_dim**-0.5
            )
            mem_attn = F.softmax(mem_similarity, dim=-1)

            mem_read = torch.matmul(mem_attn, mem_v)

            x = x + mem_read

            memory_update = memory_write.mean(dim=1)
            self.memory_values.data = self.memory_values * 0.95 + memory_update * 0.05

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)

        output = x + ffn_out

        info = {
            "self_attention": self_attn_weights,
            "memory_attention": mem_attn if memory_write is not None else None,
            "memory": self.memory_values,
        }

        return output, info


class RoutingAttention(nn.Module):
    """
    Attention with routing mechanism for dynamic memory selection.

    Uses routing-by-agreement to select which memory bank to attend to.
    """

    def __init__(
        self,
        embed_dim: int,
        num_banks: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_banks = num_banks
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.memory_banks = nn.ModuleList(
            [nn.Parameter(torch.randn(128, embed_dim) * 0.01) for _ in range(num_banks)]
        )

        self.routing_weights = nn.Parameter(torch.ones(num_banks) / num_banks)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Routing attention forward.

        Args:
            x: Input [batch, seq_len, embed_dim]

        Returns:
            output: Attended output [batch, seq_len, embed_dim]
            info: Routing weights and attention
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        bank_outputs = []
        bank_attentions = []

        for bank in self.memory_banks:
            bank_k = (
                bank.unsqueeze(0)
                .unsqueeze(2)
                .expand(batch_size, seq_len, self.num_heads, -1, self.head_dim)
            )
            bank_v = (
                bank.unsqueeze(0)
                .unsqueeze(2)
                .expand(batch_size, seq_len, self.num_heads, -1, self.head_dim)
            )

            energy = torch.matmul(q, bank_k.transpose(-2, -1)) * self.scale
            attn = F.softmax(energy, dim=-1)

            bank_out = torch.matmul(attn, bank_v)

            bank_outputs.append(bank_out)
            bank_attentions.append(attn)

        bank_outputs = torch.stack(bank_outputs, dim=0)

        routing_weights = F.softmax(self.routing_weights, dim=-1)

        weighted_output = torch.einsum("bknsd,b->bknsd", bank_outputs, routing_weights)

        output = weighted_output.sum(dim=0)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        info = {
            "routing_weights": routing_weights,
            "bank_attentions": bank_attentions,
        }

        return output, info


class SetAttention(nn.Module):
    """
    Attention for unordered set inputs (permutation invariant).

    Uses mean pooling over attention outputs for set-level representation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Set attention (permutation invariant).

        Args:
            x: Input [batch, set_size, embed_dim]

        Returns:
            output: Set representation [batch, embed_dim]
            info: Attention weights
        """
        batch_size, set_size, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, set_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, set_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, set_size, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(energy, dim=-1)

        attended = torch.matmul(attn_weights, v)

        output = attended.mean(dim=2)

        output = self.out_proj(output)

        info = {
            "attention_weights": attn_weights,
            "set_attention": attended,
        }

        return output, info
