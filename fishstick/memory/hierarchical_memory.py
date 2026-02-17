"""
Hierarchical Memory Systems.

Implements multi-level memory architecture:
- Working memory (short-term, limited capacity)
- Episodic memory (medium-term, event-based)
- Semantic memory (long-term, structured knowledge)
- Memory consolidation mechanisms

Based on:
- Differentiable Neural Computers (Graves et al., 2016)
- Memory Networks (Weston et al., 2014)
- Hierarchical Memory (Pritzel et al., 2017)
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MemoryType(Enum):
    """Types of memory in hierarchical system."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class MemoryState:
    """State of a memory module."""

    content: Tensor
    attention_weights: Optional[Tensor] = None
    timestamps: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemory(nn.Module):
    """
    Working memory with limited capacity.

    Implements attention-based selection and rapid update.
    Similar to cognitive working memory.
    """

    def __init__(
        self,
        capacity: int = 7,
        item_size: int = 64,
        num_slots: int = 4,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size
        self.num_slots = num_slots

        self.slots = nn.Parameter(torch.randn(num_slots, item_size) * 0.01)
        self.slot_strengths = nn.Parameter(torch.ones(num_slots))

        self.query_net = nn.Linear(item_size, item_size)
        self.key_net = nn.Linear(item_size, item_size)
        self.value_net = nn.Linear(item_size, item_size)

        self.gate_net = nn.Linear(item_size * 2, item_size)

        self.usage = nn.Parameter(torch.zeros(num_slots))

    def forward(
        self,
        item: Tensor,
        operation: str = "write",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Working memory operation.

        Args:
            item: Item to process [batch, item_size]
            operation: "write" or "read"

        Returns:
            output: Processed item or read content
            info: Debug information
        """
        batch_size = item.size(0)

        if operation == "write":
            return self._write(item)
        elif operation == "read":
            return self._read(item)
        elif operation == "read_all":
            return self._read_all()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _write(self, item: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Write item to working memory with slot-based attention."""
        batch_size = item.size(0)

        query = torch.tanh(self.query_net(item))
        keys = torch.tanh(self.key_net(self.slots))

        similarity = torch.matmul(query, keys.T)
        slot_weights = F.softmax(similarity, dim=-1)

        strengths = F.softplus(self.slot_strengths)
        strengths = strengths / strengths.sum(dim=-1, keepdim=True)

        usage_weighted = slot_weights * strengths

        usage_weighted = usage_weighted / usage_weighted.sum(dim=-1, keepdim=True)

        selected_slot = usage_weighted.max(dim=-1).indices

        new_slots = self.slots.data.clone()

        for i in range(batch_size):
            idx = selected_slot[i].item()
            new_slots[idx] = item[i].detach()

        self.slots.data = new_slots

        self.usage.data = self.usage.data * 0.9 + usage_weighted.mean(dim=0).detach()

        output = torch.tanh(self.value_net(self.slots.mean(dim=0, keepdim=True)))
        output = output.expand(batch_size, -1)

        info = {
            "slot_weights": slot_weights,
            "selected_slot": selected_slot,
            "strengths": strengths,
            "usage": self.usage,
        }

        return output, info

    def _read(self, query: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Read from working memory using query."""
        batch_size = query.size(0)

        query_proj = torch.tanh(self.query_net(query))
        keys = torch.tanh(self.key_net(self.slots))

        similarity = torch.matmul(query_proj, keys.T)
        attention = F.softmax(similarity, dim=-1)

        read_vector = torch.matmul(attention, self.slots)

        output = torch.tanh(self.value_net(read_vector))

        info = {
            "attention": attention,
            "keys": keys,
        }

        return output, info

    def _read_all(self) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Read all slots."""
        output = torch.tanh(self.value_net(self.slots))

        info = {
            "slots": self.slots,
            "usage": self.usage,
        }

        return output, info

    def clear(self) -> None:
        """Clear working memory."""
        self.slots.data = torch.randn_like(self.slots) * 0.01
        self.usage.data = torch.zeros_like(self.usage)


class EpisodicMemory(nn.Module):
    """
    Episodic memory for storing sequences of events.

    Implements memory traces with temporal ordering and retrieval.
    """

    def __init__(
        self,
        capacity: int = 100,
        item_size: int = 64,
        memory_size: int = 128,
    ):
        super().__init__()
        self.capacity = capacity
        self.item_size = item_size
        self.memory_size = memory_size

        self.memory = nn.Parameter(torch.randn(memory_size, item_size) * 0.01)
        self.timestamps = nn.Parameter(torch.zeros(memory_size))
        self.importance = nn.Parameter(torch.zeros(memory_size))

        self.write_idx = 0
        self.num_writes = 0

        self.key_net = nn.Linear(item_size, item_size)
        self.timestamp_net = nn.Linear(1, 1)
        self.importance_net = nn.Linear(item_size, 1)

    def store(
        self,
        item: Tensor,
        timestamp: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Store item in episodic memory.

        Args:
            item: Item to store [batch, item_size]
            timestamp: Optional timestamp [batch]
            importance: Optional importance score [batch]

        Returns:
            Info dictionary
        """
        batch_size = item.size(0)

        if timestamp is None:
            timestamp = torch.full(
                (batch_size,), self.num_writes, dtype=torch.float32, device=item.device
            )

        if importance is None:
            importance = torch.ones(batch_size, device=item.device)

        key = torch.tanh(self.key_net(item))

        for i in range(batch_size):
            idx = self.write_idx % self.memory_size

            self.memory.data[idx] = key[i].detach()
            self.timestamps.data[idx] = timestamp[i].detach()
            self.importance.data[idx] = importance[i].detach()

            self.write_idx += 1

        self.num_writes += 1

        info = {
            "write_idx": self.write_idx,
            "timestamp": timestamp,
        }

        return info

    def retrieve(
        self,
        query: Tensor,
        k: int = 5,
        recency_weight: float = 0.5,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Retrieve memories based on query.

        Args:
            query: Query tensor [batch, item_size]
            k: Number of memories to retrieve
            recency_weight: Weight for recency (0 = ignore, 1 = only recent)

        Returns:
            retrieved: Retrieved memories [batch, k, item_size]
            info: Debug information
        """
        batch_size = query.size(0)

        query_proj = torch.tanh(self.key_net(query))

        similarity = torch.matmul(query_proj, self.memory.T)

        time_weights = torch.sigmoid(-self.timestamps.unsqueeze(0) * 0.01)

        combined = (1 - recency_weight) * similarity + recency_weight * time_weights

        topk_weights, topk_indices = torch.topk(
            combined, min(k, self.memory_size), dim=-1
        )

        retrieved = []
        for i in range(batch_size):
            mem_idx = topk_indices[i]
            retrieved.append(self.memory[mem_idx])

        retrieved = torch.stack(retrieved)

        info = {
            "similarity": similarity,
            "time_weights": time_weights,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }

        return retrieved, info

    def retrieve_by_time(
        self,
        start_time: float,
        end_time: float,
        k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Retrieve memories within time range.

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            k: Max memories to retrieve

        Returns:
            memories: Retrieved memories
            times: Their timestamps
        """
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        indices = mask.nonzero().squeeze(-1)

        if len(indices) == 0:
            return torch.zeros(k, self.item_size), torch.zeros(k)

        selected_memory = self.memory[indices]
        selected_time = self.timestamps[indices]

        if len(indices) > k:
            topk_idx = torch.randperm(len(indices))[:k]
            selected_memory = selected_memory[topk_idx]
            selected_time = selected_time[topk_idx]

        return selected_memory, selected_time

    def get_recent(self, k: int = 10) -> Tuple[Tensor, Tensor]:
        """Get k most recent memories."""
        _, recent_indices = torch.topk(self.timestamps, min(k, self.memory_size))

        return self.memory[recent_indices], self.timestamps[recent_indices]

    def clear(self) -> None:
        """Clear episodic memory."""
        self.memory.data = torch.randn_like(self.memory) * 0.01
        self.timestamps.data = torch.zeros_like(self.timestamps)
        self.importance.data = torch.zeros_like(self.importance)
        self.write_idx = 0
        self.num_writes = 0


class SemanticMemory(nn.Module):
    """
    Semantic memory for structured knowledge.

    Implements knowledge graph-like storage with relational reasoning.
    """

    def __init__(
        self,
        num_entities: int = 100,
        entity_size: int = 64,
        relation_size: int = 32,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.entity_size = entity_size
        self.relation_size = relation_size

        self.entities = nn.Parameter(torch.randn(num_entities, entity_size) * 0.01)

        self.relation_proj = nn.Linear(relation_size, entity_size * 2)

        self.entity_keys = nn.Parameter(torch.randn(num_entities, entity_size) * 0.01)

        self.query_net = nn.Linear(entity_size, entity_size)
        self.key_net = nn.Linear(entity_size, entity_size)

    def store_entity(
        self,
        entity_id: int,
        embedding: Tensor,
    ) -> None:
        """
        Store entity embedding.

        Args:
            entity_id: Entity index
            embedding: Entity embedding [entity_size]
        """
        self.entities.data[entity_id] = embedding.detach()

    def store_relation(
        self,
        subject_id: int,
        relation: Tensor,
        object_id: int,
    ) -> None:
        """
        Store relational fact (subject, relation, object).

        Args:
            subject_id: Subject entity index
            relation: Relation embedding [relation_size]
            object_id: Object entity index
        """
        relation_proj = torch.tanh(self.relation_proj(relation))

        subject_emb = self.entities[subject_id]
        object_emb = self.entities[object_id]

        combined = torch.cat([subject_emb, relation_proj, object_emb])

        key = torch.tanh(self.key_net(combined))

        similarity = torch.matmul(key, self.entity_keys.T)
        attention = F.softmax(similarity, dim=-1)

        updated = attention @ self.entities
        self.entities.data = self.entities * 0.99 + updated * 0.01

    def query_entity(self, entity_id: int) -> Tensor:
        """Get entity embedding."""
        return self.entities[entity_id]

    def find_similar(
        self,
        query: Tensor,
        k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Find k most similar entities.

        Args:
            query: Query embedding [batch, entity_size] or [entity_size]
            k: Number to retrieve

        Returns:
            embeddings: Entity embeddings [batch, k, entity_size] or [k, entity_size]
            entity_ids: Entity indices [batch, k] or [k]
        """
        query_proj = torch.tanh(self.query_net(query))

        if query_proj.dim() == 1:
            query_proj = query_proj.unsqueeze(0)

        similarity = torch.matmul(query_proj, self.entity_keys.T)

        topk_sim, topk_idx = torch.topk(similarity, min(k, self.num_entities), dim=-1)

        embeddings = self.entities[topk_idx]

        if embeddings.size(0) == 1 and query.dim() == 1:
            embeddings = embeddings.squeeze(0)
            topk_idx = topk_idx.squeeze(0)

        return embeddings, topk_idx

    def reason(
        self,
        subject: Tensor,
        relation: Tensor,
    ) -> Tensor:
        """
        Relational reasoning: given subject and relation, predict object.

        Args:
            subject: Subject embedding [entity_size]
            relation: Relation embedding [relation_size]

        Returns:
            Predicted object embedding
        """
        relation_proj = torch.tanh(self.relation_proj(relation))

        combined = torch.cat([subject, relation_proj])

        pred = torch.tanh(self.key_net(combined))

        similarity = torch.matmul(pred, self.entity_keys.T)
        attention = F.softmax(similarity, dim=-1)

        return attention @ self.entities


class HierarchicalMemorySystem(nn.Module):
    """
    Complete hierarchical memory system.

    Combines working, episodic, and semantic memory with consolidation.
    """

    def __init__(
        self,
        item_size: int = 64,
        working_capacity: int = 7,
        episodic_capacity: int = 100,
        semantic_entities: int = 100,
    ):
        super().__init__()
        self.item_size = item_size

        self.working = WorkingMemory(
            capacity=working_capacity,
            item_size=item_size,
            num_slots=4,
        )

        self.episodic = EpisodicMemory(
            capacity=episodic_capacity,
            item_size=item_size,
            memory_size=128,
        )

        self.semantic = SemanticMemory(
            num_entities=semantic_entities,
            entity_size=item_size,
        )

        self.consolidation_gate = nn.Linear(item_size * 3, 3)

        self.working_to_episodic = nn.Linear(item_size, item_size)
        self.episodic_to_semantic = nn.Linear(item_size, item_size)

        self.usage_count = 0

    def forward(
        self,
        item: Tensor,
        operation: str = "encode",
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Hierarchical memory operation.

        Args:
            item: Input item [batch, item_size]
            operation: "encode", "retrieve", "consolidate"

        Returns:
            output: Processed output
            info: Debug information
        """
        if operation == "encode":
            return self._encode(item)
        elif operation == "retrieve":
            return self._retrieve(item)
        elif operation == "consolidate":
            return self._consolidate()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _encode(self, item: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Encode item through memory hierarchy."""
        working_out, working_info = self.working(item, "write")

        self.episodic.store(item)

        gate_values = torch.sigmoid(
            self.consolidation_gate(torch.cat([item, working_out, item], dim=-1))
        )

        info = {
            "working": working_info,
            "gate_values": gate_values,
            "item": item,
        }

        return working_out, info

    def _retrieve(self, query: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Retrieve from memory hierarchy."""
        working_out, working_info = self.working._read(query)

        episodic_retrieved, episodic_info = self.episodic.retrieve(query)

        episodic_summary = episodic_retrieved.mean(dim=1)

        semantic_similar, semantic_ids = self.semantic.find_similar(query)

        semantic_summary = semantic_similar.mean(dim=1)

        combined = torch.cat([working_out, episodic_summary, semantic_summary], dim=-1)

        gate_values = torch.sigmoid(self.consolidation_gate(combined))

        output = (
            gate_values[:, 0:1] * working_out
            + gate_values[:, 1:2] * episodic_summary
            + gate_values[:, 2:3] * semantic_summary
        )

        info = {
            "working": working_info,
            "episodic_retrieved": episodic_retrieved,
            "episodic_info": episodic_info,
            "semantic_similar": semantic_similar,
            "semantic_ids": semantic_ids,
            "gate_values": gate_values,
        }

        return output, info

    def _consolidate(self) -> Dict[str, Any]:
        """Consolidate episodic to semantic memory."""
        recent_memories, recent_times = self.episodic.get_recent(k=10)

        importance = self.episodic.importance.data

        important_idx = torch.topk(
            importance, min(10, self.semantic.num_entities)
        ).indices

        for idx in important_idx:
            if idx < len(recent_memories):
                entity_id = idx.item() % self.semantic.num_entities
                self.semantic.store_entity(
                    entity_id, recent_memories[idx % len(recent_memories)]
                )

        info = {
            "consolidated_entities": important_idx,
            "recent_memories": recent_memories,
        }

        return info

    def write_semantic(
        self,
        entity_id: int,
        embedding: Tensor,
    ) -> None:
        """Write directly to semantic memory."""
        self.semantic.store_entity(entity_id, embedding)

    def clear_working(self) -> None:
        """Clear working memory."""
        self.working.clear()

    def clear_episodic(self) -> None:
        """Clear episodic memory."""
        self.episodic.clear()


class ContinualLearningMemory(nn.Module):
    """
    Memory system designed for continual learning.

    Implements experience replay with reservoir sampling.
    """

    def __init__(
        self,
        max_size: int = 1000,
        item_size: int = 64,
        sample_size: int = 32,
    ):
        super().__init__()
        self.max_size = max_size
        self.item_size = item_size
        self.sample_size = sample_size

        self.buffer = nn.Parameter(torch.zeros(max_size, item_size))
        self.buffer_targets = nn.Parameter(torch.zeros(max_size))

        self.buffer_idx = 0
        self.filled = 0

        self.priorities = nn.Parameter(torch.zeros(max_size))

    def store(
        self,
        items: Tensor,
        targets: Optional[Tensor] = None,
        priorities: Optional[Tensor] = None,
    ) -> None:
        """
        Store items using reservoir sampling.

        Args:
            items: Items to store [batch, item_size]
            targets: Optional targets [batch]
            priorities: Optional priorities [batch]
        """
        batch_size = items.size(0)

        for i in range(batch_size):
            idx = self.buffer_idx % self.max_size

            self.buffer.data[idx] = items[i].detach()

            if targets is not None:
                self.buffer_targets.data[idx] = targets[i].detach()

            if priorities is not None:
                self.priorities.data[idx] = priorities[i].detach()
            else:
                self.priorities.data[idx] = 1.0

            self.buffer_idx += 1

            if self.filled < self.max_size:
                self.filled += 1

    def sample(
        self,
        batch_size: int,
        priority_weight: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample from memory buffer.

        Args:
            batch_size: Number of samples
            priority_weight: Weight for priority sampling (0 = uniform, 1 = priority only)

        Returns:
            samples: Sampled items
            indices: Sample indices
        """
        if self.filled == 0:
            return torch.zeros(batch_size, self.item_size), torch.zeros(
                batch_size, dtype=torch.long
            )

        available_size = self.filled

        if priority_weight > 0:
            probs = self.priorities[:available_size]
            probs = probs / probs.sum()

            indices = torch.multinomial(probs, batch_size, replacement=True)
        else:
            indices = torch.randint(0, available_size, (batch_size,))

        samples = self.buffer[indices]
        targets = self.buffer_targets[indices]

        return samples, indices

    def update_priorities(
        self,
        indices: Tensor,
        priorities: Tensor,
    ) -> None:
        """Update priorities for sampled items."""
        self.priorities.data[indices] = priorities.detach()

    def clear(self) -> None:
        """Clear memory buffer."""
        self.buffer.data = torch.zeros_like(self.buffer)
        self.buffer_targets.data = torch.zeros_like(self.buffer_targets)
        self.priorities.data = torch.zeros_like(self.priorities)
        self.buffer_idx = 0
        self.filled = 0


class MetaLearningMemory(nn.Module):
    """
    Memory system for meta-learning.

    Supports fast adaptation with context-based memory.
    """

    def __init__(
        self,
        context_size: int = 10,
        item_size: int = 64,
        memory_size: int = 128,
    ):
        super().__init__()
        self.context_size = context_size
        self.item_size = item_size
        self.memory_size = memory_size

        self.context_buffer = nn.Parameter(torch.zeros(context_size, item_size))

        self.memory = nn.Parameter(torch.randn(memory_size, item_size) * 0.01)

        self.query_net = nn.Linear(item_size, item_size)
        self.key_net = nn.Linear(item_size, item_size)

        self.context_idx = 0

    def add_context(self, item: Tensor) -> None:
        """Add item to context buffer."""
        self.context_buffer.data[self.context_idx % self.context_size] = item.detach()
        self.context_idx += 1

    def read_memory(
        self,
        query: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Read from memory using context-augmented query."""
        context = self.context_buffer.mean(dim=0, keepdim=True)

        query_aug = query + 0.1 * context

        query_proj = torch.tanh(self.query_net(query_aug))
        keys = torch.tanh(self.key_net(self.memory))

        similarity = torch.matmul(query_proj, keys.T)
        attention = F.softmax(similarity, dim=-1)

        read = torch.matmul(attention, self.memory)

        return read, attention

    def clear_context(self) -> None:
        """Clear context buffer."""
        self.context_buffer.data = torch.zeros_like(self.context_buffer)
        self.context_idx = 0
