"""
Multi-hop Question Answering Implementation

This module provides implementations for multi-hop QA systems including
decomposition reasoners, graph reasoning, and iterative retrieval.

Author: Fishstick AI Framework
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fishstick.question_answering.types import (
    QAExample,
    QAPrediction,
    Answer,
    AnswerType,
    Context,
    Question,
    QAConfig,
    RetrievalResult,
    ReasoningChain,
    MultiHopStep,
    QATaskType,
)
from fishstick.question_answering.base import MultiHopQABase


class HopAttention(nn.Module):
    """Attention across reasoning hops.

    Aggregates information from multiple reasoning hops.
    """

    def __init__(
        self,
        hidden_size: int,
        num_hops: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize HopAttention.

        Args:
            hidden_size: Hidden dimension size
            num_hops: Maximum number of hops
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hops = num_hops

        self.hop_projection = nn.Linear(hidden_size, hidden_size)
        self.hop_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hop_hidden: List[Tensor],
        query_hidden: Tensor,
    ) -> Tensor:
        """Aggregate information across hops.

        Args:
            hop_hidden: List of hidden states for each hop [batch, hop_len, hidden]
            query_hidden: Query representation [batch, query_len, hidden]

        Returns:
            Aggregated representation [batch, hidden]
        """
        if not hop_hidden:
            return torch.zeros(
                query_hidden.size(0), self.hidden_size, device=query_hidden.device
            )

        padded_hops = torch.zeros(
            query_hidden.size(0),
            self.num_hops,
            hop_hidden[0].size(1),
            self.hidden_size,
            device=query_hidden.device,
        )

        for i, hop in enumerate(hop_hidden):
            if i < self.num_hops:
                padded_hops[:, i, : hop.size(1), :] = hop

        hop_features = self.hop_projection(padded_hops)

        query_expanded = query_hidden.unsqueeze(1).expand(-1, self.num_hops, -1, -1)

        attended, _ = self.hop_attention(
            query_expanded.reshape(-1, query_expanded.size(2), self.hidden_size),
            hop_features.reshape(-1, hop_features.size(2), self.hidden_size),
            hop_features.reshape(-1, hop_features.size(2), self.hidden_size),
        )

        output = self.output_projection(attended.mean(dim=1))

        return output


class DecompositionReasoner(nn.Module):
    """Question Decomposition for Multi-hop QA.

    Decomposes complex questions into simpler sub-questions.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_decomposition_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize DecompositionReasoner.

        Args:
            hidden_size: Hidden dimension size
            num_decomposition_layers: Number of decomposition layers
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_decomposition_layers
        )

        self.decomposition_head = nn.Linear(hidden_size, 3)

        self.subquestion_generator = nn.GRUCell(
            hidden_size + hidden_size,
            hidden_size,
        )

        self.max_subquestions = 5

    def forward(
        self,
        question_hidden: Tensor,
        question_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Decompose question into sub-questions.

        Args:
            question_hidden: [batch, seq_len, hidden]
            question_mask: [batch, seq_len]

        Returns:
            Dictionary with sub-question embeddings and types
        """
        encoded = self.encoder(question_hidden, src_key_padding_mask=question_mask == 0)

        decomposition_logits = self.decomposition_head(encoded)

        subquestions = []
        current_state = encoded.mean(dim=1)

        for step in range(self.max_subquestions):
            subquestion_repr = current_state.unsqueeze(1)

            subquestions.append(
                {
                    "embedding": subquestion_repr,
                    "step": step,
                }
            )

            new_input = torch.cat([current_state, encoded.mean(dim=1)], dim=-1)
            current_state = self.subquestion_generator(new_input)

        return {
            "subquestions": subquestions,
            "decomposition_logits": decomposition_logits,
        }

    def generate_subquestion_text(
        self,
        subquestion_embedding: Tensor,
        vocabulary: Dict[str, int],
    ) -> str:
        """Generate sub-question text from embedding.

        Args:
            subquestion_embedding: Sub-question embedding
            vocabulary: Token vocabulary

        Returns:
            Generated sub-question text
        """
        return f"subquestion_{subquestion_embedding.shape[0]}"


class GraphReasoningLayer(nn.Module):
    """Graph-based Reasoning Layer for Multi-hop QA.

    Performs reasoning over an evidence graph.
    """

    def __init__(
        self,
        node_hidden: int = 768,
        edge_hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize GraphReasoningLayer.

        Args:
            node_hidden: Node hidden dimension
            edge_hidden: Edge hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden

        self.node_projection = nn.Linear(node_hidden, node_hidden)
        self.edge_projection = nn.Linear(edge_hidden, node_hidden)

        self.message_passing = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(node_hidden * 2, node_hidden),
                    nn.LayerNorm(node_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_hidden, node_hidden),
                )
                for _ in range(num_layers)
            ]
        )

        self.graph_attention = nn.MultiheadAttention(
            node_hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Perform graph reasoning.

        Args:
            node_features: [num_nodes, hidden]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_hidden]

        Returns:
            Updated node features [num_nodes, hidden]
        """
        nodes = self.node_projection(node_features)

        for layer in self.message_passing:
            messages = []

            for i in range(edge_index.size(1)):
                src = nodes[edge_index[0, i]]
                dst = nodes[edge_index[1, i]]

                edge_msg = src + dst
                if edge_features is not None:
                    edge_msg = edge_msg + self.edge_projection(edge_features[i])

                messages.append(edge_msg)

            messages = torch.stack(messages)

            aggregated = torch.zeros_like(nodes)
            aggregated.index_add_(0, edge_index[1], messages)

            combined = torch.cat([nodes, aggregated], dim=-1)
            nodes = layer(combined) + nodes

        attended, _ = self.graph_attention(
            nodes.unsqueeze(0), nodes.unsqueeze(0), nodes.unsqueeze(0)
        )
        nodes = nodes + attended.squeeze(0)

        return nodes


class EntityLinking(nn.Module):
    """Entity Linking for Multi-hop QA.

    Links entities across documents for multi-hop reasoning.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_entity_types: int = 20,
        dropout: float = 0.1,
    ) -> None:
        """Initialize EntityLinking.

        Args:
            hidden_size: Hidden dimension size
            num_entity_types: Number of entity types
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_entity_types = num_entity_types

        self.entity_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.entity_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_entity_types),
        )

        self.entity_similarity = nn.Bilinear(hidden_size, hidden_size, 1)

    def extract_entities(
        self,
        tokens: List[str],
        hidden_states: Tensor,
    ) -> List[Dict[str, Any]]:
        """Extract entities from text.

        Args:
            tokens: List of tokens
            hidden_states: [seq_len, hidden]

        Returns:
            List of extracted entities
        """
        output, (h_n, _) = self.entity_encoder(hidden_states.unsqueeze(0))
        entity_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1).squeeze(0)

        type_logits = self.entity_type_classifier(entity_repr)
        entity_type = type_logits.argmax(dim=-1).item()

        return [
            {
                "text": " ".join(tokens[: min(5, len(tokens))]),
                "type_id": entity_type,
                "embedding": entity_repr.detach().cpu().numpy().tolist(),
            }
        ]

    def link_entities(
        self,
        entities_a: List[Dict[str, Any]],
        entities_b: List[Dict[str, Any]],
    ) -> List[Tuple[int, int, float]]:
        """Link entities between two sets.

        Args:
            entities_a: First set of entities
            entities_b: Second set of entities

        Returns:
            List of (idx_a, idx_b, score) tuples
        """
        links = []

        for i, ent_a in enumerate(entities_a):
            emb_a = torch.tensor(ent_a["embedding"], dtype=torch.float32)

            best_score = -float("inf")
            best_j = -1

            for j, ent_b in enumerate(entities_b):
                emb_b = torch.tensor(ent_b["embedding"], dtype=torch.float32)

                score = self.entity_similarity(emb_a, emb_b).item()

                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0 and best_score > 0.0:
                links.append((i, best_j, best_score))

        return links


class MultiHopReasoner(MultiHopQABase[nn.Module]):
    """Multi-hop QA Reasoner.

    Implements multi-hop reasoning by decomposing questions and
    aggregating answers from multiple hops.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize Multi-hop Reasoner.

        Args:
            config: QA configuration
        """
        super().__init__(config)

        self.hidden_size = config.metadata.get("hidden_size", 768)

        self.decomposer = DecompositionReasoner(
            hidden_size=self.hidden_size,
            num_decomposition_layers=config.metadata.get("num_decomp_layers", 4),
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.graph_reasoning = GraphReasoningLayer(
            node_hidden=self.hidden_size,
            num_layers=config.metadata.get("graph_layers", 3),
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.hop_attention = HopAttention(
            hidden_size=self.hidden_size,
            num_hops=self.max_hops,
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.entity_linker = EntityLinking(
            hidden_size=self.hidden_size,
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.answer_aggregator = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.answer_head = nn.Linear(self.hidden_size * 2, 1)

        self.hop_models: Dict[int, Any] = {}

    def decompose_question(self, question: Question) -> List[str]:
        """Decompose question into sub-questions.

        Args:
            question: The complex question

        Returns:
            List of sub-questions
        """
        return [f"What is {question.text}?", "What else is related?"]

    def retrieve_for_hop(
        self,
        question: str,
        evidence_so_far: List[str],
    ) -> List[RetrievalResult]:
        """Retrieve evidence for a reasoning hop.

        Args:
            question: Current sub-question
            evidence_so_far: Accumulated evidence

        Returns:
            List of retrieved documents
        """
        return [
            RetrievalResult(
                document_id=f"doc_{len(evidence_so_far)}",
                document=f"Evidence for: {question}",
                score=0.8,
            )
        ]

    def aggregate_hop_answers(
        self,
        hop_answers: List[str],
        reasoning_chain: List[str],
    ) -> str:
        """Aggregate answers from multiple hops.

        Args:
            hop_answers: Answers from each hop
            reasoning_chain: Reasoning chain

        Returns:
            Final answer
        """
        return " | ".join(hop_answers)

    def forward(
        self,
        question: Union[str, Question],
        context: Union[str, Context],
    ) -> Answer:
        """Forward pass to generate answer.

        Args:
            question: The question to answer
            context: The context to extract answer from

        Returns:
            Answer object with the predicted answer
        """
        q_text = question.text if isinstance(question, Question) else question
        c_text = context.text if isinstance(context, Context) else context

        sub_questions = self.decompose_question(
            Question(text=q_text) if isinstance(question, str) else question
        )

        all_evidence = []
        hop_answers = []

        for i, sub_q in enumerate(sub_questions[: self.max_hops]):
            retrieved = self.retrieve_for_hop(sub_q, all_evidence)

            for result in retrieved:
                all_evidence.append(result.document)

            answer = f"Answer to: {sub_q}"
            hop_answers.append(answer)

        final_answer = self.aggregate_hop_answers(hop_answers, sub_questions)

        return Answer(
            text=final_answer,
            type=AnswerType.FREE_FORM,
            confidence=0.7,
            reasoning_chain=hop_answers,
        )

    def predict(
        self,
        examples: List[QAExample],
    ) -> List[QAPrediction]:
        """Generate predictions for a batch of examples.

        Args:
            examples: List of QA examples to predict

        Returns:
            List of predictions
        """
        predictions = []

        for example in examples:
            answer = self.forward(example.question, example.context)

            pred = QAPrediction(
                id=example.id,
                question=example.question.text
                if isinstance(example.question, Question)
                else example.question,
                answer=answer,
                context_used=example.context.text
                if isinstance(example.context, Context)
                else example.context,
            )
            predictions.append(pred)

        return predictions

    def train_model(
        self,
        train_examples: List[QAExample],
        eval_examples: Optional[List[QAExample]] = None,
    ) -> Dict[str, Any]:
        """Train the QA model.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples

        Returns:
            Training history dictionary
        """
        raise NotImplementedError("Training not implemented. Use QATrainer.")

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "decomposer": self.decomposer.state_dict(),
                "graph_reasoning": self.graph_reasoning.state_dict(),
                "hop_attention": self.hop_attention.state_dict(),
                "entity_linker": self.entity_linker.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.decomposer.load_state_dict(checkpoint["decomposer"])
        self.graph_reasoning.load_state_dict(checkpoint["graph_reasoning"])
        self.hop_attention.load_state_dict(checkpoint["hop_attention"])
        self.entity_linker.load_state_dict(checkpoint["entity_linker"])


class IterativeRetrieval(nn.Module):
    """Iterative Evidence Retrieval for Multi-hop QA.

    Retrieves evidence iteratively, using previous findings to
    inform subsequent retrievals.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_iterations: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize IterativeRetrieval.

        Args:
            hidden_size: Hidden dimension size
            num_iterations: Number of retrieval iterations
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations

        self.query_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.context_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.iteration_gate = nn.GRUCell(hidden_size * 2, hidden_size)

        self.relevance_scorer = nn.Linear(hidden_size * 2, 1)

    def encode_query(self, query: str) -> Tensor:
        """Encode query.

        Args:
            query: Query string

        Returns:
            Query representation
        """
        return torch.randn(1, len(query.split()), self.hidden_size)

    def encode_context(self, context: str) -> Tensor:
        """Encode context.

        Args:
            context: Context string

        Returns:
            Context representation
        """
        return torch.randn(1, len(context.split()), self.hidden_size)

    def retrieve_iteratively(
        self,
        query: str,
        candidates: List[str],
        num_iterations: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Perform iterative retrieval.

        Args:
            query: Initial query
            candidates: Candidate documents
            num_iterations: Number of iterations

        Returns:
            List of (document, score) tuples
        """
        iters = num_iterations or self.num_iterations

        query_repr = self.encode_query(query)

        for iteration in range(iters):
            scores = []

            for doc in candidates:
                doc_repr = self.encode_context(doc)

                combined = torch.cat(
                    [query_repr.mean(dim=1), doc_repr.mean(dim=1)], dim=-1
                )
                score = self.relevance_scorer(combined).item()
                scores.append(score)

            query_repr = self.iteration_gate(
                torch.cat(
                    [
                        query_repr.mean(dim=1, keepdim=True),
                        self.encode_context(candidates[scores.index(max(scores))]).mean(
                            dim=0, keepdim=True
                        ),
                    ],
                    dim=-1,
                ),
                query_repr.mean(dim=1, keepdim=True).squeeze(0),
            ).unsqueeze(1)

        results = list(zip(candidates, scores))
        return sorted(results, key=lambda x: x[1], reverse=True)
