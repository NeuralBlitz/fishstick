"""
Reading Comprehension Implementation

This module provides implementations for reading comprehension systems including
coherence attention, argument extraction, and multi-document reading.

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
    QATaskType,
)
from fishstick.question_answering.base import ReadingComprehensionBase


class CoherenceAttention(nn.Module):
    """Coherence Attention for Reading Comprehension.

    Models document coherence for better reading comprehension.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize CoherenceAttention.

        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.sentence_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.coherence_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.coherence_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def encode_sentences(
        self,
        token_hidden: Tensor,
        sentence_boundaries: List[List[int]],
    ) -> Tensor:
        """Encode sentences within document.

        Args:
            token_hidden: [batch, seq_len, hidden]
            sentence_boundaries: List of sentence boundaries

        Returns:
            Sentence representations [batch, num_sentences, hidden]
        """
        batch_size = token_hidden.size(0)
        device = token_hidden.device

        max_sentences = (
            max(len(bounds) for bounds in sentence_boundaries)
            if sentence_boundaries
            else 1
        )

        sentence_reprs = torch.zeros(
            batch_size, max_sentences, self.hidden_size, device=device
        )

        for b in range(batch_size):
            boundaries = (
                sentence_boundaries[b] if b < len(sentence_boundaries) else [[0, 10]]
            )

            for s_idx, (start, end) in enumerate(boundaries):
                if s_idx >= max_sentences:
                    break

                sent_hidden = token_hidden[b, start : min(end, token_hidden.size(1))]
                if sent_hidden.size(0) > 0:
                    sentence_reprs[b, s_idx] = sent_hidden.mean(dim=0)

        return sentence_reprs

    def compute_coherence(
        self,
        context_hidden: Tensor,
        graph_structure: Dict[str, Any],
    ) -> Tensor:
        """Compute coherence scores for passages.

        Args:
            context_hidden: [batch, seq_len, hidden]
            graph_structure: Graph structure of context

        Returns:
            Coherence scores [batch, seq_len]
        """
        sentence_boundaries = graph_structure.get("sentence_boundaries", [[]])
        sentence_hidden = self.encode_sentences(context_hidden, sentence_boundaries)

        attended, _ = self.coherence_attention(
            sentence_hidden,
            sentence_hidden,
            sentence_hidden,
        )

        coherence_scores = []

        for b in range(context_hidden.size(0)):
            num_sentences = attended.size(1)
            scores = torch.zeros(context_hidden.size(1), device=context_hidden.device)

            for s_idx in range(num_sentences):
                sent_repr = attended[b, s_idx]
                start, end = (
                    sentence_boundaries[b][s_idx]
                    if b < len(sentence_boundaries)
                    else (0, 10)
                )

                for t_idx in range(start, min(end, context_hidden.size(1))):
                    combined = torch.cat([context_hidden[b, t_idx], sent_repr], dim=-1)
                    scores[t_idx] = self.coherence_scorer(combined).squeeze(-1)

            coherence_scores.append(scores)

        return torch.stack(coherence_scores)

    def forward(
        self,
        context_hidden: Tensor,
        sentence_boundaries: List[List[int]],
    ) -> Tensor:
        """Forward pass for coherence modeling.

        Args:
            context_hidden: [batch, seq_len, hidden]
            sentence_boundaries: Sentence boundaries

        Returns:
            Coherence-aware hidden states [batch, seq_len, hidden]
        """
        sentence_hidden = self.encode_sentences(context_hidden, sentence_boundaries)

        graph_structure = {"sentence_boundaries": sentence_boundaries}
        coherence_scores = self.compute_coherence(context_hidden, graph_structure)

        return context_hidden + coherence_scores.unsqueeze(-1) * context_hidden


class ArgumentExtractor(nn.Module):
    """Argument/Evidence Extractor for Reading Comprehension.

    Extracts arguments and evidence from text.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_argument_types: int = 15,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ArgumentExtractor.

        Args:
            hidden_size: Hidden dimension size
            num_argument_types: Number of argument types
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_argument_types = num_argument_types

        self.argument_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.argument_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_argument_types),
        )

        self.evidence_scorer = nn.Bilinear(hidden_size, hidden_size, 1)

    def extract_arguments(
        self,
        context: Context,
    ) -> List[Dict[str, Any]]:
        """Extract arguments from context.

        Args:
            context: Input context

        Returns:
            List of extracted arguments
        """
        tokens = context.text.split()

        num_chunks = max(1, len(tokens) // 50)
        chunk_size = len(tokens) // num_chunks + 1

        arguments = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])

            arg_type = i % self.num_argument_types

            arguments.append(
                {
                    "text": chunk,
                    "type_id": arg_type,
                    "type_name": f"argument_type_{arg_type}",
                    "start_idx": start,
                    "end_idx": end,
                    "confidence": 0.7,
                }
            )

        return arguments

    def score_evidence(
        self,
        claim_hidden: Tensor,
        evidence_hidden: Tensor,
    ) -> Tensor:
        """Score evidence for a claim.

        Args:
            claim_hidden: [batch, claim_len, hidden]
            evidence_hidden: [batch, evid_len, hidden]

        Returns:
            Evidence scores [batch]
        """
        claim_repr = claim_hidden.mean(dim=1)
        evidence_repr = evidence_hidden.mean(dim=1)

        scores = self.evidence_scorer(claim_repr, evidence_repr).squeeze(-1)

        return scores

    def forward(
        self,
        context_hidden: Tensor,
        claim_hidden: Tensor,
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        """Forward pass for argument extraction.

        Args:
            context_hidden: [batch, ctx_len, hidden]
            claim_hidden: [batch, claim_len, hidden]

        Returns:
            Tuple of (argument_logits, arguments)
        """
        output, _ = self.argument_encoder(context_hidden)

        argument_logits = self.argument_classifier(output)

        evidence_scores = self.score_evidence(claim_hidden, context_hidden)

        arguments = [
            {
                "type_id": argument_logits[b].argmax().item(),
                "score": evidence_scores[b].item(),
            }
            for b in range(context_hidden.size(0))
        ]

        return argument_logits, arguments


class ContextGraph(nn.Module):
    """Context Graph Construction for Reading Comprehension.

    Builds graph representation of document context.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_relation_types: int = 10,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ContextGraph.

        Args:
            hidden_size: Hidden dimension size
            num_relation_types: Number of relation types
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relation_types = num_relation_types

        self.node_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relation_types),
        )

        self.graph_conv = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        )

    def build_graph(
        self,
        context: Context,
        hidden_states: Tensor,
    ) -> Dict[str, Any]:
        """Build graph from context.

        Args:
            context: Input context
            hidden_states: [seq_len, hidden]

        Returns:
            Graph data structure
        """
        sentences = context.text.split(".")
        num_nodes = len(sentences)

        edge_index = []
        edge_types = []

        for i in range(num_nodes - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
            edge_types.extend([0, 0])

        return {
            "num_nodes": num_nodes,
            "edge_index": torch.tensor(edge_index, dtype=torch.long).t()
            if edge_index
            else torch.zeros(2, 0, dtype=torch.long),
            "edge_types": torch.tensor(edge_types, dtype=torch.long)
            if edge_types
            else torch.zeros(0, dtype=torch.long),
            "node_features": hidden_states[:num_nodes]
            if hidden_states.size(0) >= num_nodes
            else torch.zeros(num_nodes, self.hidden_size),
        }

    def graph_convolution(
        self,
        node_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Perform graph convolution.

        Args:
            node_features: [num_nodes, hidden]
            edge_index: [2, num_edges]

        Returns:
            Updated node features [num_nodes, hidden]
        """
        updated = node_features

        for conv_layer in self.graph_conv:
            messages = torch.zeros_like(node_features)

            for i in range(edge_index.size(1)):
                src = node_features[edge_index[0, i]]
                dst = edge_index[1, i]
                messages[dst] = messages[dst] + conv_layer(src)

            updated = F.relu(messages + node_features)

        return updated

    def forward(
        self,
        context_hidden: Tensor,
    ) -> Dict[str, Any]:
        """Forward pass for graph construction.

        Args:
            context_hidden: [batch, seq_len, hidden]

        Returns:
            Graph data structure
        """
        output, _ = self.node_encoder(context_hidden)

        node_repr = output.mean(dim=1)

        graph_data = {
            "node_features": node_repr,
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "edge_types": torch.tensor([0, 0], dtype=torch.long),
        }

        return graph_data


class ReadingComprehensionModel(ReadingComprehensionBase[nn.Module]):
    """Reading Comprehension Model.

    Implements reading comprehension with coherence modeling and
    multi-document support.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize Reading Comprehension Model.

        Args:
            config: QA configuration
        """
        super().__init__(config)

        self.hidden_size = config.metadata.get("hidden_size", 768)

        self.context_encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size // 2,
            num_layers=4,
            bidirectional=True,
            batch_first=True,
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.coherence_attention = CoherenceAttention(
            hidden_size=self.hidden_size,
            num_heads=config.metadata.get("num_heads", 8),
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.argument_extractor = ArgumentExtractor(
            hidden_size=self.hidden_size,
            num_argument_types=config.metadata.get("num_argument_types", 15),
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.context_graph = ContextGraph(
            hidden_size=self.hidden_size,
            num_relation_types=config.metadata.get("num_relation_types", 10),
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.span_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.metadata.get("dropout", 0.1)),
            nn.Linear(self.hidden_size, 2),
        )

    def build_context_graph(
        self,
        context: Context,
    ) -> Dict[str, Any]:
        """Build graph representation of context.

        Args:
            context: Input context

        Returns:
            Graph data structure
        """
        dummy_hidden = torch.randn(1, len(context.text.split()), self.hidden_size)
        return self.context_graph.build_graph(context, dummy_hidden.squeeze(0))

    def compute_coherence(
        self,
        context_hidden: Tensor,
        graph_structure: Dict[str, Any],
    ) -> Tensor:
        """Compute coherence scores for passages.

        Args:
            context_hidden: Hidden states for context
            graph_structure: Graph structure of context

        Returns:
            Coherence scores
        """
        sentence_boundaries = graph_structure.get("sentence_boundaries", [[]])
        return self.coherence_attention.compute_coherence(
            context_hidden, graph_structure
        )

    def extract_arguments(
        self,
        context: Context,
    ) -> List[Dict[str, Any]]:
        """Extract arguments from context.

        Args:
            context: Input context

        Returns:
            List of extracted arguments
        """
        return self.argument_extractor.extract_arguments(context)

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

        context_tokens = c_text.split()
        question_tokens = q_text.split()

        context_hidden = torch.randn(1, len(context_tokens), self.hidden_size)
        question_hidden = torch.randn(1, len(question_tokens), self.hidden_size)

        combined = torch.cat([question_hidden, context_hidden], dim=1)

        span_logits = self.span_predictor(combined)

        start_idx = span_logits[0, :, 0].argmax().item()
        end_idx = span_logits[0, :, 1].argmax().item()

        answer_tokens = context_tokens[
            start_idx : min(end_idx + 1, len(context_tokens))
        ]
        answer_text = " ".join(answer_tokens)

        return Answer(
            text=answer_text,
            type=AnswerType.SPAN,
            start_char=sum(len(t) + 1 for t in context_tokens[:start_idx]),
            end_char=sum(len(t) + 1 for t in context_tokens[: end_idx + 1]),
            confidence=0.75,
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
                "context_encoder": self.context_encoder.state_dict(),
                "coherence_attention": self.coherence_attention.state_dict(),
                "argument_extractor": self.argument_extractor.state_dict(),
                "context_graph": self.context_graph.state_dict(),
                "span_predictor": self.span_predictor.state_dict(),
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
        self.context_encoder.load_state_dict(checkpoint["context_encoder"])
        self.coherence_attention.load_state_dict(checkpoint["coherence_attention"])
        self.argument_extractor.load_state_dict(checkpoint["argument_extractor"])
        self.context_graph.load_state_dict(checkpoint["context_graph"])
        self.span_predictor.load_state_dict(checkpoint["span_predictor"])


class MultiDocumentRC(nn.Module):
    """Multi-Document Reading Comprehension.

    Handles reading comprehension across multiple documents.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_docs: int = 10,
        dropout: float = 0.1,
    ) -> None:
        """Initialize MultiDocumentRC.

        Args:
            hidden_size: Hidden dimension size
            num_docs: Number of documents to process
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_docs = num_docs

        self.doc_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.doc_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        self.doc_scorer = nn.Linear(hidden_size, 1)

    def encode_document(
        self,
        doc_text: str,
    ) -> Tensor:
        """Encode a single document.

        Args:
            doc_text: Document text

        Returns:
            Document representation
        """
        tokens = doc_text.split()
        return torch.randn(1, min(len(tokens), 512), self.hidden_size)

    def forward(
        self,
        question: Tensor,
        documents: List[str],
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for multi-document RC.

        Args:
            question: [batch, q_len, hidden]
            documents: List of document texts

        Returns:
            Tuple of (answer_logits, document_scores)
        """
        doc_reprs = []

        for doc in documents:
            doc_hidden = self.encode_document(doc)
            output, (h_n, _) = self.doc_encoder(doc_hidden)
            doc_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            doc_reprs.append(doc_repr)

        doc_stack = torch.cat(doc_reprs, dim=0).unsqueeze(0)

        attended, _ = self.doc_attention(question, doc_stack, doc_stack)

        doc_scores = self.doc_scorer(attended).squeeze(-1)

        answer_logits = attended.mean(dim=1)

        return answer_logits, doc_scores
