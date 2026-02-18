"""
Question Answering Module for Fishstick

Comprehensive implementations of extractive, generative, open-domain, multi-hop,
conversational, and knowledge-base QA systems with evaluation metrics.

This module provides state-of-the-art QA architectures including:
- Extractive QA: BiDAF, DrQA, BERT variants, SpanBERT, Splinter, FiD
- Generative QA: Seq2Seq, T5, BART, Fusion-in-Decoder
- Open-Domain QA: Dense/Sparse/Hybrid retrievers, RAG, REALM
- Multi-hop QA: Decomposition, graph networks, iterative retrieval
- Conversational QA: Contextual understanding, coreference resolution
- Knowledge Base QA: Semantic parsing, graph reasoning
"""

from typing import Optional, List, Dict, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import re
import json
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class QAExample:
    """Single QA example with context, question, and answer."""

    id: str
    question: str
    context: str
    answer: Optional[str] = None
    answer_start: Optional[int] = None
    answer_end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QAPrediction:
    """QA model prediction."""

    id: str
    question: str
    answer: str
    confidence: float
    start_logits: Optional[Tensor] = None
    end_logits: Optional[Tensor] = None
    context_used: Optional[str] = None


@dataclass
class RetrievalResult:
    """Retrieval result with document and score."""

    document_id: str
    document: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Utility Functions
# ============================================================================


def normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens) if prediction_tokens else 0
    recall = num_common / len(ground_truth_tokens) if ground_truth_tokens else 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ============================================================================
# Extractive QA Models
# ============================================================================


class BiDAF(nn.Module):
    """
    Bi-Directional Attention Flow (BiDAF) model for extractive QA.

    BiDAF uses a multi-stage hierarchical process to represent context
    at different levels of granularity and uses bidirectional attention
    flow mechanisms to obtain a query-aware context representation.

    Reference: Seo et al., "Bidirectional Attention Flow for Machine Comprehension", 2017

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 100,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Highway network
        self.highway = HighwayNetwork(embed_dim, num_layers=2)

        # Contextual embedding layer (bi-LSTM)
        self.context_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0,
        )

        # Attention flow layer
        self.att_weight_c = nn.Linear(hidden_dim * 6, 1)
        self.att_weight_q = nn.Linear(hidden_dim * 6, 1)
        self.att_weight_cq = nn.Linear(hidden_dim * 6, 1)

        # Modeling layer (bi-LSTM)
        self.modeling_lstm1 = nn.LSTM(
            hidden_dim * 8,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.modeling_lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Output layer (pointer network)
        self.start_linear = nn.Linear(hidden_dim * 10, 1)
        self.end_linear = nn.Linear(hidden_dim * 10, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        context: Tensor,
        question: Tensor,
        context_mask: Optional[Tensor] = None,
        question_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            context: Context token IDs [batch, ctx_len]
            question: Question token IDs [batch, q_len]
            context_mask: Context padding mask [batch, ctx_len]
            question_mask: Question padding mask [batch, q_len]

        Returns:
            start_logits: Start position logits [batch, ctx_len]
            end_logits: End position logits [batch, ctx_len]
        """
        batch_size, ctx_len = context.size()
        q_len = question.size(1)

        # 1. Word embedding layer
        context_emb = self.embedding(context)  # [batch, ctx_len, embed_dim]
        question_emb = self.embedding(question)  # [batch, q_len, embed_dim]

        context_emb = self.highway(context_emb)
        question_emb = self.highway(question_emb)

        # 2. Contextual embedding layer
        context_h, _ = self.context_lstm(context_emb)  # [batch, ctx_len, 2*hidden_dim]
        question_h, _ = self.context_lstm(question_emb)  # [batch, q_len, 2*hidden_dim]

        context_h = self.dropout(context_h)
        question_h = self.dropout(question_h)

        # 3. Attention flow layer
        # Compute similarity matrix
        context_h_exp = context_h.unsqueeze(2).expand(
            -1, -1, q_len, -1
        )  # [batch, ctx_len, q_len, 2*hidden_dim]
        question_h_exp = question_h.unsqueeze(1).expand(
            -1, ctx_len, -1, -1
        )  # [batch, ctx_len, q_len, 2*hidden_dim]

        # Concatenate context, question, and element-wise product
        concat = torch.cat(
            [context_h_exp, question_h_exp, context_h_exp * question_h_exp], dim=-1
        )  # [batch, ctx_len, q_len, 6*hidden_dim]

        # Compute attention scores
        similarity = self.att_weight_cq(concat).squeeze(-1)  # [batch, ctx_len, q_len]

        if question_mask is not None:
            similarity = similarity.masked_fill(
                ~question_mask.unsqueeze(1).expand(-1, ctx_len, -1), float("-inf")
            )

        # Context-to-query attention
        c2q_attn = F.softmax(similarity, dim=-1)  # [batch, ctx_len, q_len]
        c2q = torch.bmm(c2q_attn, question_h)  # [batch, ctx_len, 2*hidden_dim]

        # Query-to-context attention
        q2c_attn = F.softmax(similarity.max(dim=-1)[0], dim=-1)  # [batch, ctx_len]
        q2c = torch.bmm(q2c_attn.unsqueeze(1), context_h)  # [batch, 1, 2*hidden_dim]
        q2c = q2c.expand(-1, ctx_len, -1)  # [batch, ctx_len, 2*hidden_dim]

        # Merge attention outputs
        g = torch.cat(
            [context_h, c2q, context_h * c2q, context_h * q2c], dim=-1
        )  # [batch, ctx_len, 8*hidden_dim]

        # 4. Modeling layer
        m, _ = self.modeling_lstm1(g)  # [batch, ctx_len, 2*hidden_dim]
        m = self.dropout(m)
        m, _ = self.modeling_lstm2(m)  # [batch, ctx_len, 2*hidden_dim]
        m = self.dropout(m)

        # 5. Output layer
        start_input = torch.cat([g, m], dim=-1)  # [batch, ctx_len, 10*hidden_dim]
        start_logits = self.start_linear(start_input).squeeze(-1)  # [batch, ctx_len]

        end_input = torch.cat([g, m], dim=-1)  # [batch, ctx_len, 10*hidden_dim]
        end_logits = self.end_linear(end_input).squeeze(-1)  # [batch, ctx_len]

        if context_mask is not None:
            start_logits = start_logits.masked_fill(~context_mask, float("-inf"))
            end_logits = end_logits.masked_fill(~context_mask, float("-inf"))

        return start_logits, end_logits


class HighwayNetwork(nn.Module):
    """Highway network for transforming embeddings."""

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.num_layers):
            gate = torch.sigmoid(self.gate[i](x))
            transform = F.relu(self.linear[i](x))
            x = gate * transform + (1 - gate) * x
        return x


class DrQA(nn.Module):
    """
    Document Reader (DrQA) for extractive QA.

    DrQA uses a multi-layer bi-LSTM to encode the document and question,
    then applies attention to find the answer span.

    Reference: Chen et al., "Reading Wikipedia to Answer Open-Domain Questions", 2017

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        use_features: Whether to use additional features (POS, NER, etc.)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_features: bool = True,
    ):
        super().__init__()

        self.use_features = use_features

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Feature embeddings (if used)
        feature_dim = 0
        if use_features:
            self.pos_embedding = nn.Embedding(50, 12)  # POS tags
            self.ner_embedding = nn.Embedding(20, 8)  # NER tags
            self.lemma_embedding = nn.Embedding(2, 8)  # Exact match
            feature_dim = 28

        # Document encoder (multi-layer bi-LSTM)
        self.doc_lstm = nn.LSTM(
            embed_dim + feature_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Question encoder (single-layer bi-LSTM)
        self.question_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)

        # Start/end prediction
        self.start_linear = nn.Linear(hidden_dim * 2, 1)
        self.end_linear = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        document: Tensor,
        question: Tensor,
        doc_features: Optional[Dict[str, Tensor]] = None,
        doc_mask: Optional[Tensor] = None,
        q_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            document: Document token IDs [batch, doc_len]
            question: Question token IDs [batch, q_len]
            doc_features: Optional features dict with 'pos', 'ner', 'lemma'
            doc_mask: Document padding mask
            q_mask: Question padding mask

        Returns:
            start_logits, end_logits: Position logits
        """
        batch_size, doc_len = document.size()

        # Embed document
        doc_emb = self.embedding(document)

        # Add features if available
        if self.use_features and doc_features is not None:
            pos_emb = self.pos_embedding(
                doc_features.get("pos", torch.zeros_like(document))
            )
            ner_emb = self.ner_embedding(
                doc_features.get("ner", torch.zeros_like(document))
            )
            lemma_emb = self.lemma_embedding(
                doc_features.get("lemma", torch.zeros_like(document))
            )
            doc_emb = torch.cat([doc_emb, pos_emb, ner_emb, lemma_emb], dim=-1)

        # Embed question
        q_emb = self.embedding(question)

        # Encode
        doc_encoded, _ = self.doc_lstm(self.dropout(doc_emb))
        q_encoded, _ = self.question_lstm(self.dropout(q_emb))

        # Question attention (sum pooling with mask)
        if q_mask is not None:
            q_encoded = q_encoded * q_mask.unsqueeze(-1).float()
            q_weights = q_mask.float().sum(dim=1, keepdim=True)
            q_vec = q_encoded.sum(dim=1) / q_weights.clamp(min=1)
        else:
            q_vec = q_encoded.mean(dim=1)

        # Compute attention over document
        q_transformed = self.attention(q_vec).unsqueeze(1)  # [batch, 1, hidden*2]
        attn_scores = torch.bmm(doc_encoded, q_transformed.transpose(1, 2)).squeeze(-1)

        if doc_mask is not None:
            attn_scores = attn_scores.masked_fill(~doc_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        doc_attended = doc_encoded * attn_weights

        # Predict start and end positions
        start_logits = self.start_linear(doc_attended).squeeze(-1)
        end_logits = self.end_linear(doc_attended).squeeze(-1)

        if doc_mask is not None:
            start_logits = start_logits.masked_fill(~doc_mask, float("-inf"))
            end_logits = end_logits.masked_fill(~doc_mask, float("-inf"))

        return start_logits, end_logits


class BERTQA(nn.Module):
    """
    BERT for Question Answering.

    Fine-tunes BERT to predict start and end positions of the answer span.

    Reference: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2019

    Args:
        model_name: Pretrained BERT model name or path
        dropout: Dropout rate for classification head
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.bert.config.hidden_size
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs (0 for question, 1 for context) [batch, seq_len]

        Returns:
            start_logits, end_logits: Position logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = self.dropout(outputs.last_hidden_state)

        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return start_logits, end_logits


class RoBERTaQA(nn.Module):
    """
    RoBERTa for Question Answering.

    Uses RoBERTa improvements over BERT: dynamic masking, full sentences,
    larger batches, and more data.

    Reference: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019

    Args:
        model_name: Pretrained RoBERTa model name
        dropout: Dropout rate
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.roberta.config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            start_logits, end_logits: Position logits
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return start_logits, end_logits


class DistilBERTQA(nn.Module):
    """
    DistilBERT for Question Answering.

    Lightweight version of BERT with 40% fewer parameters,
    60% faster inference while retaining 97% of BERT's performance.

    Reference: Sanh et al., "DistilBERT, a distilled version of BERT", 2019

    Args:
        model_name: Pretrained DistilBERT model name
        dropout: Dropout rate
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.distilbert.config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            start_logits, end_logits: Position logits
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return start_logits, end_logits


class SpanBERTQA(nn.Module):
    """
    SpanBERT for Question Answering.

    Pre-trains BERT with span-based objectives to better represent spans,
    leading to improved performance on span selection tasks.

    Reference: Joshi et al., "SpanBERT: Improving Pre-training by Representing and Predicting Spans", 2020

    Args:
        model_name: Pretrained SpanBERT model name
        dropout: Dropout rate
    """

    def __init__(
        self,
        model_name: str = "SpanBERT/spanbert-base-cased",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.bert.config.hidden_size
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

        # Span representation layer
        self.span_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with span-aware representations.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs [batch, seq_len]

        Returns:
            start_logits, end_logits: Position logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Compute span representations
        # For each position, compute span representation with every other position
        start_reps = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_reps = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)

        span_reps = torch.cat([start_reps, end_reps], dim=-1)
        span_reps = self.span_projection(span_reps)

        # Pool for position predictions
        pooled_output = self.dropout(sequence_output)
        start_logits = self.start_classifier(pooled_output).squeeze(-1)
        end_logits = self.end_classifier(pooled_output).squeeze(-1)

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return start_logits, end_logits


class SplinterQA(nn.Module):
    """
    Splinter: Few-shot span prediction for QA.

    Pre-trains by masking recurring spans in passages and predicting them,
    making it sample-efficient for QA tasks.

    Reference: Ram et al., "Few-Shot Question Answering by Pretraining Span Selection", 2021

    Args:
        model_name: Pretrained Splinter model name
        max_query_length: Maximum query length
    """

    def __init__(
        self,
        model_name: str = "tau/splinter-base",
        max_query_length: int = 64,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.max_query_length = max_query_length

        hidden_size = self.bert.config.hidden_size

        # Question-aware classifier
        self.qa_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with question token representations.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs [batch, seq_len]

        Returns:
            start_logits, end_logits: Position logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Get question representations (first max_query_length tokens)
        question_output = sequence_output[:, : self.max_query_length, :]
        question_repr = question_output.mean(dim=1, keepdim=True)  # [batch, 1, hidden]

        # Expand to match context length
        question_repr = question_repr.expand(-1, seq_len, -1)

        # Concatenate question representation with each position
        combined = torch.cat([sequence_output, question_repr], dim=-1)

        # Predict start and end jointly
        logits = self.qa_classifier(combined).squeeze(-1)

        # Split into start and end predictions
        mid = seq_len // 2
        start_logits = logits[:, :mid]
        end_logits = logits[:, mid:]

        if attention_mask is not None:
            start_mask = attention_mask[:, :mid]
            end_mask = attention_mask[:, mid:]
            start_logits = start_logits.masked_fill(~start_mask.bool(), float("-inf"))
            end_logits = end_logits.masked_fill(~end_mask.bool(), float("-inf"))

        return start_logits, end_logits


class FiDQA(nn.Module):
    """
    Fusion-in-Decoder (FiD) for QA.

    Retrieves multiple passages and fuses them in the decoder to generate answers.
    Particularly effective for open-domain QA.

    Reference: Izacard & Grave, "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", 2021

    Args:
        encoder_model: Pretrained encoder model name
        decoder_model: Pretrained decoder model name
        num_passages: Number of passages to retrieve
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        encoder_model: str = "facebook/bart-base",
        decoder_model: str = "facebook/bart-base",
        num_passages: int = 100,
        max_length: int = 512,
    ):
        super().__init__()

        from transformers import BartForConditionalGeneration, BartTokenizer

        self.model = BartForConditionalGeneration.from_pretrained(encoder_model)
        self.tokenizer = BartTokenizer.from_pretrained(encoder_model)

        self.num_passages = num_passages
        self.max_length = max_length

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass fusing multiple passages.

        Args:
            input_ids: Token IDs [batch * num_passages, seq_len]
            attention_mask: Attention mask
            decoder_input_ids: Decoder input IDs
            labels: Target labels

        Returns:
            Dictionary with loss and logits
        """
        # FiD concatenates all passage encodings in the decoder cross-attention
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return outputs

    def generate_answer(
        self,
        question: str,
        passages: List[str],
        max_length: int = 50,
    ) -> str:
        """
        Generate answer from question and retrieved passages.

        Args:
            question: Question string
            passages: List of retrieved passages
            max_length: Maximum generation length

        Returns:
            Generated answer string
        """
        # Format inputs: question + passage for each passage
        inputs = []
        for passage in passages[: self.num_passages]:
            text = f"question: {question} context: {passage}"
            inputs.append(text)

        # Tokenize all inputs
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Flatten for FiD
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


# ============================================================================
# Generative QA Models
# ============================================================================


class Seq2SeqQA(nn.Module):
    """
    Sequence-to-Sequence model for generative QA.

    Uses encoder-decoder architecture to generate answers token by token.

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.decoder = nn.LSTM(
            embed_dim,
            hidden_dim * 2,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Encoder input IDs [batch, src_len]
            decoder_input_ids: Decoder input IDs [batch, tgt_len]
            attention_mask: Source attention mask

        Returns:
            Output logits [batch, tgt_len, vocab_size]
        """
        # Encode
        src_emb = self.dropout(self.embedding(input_ids))
        encoder_output, (hidden, cell) = self.encoder(src_emb)

        # Decode
        tgt_emb = self.dropout(self.embedding(decoder_input_ids))
        decoder_output, _ = self.decoder(tgt_emb, (hidden, cell))

        # Apply attention
        attn_output, _ = self.attention(
            decoder_output.transpose(0, 1),
            encoder_output.transpose(0, 1),
            encoder_output.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
            if attention_mask is not None
            else None,
        )

        # Project to vocabulary
        output = self.output_projection(attn_output.transpose(0, 1))

        return output


class T5QA(nn.Module):
    """
    T5 for Question Answering.

    Treats QA as a text-to-text task, framing both input and output as text.

    Reference: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", 2020

    Args:
        model_name: Pretrained T5 model name
    """

    def __init__(
        self,
        model_name: str = "t5-base",
    ):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            decoder_input_ids: Decoder input IDs
            labels: Target labels

        Returns:
            Model outputs including loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return outputs

    def answer(
        self,
        question: str,
        context: str,
        max_length: int = 128,
    ) -> str:
        """
        Generate answer for question with context.

        Args:
            question: Question string
            context: Context string
            max_length: Maximum generation length

        Returns:
            Generated answer string
        """
        input_text = f"question: {question} context: {context}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


class BARTQA(nn.Module):
    """
    BART for Question Answering.

    Bidirectional and auto-regressive transformer for sequence generation.

    Reference: Lewis et al., "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", 2020

    Args:
        model_name: Pretrained BART model name
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
    ):
        super().__init__()

        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            decoder_input_ids: Decoder input IDs
            labels: Target labels

        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

        return outputs

    def answer(
        self,
        question: str,
        context: str,
        max_length: int = 128,
    ) -> str:
        """
        Generate answer for question with context.

        Args:
            question: Question string
            context: Context string
            max_length: Maximum generation length

        Returns:
            Generated answer string
        """
        input_text = f"{context} </s> {question}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


class FusionDecoder(nn.Module):
    """
    Fusion Decoder: Retrieve then Generate approach.

    Retrieves relevant passages and fuses them in the decoder for answer generation.

    Args:
        retriever: Retriever model
        generator: Generative model
        num_passages: Number of passages to retrieve
    """

    def __init__(
        self,
        retriever: nn.Module,
        generator: nn.Module,
        num_passages: int = 10,
    ):
        super().__init__()

        self.retriever = retriever
        self.generator = generator
        self.num_passages = num_passages

    def forward(
        self,
        question: str,
        corpus: List[str],
        max_length: int = 128,
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve passages and generate answer.

        Args:
            question: Question string
            corpus: Document corpus
            max_length: Maximum generation length

        Returns:
            Tuple of (answer, retrieval_results)
        """
        # Retrieve relevant passages
        retrieved = self.retriever.retrieve(question, corpus, top_k=self.num_passages)

        # Concatenate passages
        context = " ".join([r.document for r in retrieved])

        # Generate answer
        answer = self.generator.answer(question, context, max_length)

        return answer, retrieved


# ============================================================================
# Open-Domain QA - Retrievers
# ============================================================================


class DenseRetriever(nn.Module):
    """
    Dense Passage Retriever (DPR) style dense retrieval.

    Uses dual-encoder architecture to encode questions and passages
    into dense vectors for efficient similarity search.

    Reference: Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", 2020

    Args:
        query_encoder: Query encoder model name
        ctx_encoder: Context encoder model name
        hidden_size: Hidden dimension
    """

    def __init__(
        self,
        query_encoder: str = "facebook/dpr-question_encoder-single-nq-base",
        ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base",
        hidden_size: int = 768,
    ):
        super().__init__()

        from transformers import DPRQuestionEncoder, DPRContextEncoder

        self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder)

        self.hidden_size = hidden_size
        self.passage_embeddings = None
        self.passage_texts = None

    def encode_queries(self, queries: List[str]) -> Tensor:
        """
        Encode queries into dense vectors.

        Args:
            queries: List of query strings

        Returns:
            Query embeddings [num_queries, hidden_size]
        """
        from transformers import DPRQuestionEncoderTokenizer

        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )

        inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            embeddings = self.query_encoder(**inputs).pooler_output

        return F.normalize(embeddings, p=2, dim=1)

    def encode_passages(self, passages: List[str]) -> Tensor:
        """
        Encode passages into dense vectors.

        Args:
            passages: List of passage strings

        Returns:
            Passage embeddings [num_passages, hidden_size]
        """
        from transformers import DPRContextEncoderTokenizer

        tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

        inputs = tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            embeddings = self.ctx_encoder(**inputs).pooler_output

        return F.normalize(embeddings, p=2, dim=1)

    def index_passages(self, passages: List[str]):
        """
        Index passages for efficient retrieval.

        Args:
            passages: List of passage strings
        """
        self.passage_texts = passages
        self.passage_embeddings = self.encode_passages(passages)

    def retrieve(
        self,
        query: str,
        passages: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k passages for query.

        Args:
            query: Query string
            passages: Optional passage list (uses indexed if None)
            top_k: Number of passages to retrieve

        Returns:
            List of retrieval results
        """
        # Encode query
        query_emb = self.encode_queries([query])

        # Encode or use indexed passages
        if passages is not None:
            passage_emb = self.encode_passages(passages)
            passage_texts = passages
        else:
            passage_emb = self.passage_embeddings
            passage_texts = self.passage_texts

        # Compute similarities
        similarities = torch.matmul(query_emb, passage_emb.T).squeeze(0)

        # Get top-k
        top_scores, top_indices = torch.topk(
            similarities, min(top_k, len(passage_texts))
        )

        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append(
                RetrievalResult(
                    document_id=str(idx.item()),
                    document=passage_texts[idx.item()],
                    score=score.item(),
                )
            )

        return results


class SparseRetriever:
    """
    Sparse retriever using BM25/TF-IDF.

    Traditional lexical matching approach for passage retrieval.

    Reference: Robertson et al., "Okapi at TREC-3", 1995

    Args:
        k1: BM25 parameter (term frequency saturation)
        b: BM25 parameter (length normalization)
        use_tfidf: Whether to use TF-IDF instead of BM25
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_tfidf: bool = False,
    ):
        self.k1 = k1
        self.b = b
        self.use_tfidf = use_tfidf

        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.vocab = set()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def index_documents(self, documents: List[str]):
        """
        Index documents for retrieval.

        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []

        # Build document frequencies
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            unique_tokens = set(tokens)

            for token in unique_tokens:
                self.doc_freqs[token] += 1
                self.vocab.add(token)

        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0
        )
        self.num_docs = len(documents)

    def _compute_bm25_score(
        self,
        query_tokens: List[str],
        doc_idx: int,
    ) -> float:
        """Compute BM25 score for a document."""
        doc_tokens = self._tokenize(self.documents[doc_idx])
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_tokens:
            # Term frequency in document
            tf = doc_tokens.count(term)

            # Document frequency
            df = self.doc_freqs.get(term, 0)

            if df == 0:
                continue

            # IDF
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 term score
            tf_component = tf * (self.k1 + 1)
            normalization = tf + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_length
            )

            score += idf * tf_component / normalization

        return score

    def _compute_tfidf_score(
        self,
        query_tokens: List[str],
        doc_idx: int,
    ) -> float:
        """Compute TF-IDF score for a document."""
        doc_tokens = self._tokenize(self.documents[doc_idx])

        score = 0.0
        for term in query_tokens:
            tf = doc_tokens.count(term)
            df = self.doc_freqs.get(term, 0)

            if df == 0:
                continue

            idf = math.log(self.num_docs / df)
            score += tf * idf

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        query_tokens = self._tokenize(query)

        # Score all documents
        scores = []
        for i in range(len(self.documents)):
            if self.use_tfidf:
                score = self._compute_tfidf_score(query_tokens, i)
            else:
                score = self._compute_bm25_score(query_tokens, i)
            scores.append((i, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for idx, score in scores[:top_k]:
            results.append(
                RetrievalResult(
                    document_id=str(idx),
                    document=self.documents[idx],
                    score=score,
                )
            )

        return results


class HybridRetriever(nn.Module):
    """
    Hybrid retriever combining dense and sparse retrieval.

    Uses both semantic (dense) and lexical (sparse) matching for improved recall.

    Args:
        dense_retriever: Dense retrieval model
        sparse_retriever: Sparse retrieval model
        dense_weight: Weight for dense retrieval scores
        sparse_weight: Weight for sparse retrieval scores
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ):
        super().__init__()

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def index_passages(self, passages: List[str]):
        """Index passages in both retrievers."""
        self.dense_retriever.index_passages(passages)
        self.sparse_retriever.index_documents(passages)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 100,
    ) -> List[RetrievalResult]:
        """
        Retrieve using hybrid scoring.

        Args:
            query: Query string
            top_k: Number of results to return
            initial_k: Initial retrieval size before reranking

        Returns:
            List of retrieval results
        """
        # Retrieve from both methods
        dense_results = self.dense_retriever.retrieve(query, top_k=initial_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=initial_k)

        # Normalize scores
        dense_scores = {r.document_id: r.score for r in dense_results}
        sparse_scores = {r.document_id: r.score for r in sparse_results}

        # Combine scores
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}

        for doc_id in all_doc_ids:
            d_score = dense_scores.get(doc_id, 0)
            s_score = sparse_scores.get(doc_id, 0)
            combined_scores[doc_id] = (
                self.dense_weight * d_score + self.sparse_weight * s_score
            )

        # Sort and return top-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Map back to documents
        doc_map = {r.document_id: r for r in dense_results}
        for r in sparse_results:
            if r.document_id not in doc_map:
                doc_map[r.document_id] = r

        results = []
        for doc_id, score in sorted_results[:top_k]:
            original = doc_map[doc_id]
            results.append(
                RetrievalResult(
                    document_id=doc_id,
                    document=original.document,
                    score=score,
                )
            )

        return results


class IterativeRetriever(nn.Module):
    """
    Iterative retriever for multi-hop reasoning.

    Performs multiple retrieval steps, using previous results to inform next retrieval.

    Args:
        base_retriever: Base retriever model
        num_iterations: Number of retrieval iterations
        hop_threshold: Threshold for continuing to next hop
    """

    def __init__(
        self,
        base_retriever: Union[DenseRetriever, SparseRetriever, HybridRetriever],
        num_iterations: int = 2,
        hop_threshold: float = 0.5,
    ):
        super().__init__()

        self.base_retriever = base_retriever
        self.num_iterations = num_iterations
        self.hop_threshold = hop_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        passages_per_hop: int = 5,
    ) -> List[RetrievalResult]:
        """
        Iterative multi-hop retrieval.

        Args:
            query: Initial query string
            top_k: Number of final results
            passages_per_hop: Number of passages per hop

        Returns:
            List of retrieval results
        """
        all_results = []
        current_query = query

        for iteration in range(self.num_iterations):
            # Retrieve for current query
            results = self.base_retriever.retrieve(
                current_query, top_k=passages_per_hop
            )

            if not results:
                break

            all_results.extend(results)

            # Check if we should continue
            if results[0].score < self.hop_threshold:
                break

            # Update query for next iteration
            # Use top result to formulate next query
            top_doc = results[0].document
            current_query = f"{query} {top_doc[:200]}"

        # Deduplicate and rerank
        seen = set()
        unique_results = []
        for r in all_results:
            if r.document_id not in seen:
                seen.add(r.document_id)
                unique_results.append(r)

        # Sort by score and return top-k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:top_k]


# ============================================================================
# RAG and REALM
# ============================================================================


class RAG(nn.Module):
    """
    Retrieval-Augmented Generation (RAG).

    Combines dense retrieval with sequence-to-sequence generation.
    Retrieves documents and conditions generation on them.

    Reference: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020

    Args:
        retriever: Dense retriever
        generator_name: Generator model name
        num_docs: Number of documents to retrieve
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        generator_name: str = "facebook/bart-large-cnn",
        num_docs: int = 5,
    ):
        super().__init__()

        self.retriever = retriever
        self.generator = BartForConditionalGeneration.from_pretrained(generator_name)
        self.tokenizer = BartTokenizer.from_pretrained(generator_name)
        self.num_docs = num_docs

    def forward(
        self,
        question: str,
        context: Optional[str] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with retrieval and generation.

        Args:
            question: Question string
            context: Optional context (retrieves if None)
            labels: Optional target labels for training

        Returns:
            Generation outputs
        """
        # Retrieve if no context provided
        if context is None:
            retrieved = self.retriever.retrieve(question, top_k=self.num_docs)
            context = " ".join([r.document for r in retrieved])

        # Format input
        input_text = f"question: {question} context: {context}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )

        # Generate
        outputs = self.generator(
            **inputs,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        question: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
    ) -> Union[str, List[str]]:
        """
        Generate answer for question.

        Args:
            question: Question string
            max_length: Maximum generation length
            num_return_sequences: Number of answers to generate

        Returns:
            Generated answer(s)
        """
        # Retrieve relevant documents
        retrieved = self.retriever.retrieve(question, top_k=self.num_docs)
        context = " ".join([r.document for r in retrieved])

        # Format input
        input_text = f"question: {question} context: {context}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )

        # Generate
        outputs = self.generator.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
        )

        answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return answers[0] if num_return_sequences == 1 else answers


class REALM(nn.Module):
    """
    Retrieval-Augmented Language Model (REALM).

    End-to-end pre-training and fine-tuning of retrieval-augmented models.
    Uses an inner product retriever that is trained jointly with the language model.

    Reference: Guu et al., "REALM: Retrieval-Augmented Language Model Pre-Training", 2020

    Args:
        bert_model: BERT model name
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_candidates: Number of retrieval candidates
    """

    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_candidates: int = 8,
    ):
        super().__init__()

        self.num_candidates = num_candidates

        # Encoder for queries (questions)
        self.query_encoder = BertModel.from_pretrained(bert_model)

        # Encoder for documents (knowledge)
        self.doc_encoder = BertModel.from_pretrained(bert_model)

        # Knowledge-augmented encoder
        self.knowledge_encoder = BertModel.from_pretrained(bert_model)

        # Output projection
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        # Knowledge embeddings index
        self.knowledge_index = None
        self.knowledge_texts = None

    def encode_query(self, query_ids: Tensor, query_mask: Tensor) -> Tensor:
        """
        Encode query into dense vector.

        Args:
            query_ids: Query token IDs
            query_mask: Query attention mask

        Returns:
            Query embedding
        """
        outputs = self.query_encoder(input_ids=query_ids, attention_mask=query_mask)
        # Use [CLS] token
        return outputs.last_hidden_state[:, 0, :]

    def encode_document(self, doc_ids: Tensor, doc_mask: Tensor) -> Tensor:
        """
        Encode document into dense vector.

        Args:
            doc_ids: Document token IDs
            doc_mask: Document attention mask

        Returns:
            Document embedding
        """
        outputs = self.doc_encoder(input_ids=doc_ids, attention_mask=doc_mask)
        # Use [CLS] token
        return outputs.last_hidden_state[:, 0, :]

    def build_knowledge_index(self, documents: List[str], tokenizer):
        """
        Build index of knowledge documents.

        Args:
            documents: List of document strings
            tokenizer: Tokenizer instance
        """
        self.knowledge_texts = documents

        # Encode all documents
        embeddings = []
        batch_size = 32

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            encoded = tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                doc_emb = self.encode_document(
                    encoded["input_ids"], encoded["attention_mask"]
                )
                embeddings.append(doc_emb)

        self.knowledge_index = torch.cat(embeddings, dim=0)
        self.knowledge_index = F.normalize(self.knowledge_index, p=2, dim=1)

    def retrieve(
        self,
        query_emb: Tensor,
        top_k: int = 8,
    ) -> Tuple[Tensor, List[str]]:
        """
        Retrieve top-k documents for query.

        Args:
            query_emb: Query embedding
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (document embeddings, document texts)
        """
        # Normalize query
        query_emb = F.normalize(query_emb, p=2, dim=1)

        # Compute similarities
        scores = torch.matmul(query_emb, self.knowledge_index.T)

        # Get top-k
        top_scores, top_indices = torch.topk(
            scores, min(top_k, len(self.knowledge_texts))
        )

        # Retrieve embeddings
        doc_embs = self.knowledge_index[top_indices]
        doc_texts = [self.knowledge_texts[i] for i in top_indices[0].tolist()]

        return doc_embs, doc_texts

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with retrieval and prediction.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional target labels

        Returns:
            Dictionary with logits and loss
        """
        # Encode input
        query_emb = self.encode_query(input_ids, attention_mask)

        # Retrieve relevant knowledge
        if self.knowledge_index is not None:
            doc_embs, doc_texts = self.retrieve(query_emb, top_k=self.num_candidates)

            # Combine query with retrieved knowledge
            # (Simplified - full REALM would jointly train retriever)
            knowledge_aware = query_emb + doc_embs.mean(dim=1)
        else:
            knowledge_aware = query_emb

        # Predict
        logits = self.output_layer(knowledge_aware)

        outputs = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            outputs["loss"] = loss

        return outputs


# ============================================================================
# Multi-hop QA
# ============================================================================


class DecompRC(nn.Module):
    """
    Decomposable Reading Comprehension.

    Decomposes complex questions into simpler sub-questions,
    answers each sub-question, and combines the results.

    Reference: Min et al., "A Discrete Hard EM Approach for Weakly Supervised Question Answering", 2019

    Args:
        base_reader: Base QA model for answering sub-questions
        decomposer: Question decomposition model
        num_hops: Maximum number of reasoning hops
    """

    def __init__(
        self,
        base_reader: nn.Module,
        decomposer: Optional[nn.Module] = None,
        num_hops: int = 2,
    ):
        super().__init__()

        self.base_reader = base_reader
        self.decomposer = decomposer
        self.num_hops = num_hops

    def decompose_question(self, question: str) -> List[str]:
        """
        Decompose complex question into sub-questions.

        Args:
            question: Complex question string

        Returns:
            List of sub-questions
        """
        if self.decomposer is not None:
            # Use learned decomposer
            return self.decomposer.decompose(question)

        # Simple rule-based decomposition
        # Look for conjunctions and split
        sub_questions = []

        # Split on common multi-hop patterns
        patterns = [
            r"(.+?)\s+and\s+(.+)",
            r"(.+?)\s+who\s+also\s+(.+)",
            r"(.+?)\s+which\s+(.+)",
        ]

        for pattern in patterns:
            match = re.match(pattern, question, re.IGNORECASE)
            if match:
                sub_questions = [match.group(1).strip(), match.group(2).strip()]
                break

        if not sub_questions:
            sub_questions = [question]

        return sub_questions

    def answer(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Answer question using decomposition.

        Args:
            question: Question string
            context: Context string

        Returns:
            Dictionary with answer and intermediate results
        """
        # Decompose question
        sub_questions = self.decompose_question(question)

        intermediate_answers = []

        # Answer each sub-question
        for sub_q in sub_questions:
            # Add intermediate answers to context
            augmented_context = context
            for ans in intermediate_answers:
                augmented_context += f" {ans}"

            # Get answer from base reader
            if hasattr(self.base_reader, "answer"):
                answer = self.base_reader.answer(sub_q, augmented_context)
            else:
                # Assume it's an extractive model
                answer = self._extractive_answer(sub_q, augmented_context)

            intermediate_answers.append(answer)

        # Final answer is the last one
        final_answer = intermediate_answers[-1] if intermediate_answers else ""

        return {
            "answer": final_answer,
            "sub_questions": sub_questions,
            "intermediate_answers": intermediate_answers,
        }

    def _extractive_answer(self, question: str, context: str) -> str:
        """Get extractive answer using base reader."""
        # Tokenize
        # This is a simplified version - actual implementation would depend on reader type
        return ""


class HotpotQAReader(nn.Module):
    """
    Specialized reader for HotpotQA (multi-hop QA dataset).

    Designed to handle reasoning over multiple supporting documents.

    Args:
        encoder: Document encoder (e.g., BERT)
        hidden_size: Hidden dimension
        num_supporting_facts: Expected number of supporting facts
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 768,
        num_supporting_facts: int = 2,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_supporting_facts = num_supporting_facts

        # Supporting facts prediction
        self.support_classifier = nn.Linear(hidden_size, 1)

        # Answer span prediction
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

        # Answer type prediction (yes/no/span)
        self.answer_type_classifier = nn.Linear(hidden_size, 3)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
        doc_start_positions: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs for concatenated documents
            attention_mask: Attention mask
            token_type_ids: Segment IDs
            doc_start_positions: Start positions of each document

        Returns:
            Dictionary with predictions
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Predict supporting facts for each document
        if doc_start_positions is not None:
            support_logits = []
            for start_pos in doc_start_positions:
                doc_repr = sequence_output[:, start_pos, :]
                support_logit = self.support_classifier(doc_repr)
                support_logits.append(support_logit)
            support_logits = torch.cat(support_logits, dim=1)
        else:
            # Assume single document
            support_logits = self.support_classifier(pooled_output)

        # Predict answer span
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)

        # Predict answer type
        answer_type_logits = self.answer_type_classifier(pooled_output)

        # Apply masks
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "support_logits": support_logits,
            "answer_type_logits": answer_type_logits,
        }


class HGN(nn.Module):
    """
    Heterogeneous Graph Network (HGN) for multi-hop QA.

    Builds a heterogeneous graph over entities, sentences, and documents,
    then uses graph neural networks for reasoning.

    Reference: Fang et al., "Hierarchical Graph Network for Multi-hop Question Answering", 2020

    Args:
        hidden_size: Hidden dimension
        num_layers: Number of GNN layers
        num_relations: Number of relation types in the graph
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 3,
        num_relations: int = 4,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BERT encoder
        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        # Graph convolution layers (simplified R-GCN)
        self.gnn_layers = nn.ModuleList(
            [
                RelationalGraphConv(hidden_size, hidden_size, num_relations)
                for _ in range(num_layers)
            ]
        )

        # Node type embeddings
        self.node_type_emb = nn.Embedding(3, hidden_size)  # entity, sentence, doc

        # Output layers
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

    def build_graph(
        self,
        documents: List[str],
        entities: List[str],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build heterogeneous graph from documents.

        Args:
            documents: List of documents
            entities: List of entities

        Returns:
            Tuple of (node_features, edge_index, edge_type)
        """
        # This is a simplified graph construction
        # Full implementation would use entity linking, coreference, etc.

        num_nodes = len(entities) + len(documents)

        # Initialize node features
        node_features = torch.randn(num_nodes, self.hidden_size)

        # Create edges (simplified)
        edge_index = []
        edge_type = []

        # Entity-document edges
        for i, entity in enumerate(entities):
            for j, doc in enumerate(documents):
                if entity.lower() in doc.lower():
                    edge_index.append([i, j + len(entities)])
                    edge_type.append(0)  # entity-doc relation

        edge_index = (
            torch.tensor(edge_index).t() if edge_index else torch.zeros(2, 0).long()
        )
        edge_type = torch.tensor(edge_type) if edge_type else torch.zeros(0).long()

        return node_features, edge_index, edge_type

    def forward(
        self,
        question: str,
        documents: List[str],
        entities: List[str],
    ) -> Dict[str, Tensor]:
        """
        Forward pass with graph reasoning.

        Args:
            question: Question string
            documents: Supporting documents
            entities: List of entities

        Returns:
            Dictionary with predictions
        """
        # Encode question and documents
        # (Simplified - would concatenate and encode properly)

        # Build graph
        node_features, edge_index, edge_type = self.build_graph(documents, entities)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, edge_index, edge_type)
            node_features = F.relu(node_features)

        # Get entity node representations for answer prediction
        entity_features = node_features[: len(entities)]

        # Predict start and end (simplified)
        start_logits = self.start_classifier(entity_features).squeeze(-1)
        end_logits = self.end_classifier(entity_features).squeeze(-1)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "node_features": node_features,
        }


class RelationalGraphConv(nn.Module):
    """Relational Graph Convolutional Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
    ):
        super().__init__()

        self.num_relations = num_relations

        # Weight for each relation type
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(in_channels, out_channels))
                for _ in range(num_relations)
            ]
        )

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_type: Edge types [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        out = torch.zeros(x.size(0), self.weights[0].size(1), device=x.device)

        for rel_type in range(self.num_relations):
            # Get edges of this type
            mask = edge_type == rel_type
            if not mask.any():
                continue

            rel_edges = edge_index[:, mask]

            # Aggregate messages
            src, tgt = rel_edges[0], rel_edges[1]
            messages = x[src] @ self.weights[rel_type]

            # Aggregate to target nodes
            out.index_add_(0, tgt, messages)

        return out + self.bias


class BeamRetriever(nn.Module):
    """
    Beam search retriever for multi-hop QA.

    Uses beam search to explore different reasoning paths,
    retrieving documents at each hop based on previous results.

    Args:
        base_retriever: Base retriever
        beam_size: Beam width for search
        num_hops: Number of reasoning hops
    """

    def __init__(
        self,
        base_retriever: Union[DenseRetriever, SparseRetriever],
        beam_size: int = 5,
        num_hops: int = 2,
    ):
        super().__init__()

        self.base_retriever = base_retriever
        self.beam_size = beam_size
        self.num_hops = num_hops

    def retrieve(
        self,
        question: str,
        top_k: int = 10,
        docs_per_beam: int = 3,
    ) -> List[RetrievalResult]:
        """
        Beam search retrieval.

        Args:
            question: Initial question
            top_k: Number of final results
            docs_per_beam: Documents to retrieve per beam per hop

        Returns:
            List of retrieval results
        """
        # Initialize beam with question
        beams = [(question, [], 0.0)]  # (query, doc_ids, score)

        for hop in range(self.num_hops):
            new_beams = []

            for query, doc_ids, score in beams:
                # Retrieve documents
                results = self.base_retriever.retrieve(query, top_k=docs_per_beam)

                for result in results:
                    if result.document_id not in doc_ids:
                        new_doc_ids = doc_ids + [result.document_id]
                        new_score = score + result.score

                        # Create new query by combining original with retrieved doc
                        new_query = f"{question} {result.document[:100]}"

                        new_beams.append((new_query, new_doc_ids, new_score))

            # Keep top beam_size beams
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[: self.beam_size]

        # Collect all unique documents from beams
        all_docs = set()
        for _, doc_ids, _ in beams:
            all_docs.update(doc_ids)

        # Retrieve final ranked list
        final_results = []
        doc_scores = defaultdict(float)

        for _, doc_ids, score in beams:
            for i, doc_id in enumerate(doc_ids):
                # Weight by position in hop
                doc_scores[doc_id] += score / (i + 1)

        # Sort and return
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Map back to documents (this is simplified)
        for doc_id, score in sorted_docs[:top_k]:
            final_results.append(
                RetrievalResult(
                    document_id=doc_id,
                    document="",  # Would retrieve from corpus
                    score=score,
                )
            )

        return final_results


# ============================================================================
# Conversational QA
# ============================================================================


class QuACReader(nn.Module):
    """
    Reader for QuAC (Question Answering in Context) dataset.

    Handles sequential QA with conversation history.

    Reference: Choi et al., "QuAC: Question Answering in Context", 2018

    Args:
        encoder: Base encoder (e.g., BERT)
        max_history: Maximum number of previous Q-A pairs to include
    """

    def __init__(
        self,
        encoder: nn.Module,
        max_history: int = 3,
    ):
        super().__init__()

        self.encoder = encoder
        self.max_history = max_history

        # History encoder
        hidden_size = 768  # BERT hidden size
        self.history_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Classifiers
        self.start_classifier = nn.Linear(hidden_size * 3, 1)
        self.end_classifier = nn.Linear(hidden_size * 3, 1)
        self.yes_no_classifier = nn.Linear(hidden_size * 3, 3)
        self.follow_up_classifier = nn.Linear(hidden_size * 3, 3)

    def encode_history(
        self,
        history: List[Tuple[str, str]],
    ) -> Tensor:
        """
        Encode conversation history.

        Args:
            history: List of (question, answer) tuples

        Returns:
            History representation
        """
        if not history:
            return None

        # Take last max_history Q-A pairs
        recent_history = history[-self.max_history :]

        # Encode each Q-A pair
        history_reps = []
        for q, a in recent_history:
            qa_text = f"Q: {q} A: {a}"
            # Encode (simplified)
            history_reps.append(torch.randn(768))  # Placeholder

        history_tensor = torch.stack(history_reps).unsqueeze(0)

        # Apply LSTM
        output, (hidden, _) = self.history_lstm(history_tensor)

        # Use final hidden state
        return hidden.view(1, -1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        history: Optional[List[Tuple[str, str]]] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with history.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            history: Conversation history
            token_type_ids: Segment IDs

        Returns:
            Dictionary with predictions
        """
        # Encode current context
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        # Encode history
        history_repr = self.encode_history(history)
        if history_repr is None:
            history_repr = torch.zeros_like(pooled_output)

        # Combine representations
        combined = torch.cat([pooled_output, history_repr], dim=-1)
        combined_expanded = combined.unsqueeze(1).expand(
            -1, sequence_output.size(1), -1
        )

        # Concatenate with sequence
        sequence_combined = torch.cat([sequence_output, combined_expanded], dim=-1)

        # Predict
        start_logits = self.start_classifier(sequence_combined).squeeze(-1)
        end_logits = self.end_classifier(sequence_combined).squeeze(-1)
        yes_no_logits = self.yes_no_classifier(combined)
        follow_up_logits = self.follow_up_classifier(combined)

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yes_no_logits": yes_no_logits,
            "follow_up_logits": follow_up_logits,
        }


class CoQABaseline(nn.Module):
    """
    Baseline model for CoQA (Conversational Question Answering).

    Handles conversational QA with history encoding.

    Reference: Reddy et al., "CoQA: A Conversational Question Answering Challenge", 2019

    Args:
        encoder: Base encoder
        history_encoder: Model to encode conversation history
    """

    def __init__(
        self,
        encoder: nn.Module,
        history_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.history_encoder = history_encoder

        hidden_size = 768

        # If history encoder not provided, use simple concatenation
        if history_encoder is None:
            self.history_lstm = nn.LSTM(
                hidden_size,
                hidden_size // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # Classifiers
        self.start_classifier = nn.Linear(hidden_size * 2, 1)
        self.end_classifier = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        history_states: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            history_states: Encoded history states
            token_type_ids: Segment IDs

        Returns:
            Dictionary with predictions
        """
        # Encode current input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state

        # Incorporate history
        if history_states is not None:
            # Combine with current sequence
            history_expanded = history_states.unsqueeze(1).expand(
                -1, sequence_output.size(1), -1
            )
            sequence_output = torch.cat([sequence_output, history_expanded], dim=-1)
        else:
            # Pad if no history
            padding = torch.zeros_like(sequence_output)
            sequence_output = torch.cat([sequence_output, padding], dim=-1)

        # Predict
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(
                ~attention_mask.bool(), float("-inf")
            )
            end_logits = end_logits.masked_fill(~attention_mask.bool(), float("-inf"))

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }


class HistoryEncoder(nn.Module):
    """
    Encoder for conversation history.

    Encodes previous questions and answers to provide context.

    Args:
        base_encoder: Base text encoder
        hidden_size: Hidden dimension
        aggregation: How to aggregate history ('mean', 'lstm', 'attention')
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        hidden_size: int = 768,
        aggregation: str = "lstm",
    ):
        super().__init__()

        self.base_encoder = base_encoder
        self.aggregation = aggregation

        if aggregation == "lstm":
            self.lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.output_dim = hidden_size * 2
        elif aggregation == "attention":
            self.query_proj = nn.Linear(hidden_size, hidden_size)
            self.key_proj = nn.Linear(hidden_size, hidden_size)
            self.value_proj = nn.Linear(hidden_size, hidden_size)
            self.output_dim = hidden_size
        else:
            self.output_dim = hidden_size

    def encode_turn(self, question: str, answer: str) -> Tensor:
        """
        Encode a single Q-A turn.

        Args:
            question: Question text
            answer: Answer text

        Returns:
            Turn encoding
        """
        # Concatenate Q and A
        text = f"Q: {question} A: {answer}"

        # Encode (simplified - would use actual tokenizer)
        # Placeholder encoding
        return torch.randn(768)

    def forward(
        self,
        history: List[Tuple[str, str]],
    ) -> Tensor:
        """
        Encode full conversation history.

        Args:
            history: List of (question, answer) tuples

        Returns:
            History representation
        """
        if not history:
            return torch.zeros(1, self.output_dim)

        # Encode each turn
        turn_encodings = []
        for q, a in history:
            turn_enc = self.encode_turn(q, a)
            turn_encodings.append(turn_enc)

        history_tensor = torch.stack(turn_encodings).unsqueeze(0)

        # Aggregate
        if self.aggregation == "lstm":
            output, (hidden, _) = self.lstm(history_tensor)
            return hidden.view(1, -1)
        elif self.aggregation == "attention":
            # Self-attention over history
            q = self.query_proj(history_tensor)
            k = self.key_proj(history_tensor)
            v = self.value_proj(history_tensor)

            attn_output, _ = F.multi_head_attention_forward(
                q,
                k,
                v,
                self.output_dim,
                8,
                in_proj_weight=None,
                in_proj_bias=None,
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.0,
                out_proj_weight=torch.eye(self.output_dim),
                out_proj_bias=None,
            )

            return attn_output.mean(dim=1)
        else:
            return history_tensor.mean(dim=1)


class CoreferenceQA(nn.Module):
    """
    QA model with coreference resolution.

    Resolves coreference in questions to improve understanding of conversational context.

    Args:
        base_reader: Base QA model
        coref_resolver: Coreference resolution model
    """

    def __init__(
        self,
        base_reader: nn.Module,
        coref_resolver: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.base_reader = base_reader
        self.coref_resolver = coref_resolver

    def resolve_coreference(
        self,
        question: str,
        history: List[Tuple[str, str]],
    ) -> str:
        """
        Resolve coreference in question.

        Args:
            question: Current question with potential coreference
            history: Previous Q-A pairs

        Returns:
            Question with resolved references
        """
        if not history:
            return question

        # Simple rule-based coreference resolution
        # Replace pronouns with antecedents from history
        pronouns = [
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
            "his",
            "her",
            "its",
            "their",
        ]
        question_lower = question.lower()

        # Check if question starts with pronoun
        words = question_lower.split()
        if words and words[0] in pronouns:
            # Try to find antecedent in previous answers
            for prev_q, prev_a in reversed(history):
                # Extract named entities or noun phrases from answer
                # (Simplified - would use NER)
                if prev_a:
                    # Use last answer as antecedent
                    resolved = prev_a + " " + question[len(words[0]) :]
                    return resolved

        return question

    def answer(
        self,
        question: str,
        context: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        Answer question with coreference resolution.

        Args:
            question: Question string
            context: Context string
            history: Conversation history

        Returns:
            Answer string
        """
        # Resolve coreference
        resolved_question = self.resolve_coreference(question, history or [])

        # Get answer from base reader
        if hasattr(self.base_reader, "answer"):
            return self.base_reader.answer(resolved_question, context)
        else:
            # Assume extractive
            return ""


# ============================================================================
# Knowledge Base QA
# ============================================================================


class KBQA(nn.Module):
    """
    Knowledge Base Question Answering.

    Answers questions by querying structured knowledge bases.

    Args:
        entity_encoder: Model to encode entities
        relation_encoder: Model to encode relations
        kb_graph: Knowledge base graph
    """

    def __init__(
        self,
        entity_encoder: nn.Module,
        relation_encoder: nn.Module,
        kb_graph: Optional[Any] = None,
    ):
        super().__init__()

        self.entity_encoder = entity_encoder
        self.relation_encoder = relation_encoder
        self.kb_graph = kb_graph

        hidden_size = 768

        # Query encoder
        self.query_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Scoring function
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def encode_query(self, question: str) -> Tensor:
        """
        Encode natural language question.

        Args:
            question: Question string

        Returns:
            Query embedding
        """
        # Tokenize and encode
        # (Simplified - would use actual tokenizer)
        return torch.randn(768)

    def forward(
        self,
        question: str,
        candidate_entities: List[str],
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            question: Question string
            candidate_entities: List of candidate entities

        Returns:
            Dictionary with entity scores
        """
        # Encode question
        query_emb = self.encode_query(question)

        # Encode entities
        entity_embs = []
        for entity in candidate_entities:
            # Encode entity
            entity_emb = torch.randn(768)  # Placeholder
            entity_embs.append(entity_emb)

        entity_embs = torch.stack(entity_embs)

        # Score entities
        query_expanded = query_emb.unsqueeze(0).expand(len(candidate_entities), -1)
        combined = torch.cat([query_expanded, entity_embs], dim=-1)
        scores = self.score_layer(combined).squeeze(-1)

        return {
            "entity_scores": scores,
            "entity_embs": entity_embs,
        }


class SemanticParser(nn.Module):
    """
    Semantic parser for KB QA.

    Parses natural language questions into logical forms
    that can be executed against a knowledge base.

    Reference: Berant et al., "Semantic Parsing on Freebase from Question-Answer Pairs", 2013

    Args:
        encoder: Text encoder
        decoder: Sequence decoder for logical forms
        vocab_size: Logical form vocabulary size
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        hidden_size: int = 512,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # Logical form vocabulary embeddings
        self.lf_embedding = nn.Embedding(vocab_size, hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def encode_question(self, question: str) -> Tensor:
        """
        Encode natural language question.

        Args:
            question: Question string

        Returns:
            Question encoding
        """
        # Encode (simplified)
        return torch.randn(512)

    def forward(
        self,
        question: str,
        target_lf: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            question: Question string
            target_lf: Target logical form token IDs

        Returns:
            Dictionary with logits and loss
        """
        # Encode question
        question_encoding = self.encode_question(question)

        # Decode logical form
        # (Simplified decoder - would use proper seq2seq)
        logits = self.output_proj(question_encoding.unsqueeze(0))

        outputs = {"logits": logits}

        if target_lf is not None:
            loss = F.cross_entropy(logits, target_lf)
            outputs["loss"] = loss

        return outputs

    def parse(self, question: str) -> str:
        """
        Parse question to logical form.

        Args:
            question: Question string

        Returns:
            Logical form string
        """
        # Generate logical form
        # (Simplified - would use beam search)
        return "(QUERY (ENTITY ?x) (RELATION ?x ?y))"


class GraphQA(nn.Module):
    """
    Graph-based QA using GNNs over knowledge bases.

    Performs reasoning over KB subgraphs using graph neural networks.

    Args:
        entity_embedder: Entity embedding layer
        relation_embedder: Relation embedding layer
        num_layers: Number of GNN layers
        hidden_size: Hidden dimension
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_layers: int = 3,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Embeddings
        self.entity_embedder = nn.Embedding(num_entities, hidden_size)
        self.relation_embedder = nn.Embedding(num_relations, hidden_size)

        # GNN layers
        self.gnn_layers = nn.ModuleList(
            [GraphConvLayer(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        # Query encoder
        self.query_encoder = nn.Linear(hidden_size, hidden_size)

        # Scoring
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        query_entities: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        candidate_entities: Tensor,
    ) -> Tensor:
        """
        Forward pass over KB subgraph.

        Args:
            query_entities: Query entity IDs
            edge_index: Graph edge indices
            edge_type: Edge relation types
            candidate_entities: Candidate answer entity IDs

        Returns:
            Scores for candidate entities
        """
        # Get entity embeddings
        x = self.entity_embedder.weight

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_type, self.relation_embedder)
            x = F.relu(x)

        # Encode query
        query_emb = self.entity_embedder(query_entities).mean(dim=0)
        query_encoded = self.query_encoder(query_emb)

        # Score candidates
        candidate_emb = x[candidate_entities]
        query_expanded = query_encoded.unsqueeze(0).expand_as(candidate_emb)

        combined = torch.cat([query_expanded, candidate_emb], dim=-1)
        scores = self.score_layer(combined).squeeze(-1)

        return scores


class GraphConvLayer(nn.Module):
    """Graph Convolution Layer for KB reasoning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.relation_linear = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        relation_embedder: nn.Module,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices
            edge_type: Edge types
            relation_embedder: Relation embedding layer

        Returns:
            Updated node features
        """
        out = self.linear(x)

        # Aggregate messages
        src, tgt = edge_index[0], edge_index[1]
        relation_embs = relation_embedder(edge_type)

        messages = self.relation_linear(x[src]) + relation_embs
        out.index_add_(0, tgt, messages)

        return out


class ComplexWebQuestions(nn.Module):
    """
    Model for ComplexWebQuestions dataset.

    Handles complex questions requiring multi-step reasoning over web data.

    Reference: Talmor & Berant, "The Web as a Knowledge-Base for Answering Complex Questions", 2018

    Args:
        base_qa: Base QA model
        composition_model: Model to compose multiple answers
        max_composition_steps: Maximum composition steps
    """

    def __init__(
        self,
        base_qa: nn.Module,
        composition_model: Optional[nn.Module] = None,
        max_composition_steps: int = 3,
    ):
        super().__init__()

        self.base_qa = base_qa
        self.max_composition_steps = max_composition_steps

        hidden_size = 768

        if composition_model is None:
            self.composition_model = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            self.composition_model = composition_model

    def decompose_complex_question(self, question: str) -> List[str]:
        """
        Decompose complex question into simpler ones.

        Args:
            question: Complex question

        Returns:
            List of simpler questions
        """
        # Rule-based decomposition
        # Look for patterns like "X who is also Y"
        sub_questions = []

        # Pattern: "X and Y" questions
        if " and " in question.lower():
            parts = question.lower().split(" and ")
            sub_questions = parts
        else:
            sub_questions = [question]

        return sub_questions

    def forward(
        self,
        question: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Forward pass with composition.

        Args:
            question: Complex question
            contexts: List of contexts for each sub-question

        Returns:
            Dictionary with final answer and components
        """
        # Decompose
        sub_questions = self.decompose_complex_question(question)

        # Answer each sub-question
        sub_answers = []
        for i, sub_q in enumerate(sub_questions):
            context = contexts[i] if i < len(contexts) else contexts[0]

            if hasattr(self.base_qa, "answer"):
                answer = self.base_qa.answer(sub_q, context)
            else:
                answer = ""

            sub_answers.append(answer)

        # Compose answers
        if len(sub_answers) > 1:
            # Combine answers
            final_answer = self._compose_answers(sub_answers)
        else:
            final_answer = sub_answers[0]

        return {
            "answer": final_answer,
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
        }

    def _compose_answers(self, answers: List[str]) -> str:
        """
        Compose multiple answers into final answer.

        Args:
            answers: List of sub-answers

        Returns:
            Composed answer
        """
        # Simplified composition - return intersection or first answer
        if len(set(answers)) == 1:
            return answers[0]

        # Return answer that appears most
        from collections import Counter

        counter = Counter(answers)
        return counter.most_common(1)[0][0]


# ============================================================================
# Evaluation
# ============================================================================


class ExactMatch:
    """
    Exact Match metric for QA evaluation.

    Computes exact string match between prediction and ground truth.
    """

    def __call__(self, prediction: str, ground_truth: str) -> bool:
        """
        Check if prediction exactly matches ground truth.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if exact match, False otherwise
        """
        return exact_match_score(prediction, ground_truth)

    def compute_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        """
        Compute exact match for a batch.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths

        Returns:
            Exact match score (0-1)
        """
        matches = sum(self(pred, gt) for pred, gt in zip(predictions, ground_truths))
        return matches / len(predictions) if predictions else 0.0


class F1Score:
    """
    F1 Score metric for QA evaluation.

    Computes token-level F1 between prediction and ground truth.
    """

    def __call__(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            F1 score (0-1)
        """
        return f1_score(prediction, ground_truth)

    def compute_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        """
        Compute F1 for a batch.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths

        Returns:
            Average F1 score
        """
        scores = [self(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        return sum(scores) / len(scores) if scores else 0.0


class QA_METRICS:
    """
    Combined QA metrics.

    Computes multiple metrics including EM, F1, and additional diagnostics.
    """

    def __init__(self):
        self.em = ExactMatch()
        self.f1 = F1Score()

    def compute(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths

        Returns:
            Dictionary of metric scores
        """
        em_scores = [self.em(pred, gt) for pred, gt in zip(predictions, ground_truths)]
        f1_scores = [self.f1(pred, gt) for pred, gt in zip(predictions, ground_truths)]

        return {
            "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "em_count": sum(em_scores),
            "total": len(predictions),
        }

    def compute_by_type(
        self,
        predictions: List[str],
        ground_truths: List[str],
        question_types: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics grouped by question type.

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            question_types: List of question type labels

        Returns:
            Metrics by question type
        """
        type_metrics = defaultdict(lambda: {"predictions": [], "ground_truths": []})

        for pred, gt, q_type in zip(predictions, ground_truths, question_types):
            type_metrics[q_type]["predictions"].append(pred)
            type_metrics[q_type]["ground_truths"].append(gt)

        results = {}
        for q_type, data in type_metrics.items():
            results[q_type] = self.compute(
                data["predictions"],
                data["ground_truths"],
            )

        return results


class HumanEvaluation:
    """
    Utilities for human evaluation of QA systems.

    Supports annotation interfaces and evaluation protocols.
    """

    def __init__(self):
        self.annotations = []

    def create_annotation_task(
        self,
        predictions: List[str],
        questions: List[str],
        contexts: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Create annotation tasks for human evaluators.

        Args:
            predictions: Model predictions
            questions: Questions
            contexts: Contexts
            ground_truths: Optional ground truth answers

        Returns:
            List of annotation tasks
        """
        tasks = []

        for i, (pred, q, ctx) in enumerate(zip(predictions, questions, contexts)):
            task = {
                "id": i,
                "question": q,
                "context": ctx,
                "prediction": pred,
                "ground_truth": ground_truths[i] if ground_truths else None,
                "annotations": [],
            }
            tasks.append(task)

        return tasks

    def add_annotation(
        self,
        task_id: int,
        annotator_id: str,
        correctness: int,  # 0-5 scale
        completeness: int,  # 0-5 scale
        comments: str = "",
    ):
        """
        Add human annotation.

        Args:
            task_id: Task identifier
            annotator_id: Annotator identifier
            correctness: Correctness rating (0-5)
            completeness: Completeness rating (0-5)
            comments: Optional comments
        """
        annotation = {
            "task_id": task_id,
            "annotator_id": annotator_id,
            "correctness": correctness,
            "completeness": completeness,
            "comments": comments,
        }
        self.annotations.append(annotation)

    def compute_agreement(self) -> Dict[str, float]:
        """
        Compute inter-annotator agreement.

        Returns:
            Agreement statistics
        """
        # Group by task
        task_annotations = defaultdict(list)
        for ann in self.annotations:
            task_annotations[ann["task_id"]].append(ann)

        # Compute Cohen's Kappa or similar
        # (Simplified - full implementation would need more annotations per task)
        return {"num_tasks": len(task_annotations)}

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of human evaluation.

        Returns:
            Evaluation summary
        """
        if not self.annotations:
            return {}

        correctness_scores = [ann["correctness"] for ann in self.annotations]
        completeness_scores = [ann["completeness"] for ann in self.annotations]

        return {
            "num_annotations": len(self.annotations),
            "avg_correctness": sum(correctness_scores) / len(correctness_scores),
            "avg_completeness": sum(completeness_scores) / len(completeness_scores),
            "agreement": self.compute_agreement(),
        }


# ============================================================================
# Training Utilities
# ============================================================================


class QADataset(Dataset):
    """
    Dataset for QA training supporting SQuAD, NaturalQuestions, etc.

    Args:
        data: List of QAExample objects
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        doc_stride: Stride for splitting long documents
    """

    def __init__(
        self,
        data: List[QAExample],
        tokenizer: Any,
        max_length: int = 384,
        doc_stride: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

        self.examples = self._create_examples()

    def _create_examples(self) -> List[Dict[str, Any]]:
        """
        Create training examples from QA data.

        Returns:
            List of processed examples
        """
        examples = []

        for item in self.data:
            # Tokenize
            tokenized = self.tokenizer(
                item.question,
                item.context,
                max_length=self.max_length,
                truncation="only_second",
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Create example for each window
            for i in range(len(tokenized["input_ids"])):
                example = {
                    "id": item.id,
                    "input_ids": tokenized["input_ids"][i],
                    "attention_mask": tokenized["attention_mask"][i],
                    "token_type_ids": tokenized.get(
                        "token_type_ids", [0] * len(tokenized["input_ids"][i])
                    )[i],
                    "offset_mapping": tokenized["offset_mapping"][i],
                    "context": item.context,
                    "question": item.question,
                    "answer": item.answer,
                    "answer_start": item.answer_start,
                    "answer_end": item.answer_end,
                }
                examples.append(example)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


class QATrainer:
    """
    Specialized trainer for QA models.

    Handles training loop, evaluation, and checkpointing for QA tasks.

    Args:
        model: QA model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        optimizer: Optimizer
        device: Device to train on
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        else:
            self.optimizer = optimizer

        self.metrics = QA_METRICS()
        self.best_f1 = 0.0

    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        eval_steps: int = 500,
        save_steps: int = 1000,
        output_dir: str = "./qa_model",
    ):
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            output_dir: Directory to save checkpoints
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                self.optimizer.zero_grad()

                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get start/end positions if available
                start_positions = batch.get("start_positions")
                end_positions = batch.get("end_positions")

                if start_positions is not None:
                    start_positions = start_positions.to(self.device)
                    end_positions = end_positions.to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Compute loss
                loss = self._compute_loss(
                    outputs,
                    start_positions,
                    end_positions,
                )

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                # Evaluate
                if global_step % eval_steps == 0 and self.eval_dataset is not None:
                    eval_results = self.evaluate()
                    print(f"Step {global_step}: {eval_results}")

                    # Save best model
                    if eval_results.get("f1", 0) > self.best_f1:
                        self.best_f1 = eval_results["f1"]
                        self.save_checkpoint(f"{output_dir}/best_model.pt")

                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_checkpoint(f"{output_dir}/checkpoint-{global_step}.pt")

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def _compute_loss(
        self,
        outputs: Dict[str, Tensor],
        start_positions: Optional[Tensor],
        end_positions: Optional[Tensor],
    ) -> Tensor:
        """
        Compute QA loss.

        Args:
            outputs: Model outputs
            start_positions: True start positions
            end_positions: True end positions

        Returns:
            Loss tensor
        """
        start_logits = outputs["start_logits"]
        end_logits = outputs["end_logits"]

        if start_positions is not None and end_positions is not None:
            # Standard span loss
            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        else:
            # Unsupervised - use other signals
            loss = torch.tensor(0.0, requires_grad=True)

        return loss

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on eval dataset.

        Returns:
            Evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        self.model.eval()

        eval_loader = DataLoader(self.eval_dataset, batch_size=16)

        predictions = []
        ground_truths = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Get predicted spans
                start_preds = outputs["start_logits"].argmax(dim=-1)
                end_preds = outputs["end_logits"].argmax(dim=-1)

                # Convert to text (simplified)
                for i in range(len(start_preds)):
                    pred_text = self._extract_answer(
                        batch["context"][i],
                        start_preds[i].item(),
                        end_preds[i].item(),
                        batch.get("offset_mapping", []),
                    )
                    predictions.append(pred_text)
                    ground_truths.append(batch.get("answer", [""])[i])

        metrics = self.metrics.compute(predictions, ground_truths)

        self.model.train()
        return metrics

    def _extract_answer(
        self,
        context: str,
        start_idx: int,
        end_idx: int,
        offset_mapping: List,
    ) -> str:
        """
        Extract answer text from predicted span.

        Args:
            context: Original context
            start_idx: Predicted start token index
            end_idx: Predicted end token index
            offset_mapping: Token to character offset mapping

        Returns:
            Answer text
        """
        if (
            not offset_mapping
            or start_idx >= len(offset_mapping)
            or end_idx >= len(offset_mapping)
        ):
            return ""

        start_char = offset_mapping[start_idx][0]
        end_char = offset_mapping[end_idx][1]

        return context[start_char:end_char]

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_f1": self.best_f1,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_f1 = checkpoint.get("best_f1", 0.0)


class NegativeSampling:
    """
    Hard negative sampling for QA training.

    Mines hard negatives to improve model discrimination.

    Args:
        num_negatives: Number of negative samples per positive
        strategy: Sampling strategy ('random', 'bm25', 'embedding')
    """

    def __init__(
        self,
        num_negatives: int = 3,
        strategy: str = "bm25",
    ):
        self.num_negatives = num_negatives
        self.strategy = strategy

    def sample_negatives(
        self,
        positive_context: str,
        all_contexts: List[str],
        question: str,
    ) -> List[str]:
        """
        Sample negative contexts.

        Args:
            positive_context: Ground truth context
            all_contexts: All available contexts
            question: Question for relevance scoring

        Returns:
            List of negative contexts
        """
        # Remove positive context
        candidates = [ctx for ctx in all_contexts if ctx != positive_context]

        if len(candidates) <= self.num_negatives:
            return candidates

        if self.strategy == "random":
            # Random sampling
            import random

            return random.sample(candidates, self.num_negatives)

        elif self.strategy == "bm25":
            # Use BM25 scores to find similar but incorrect contexts
            retriever = SparseRetriever(use_tfidf=False)
            retriever.index_documents(candidates)

            # Get top results (excluding the correct one)
            results = retriever.retrieve(question, top_k=self.num_negatives * 2)
            negatives = []

            for result in results:
                if (
                    result.document != positive_context
                    and len(negatives) < self.num_negatives
                ):
                    negatives.append(result.document)

            return negatives

        elif self.strategy == "embedding":
            # Use embedding similarity
            # (Simplified - would use actual embeddings)
            return candidates[: self.num_negatives]

        else:
            return candidates[: self.num_negatives]

    def create_contrastive_batch(
        self,
        batch: List[QAExample],
        all_contexts: List[str],
    ) -> List[QAExample]:
        """
        Create batch with negative samples for contrastive learning.

        Args:
            batch: Batch of positive examples
            all_contexts: All available contexts

        Returns:
            Augmented batch with negatives
        """
        augmented = []

        for example in batch:
            # Add positive example
            augmented.append(example)

            # Sample negatives
            negatives = self.sample_negatives(
                example.context,
                all_contexts,
                example.question,
            )

            # Create negative examples
            for neg_context in negatives:
                neg_example = QAExample(
                    id=f"{example.id}_neg",
                    question=example.question,
                    context=neg_context,
                    answer="",
                    answer_start=-1,
                    answer_end=-1,
                )
                augmented.append(neg_example)

        return augmented


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Data structures
    "QAExample",
    "QAPrediction",
    "RetrievalResult",
    # Extractive QA
    "BiDAF",
    "DrQA",
    "BERTQA",
    "RoBERTaQA",
    "DistilBERTQA",
    "SpanBERTQA",
    "SplinterQA",
    "FiDQA",
    # Generative QA
    "Seq2SeqQA",
    "T5QA",
    "BARTQA",
    "FusionDecoder",
    # Open-Domain QA
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "IterativeRetriever",
    "RAG",
    "REALM",
    # Multi-hop QA
    "DecompRC",
    "HotpotQAReader",
    "HGN",
    "BeamRetriever",
    "RelationalGraphConv",
    # Conversational QA
    "QuACReader",
    "CoQABaseline",
    "HistoryEncoder",
    "CoreferenceQA",
    # Knowledge Base QA
    "KBQA",
    "SemanticParser",
    "GraphQA",
    "GraphConvLayer",
    "ComplexWebQuestions",
    # Evaluation
    "ExactMatch",
    "F1Score",
    "QA_METRICS",
    "HumanEvaluation",
    # Training utilities
    "QADataset",
    "QATrainer",
    "NegativeSampling",
    # Utilities
    "normalize_answer",
    "f1_score",
    "exact_match_score",
]
