"""
Comprehensive Sentiment Analysis Module for Fishstick

This module provides state-of-the-art sentiment analysis capabilities including:
- Text classification with various neural architectures
- Aspect-based sentiment analysis (ABSA)
- Multimodal sentiment analysis
- Fine-grained emotion and intent detection
- Social media preprocessing
- Cross-lingual sentiment analysis
- Evaluation and training utilities

Author: Fishstick Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import warnings
import random
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, Sampler


try:
    import emoji

    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BertModel,
        BertTokenizer,
        DistilBertModel,
        DistilBertTokenizer,
        RobertaModel,
        RobertaTokenizer,
        XLNetModel,
        XLNetTokenizer,
        XLMRobertaModel,
        XLMRobertaTokenizer,
        AutoModelForSequenceClassification,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. BERT-based models will not work.")


# ============================================================================
# Type Definitions
# ============================================================================

T = TypeVar("T")
BatchType = Dict[str, Tensor]
SentimentLabel = Literal["negative", "neutral", "positive"]
EmotionLabel = Literal[
    "anger", "joy", "sadness", "fear", "surprise", "neutral", "disgust", "love"
]
AspectLabel = Literal["positive", "neutral", "negative", "conflict"]
StanceLabel = Literal["favor", "against", "none"]


@dataclass
class SentimentOutput:
    """Output container for sentiment predictions."""

    label: str
    confidence: float
    scores: Dict[str, float]


@dataclass
class AspectOutput:
    """Output container for aspect-based sentiment."""

    aspect: str
    sentiment: str
    confidence: float
    opinion_terms: List[str] = field(default_factory=list)


@dataclass
class MultimodalOutput:
    """Output container for multimodal sentiment."""

    sentiment: str
    confidence: float
    text_contribution: float
    modality_weights: Dict[str, float]


# ============================================================================
# Sentiment Classification
# ============================================================================


class BasicLSTM(nn.Module):
    """
    Basic LSTM-based sentiment classifier.

    A foundational recurrent neural network architecture for text classification.
    Uses bidirectional LSTM layers with dropout for regularization.

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension of LSTM layers
        num_layers: Number of LSTM layers
        num_classes: Number of sentiment classes
        dropout: Dropout probability
        pretrained_embeddings: Optional pretrained embedding matrix

    Example:
        >>> model = BasicLSTM(vocab_size=10000, embedding_dim=300, hidden_dim=128)
        >>> logits = model(input_ids, lengths)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[Tensor] = None,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            lengths: Actual lengths of sequences (batch_size,)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        embedded = self.dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)

        return self.fc(hidden)


class AttentionMechanism(nn.Module):
    """Self-attention mechanism for LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply attention to LSTM outputs.

        Args:
            lstm_output: LSTM output of shape (batch_size, seq_len, hidden_dim*2)
            mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Weighted context vector and attention weights
        """
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)

        return context, attention_weights


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with self-attention mechanism.

    Extends BasicLSTM with an attention layer that learns to focus on
    sentiment-bearing words in the text.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        num_classes: Number of sentiment classes
        dropout: Dropout rate

    Example:
        >>> model = BiLSTMAttention(vocab_size=10000, hidden_dim=256)
        >>> logits, attention = model(input_ids, lengths)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = AttentionMechanism(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning logits and attention weights."""
        embedded = self.dropout(self.embedding(input_ids))

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        mask = input_ids != 0
        context, attention_weights = self.attention(output, mask)

        context = self.dropout(context)
        logits = self.fc(context)

        return logits, attention_weights


class CNNText(nn.Module):
    """
    CNN-based text classifier for sentiment analysis.

    Uses multiple convolutional filters of different sizes to capture
    n-gram features, followed by max pooling and fully connected layers.

    Args:
        vocab_size: Vocabulary size
        embedding_dim: Word embedding dimension
        num_filters: Number of filters per kernel size
        filter_sizes: List of kernel sizes
        num_classes: Number of sentiment classes
        dropout: Dropout rate

    Example:
        >>> model = CNNText(vocab_size=10000, filter_sizes=[3, 4, 5])
        >>> logits = model(input_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_filters: int = 100,
        filter_sizes: List[int] = None,
        num_classes: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass."""
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pooled.squeeze(2))

        concatenated = torch.cat(conv_outputs, dim=1)
        concatenated = self.dropout(concatenated)

        return self.fc(concatenated)


class BERTSentiment(nn.Module):
    """
    BERT-based sentiment classifier.

    Fine-tunes a pretrained BERT model for sentiment classification.
    Uses the [CLS] token representation for classification.

    Args:
        model_name: HuggingFace model name or path
        num_classes: Number of sentiment classes
        dropout: Dropout rate
        freeze_bert: Whether to freeze BERT parameters

    Example:
        >>> model = BERTSentiment('bert-base-uncased', num_classes=3)
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for BERT models")

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class RoBERTaSentiment(nn.Module):
    """
    RoBERTa-based sentiment classifier.

    Uses RoBERTa (Robustly optimized BERT approach) for sentiment analysis.
    RoBERTa removes the NSP objective and uses dynamic masking.

    Args:
        model_name: HuggingFace model name
        num_classes: Number of sentiment classes
        dropout: Dropout rate

    Example:
        >>> model = RoBERTaSentiment('roberta-base')
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class XLNetSentiment(nn.Module):
    """
    XLNet-based sentiment classifier.

    Uses XLNet with permutation language modeling for sentiment analysis.
    Better at capturing bidirectional context than autoregressive models.

    Args:
        model_name: HuggingFace model name
        num_classes: Number of sentiment classes
        dropout: Dropout rate

    Example:
        >>> model = XLNetSentiment('xlnet-base-cased')
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "xlnet-base-cased",
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.xlnet = XLNetModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlnet.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


class DistilBERTSentiment(nn.Module):
    """
    DistilBERT-based sentiment classifier (lightweight).

    DistilBERT is a smaller, faster version of BERT (40% smaller, 60% faster)
    while retaining 97% of language understanding capabilities.

    Args:
        model_name: HuggingFace model name
        num_classes: Number of sentiment classes
        dropout: Dropout rate

    Example:
        >>> model = DistilBERTSentiment('distilbert-base-uncased')
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
        self.pre_classifier = nn.Linear(
            self.distilbert.config.hidden_size, self.distilbert.config.hidden_size
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        pooled_output = self.pre_classifier(hidden_state)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


# ============================================================================
# Aspect-Based Sentiment Analysis (ABSA)
# ============================================================================


class AspectExtractor(nn.Module):
    """
    Extracts aspect terms from text using BiLSTM-CRF.

    Identifies aspect terms (e.g., "battery life", "screen quality") in reviews
    that users express opinions about.

    Args:
        vocab_size: Vocabulary size
        embedding_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        num_tags: Number of BIO tags

    Example:
        >>> extractor = AspectExtractor(vocab_size=10000, num_tags=3)
        >>> tags = extractor(input_ids, lengths)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_tags: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.transitions.data[0, :] = -10000  # Cannot start with I

    def _forward_alg(self, feats: Tensor, mask: Tensor) -> Tensor:
        """Forward algorithm for CRF."""
        batch_size, seq_len, num_tags = feats.shape

        alpha = feats[:, 0] + self.transitions[0].unsqueeze(0)

        for t in range(1, seq_len):
            emit = feats[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            alpha = torch.logsumexp(alpha.unsqueeze(2) + emit + trans, dim=1)
            alpha = alpha * mask[:, t].unsqueeze(1) + alpha.detach() * (
                ~mask[:, t]
            ).unsqueeze(1)

        return torch.logsumexp(alpha, dim=1)

    def _score_sentence(self, feats: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Score a tagged sentence."""
        batch_size = feats.shape[0]
        score = torch.zeros(batch_size, device=feats.device)

        score += feats[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for t in range(1, feats.shape[1]):
            score += self.transitions[tags[:, t - 1], tags[:, t]] * mask[:, t]
            score += feats[:, t].gather(1, tags[:, t : t + 1]).squeeze(1) * mask[:, t]

        return score

    def forward(
        self, input_ids: Tensor, lengths: Tensor, tags: Optional[Tensor] = None
    ) -> Union[Tensor, List[List[int]]]:
        """
        Forward pass.

        Args:
            input_ids: Token indices
            lengths: Sequence lengths
            tags: Optional gold tags for training

        Returns:
            Loss during training, predicted tags during inference
        """
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        feats = self.hidden2tag(lstm_out)
        mask = input_ids != 0

        if tags is not None:
            forward_score = self._forward_alg(feats, mask)
            gold_score = self._score_sentence(feats, tags, mask)
            return forward_score - gold_score
        else:
            return self._viterbi_decode(feats, mask)

    def _viterbi_decode(self, feats: Tensor, mask: Tensor) -> List[List[int]]:
        """Viterbi decoding for CRF."""
        batch_size, seq_len, num_tags = feats.shape

        backpointers = []
        viterbi_vars = feats[:, 0] + self.transitions[0].unsqueeze(0)

        for t in range(1, seq_len):
            viterbi_vars_t = viterbi_vars.unsqueeze(2) + self.transitions.unsqueeze(0)
            best_tag_id = torch.argmax(viterbi_vars_t, dim=1)
            backpointers.append(best_tag_id)

            viterbi_vars = feats[:, t].gather(1, best_tag_id) * mask[:, t].unsqueeze(1)
            viterbi_vars = viterbi_vars + viterbi_vars.detach() * (
                ~mask[:, t]
            ).unsqueeze(1)

        best_path = [torch.argmax(viterbi_vars, dim=1)]
        for backpointer in reversed(backpointers):
            best_path.insert(
                0, backpointer.gather(1, best_path[0].unsqueeze(1)).squeeze(1)
            )

        return torch.stack(best_path, dim=1).tolist()


class AspectSentiment(nn.Module):
    """
    Aspect-level sentiment classifier.

    Classifies sentiment toward specific aspects using aspect embeddings
    and context encoding.

    Args:
        vocab_size: Vocabulary size
        embedding_dim: Word embedding dimension
        hidden_dim: LSTM hidden dimension
        num_aspects: Number of aspect categories
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = AspectSentiment(vocab_size=10000, num_aspects=10)
        >>> logits = model(input_ids, aspect_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_aspects: int = 10,
        num_sentiments: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.aspect_embedding = nn.Embedding(num_aspects, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_sentiments)

    def forward(self, input_ids: Tensor, aspect_ids: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass."""
        embedded = self.dropout(self.embedding(input_ids))
        aspect_emb = self.aspect_embedding(aspect_ids)

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        aspect_expanded = aspect_emb.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        combined = torch.cat([lstm_out, aspect_expanded], dim=-1)

        attention_scores = self.attention(combined).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)

        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        context = self.dropout(context)

        return self.classifier(context)


class GCN_ABSA(nn.Module):
    """
    Graph Convolutional Network for Aspect-Based Sentiment Analysis.

    Uses graph convolutional layers to model syntactic dependencies
    between aspect terms and opinion words.

    Args:
        vocab_size: Vocabulary size
        embedding_dim: Word embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of GCN layers
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = GCN_ABSA(vocab_size=10000, num_layers=2)
        >>> logits = model(input_ids, dependency_graph, aspect_mask)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_sentiments: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.layers.append(GCNLayer(in_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_sentiments)

    def forward(
        self,
        input_ids: Tensor,
        adj_matrix: Tensor,
        aspect_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices
            adj_matrix: Dependency graph adjacency matrix
            aspect_mask: Binary mask for aspect term positions
        """
        h = self.dropout(self.embedding(input_ids))

        for layer in self.layers:
            h = layer(h, adj_matrix)
            h = F.relu(h)
            h = self.dropout(h)

        aspect_repr = torch.sum(h * aspect_mask.unsqueeze(-1), dim=1)
        aspect_repr = aspect_repr / (aspect_mask.sum(dim=1, keepdim=True) + 1e-8)

        return self.classifier(aspect_repr)


class GCNLayer(nn.Module):
    """Single GCN layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Forward pass with adjacency matrix."""
        support = self.linear(x)
        output = torch.bmm(adj, support)
        return output


class LCF(nn.Module):
    """
    Local Context Focus for ABSA.

    Dynamically focuses on local context around aspect terms while
    maintaining global context information.

    Args:
        model_name: BERT model name
        num_sentiments: Number of sentiment classes
        max_seq_len: Maximum sequence length
        local_context_focus: Context focus mechanism

    Example:
        >>> model = LCF('bert-base-uncased', local_context_focus='cdw')
        >>> logits = model(input_ids, attention_mask, aspect_indices)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_sentiments: int = 3,
        max_seq_len: int = 80,
        local_context_focus: str = "cdw",
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.local_context_focus = local_context_focus
        self.max_seq_len = max_seq_len

        self.dropout = nn.Dropout(dropout)
        self.bert_dim = self.bert.config.hidden_size

        self.local_fc = nn.Linear(self.bert_dim, self.bert_dim)
        self.global_fc = nn.Linear(self.bert_dim, self.bert_dim)

        self.classifier = nn.Linear(self.bert_dim * 2, num_sentiments)

    def _dynamic_mask(self, aspect_indices: Tensor) -> Tensor:
        """Create dynamic mask for local context."""
        batch_size = aspect_indices.size(0)
        mask = torch.zeros(batch_size, self.max_seq_len, device=aspect_indices.device)

        for i in range(batch_size):
            start, end = aspect_indices[i]
            if self.local_context_focus == "cdw":
                weights = self._calculate_scdw(
                    start.item(), end.item(), self.max_seq_len
                )
            else:
                weights = self._calculate_cdm(
                    start.item(), end.item(), self.max_seq_len
                )
            mask[i] = weights

        return mask

    def _calculate_cdm(self, start: int, end: int, seq_len: int) -> Tensor:
        """Context Dynamic Mask."""
        mask = torch.zeros(seq_len)
        for i in range(start, end):
            mask[i] = 1.0
        return mask

    def _calculate_scdw(self, start: int, end: int, seq_len: int) -> Tensor:
        """Syntactic Context Dynamic Weight."""
        mask = torch.zeros(seq_len)
        aspect_center = (start + end) / 2
        for i in range(seq_len):
            distance = abs(i - aspect_center)
            mask[i] = 1 - distance / seq_len
        mask[start:end] = 1.0
        return mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        aspect_indices: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        dynamic_weights = self._dynamic_mask(aspect_indices).unsqueeze(-1)
        local_features = sequence_output * dynamic_weights

        local_pooled = torch.mean(local_features, dim=1)
        global_pooled = torch.mean(sequence_output, dim=1)

        local_repr = F.relu(self.local_fc(local_pooled))
        global_repr = F.relu(self.global_fc(global_pooled))

        combined = torch.cat([local_repr, global_repr], dim=-1)
        combined = self.dropout(combined)

        return self.classifier(combined)


class AEN(nn.Module):
    """
    Attentional Encoder Network for ABSA.

    Uses multiple attention layers to model interactions between
    aspect terms and context words.

    Args:
        model_name: BERT model name
        num_sentiments: Number of sentiment classes
        num_heads: Number of attention heads
        num_layers: Number of encoder layers

    Example:
        >>> model = AEN('bert-base-uncased', num_heads=8)
        >>> logits = model(input_ids, attention_mask, aspect_indices)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_sentiments: int = 3,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size

        self.aspect_attention = nn.MultiheadAttention(
            self.bert_dim, num_heads, dropout=dropout, batch_first=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bert_dim,
            nhead=num_heads,
            dim_feedforward=self.bert_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert_dim, num_sentiments)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        aspect_indices: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        aspect_reprs = []
        for i, (start, end) in enumerate(aspect_indices):
            aspect_reprs.append(torch.mean(sequence_output[i, start:end], dim=0))
        aspect_reprs = torch.stack(aspect_reprs).unsqueeze(1)

        attended, _ = self.aspect_attention(
            aspect_reprs,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool(),
        )

        encoded = self.context_encoder(attended, src_key_padding_mask=None)
        pooled = encoded.squeeze(1)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)


class BertABSA(nn.Module):
    """
    BERT for Aspect-Based Sentiment Analysis.

    Fine-tunes BERT specifically for ABSA by incorporating
    aspect position information into the input.

    Args:
        model_name: BERT model name
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = BertABSA('bert-base-uncased')
        >>> logits = model(input_ids, attention_mask, aspect_positions)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_sentiments: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_sentiments)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        aspect_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices
            attention_mask: Attention mask
            aspect_positions: Optional positions of aspect terms
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if aspect_positions is not None:
            aspect_vectors = []
            for i, pos in enumerate(aspect_positions):
                start, end = pos
                aspect_vec = torch.mean(outputs.last_hidden_state[i, start:end], dim=0)
                aspect_vectors.append(aspect_vec)
            pooled = torch.stack(aspect_vectors)
        else:
            pooled = outputs.pooler_output

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# ============================================================================
# Multimodal Sentiment Analysis
# ============================================================================


class TextImageSentiment(nn.Module):
    """
    Multimodal sentiment analysis using text and images.

    Combines textual and visual features using late fusion
    with cross-modal attention.

    Args:
        text_model: Text encoder model
        image_encoder: Image encoder (e.g., ResNet)
        fusion_dim: Dimension after fusion
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = TextImageSentiment(text_model, image_encoder)
        >>> logits = model(input_ids, images, attention_mask)
    """

    def __init__(
        self,
        text_model: nn.Module,
        image_encoder: nn.Module,
        fusion_dim: int = 512,
        num_sentiments: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.text_model = text_model
        self.image_encoder = image_encoder

        self.text_dim = self._get_text_dim()
        self.image_dim = self._get_image_dim()

        self.text_proj = nn.Linear(self.text_dim, fusion_dim)
        self.image_proj = nn.Linear(self.image_dim, fusion_dim)

        self.cross_attention = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(fusion_dim, num_sentiments)

    def _get_text_dim(self) -> int:
        """Get text model output dimension."""
        if hasattr(self.text_model, "config"):
            return self.text_model.config.hidden_size
        return 768

    def _get_image_dim(self) -> int:
        """Get image encoder output dimension."""
        return 2048

    def forward(
        self,
        input_ids: Tensor,
        images: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> MultimodalOutput:
        """Forward pass."""
        text_features = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        text_features = self.text_proj(text_features)

        image_features = self.image_encoder(images)
        if len(image_features.shape) > 2:
            image_features = (
                F.adaptive_avg_pool2d(image_features, (1, 1)).squeeze(-1).squeeze(-1)
            )
        image_features = self.image_proj(image_features)

        text_attended, _ = self.cross_attention(
            text_features.unsqueeze(1),
            image_features.unsqueeze(1),
            image_features.unsqueeze(1),
        )

        combined = torch.cat([text_features, text_attended.squeeze(1)], dim=-1)
        fused = self.fusion(combined)

        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        sentiment_idx = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values

        sentiments = ["negative", "neutral", "positive"]

        return MultimodalOutput(
            sentiment=sentiments[sentiment_idx.item()],
            confidence=confidence.item(),
            text_contribution=0.5,
            modality_weights={"text": 0.5, "image": 0.5},
        )


class VideoSentiment(nn.Module):
    """
    Video-based multimodal sentiment analysis.

    Analyzes sentiment from video by combining visual, audio,
    and text modalities extracted from video content.

    Args:
        video_encoder: Video feature extractor
        audio_encoder: Audio feature extractor
        text_encoder: Text encoder
        fusion_dim: Fusion layer dimension
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = VideoSentiment(video_enc, audio_enc, text_enc)
        >>> output = model(video_frames, audio, transcript)
    """

    def __init__(
        self,
        video_encoder: nn.Module,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion_dim: int = 512,
        num_sentiments: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder

        self.video_proj = nn.Linear(512, fusion_dim)
        self.audio_proj = nn.Linear(128, fusion_dim)
        self.text_proj = nn.Linear(768, fusion_dim)

        self.temporal_fusion = nn.LSTM(fusion_dim * 3, fusion_dim, batch_first=True)

        self.attention_pool = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_sentiments),
        )

    def forward(
        self,
        video_frames: Tensor,
        audio_features: Tensor,
        text_input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> MultimodalOutput:
        """Forward pass."""
        video_feat = self.video_encoder(video_frames)
        video_feat = self.video_proj(video_feat)

        audio_feat = self.audio_encoder(audio_features)
        audio_feat = self.audio_proj(audio_feat)

        text_feat = self.text_encoder(
            input_ids=text_input_ids, attention_mask=attention_mask
        ).pooler_output
        text_feat = self.text_proj(text_feat)

        if video_feat.dim() == 2:
            video_feat = video_feat.unsqueeze(1)
        if audio_feat.dim() == 2:
            audio_feat = audio_feat.unsqueeze(1)
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)

        combined = torch.cat([video_feat, audio_feat, text_feat], dim=-1)

        fused, _ = self.temporal_fusion(combined)

        attn_scores = self.attention_pool(fused).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)

        pooled = torch.bmm(attn_weights.unsqueeze(1), fused).squeeze(1)

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)

        sentiment_idx = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values

        sentiments = ["negative", "neutral", "positive"]

        return MultimodalOutput(
            sentiment=sentiments[sentiment_idx.item()],
            confidence=confidence.item(),
            text_contribution=0.33,
            modality_weights={"video": 0.33, "audio": 0.33, "text": 0.34},
        )


class AudioSentiment(nn.Module):
    """
    Speech sentiment analysis from audio.

    Analyzes emotional content and sentiment from speech audio
    using acoustic features.

    Args:
        feature_dim: Input audio feature dimension (e.g., MFCCs)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = AudioSentiment(feature_dim=40)
        >>> logits = model(audio_features)
    """

    def __init__(
        self,
        feature_dim: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_sentiments: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sentiments),
        )

    def forward(
        self, audio_features: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass."""
        if lengths is not None:
            packed = pack_padded_sequence(
                audio_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(audio_features)

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        return self.classifier(context)


class MAG(nn.Module):
    """
    Multimodal Adaptation Gates for sentiment analysis.

    Uses gating mechanisms to adapt text representations based on
    non-verbal signals (visual and audio).

    Args:
        text_dim: Text feature dimension
        visual_dim: Visual feature dimension
        acoustic_dim: Acoustic feature dimension
        hidden_dim: Hidden dimension
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = MAG(text_dim=768, visual_dim=512, acoustic_dim=128)
        >>> logits = model(text, visual, acoustic, attention_mask)
    """

    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 512,
        acoustic_dim: int = 128,
        hidden_dim: int = 768,
        num_sentiments: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, hidden_dim)

        self.visual_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.acoustic_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dropout=dropout),
            num_layers=4,
        )

        self.classifier = nn.Linear(hidden_dim, num_sentiments)

    def forward(
        self,
        text: Tensor,
        visual: Tensor,
        acoustic: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        text_h = self.text_proj(text)
        visual_h = self.visual_proj(visual)
        acoustic_h = self.acoustic_proj(acoustic)

        if visual_h.dim() == 2:
            visual_h = visual_h.unsqueeze(1).expand(-1, text_h.size(1), -1)
        if acoustic_h.dim() == 2:
            acoustic_h = acoustic_h.unsqueeze(1).expand(-1, text_h.size(1), -1)

        visual_gate = self.visual_gate(torch.cat([text_h, visual_h], dim=-1))
        acoustic_gate = self.acoustic_gate(torch.cat([text_h, acoustic_h], dim=-1))

        adapted = text_h + visual_gate * visual_h + acoustic_gate * acoustic_h

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        encoded = self.text_encoder(adapted, src_key_padding_mask=src_key_padding_mask)

        pooled = encoded.mean(dim=1)

        return self.classifier(pooled)


class MulT(nn.Module):
    """
    Multimodal Transformer for sentiment analysis.

    Uses cross-modal transformers to model interactions between
    different modalities without early fusion.

    Args:
        text_dim: Text feature dimension
        visual_dim: Visual feature dimension
        acoustic_dim: Acoustic feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_sentiments: Number of sentiment classes

    Example:
        >>> model = MulT(text_dim=768, visual_dim=512, acoustic_dim=128)
        >>> logits = model(text, visual, acoustic)
    """

    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 512,
        acoustic_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_sentiments: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, hidden_dim)

        self.cross_modal_tv = nn.ModuleList(
            [
                CrossModalTransformer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.cross_modal_ta = nn.ModuleList(
            [
                CrossModalTransformer(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sentiments),
        )

    def forward(
        self,
        text: Tensor,
        visual: Tensor,
        acoustic: Tensor,
    ) -> Tensor:
        """Forward pass."""
        text_h = self.text_proj(text)
        visual_h = self.visual_proj(visual)
        acoustic_h = self.acoustic_proj(acoustic)

        if visual_h.dim() == 2:
            visual_h = visual_h.unsqueeze(1)
        if acoustic_h.dim() == 2:
            acoustic_h = acoustic_h.unsqueeze(1)

        for layer in self.cross_modal_tv:
            text_h = layer(text_h, visual_h)

        text_h_acoustic = self.text_proj(text)
        for layer in self.cross_modal_ta:
            text_h_acoustic = layer(text_h_acoustic, acoustic_h)

        text_pooled = text_h.mean(dim=1)
        text_acoustic_pooled = text_h_acoustic.mean(dim=1)
        visual_pooled = (
            visual_h.mean(dim=1) if visual_h.dim() > 2 else visual_h.squeeze(1)
        )

        combined = torch.cat([text_pooled, text_acoustic_pooled, visual_pooled], dim=-1)
        fused = self.fusion_layer(combined)

        return self.classifier(fused)


class CrossModalTransformer(nn.Module):
    """Cross-modal transformer layer."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """Forward pass with cross-modal attention."""
        residual = query
        query = self.norm1(query)
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        query = residual + self.dropout(attn_out)

        residual = query
        query = self.norm2(query)
        self_attn_out, _ = self.self_attn(query, query, query)
        query = residual + self.dropout(self_attn_out)

        residual = query
        query = self.norm3(query)
        ffn_out = self.ffn(query)
        query = residual + self.dropout(ffn_out)

        return query


# ============================================================================
# Fine-grained Sentiment Analysis
# ============================================================================


class SentimentIntensity(nn.Module):
    """
    Continuous sentiment intensity scorer.

    Predicts sentiment on a continuous scale from -1 (negative)
    to +1 (positive), allowing for fine-grained analysis.

    Args:
        model_name: Pretrained model name

    Example:
        >>> model = SentimentIntensity('bert-base-uncased')
        >>> score = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass returning continuous score."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        score = torch.tanh(self.regressor(pooled))
        return score.squeeze(-1)


class EmotionDetection(nn.Module):
    """
    Fine-grained emotion detection.

    Detects specific emotions: joy, anger, sadness, fear, surprise,
    disgust, love, and neutral.

    Args:
        model_name: Pretrained model name
        num_emotions: Number of emotion classes

    Example:
        >>> model = EmotionDetection('bert-base-uncased', num_emotions=8)
        >>> logits = model(input_ids, attention_mask)
    """

    EMOTIONS = [
        "anger",
        "joy",
        "sadness",
        "fear",
        "surprise",
        "neutral",
        "disgust",
        "love",
    ]

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_emotions: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def predict_emotion(self, logits: Tensor) -> List[str]:
        """Convert logits to emotion labels."""
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        return [self.EMOTIONS[p] for p in predictions.tolist()]


class SarcasmDetection(nn.Module):
    """
    Sarcasm detection model.

    Detects sarcastic or ironic content that may convey opposite
    sentiment from the literal meaning.

    Args:
        model_name: Pretrained model name
        use_context: Whether to use conversational context

    Example:
        >>> model = SarcasmDetection('bert-base-uncased')
        >>> logits = model(input_ids, attention_mask, context_ids)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        use_context: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.use_context = use_context

        input_dim = self.bert.config.hidden_size
        if use_context:
            self.context_encoder = nn.LSTM(
                self.bert.config.hidden_size,
                self.bert.config.hidden_size // 2,
                bidirectional=True,
                batch_first=True,
            )
            input_dim *= 2

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 2),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        context_ids: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = outputs.pooler_output

        if self.use_context and context_ids is not None:
            context_outputs = self.bert(
                input_ids=context_ids,
                attention_mask=context_mask,
            )
            context_seq = context_outputs.last_hidden_state
            context_repr, _ = self.context_encoder(context_seq)
            context_repr = context_repr[:, -1]

            combined = torch.cat([text_repr, context_repr], dim=-1)
        else:
            combined = text_repr

        combined = self.dropout(combined)
        return self.classifier(combined)


class StanceDetection(nn.Module):
    """
    Stance detection for topic classification.

    Classifies stance toward a target topic as favor, against, or none.

    Args:
        model_name: Pretrained model name
        num_targets: Number of target topics

    Example:
        >>> model = StanceDetection('bert-base-uncased', num_targets=10)
        >>> logits = model(input_ids, attention_mask, target_id)
    """

    STANCES = ["favor", "against", "none"]

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_targets: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.target_embedding = nn.Embedding(num_targets, self.bert.config.hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, 3)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = outputs.pooler_output

        target_repr = self.target_embedding(target_ids)

        combined = torch.cat([text_repr, target_repr], dim=-1)
        combined = self.dropout(combined)

        return self.classifier(combined)


class IntentClassification(nn.Module):
    """
    User intent classification.

    Classifies user intent in conversational contexts (e.g.,
    inquiry, complaint, purchase, etc.).

    Args:
        model_name: Pretrained model name
        num_intents: Number of intent classes
        use_hierarchy: Whether to use hierarchical intent structure

    Example:
        >>> model = IntentClassification('bert-base-uncased', num_intents=50)
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_intents: int = 50,
        num_coarse_intents: int = 10,
        use_hierarchy: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.bert = BertModel.from_pretrained(model_name)
        self.use_hierarchy = use_hierarchy

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        if use_hierarchy:
            self.coarse_classifier = nn.Linear(hidden_size, num_coarse_intents)
            self.fine_classifier = nn.Linear(
                hidden_size + num_coarse_intents, num_intents
            )
        else:
            self.classifier = nn.Linear(hidden_size, num_intents)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)

        if self.use_hierarchy:
            coarse_logits = self.coarse_classifier(pooled)
            coarse_probs = F.softmax(coarse_logits, dim=-1)

            combined = torch.cat([pooled, coarse_probs], dim=-1)
            fine_logits = self.fine_classifier(combined)

            return coarse_logits, fine_logits
        else:
            return self.classifier(pooled)


# ============================================================================
# Social Media Processing
# ============================================================================


class EmojiHandling:
    """
    Emoji processing for social media text.

    Handles emoji conversion, sentiment analysis, and normalization.

    Example:
        >>> handler = EmojiHandling()
        >>> text = handler.convert_emojis("I love this! 😍")
    """

    EMOJI_SENTIMENT = {
        "😍": 1.0,
        "❤️": 1.0,
        "👍": 0.8,
        "😊": 0.8,
        "🎉": 0.9,
        "😢": -0.8,
        "😠": -0.9,
        "👎": -0.8,
        "😞": -0.7,
        "😡": -1.0,
        "😐": 0.0,
        "🤔": 0.0,
        "😂": 0.5,
        "🤣": 0.6,
        "🙄": -0.3,
    }

    def __init__(self, convert_to_text: bool = True):
        self.convert_to_text = convert_to_text

    def extract_emojis(self, text: str) -> List[str]:
        """Extract all emojis from text."""
        if EMOJI_AVAILABLE:
            return [c for c in text if c in emoji.EMOJI_DATA]
        else:
            return [c for c in text if unicodedata.category(c) == "So"]

    def convert_emojis(self, text: str) -> str:
        """Convert emojis to text descriptions."""
        if not EMOJI_AVAILABLE:
            return text

        if self.convert_to_text:
            return emoji.demojize(text)
        return text

    def get_emoji_sentiment(self, text: str) -> float:
        """Calculate sentiment score from emojis."""
        emojis = self.extract_emojis(text)
        if not emojis:
            return 0.0

        scores = [self.EMOJI_SENTIMENT.get(e, 0.0) for e in emojis]
        return sum(scores) / len(scores)

    def remove_emojis(self, text: str) -> str:
        """Remove all emojis from text."""
        if EMOJI_AVAILABLE:
            return emoji.replace_emoji(text, replace="")
        else:
            return "".join(c for c in text if unicodedata.category(c) != "So")


class HashtagProcessing:
    """
    Hashtag processing utilities.

    Extracts, normalizes, and expands hashtags for better
    sentiment analysis on social media.

    Example:
        >>> processor = HashtagProcessing()
        >>> tags = processor.extract_hashtags("Love this #AmazingProduct!")
    """

    HASHTAG_PATTERN = re.compile(r"#\w+")

    def extract_hashtags(self, text: str) -> List[str]:
        """Extract all hashtags from text."""
        return self.HASHTAG_PATTERN.findall(text)

    def normalize_hashtag(self, hashtag: str) -> str:
        """Normalize hashtag by removing # and splitting camelCase."""
        text = hashtag.lstrip("#")

        words = []
        current_word = text[0] if text else ""

        for i in range(1, len(text)):
            if text[i].isupper() and text[i - 1].islower():
                words.append(current_word)
                current_word = text[i]
            elif text[i].isupper() and i + 1 < len(text) and text[i + 1].islower():
                words.append(current_word)
                current_word = text[i]
            else:
                current_word += text[i]
        words.append(current_word)

        return " ".join(words).lower()

    def expand_hashtags(self, text: str) -> str:
        """Expand hashtags to normalized text."""
        hashtags = self.extract_hashtags(text)
        for tag in hashtags:
            expanded = self.normalize_hashtag(tag)
            text = text.replace(tag, expanded)
        return text

    def remove_hashtags(self, text: str) -> str:
        """Remove all hashtags from text."""
        return self.HASHTAG_PATTERN.sub("", text)


class MentionProcessing:
    """
    User mention processing for social media.

    Handles @mentions in social media text.

    Example:
        >>> processor = MentionProcessing()
        >>> mentions = processor.extract_mentions("Thanks @user1 and @user2!")
    """

    MENTION_PATTERN = re.compile(r"@\w+")

    def extract_mentions(self, text: str) -> List[str]:
        """Extract all mentions from text."""
        return self.MENTION_PATTERN.findall(text)

    def remove_mentions(self, text: str) -> str:
        """Remove all mentions from text."""
        return self.MENTION_PATTERN.sub("", text)

    def replace_mentions(self, text: str, replacement: str = "@user") -> str:
        """Replace mentions with placeholder."""
        return self.MENTION_PATTERN.sub(replacement, text)


class URLRemoval:
    """
    URL removal and normalization.

    Cleans URLs from text for cleaner sentiment analysis.

    Example:
        >>> cleaner = URLRemoval()
        >>> clean = cleaner.remove_urls("Check this out: https://example.com")
    """

    URL_PATTERN = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    def remove_urls(self, text: str) -> str:
        """Remove all URLs from text."""
        return self.URL_PATTERN.sub("", text)

    def replace_urls(self, text: str, replacement: str = "<URL>") -> str:
        """Replace URLs with placeholder."""
        return self.URL_PATTERN.sub(replacement, text)

    def extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text."""
        return self.URL_PATTERN.findall(text)


class SlangNormalization:
    """
    Slang and informal language normalization.

    Normalizes common slang, abbreviations, and informal
    expressions to standard form.

    Example:
        >>> normalizer = SlangNormalization()
        >>> text = normalizer.normalize("lol this is gr8!")
    """

    SLANG_DICT = {
        "lol": "laughing out loud",
        "lmao": "laughing my ass off",
        "rofl": "rolling on the floor laughing",
        "brb": "be right back",
        "btw": "by the way",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "tbh": "to be honest",
        "gr8": "great",
        "b4": "before",
        "u": "you",
        "r": "are",
        "ur": "your",
        "yall": "you all",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "kinda": "kind of",
        "sorta": "sort of",
        "dunno": "do not know",
        "lemme": "let me",
        "gimme": "give me",
        "cuz": "because",
        "cause": "because",
        "coz": "because",
        "tho": "though",
        "thru": "through",
        "thx": "thanks",
        "ty": "thank you",
        "tyvm": "thank you very much",
        "np": "no problem",
        "yw": "you are welcome",
        "idk": "I do not know",
        "idc": "I do not care",
        "idgaf": "I do not give a damn",
        "smh": "shaking my head",
        "fml": "screw my life",
        "ftw": "for the win",
        "irl": "in real life",
        "afaik": "as far as I know",
        "afk": "away from keyboard",
        "bff": "best friends forever",
        "ftfy": "fixed that for you",
        "hth": "hope this helps",
        "iirc": "if I recall correctly",
        "nvm": "never mind",
        "omw": "on my way",
        "tbf": "to be fair",
        "tbd": "to be determined",
        "tia": "thanks in advance",
        "ttyl": "talk to you later",
        "wdym": "what do you mean",
        "wyd": "what are you doing",
        "wya": "where are you at",
        "tfw": "that feeling when",
        "mfw": "my face when",
        "mrw": "my reaction when",
        "ifyp": "I feel your pain",
        "srsly": "seriously",
        "obvs": "obviously",
        "obvi": "obviously",
        "totes": "totally",
        "legit": "legitimate",
        "perf": "perfect",
        "fab": "fabulous",
        "delish": "delicious",
        "adorbs": "adorable",
        "cray": "crazy",
        "ridic": "ridiculous",
        "hilarios": "hilarious",
        "amazeballs": "amazing",
        "awesomesauce": "awesome",
        "awsm": "awesome",
        "awk": "awkward",
        "jelly": "jealous",
        "fam": "family",
        "squad": "group of friends",
        "bae": "before anyone else",
        "fomo": "fear of missing out",
        "lit": "exciting",
        "dope": "cool",
        "sick": "cool",
        "dank": "high quality",
        "extra": "over the top",
        "salty": "bitter or upset",
        "shook": "shocked",
        "slay": "do excellently",
        "snatched": "looking good",
        "tea": "gossip",
        "vibe": "atmosphere",
        "woke": "socially aware",
        "yolo": "you only live once",
        "gg": "good game",
        "wp": "well played",
        "gl": "good luck",
        "hf": "have fun",
        "rekt": "destroyed",
        "pwned": "dominated",
        "noob": "newbie",
        "op": "overpowered",
        "nerf": "weaken",
        "buff": "strengthen",
        "lag": "delay",
        "af": "as hell",
        "rn": "right now",
        "atm": "at the moment",
    }

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.slang_pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, self.SLANG_DICT.keys())) + r")\b",
            re.IGNORECASE,
        )

    def normalize(self, text: str) -> str:
        """Normalize slang in text."""
        if self.lowercase:
            text_lower = text.lower()
        else:
            text_lower = text

        def replace_slang(match):
            slang = match.group(0).lower()
            return self.SLANG_DICT.get(slang, match.group(0))

        normalized = self.slang_pattern.sub(replace_slang, text_lower)
        return normalized

    def is_slang(self, word: str) -> bool:
        """Check if word is slang."""
        return word.lower() in self.SLANG_DICT

    def get_expansion(self, slang: str) -> Optional[str]:
        """Get expansion for slang term."""
        return self.SLANG_DICT.get(slang.lower())


class SocialMediaPreprocessor:
    """
    Complete social media text preprocessor.

    Combines all social media processing utilities into a
    single preprocessing pipeline.

    Args:
        remove_urls: Whether to remove URLs
        expand_hashtags: Whether to expand hashtags
        convert_emojis: Whether to convert emojis to text
        normalize_slang: Whether to normalize slang
        remove_mentions: Whether to remove @mentions

    Example:
        >>> preprocessor = SocialMediaPreprocessor()
        >>> clean = preprocessor.preprocess("Love this! 😍 #AmazingProduct https://t.co/xyz")
    """

    def __init__(
        self,
        remove_urls: bool = True,
        expand_hashtags: bool = True,
        convert_emojis: bool = True,
        normalize_slang: bool = True,
        remove_mentions: bool = True,
    ):
        self.remove_urls = remove_urls
        self.expand_hashtags = expand_hashtags
        self.convert_emojis = convert_emojis
        self.normalize_slang = normalize_slang
        self.remove_mentions = remove_mentions

        self.url_processor = URLRemoval()
        self.hashtag_processor = HashtagProcessing()
        self.emoji_processor = EmojiHandling(convert_to_text=convert_emojis)
        self.mention_processor = MentionProcessing()
        self.slang_processor = SlangNormalization()

    def preprocess(self, text: str) -> str:
        """
        Run full preprocessing pipeline.

        Args:
            text: Raw social media text

        Returns:
            Preprocessed text
        """
        if self.remove_urls:
            text = self.url_processor.replace_urls(text)

        if self.remove_mentions:
            text = self.mention_processor.replace_mentions(text)

        if self.expand_hashtags:
            text = self.hashtag_processor.expand_hashtags(text)

        if self.convert_emojis:
            text = self.emoji_processor.convert_emojis(text)

        if self.normalize_slang:
            text = self.slang_processor.normalize(text)

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def get_social_features(self, text: str) -> Dict[str, Any]:
        """Extract social media specific features."""
        features = {
            "has_emoji": len(self.emoji_processor.extract_emojis(text)) > 0,
            "emoji_count": len(self.emoji_processor.extract_emojis(text)),
            "emoji_sentiment": self.emoji_processor.get_emoji_sentiment(text),
            "has_hashtags": len(self.hashtag_processor.extract_hashtags(text)) > 0,
            "hashtag_count": len(self.hashtag_processor.extract_hashtags(text)),
            "has_mentions": len(self.mention_processor.extract_mentions(text)) > 0,
            "mention_count": len(self.mention_processor.extract_mentions(text)),
            "has_urls": len(self.url_processor.extract_urls(text)) > 0,
            "url_count": len(self.url_processor.extract_urls(text)),
        }
        return features


# ============================================================================
# Cross-lingual Sentiment Analysis
# ============================================================================


class ZeroShotSentiment(nn.Module):
    """
    Zero-shot cross-lingual sentiment analysis.

    Performs sentiment analysis on languages not seen during training
    using multilingual models and zero-shot transfer.

    Args:
        model_name: Multilingual model name
        num_classes: Number of sentiment classes

    Example:
        >>> model = ZeroShotSentiment('xlm-roberta-base')
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class CrossLingualBERT(nn.Module):
    """
    XLM-RoBERTa for cross-lingual sentiment analysis.

    Uses XLM-RoBERTa, a multilingual model trained on 100 languages,
    for cross-lingual transfer.

    Args:
        model_name: XLM-RoBERTa model name
        num_classes: Number of sentiment classes
        freeze_layers: Number of layers to freeze

    Example:
        >>> model = CrossLingualBERT('xlm-roberta-large')
        >>> logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_classes: int = 3,
        freeze_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

        if freeze_layers > 0:
            for param in list(self.model.parameters())[:freeze_layers]:
                param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class TranslationPivot:
    """
    Translation-based cross-lingual sentiment analysis.

    Translates text to a pivot language (usually English),
    then performs sentiment analysis.

    Args:
        target_lang: Target language code for translation
        sentiment_model: Sentiment analysis model

    Example:
        >>> pivot = TranslationPivot('en', sentiment_model)
        >>> result = pivot.analyze("J'aime ce produit", translator)
    """

    def __init__(
        self,
        target_lang: str = "en",
        sentiment_model: Optional[nn.Module] = None,
    ):
        self.target_lang = target_lang
        self.sentiment_model = sentiment_model

    def translate(
        self,
        text: str,
        translator: Callable[[str, str, str], str],
        source_lang: str,
    ) -> str:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            translator: Translation function (text, source, target) -> translated
            source_lang: Source language code

        Returns:
            Translated text
        """
        return translator(text, source_lang, self.target_lang)

    def analyze(
        self,
        text: str,
        translator: Callable[[str, str, str], str],
        source_lang: str,
        tokenizer: Any,
        device: str = "cpu",
    ) -> SentimentOutput:
        """
        Translate and analyze sentiment.

        Args:
            text: Text to analyze
            translator: Translation function
            source_lang: Source language code
            tokenizer: Tokenizer for sentiment model
            device: Device for inference

        Returns:
            Sentiment output
        """
        translated = self.translate(text, translator, source_lang)

        if self.sentiment_model is None:
            raise ValueError("sentiment_model must be provided")

        inputs = tokenizer(
            translated,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.sentiment_model.eval()
        with torch.no_grad():
            logits = self.sentiment_model(**inputs)
            probs = F.softmax(logits, dim=-1)

        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

        labels = ["negative", "neutral", "positive"]
        scores = {labels[i]: probs[0, i].item() for i in range(len(labels))}

        return SentimentOutput(
            label=labels[pred_idx],
            confidence=confidence,
            scores=scores,
        )


# ============================================================================
# Evaluation Utilities
# ============================================================================


class SentimentEvaluator:
    """
    Comprehensive sentiment analysis evaluation.

    Provides accuracy, F1, precision, recall, confusion matrix,
    and ABSA-specific metrics.

    Example:
        >>> evaluator = SentimentEvaluator(num_classes=3)
        >>> metrics = evaluator.compute_metrics(predictions, labels)
    """

    def __init__(self, num_classes: int = 3, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.predictions = []
        self.labels = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def add_batch(self, predictions: np.ndarray, labels: np.ndarray):
        """Add a batch of predictions and labels."""
        self.predictions.extend(predictions.tolist())
        self.labels.extend(labels.tolist())

        for pred, label in zip(predictions, labels):
            self.confusion_matrix[label, pred] += 1

    def compute_accuracy(self) -> float:
        """Compute accuracy."""
        correct = sum(1 for p, l in zip(self.predictions, self.labels) if p == l)
        return correct / len(self.labels) if self.labels else 0.0

    def compute_precision(self) -> Union[float, np.ndarray]:
        """Compute precision per class."""
        precision = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if self.average == "macro":
            return precision.mean()
        elif self.average == "micro":
            tp = np.diag(self.confusion_matrix).sum()
            total_pred = self.confusion_matrix.sum()
            return tp / total_pred if total_pred > 0 else 0.0
        else:
            return precision

    def compute_recall(self) -> Union[float, np.ndarray]:
        """Compute recall per class."""
        recall = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            fn = self.confusion_matrix[i, :].sum() - tp
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if self.average == "macro":
            return recall.mean()
        elif self.average == "micro":
            tp = np.diag(self.confusion_matrix).sum()
            total = self.confusion_matrix.sum()
            return tp / total if total > 0 else 0.0
        else:
            return recall

    def compute_f1(self) -> Union[float, np.ndarray]:
        """Compute F1 score."""
        precision = self.compute_precision()
        recall = self.compute_recall()

        if isinstance(precision, np.ndarray):
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            return f1.mean() if self.average == "macro" else f1
        else:
            return 2 * (precision * recall) / (precision + recall + 1e-10)

    def compute_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.confusion_matrix.copy()

    def compute_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each class."""
        precision = self.compute_precision()
        recall = self.compute_recall()
        f1 = self.compute_f1()

        if not isinstance(precision, np.ndarray):
            precision = np.array([precision])
            recall = np.array([recall])
            f1 = np.array([f1])

        metrics = {}
        for i in range(self.num_classes):
            metrics[f"class_{i}"] = {
                "precision": precision[i] if len(precision) > 1 else precision[0],
                "recall": recall[i] if len(recall) > 1 else recall[0],
                "f1": f1[i] if len(f1) > 1 else f1[0],
                "support": self.confusion_matrix[i, :].sum(),
            }
        return metrics

    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            "accuracy": self.compute_accuracy(),
            "precision": self.compute_precision()
            if isinstance(self.compute_precision(), float)
            else self.compute_precision().mean(),
            "recall": self.compute_recall()
            if isinstance(self.compute_recall(), float)
            else self.compute_recall().mean(),
            "f1": self.compute_f1()
            if isinstance(self.compute_f1(), float)
            else self.compute_f1().mean(),
        }


class AspectF1Evaluator:
    """
    F1 evaluation specifically for ABSA.

    Computes F1 for aspect extraction, opinion extraction,
    and aspect sentiment classification.

    Example:
        >>> evaluator = AspectF1Evaluator()
        >>> metrics = evaluator.compute_aspect_f1(pred_aspects, true_aspects)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset statistics."""
        self.aspect_predictions = []
        self.aspect_labels = []
        self.sentiment_predictions = []
        self.sentiment_labels = []

    def add_batch(
        self,
        pred_aspects: List[List[Tuple[str, int, int]]],
        true_aspects: List[List[Tuple[str, int, int]]],
        pred_sentiments: Optional[List[List[int]]] = None,
        true_sentiments: Optional[List[List[int]]] = None,
    ):
        """
        Add batch of predictions.

        Args:
            pred_aspects: List of predicted aspects per sample
            true_aspects: List of true aspects per sample
            pred_sentiments: Optional predicted sentiments
            true_sentiments: Optional true sentiments
        """
        self.aspect_predictions.extend(pred_aspects)
        self.aspect_labels.extend(true_aspects)

        if pred_sentiments is not None:
            self.sentiment_predictions.extend(pred_sentiments)
            self.sentiment_labels.extend(true_sentiments)

    def compute_exact_match_f1(self) -> Dict[str, float]:
        """Compute exact match F1 for aspect extraction."""
        tp, fp, fn = 0, 0, 0

        for pred, true in zip(self.aspect_predictions, self.aspect_labels):
            pred_set = set(pred)
            true_set = set(true)

            tp += len(pred_set & true_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_partial_match_f1(
        self, overlap_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute partial match F1 with overlap threshold."""
        tp, fp, fn = 0, 0, 0

        for pred, true in zip(self.aspect_predictions, self.aspect_labels):
            matched_true = set()

            for p_aspect in pred:
                p_text, p_start, p_end = p_aspect

                matched = False
                for t_aspect in true:
                    if t_aspect in matched_true:
                        continue
                    t_text, t_start, t_end = t_aspect

                    overlap_start = max(p_start, t_start)
                    overlap_end = min(p_end, t_end)
                    overlap_len = max(0, overlap_end - overlap_start)

                    p_len = p_end - p_start
                    t_len = t_end - t_start

                    if overlap_len / max(p_len, t_len) >= overlap_threshold:
                        matched = True
                        matched_true.add(t_aspect)
                        break

                if matched:
                    tp += 1
                else:
                    fp += 1

            fn += len(true) - len(matched_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute_sentiment_accuracy(self) -> float:
        """Compute sentiment classification accuracy."""
        if not self.sentiment_predictions:
            return 0.0

        correct = sum(
            sum(p == t for p, t in zip(pred, true))
            for pred, true in zip(self.sentiment_predictions, self.sentiment_labels)
        )
        total = sum(len(true) for true in self.sentiment_labels)

        return correct / total if total > 0 else 0.0


# ============================================================================
# Training Utilities
# ============================================================================


class SentimentDataset(Dataset):
    """
    Dataset for sentiment analysis with support for various formats.

    Supports IMDB, SST, and Yelp dataset formats.

    Args:
        texts: List of text samples
        labels: List of labels
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        format_type: Dataset format type

    Example:
        >>> dataset = SentimentDataset(texts, labels, tokenizer)
        >>> sample = dataset[0]
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
        format_type: str = "standard",
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)

        return item


class ABSA_Dataset(Dataset):
    """
    Dataset for Aspect-Based Sentiment Analysis.

    Args:
        texts: List of texts
        aspect_lists: List of aspect terms per text
        sentiment_lists: List of sentiments per aspect
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Example:
        >>> dataset = ABSA_Dataset(texts, aspects, sentiments, tokenizer)
    """

    def __init__(
        self,
        texts: List[str],
        aspect_lists: List[List[str]],
        sentiment_lists: List[List[int]],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = texts
        self.aspect_lists = aspect_lists
        self.sentiment_lists = sentiment_lists
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        aspects = self.aspect_lists[idx]
        sentiments = self.sentiment_lists[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["aspects"] = aspects
        item["sentiments"] = torch.tensor(sentiments, dtype=torch.long)

        return item


class ClassWeights:
    """
    Compute class weights for imbalanced datasets.

    Example:
        >>> weights = ClassWeights.compute_weights(labels, method='inverse')
    """

    @staticmethod
    def compute_weights(
        labels: List[int],
        num_classes: int,
        method: str = "inverse",
    ) -> Tensor:
        """
        Compute class weights.

        Args:
            labels: List of class labels
            num_classes: Number of classes
            method: Weight computation method ('inverse', 'sqrt_inv', 'effective')

        Returns:
            Class weights tensor
        """
        counts = Counter(labels)

        weights = torch.zeros(num_classes)

        if method == "inverse":
            for i in range(num_classes):
                weights[i] = 1.0 / (counts.get(i, 1))
        elif method == "sqrt_inv":
            for i in range(num_classes):
                weights[i] = 1.0 / np.sqrt(counts.get(i, 1))
        elif method == "effective":
            beta = 0.9999
            for i in range(num_classes):
                count = counts.get(i, 1)
                weights[i] = (1 - beta) / (1 - beta**count)
        else:
            weights = torch.ones(num_classes)

        weights = weights / weights.sum() * num_classes

        return weights


class DataAugmentation:
    """
    Data augmentation for sentiment analysis.

    Implements paraphrasing, back-translation, and synonym replacement.

    Example:
        >>> aug = DataAugmentation()
        >>> augmented = aug.back_translate(texts, translator)
    """

    def __init__(self, aug_probability: float = 0.3):
        self.aug_probability = aug_probability

    def synonym_replacement(
        self,
        text: str,
        n: int = 2,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Replace words with synonyms.

        Args:
            text: Input text
            n: Number of words to replace
            synonym_dict: Dictionary of word -> synonyms

        Returns:
            Augmented text
        """
        if synonym_dict is None:
            synonym_dict = {
                "good": ["great", "excellent", "nice"],
                "bad": ["terrible", "awful", "poor"],
                "happy": ["joyful", "cheerful", "glad"],
                "sad": ["unhappy", "sorrowful", "gloomy"],
                "big": ["large", "huge", "enormous"],
                "small": ["tiny", "little", "miniature"],
                "like": ["love", "enjoy", "appreciate"],
                "hate": ["dislike", "despise", "loathe"],
            }

        words = text.split()
        new_words = words.copy()

        replaceable = [i for i, w in enumerate(words) if w.lower() in synonym_dict]

        if not replaceable:
            return text

        n = min(n, len(replaceable))
        indices = random.sample(replaceable, n)

        for idx in indices:
            word = words[idx].lower()
            synonyms = synonym_dict[word]
            new_word = random.choice(synonyms)

            if words[idx][0].isupper():
                new_word = new_word.capitalize()
            new_words[idx] = new_word

        return " ".join(new_words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()

        if len(words) == 1:
            return text

        new_words = [w for w in words if random.random() > p]

        if not new_words:
            return random.choice(words)

        return " ".join(new_words)

    def random_swap(self, text: str, n: int = 2) -> str:
        """Randomly swap n pairs of words."""
        words = text.split()

        if len(words) < 2:
            return text

        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)

    def back_translate(
        self,
        texts: List[str],
        translator: Callable[[List[str], str, str], List[str]],
        intermediate_lang: str = "fr",
        source_lang: str = "en",
    ) -> List[str]:
        """
        Back-translation augmentation.

        Translates text to intermediate language and back.

        Args:
            texts: List of texts to augment
            translator: Translation function
            intermediate_lang: Intermediate language code
            source_lang: Source language code

        Returns:
            Augmented texts
        """
        translated = translator(texts, source_lang, intermediate_lang)
        back_translated = translator(translated, intermediate_lang, source_lang)

        return back_translated

    def augment_batch(
        self,
        texts: List[str],
        labels: List[int],
        augment_ratio: float = 0.3,
    ) -> Tuple[List[str], List[int]]:
        """
        Augment a batch of texts.

        Args:
            texts: Original texts
            labels: Original labels
            augment_ratio: Ratio of samples to augment

        Returns:
            Augmented texts and labels
        """
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()

        n_augment = int(len(texts) * augment_ratio)
        indices = random.sample(range(len(texts)), n_augment)

        for idx in indices:
            text = texts[idx]
            label = labels[idx]

            aug_type = random.choice(["synonym", "deletion", "swap"])

            if aug_type == "synonym":
                new_text = self.synonym_replacement(text)
            elif aug_type == "deletion":
                new_text = self.random_deletion(text)
            else:
                new_text = self.random_swap(text)

            augmented_texts.append(new_text)
            augmented_labels.append(label)

        return augmented_texts, augmented_labels


class SentimentTrainer:
    """
    Specialized trainer for sentiment analysis models.

    Handles training loop, evaluation, checkpointing, and logging.

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device to train on

    Example:
        >>> trainer = SentimentTrainer(model, train_ds, val_ds, optimizer)
        >>> trainer.train(epochs=5)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        class_weights: Optional[Tensor] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 5,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        else:
            self.val_loader = None

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            batch = {
                k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }

            labels = batch.pop("labels")

            outputs = self.model(**batch)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }

                labels = batch.pop("labels")

                outputs = self.model(**batch)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self, epochs: int, save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            save_path: Path to save best model

        Returns:
            Training history
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                if save_path:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                        },
                        save_path,
                    )
                    print(f"  Saved best model to {save_path}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        return self.history


class WeightedRandomSampler(Sampler):
    """
    Weighted random sampler for imbalanced datasets.

    Example:
        >>> sampler = WeightedRandomSampler(labels, num_samples)
        >>> loader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(self, weights: Tensor, num_samples: int, replacement: bool = True):
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(
            torch.multinomial(self.weights, self.num_samples, self.replacement).tolist()
        )

    def __len__(self):
        return self.num_samples


# ============================================================================
# Utility Functions
# ============================================================================


def get_model(model_name: str, num_classes: int = 3, **kwargs) -> nn.Module:
    """
    Factory function to get sentiment models by name.

    Args:
        model_name: Name of the model
        num_classes: Number of sentiment classes
        **kwargs: Additional model arguments

    Returns:
        Model instance

    Example:
        >>> model = get_model('bert', num_classes=3)
        >>> model = get_model('bilstm_attention', vocab_size=10000)
    """
    models = {
        "lstm": BasicLSTM,
        "bilstm_attention": BiLSTMAttention,
        "cnn": CNNText,
        "bert": BERTSentiment,
        "roberta": RoBERTaSentiment,
        "xlnet": XLNetSentiment,
        "distilbert": DistilBERTSentiment,
        "bert_absa": BertABSA,
        "gcn_absa": GCN_ABSA,
        "lcf": LCF,
        "aen": AEN,
        "emotion": EmotionDetection,
        "sarcasm": SarcasmDetection,
        "stance": StanceDetection,
        "intent": IntentClassification,
        "zeroshot": ZeroShotSentiment,
        "xlm": CrossLingualBERT,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    return models[model_name](num_classes=num_classes, **kwargs)


def create_optimizer(
    model: nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """
    Create optimizer for sentiment models.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        optimizer_type: Type of optimizer

    Returns:
        Optimizer instance
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_type == "adamw":
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    elif optimizer_type == "adam":
        return torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Types
    "SentimentOutput",
    "AspectOutput",
    "MultimodalOutput",
    "SentimentLabel",
    "EmotionLabel",
    "AspectLabel",
    "StanceLabel",
    # Sentiment Classification
    "BasicLSTM",
    "BiLSTMAttention",
    "CNNText",
    "BERTSentiment",
    "RoBERTaSentiment",
    "XLNetSentiment",
    "DistilBERTSentiment",
    # Aspect-Based
    "AspectExtractor",
    "AspectSentiment",
    "GCN_ABSA",
    "LCF",
    "AEN",
    "BertABSA",
    # Multimodal
    "TextImageSentiment",
    "VideoSentiment",
    "AudioSentiment",
    "MAG",
    "MulT",
    # Fine-grained
    "SentimentIntensity",
    "EmotionDetection",
    "SarcasmDetection",
    "StanceDetection",
    "IntentClassification",
    # Social Media
    "EmojiHandling",
    "HashtagProcessing",
    "MentionProcessing",
    "URLRemoval",
    "SlangNormalization",
    "SocialMediaPreprocessor",
    # Cross-lingual
    "ZeroShotSentiment",
    "CrossLingualBERT",
    "TranslationPivot",
    # Evaluation
    "SentimentEvaluator",
    "AspectF1Evaluator",
    # Training
    "SentimentDataset",
    "ABSA_Dataset",
    "ClassWeights",
    "DataAugmentation",
    "SentimentTrainer",
    "WeightedRandomSampler",
    # Utilities
    "get_model",
    "create_optimizer",
]
