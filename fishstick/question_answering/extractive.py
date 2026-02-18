"""
Extractive Question Answering Implementation

This module provides implementations for extractive QA systems including
BERT-based models, BiDAF+, and document ranking.

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
from fishstick.question_answering.base import ExtractiveQABase


class SpanExtractor(nn.Module):
    """Span extraction head for extractive QA.

    Extracts answer spans from encoded question and context representations
    by predicting start and end positions.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize SpanExtractor.

        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.start_logits = nn.Linear(hidden_size, 1)
        self.end_logits = nn.Linear(hidden_size, 1)

        self.answer_type_logits = nn.Linear(hidden_size, 4)

    def forward(
        self,
        sequence_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract spans from hidden states.

        Args:
            sequence_hidden: [batch, seq_len, hidden]
            attention_mask: [batch, seq_len]

        Returns:
            Tuple of (start_logits, end_logits, answer_type_logits)
        """
        hidden = self.dropout(sequence_hidden)

        start_logits = self.start_logits(hidden).squeeze(-1)
        end_logits = self.end_logits(hidden).squeeze(-1)

        answer_type_logits = self.answer_type_logits(sequence_hidden[:, 0])

        if attention_mask is not None:
            start_logits = start_logits.masked_fill(attention_mask == 0, float("-inf"))
            end_logits = end_logits.masked_fill(attention_mask == 0, float("-inf"))

        return start_logits, end_logits, answer_type_logits

    def predict_spans(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        max_answer_length: int = 30,
        n_best_size: int = 20,
    ) -> List[List[Dict[str, Any]]]:
        """Predict answer spans from logits.

        Args:
            start_logits: [batch, seq_len]
            end_logits: [batch, seq_len]
            max_answer_length: Maximum answer length
            n_best_size: Number of best spans to return

        Returns:
            List of best spans for each batch item
        """
        batch_size = start_logits.size(0)
        results = []

        for i in range(batch_size):
            start_logit = start_logits[i]
            end_logit = end_logits[i]

            start_indexes = torch.argsort(start_logit, descending=True)[:n_best_size]
            end_indexes = torch.argsort(end_logit, descending=True)[:n_best_size]

            best_spans = []

            for start_idx in start_indexes:
                for end_idx in end_indexes:
                    if start_idx > end_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue

                    score = start_logit[start_idx] + end_logit[end_idx]
                    best_spans.append(
                        {
                            "start_idx": start_idx.item(),
                            "end_idx": end_idx.item(),
                            "score": score.item(),
                        }
                    )

            best_spans = sorted(best_spans, key=lambda x: x["score"], reverse=True)[
                :n_best_size
            ]
            results.append(best_spans)

        return results


class BERTExtractiveQA(ExtractiveQABase[nn.Module]):
    """BERT-based Extractive QA Model.

    Implements extractive QA using BERT as the encoder with span prediction.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize BERT Extractive QA.

        Args:
            config: QA configuration
        """
        super().__init__(config)
        self.hidden_size = config.metadata.get("hidden_size", 768)

        try:
            from transformers import AutoModel, AutoConfig

            model_cfg = AutoConfig.from_pretrained(config.model_name)
            self.encoder = AutoModel.from_pretrained(config.model_name)
            self.hidden_size = model_cfg.hidden_size
        except ImportError:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=8,
                    dim_feedforward=self.hidden_size * 4,
                ),
                num_layers=6,
            )

        self.span_extractor = SpanExtractor(
            hidden_size=self.hidden_size,
            dropout=config.metadata.get("dropout", 0.1),
        )

        self.tokenizer: Optional[Any] = None
        self.model = nn.ModuleDict(
            {
                "encoder": self.encoder,
                "span_extractor": self.span_extractor,
            }
        )

    def encode(
        self,
        questions: List[str],
        contexts: List[str],
    ) -> Dict[str, Tensor]:
        """Encode question and context.

        Args:
            questions: List of question strings
            contexts: List of context strings

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                questions,
                contexts,
                padding=True,
                truncation="only_second",
                max_length=self.max_seq_length,
                stride=self.doc_stride,
                return_tensors="pt",
            )
            return encoding

        raise RuntimeError("Tokenizer not set. Call set_tokenizer() first.")

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set tokenizer for the model.

        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer

    def extract_spans(
        self,
        question_hidden: Tensor,
        context_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract answer spans from hidden states.

        Args:
            question_hidden: Question hidden states
            context_hidden: Context hidden states
            attention_mask: Attention mask

        Returns:
            Tuple of (start_logits, end_logits)
        """
        sequence_hidden = torch.cat([question_hidden, context_hidden], dim=1)
        seq_attention_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(sequence_hidden[:, :, 0])
        )

        start_logits, end_logits, _ = self.span_extractor(
            sequence_hidden, seq_attention_mask
        )
        return start_logits, end_logits

    def predict_spans(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        context_tokens: List[str],
        n_best_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Predict answer spans from logits.

        Args:
            start_logits: Logits for start positions
            end_logits: Logits for end positions
            context_tokens: List of context tokens
            n_best_size: Number of best spans to return

        Returns:
            List of best spans with scores
        """
        return self.span_extractor.predict_spans(
            start_logits,
            end_logits,
            max_answer_length=self.max_answer_length,
            n_best_size=n_best_size,
        )

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

        encoding = self.encode([q_text], [c_text])
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.encoder(input_ids, attention_mask)
        sequence_hidden = outputs.last_hidden_state

        start_logits, end_logits, _ = self.span_extractor(
            sequence_hidden, attention_mask
        )

        best_spans = self.span_extractor.predict_spans(
            start_logits.unsqueeze(0),
            end_logits.unsqueeze(0),
            max_answer_length=self.max_answer_length,
            n_best_size=self.config.n_best_size,
        )

        if not best_spans or not best_spans[0]:
            return Answer(text="", type=AnswerType.SPAN, confidence=0.0)

        best = best_spans[0][0]

        tokens = (
            self.tokenizer.convert_ids_to_tokens(input_ids[0]) if self.tokenizer else []
        )
        answer_text = "".join(tokens[best["start_idx"] : best["end_idx"] + 1])
        answer_text = (
            answer_text.replace("##", "")
            .replace("[CLS]", "")
            .replace("[SEP]", "")
            .replace(" ", "")
        )

        return Answer(
            text=answer_text,
            type=AnswerType.SPAN,
            confidence=torch.sigmoid(torch.tensor(best["score"])).item(),
            start_char=best["start_idx"],
            end_char=best["end_idx"],
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
                "model_state": self.model.state_dict(),
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
        self.model.load_state_dict(checkpoint["model_state"])


class BiDAFPlus(nn.Module):
    """BiDAF+ - Bidirectional Attention Flow with enhancements.

    Implements enhanced BiDAF with contextual embeddings and attention.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize BiDAF+.

        Args:
            hidden_size: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.contextualizer = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.query_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.context_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.modeling_layers = nn.ModuleList(
            [
                nn.LSTM(
                    hidden_size * 4,
                    hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )

        self.span_predictor = SpanExtractor(hidden_size * 2, dropout)

    def forward(
        self,
        context_hidden: Tensor,
        query_hidden: Tensor,
        context_mask: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of BiDAF+.

        Args:
            context_hidden: [batch, ctx_len, hidden]
            query_hidden: [batch, query_len, hidden]
            context_mask: [batch, ctx_len]
            query_mask: [batch, query_len]

        Returns:
            Tuple of (start_logits, end_logits)
        """
        context_ctx, _ = self.contextualizer(context_hidden)
        query_ctx, _ = self.contextualizer(query_hidden)

        query_attended, _ = self.query_attention(
            query_ctx,
            context_ctx,
            context_ctx,
            key_padding_mask=context_mask,
        )

        context_attended, _ = self.context_attention(
            context_ctx,
            query_ctx,
            query_ctx,
            key_padding_mask=query_mask,
        )

        merged = torch.cat(
            [
                context_ctx,
                query_attended,
                context_ctx * query_attended,
                context_ctx - query_attended,
            ],
            dim=-1,
        )

        for layer in self.modeling_layers:
            merged, _ = layer(merged)

        start_logits, end_logits, _ = self.span_predictor(merged, context_mask)

        return start_logits, end_logits


class DocumentRanker(nn.Module):
    """Document/Passage Ranker for QA.

    Ranks documents or passages by relevance to a query.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """Initialize DocumentRanker.

        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.query_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.doc_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.similarity = nn.Bilinear(
            hidden_size,
            hidden_size,
            1,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

    def encode_query(
        self,
        query_hidden: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode query representation.

        Args:
            query_hidden: [batch, seq_len, hidden]
            mask: [batch, seq_len]

        Returns:
            Query representation [batch, hidden]
        """
        output, (h_n, _) = self.query_encoder(query_hidden)
        query_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return query_repr

    def encode_document(
        self,
        doc_hidden: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode document representation.

        Args:
            doc_hidden: [batch, seq_len, hidden]
            mask: [batch, seq_len]

        Returns:
            Document representation [batch, hidden]
        """
        output, (h_n, _) = self.doc_encoder(doc_hidden)
        doc_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return doc_repr

    def compute_relevance(
        self,
        query_repr: Tensor,
        doc_repr: Tensor,
    ) -> Tensor:
        """Compute relevance score between query and document.

        Args:
            query_repr: [batch, hidden]
            doc_repr: [batch, hidden]

        Returns:
            Relevance scores [batch]
        """
        query_repr = self.dropout(query_repr)
        doc_repr = self.dropout(doc_repr)

        scores = self.similarity(query_repr, doc_repr).squeeze(-1)
        return scores

    def forward(
        self,
        query_hidden: Tensor,
        doc_hidden: Tensor,
        query_mask: Optional[Tensor] = None,
        doc_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass to compute relevance scores.

        Args:
            query_hidden: [batch, query_len, hidden]
            doc_hidden: [batch, doc_len, hidden]
            query_mask: [batch, query_len]
            doc_mask: [batch, doc_len]

        Returns:
            Relevance scores [batch]
        """
        query_repr = self.encode_query(query_hidden, query_mask)
        doc_repr = self.encode_document(doc_hidden, doc_mask)

        return self.compute_relevance(query_repr, doc_repr)


class CrossAttentionQA(nn.Module):
    """Cross-Attention based QA Model.

    Uses cross-attention between question and context for span extraction.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Cross-Attention QA.

        Args:
            hidden_size: Hidden dimension size
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        self.cross_attention_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.span_extractor = SpanExtractor(hidden_size, dropout)

    def forward(
        self,
        question_hidden: Tensor,
        context_hidden: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with cross-attention.

        Args:
            question_hidden: [batch, query_len, hidden]
            context_hidden: [batch, ctx_len, hidden]
            attention_mask: [batch, ctx_len]

        Returns:
            Tuple of (start_logits, end_logits)
        """
        q = self.query_proj(question_hidden)
        k = self.key_proj(context_hidden)
        v = self.value_proj(context_hidden)

        cross_attn_output = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_size**0.5)
        cross_attn_output = F.softmax(cross_attn_output, dim=-1)
        attended = torch.bmm(cross_attn_output, v)

        merged = torch.cat([context_hidden, attended], dim=-1)
        merged = merged + context_hidden

        for layer in self.cross_attention_layers:
            merged = layer(merged)

        start_logits, end_logits, _ = self.span_extractor(merged, attention_mask)

        return start_logits, end_logits
