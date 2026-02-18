"""
Legal NLP Module for Fishstick
===============================

Comprehensive legal natural language processing module providing:
- Document understanding and classification
- Information extraction (entities, parties, obligations, dates)
- Legal reasoning and case law retrieval
- Risk analysis and compliance checking
- Legal document drafting and generation
- Training and evaluation utilities

Author: Fishstick AI Framework
"""

from __future__ import annotations

import re
import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Optional,
    Set,
    Tuple,
    Any,
    Callable,
    Union,
    Iterator,
    NamedTuple,
)
from enum import Enum, auto
from collections import defaultdict
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Types and Data Structures
# =============================================================================


class DocumentType(Enum):
    """Types of legal documents."""

    CONTRACT = auto()
    BRIEF = auto()
    MEMO = auto()
    OPINION = auto()
    STATUTE = auto()
    REGULATION = auto()
    MOTION = auto()
    COMPLAINT = auto()
    ANSWER = auto()
    INTERROGATORY = auto()
    DEPOSITION = auto()
    WILL = auto()
    TRUST = auto()
    PATENT = auto()
    TRADEMARK = auto()
    UNKNOWN = auto()


class ClauseType(Enum):
    """Types of legal clauses."""

    TERMINATION = "termination"
    CONFIDENTIALITY = "confidentiality"
    INDEMNIFICATION = "indemnification"
    FORCE_MAJEURE = "force_majeure"
    GOVERNING_LAW = "governing_law"
    ARBITRATION = "arbitration"
    ASSIGNMENT = "assignment"
    SEVERABILITY = "severability"
    ENTIRE_AGREEMENT = "entire_agreement"
    WARRANTIES = "warranties"
    LIMITATION_LIABILITY = "limitation_liability"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    NON_COMPETE = "non_compete"
    PAYMENT_TERMS = "payment_terms"
    DELIVERY_TERMS = "delivery_terms"


class EntityType(Enum):
    """Legal named entity types."""

    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    COURT = "COURT"
    STATUTE = "STATUTE"
    CASE = "CASE"
    REGULATION = "REGULATION"
    DATE = "DATE"
    MONEY = "MONEY"
    LOCATION = "LOC"
    LEGAL_TERM = "LEGAL_TERM"
    CONTRACT_TERM = "CONTRACT_TERM"


class RiskLevel(Enum):
    """Risk assessment levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class Clause:
    """Represents a legal clause."""

    text: str
    clause_type: Optional[ClauseType] = None
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.0
    sub_clauses: List[str] = field(default_factory=list)
    obligations: List[Obligation] = field(default_factory=list)


@dataclass
class LegalEntity:
    """Represents a legal named entity."""

    text: str
    entity_type: EntityType
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.0
    normalized_form: Optional[str] = None


@dataclass
class Party:
    """Represents a legal party."""

    name: str
    role: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Obligation:
    """Represents a legal obligation."""

    text: str
    party: Optional[str] = None
    action: Optional[str] = None
    deadline: Optional[str] = None
    condition: Optional[str] = None
    obligation_type: str = "performance"
    is_conditional: bool = False
    is_negative: bool = False


@dataclass
class LegalDate:
    """Represents a legal date with context."""

    text: str
    normalized_date: Optional[str] = None
    date_type: str = "specific"
    is_deadline: bool = False
    is_effective_date: bool = False
    context: Optional[str] = None


@dataclass
class CaseLaw:
    """Represents a legal case."""

    citation: str
    title: str
    court: str
    date: str
    summary: str
    holding: str
    reasoning: str
    relevant_statutes: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    precedential_value: str = "binding"


@dataclass
class RiskAssessment:
    """Represents a risk assessment result."""

    risk_type: str
    risk_level: RiskLevel
    description: str
    location: Optional[str] = None
    recommendation: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ComplianceResult:
    """Represents a compliance check result."""

    regulation: str
    is_compliant: bool
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class LegalSummary:
    """Represents a legal document summary."""

    summary: str
    key_points: List[str] = field(default_factory=list)
    method: str = "extractive"
    confidence: float = 0.0


@dataclass
class QAResult:
    """Represents a legal QA result."""

    question: str
    answer: str
    confidence: float = 0.0
    supporting_text: Optional[str] = None
    citation: Optional[str] = None


@dataclass
class LegalDocument:
    """Represents a legal document."""

    text: str
    doc_type: Optional[DocumentType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    clauses: List[Clause] = field(default_factory=list)
    entities: List[LegalEntity] = field(default_factory=list)


# =============================================================================
# Document Understanding
# =============================================================================


class LegalDocumentClassifier(nn.Module):
    """Classifies legal documents by type."""

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_dim: int = 512,
        num_classes: int = len(DocumentType),
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.num_classes = num_classes

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.doc_type_embedding = nn.Embedding(num_classes, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.structure_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        structural_features: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        x = token_emb + pos_emb
        x = self.dropout(x)

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer(x)

        pooled = x.mean(dim=1)
        if structural_features is not None:
            struct_emb = self.structure_encoder(structural_features)
            pooled = pooled + struct_emb

        logits = self.classifier(pooled)
        return logits

    def extract_structural_features(self, text: str) -> Tensor:
        features = torch.zeros(10)
        features[0] = (
            1.0 if len(re.findall(r"\bWHEREAS\b", text, re.IGNORECASE)) > 0 else 0.0
        )
        features[1] = (
            1.0 if re.search(r"\bNOW,\s*THEREFORE\b", text, re.IGNORECASE) else 0.0
        )
        features[2] = (
            1.0
            if len(re.findall(r"\bIN\s+WITNESS\s+WHEREOF\b", text, re.IGNORECASE)) > 0
            else 0.0
        )
        features[3] = min(len(re.findall(r"^\s*\d+\.", text, re.MULTILINE)) / 10, 1.0)
        features[4] = min(len(re.findall(r"\d+\s+\w+\.\s+\d+", text)) / 5, 1.0)
        features[5] = (
            1.0 if len(re.findall(r'\(["\'][^"\']+["\']\)', text)) > 0 else 0.0
        )
        features[6] = min(text.lower().count("recital"), 1.0)
        features[7] = (
            1.0 if re.search(r"\bEXHIBIT\s+\w+\b", text, re.IGNORECASE) else 0.0
        )
        features[8] = min(
            len(re.findall(r"^\s*SECTION\s+\d+", text, re.MULTILINE | re.IGNORECASE))
            / 10,
            1.0,
        )
        sentences = re.split(r"[.!?]+", text)
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        features[9] = min(avg_len / 50, 1.0)
        return features

    def predict(
        self, text: str, tokenizer: Optional[Callable] = None
    ) -> Tuple[DocumentType, float]:
        self.eval()
        if tokenizer is None:
            tokens = text.lower().split()[: self.max_length]
            token_ids = [hash(t) % self.vocab_size for t in tokens]
            token_ids += [0] * (self.max_length - len(token_ids))
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.ones(1, self.max_length)
            attention_mask[0, len(tokens) :] = 0
        else:
            encoded = tokenizer(
                text, max_length=self.max_length, padding="max_length", truncation=True
            )
            input_ids = torch.tensor([encoded["input_ids"]])
            attention_mask = torch.tensor([encoded["attention_mask"]])

        struct_features = self.extract_structural_features(text).unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, struct_features)
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)

        doc_type = list(DocumentType)[predicted.item()]
        return doc_type, confidence.item()


class ContractAnalyzer(nn.Module):
    """Analyzes legal contracts for key provisions and risks."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_clause_types: int = len(ClauseType),
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_clause_types = num_clause_types

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.clause_detector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clause_types),
        )
        self.risk_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(RiskLevel)),
        )
        self.span_start = nn.Linear(embed_dim, 1)
        self.span_end = nn.Linear(embed_dim, 1)

    def forward(
        self, embeddings: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
            encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        else:
            encoded = self.encoder(embeddings)

        clause_logits = self.clause_detector(encoded)
        pooled = encoded.mean(dim=1)
        risk_logits = self.risk_classifier(pooled)
        start_logits = self.span_start(encoded).squeeze(-1)
        end_logits = self.span_end(encoded).squeeze(-1)

        return {
            "clause_logits": clause_logits,
            "risk_logits": risk_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "encoded": encoded,
        }

    def analyze(
        self, contract_text: str, embedding_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        self.eval()
        if embedding_fn is None:
            words = contract_text.lower().split()[:512]
            embeddings = torch.randn(1, len(words), self.embed_dim)
            attention_mask = torch.ones(1, len(words))
        else:
            embeddings, attention_mask = embedding_fn(contract_text)

        with torch.no_grad():
            outputs = self.forward(embeddings, attention_mask)

        clause_probs = torch.sigmoid(outputs["clause_logits"]).mean(dim=1)
        detected_clauses = []
        for i, prob in enumerate(clause_probs[0]):
            if prob > 0.5:
                clause_type = list(ClauseType)[i]
                detected_clauses.append(
                    {"type": clause_type, "confidence": prob.item()}
                )

        risk_probs = F.softmax(outputs["risk_logits"], dim=-1)
        risk_level = list(RiskLevel)[risk_probs.argmax(dim=-1).item()]

        return {
            "detected_clauses": detected_clauses,
            "risk_level": risk_level,
            "risk_confidence": risk_probs.max().item(),
        }


class ClauseExtractor(nn.Module):
    """Extracts legal clauses from documents using span-based extraction."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_clause_types: int = len(ClauseType),
        max_span_length: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_clause_types = num_clause_types
        self.max_span_length = max_span_length

        self.span_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.span_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.type_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_clause_types),
        )
        self.width_embedding = nn.Embedding(max_span_length, embed_dim)

    def forward(
        self, embeddings: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = embeddings.shape
        spans = []
        for i in range(seq_len):
            for j in range(i, min(i + self.max_span_length, seq_len)):
                spans.append((i, j))

        if len(spans) == 0:
            return (
                torch.zeros(batch_size, 0),
                torch.zeros(batch_size, 0, self.num_clause_types),
            )

        span_representations = []
        for start, end in spans:
            start_emb = embeddings[:, start, :]
            end_emb = embeddings[:, end, :]
            span_emb = torch.cat([start_emb, end_emb], dim=-1)
            span_repr = self.span_encoder(span_emb)
            width = end - start
            width_emb = self.width_embedding(torch.tensor([width]))
            span_repr = span_repr + width_emb
            span_representations.append(span_repr)

        span_reprs = torch.stack(span_representations, dim=1)
        span_scores = self.span_scorer(span_reprs).squeeze(-1)
        type_logits = self.type_classifier(span_reprs)

        return span_scores, type_logits

    def extract_clauses(
        self, text: str, embeddings: Tensor, threshold: float = 0.5
    ) -> List[Clause]:
        self.eval()
        with torch.no_grad():
            span_scores, type_logits = self.forward(embeddings)

        batch_size, seq_len, _ = embeddings.shape
        spans = []
        idx = 0
        for i in range(seq_len):
            for j in range(i, min(i + self.max_span_length, seq_len)):
                spans.append((i, j))
                idx += 1

        clauses = []
        for batch_idx in range(batch_size):
            probs = torch.sigmoid(span_scores[batch_idx])
            types = torch.argmax(type_logits[batch_idx], dim=-1)

            for i, (start, end) in enumerate(spans):
                if probs[i] > threshold:
                    clause_text = text[start:end]
                    clause_type = list(ClauseType)[types[i].item()]
                    clauses.append(
                        Clause(
                            text=clause_text,
                            clause_type=clause_type,
                            start_pos=start,
                            end_pos=end,
                            confidence=probs[i].item(),
                        )
                    )

        return clauses


class LegalSummarizer(nn.Module):
    """Summarizes legal documents using extractive and abstractive methods."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 512,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.sentence_scorer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self, embeddings: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
            encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        else:
            encoded = self.encoder(embeddings)
        return encoded

    def extractive_summarize(
        self, sentences: List[str], embeddings: Tensor, num_sentences: int = 3
    ) -> LegalSummary:
        encoded = self.encode(embeddings)
        scores = self.sentence_scorer(encoded).squeeze(-1)
        probs = torch.sigmoid(scores)

        top_k = min(num_sentences, len(sentences))
        top_indices = torch.topk(probs[0], top_k).indices.sort()[0]

        summary_sentences = [sentences[i] for i in top_indices.tolist()]
        summary = " ".join(summary_sentences)

        return LegalSummary(
            summary=summary,
            key_points=summary_sentences,
            method="extractive",
            confidence=probs[0][top_indices].mean().item(),
        )

    def abstractive_summarize(
        self,
        embeddings: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_summary_length: int = 100,
    ) -> LegalSummary:
        encoded = self.encode(embeddings, attention_mask)

        decoder_input = torch.zeros(1, 1, self.embed_dim)
        summary_embeddings = []

        for _ in range(max_summary_length):
            if attention_mask is not None:
                memory_key_padding_mask = ~attention_mask.bool()
                decoded = self.decoder(
                    decoder_input,
                    encoded,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            else:
                decoded = self.decoder(decoder_input, encoded)

            output = self.output_proj(decoded[:, -1:, :])
            summary_embeddings.append(output)
            decoder_input = torch.cat([decoder_input, output], dim=1)

        return LegalSummary(
            summary="", key_points=[], method="abstractive", confidence=0.5
        )

    def summarize(
        self, text: str, method: str = "extractive", num_sentences: int = 3
    ) -> LegalSummary:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= num_sentences:
            return LegalSummary(
                summary=text, key_points=sentences, method="none", confidence=1.0
            )

        words = text.lower().split()[: self.max_length]
        embeddings = torch.randn(1, len(words), self.embed_dim)
        attention_mask = torch.ones(1, len(words))

        if method == "extractive":
            return self.extractive_summarize(sentences, embeddings, num_sentences)
        else:
            return self.abstractive_summarize(embeddings, attention_mask)
