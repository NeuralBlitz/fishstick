"""
Retrieval and Pipeline Components for QA Systems

This module provides implementations for dense retrieval, hybrid retrieval,
and RAG pipeline integration.

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
    QATaskType,
)
from fishstick.question_answering.base import RetrieverBase, QAPipeline


class DenseRetriever(RetrieverBase):
    """Dense Passage Retriever.

    Uses dense embeddings for document retrieval.
    """

    def __init__(
        self,
        config: Optional[QAConfig] = None,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """Initialize DenseRetriever.

        Args:
            config: Optional QA configuration
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__(config)
        self.hidden_size = hidden_size

        self.query_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.passage_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.similarity = nn.CosineSimilarity(dim=-1)

        self.index: Optional[Dict[str, Tensor]] = None
        self.passage_ids: List[str] = []

    def encode_query(
        self,
        query: str,
    ) -> Tensor:
        """Encode query to dense representation.

        Args:
            query: Query string

        Returns:
            Query embedding
        """
        tokens = query.split()
        return torch.randn(1, len(tokens), self.hidden_size)

    def encode_passage(
        self,
        passage: str,
    ) -> Tensor:
        """Encode passage to dense representation.

        Args:
            passage: Passage string

        Returns:
            Passage embedding
        """
        tokens = passage.split()
        return torch.randn(1, len(tokens), self.hidden_size)

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of documents to index
        """
        self.passage_ids = [
            doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)
        ]

        embeddings = []

        for doc in documents:
            text = doc.get("text", doc.get("document", ""))
            emb = self.encode_passage(text)
            embeddings.append(emb.mean(dim=1))

        self.index = {
            "embeddings": torch.cat(embeddings, dim=0),
            "documents": documents,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        if self.index is None:
            return []

        query_emb = self.encode_query(query)
        query_vec = query_emb.mean(dim=1)

        doc_embs = self.index["embeddings"]

        scores = self.similarity(query_vec.unsqueeze(0), doc_embs.unsqueeze(0)).squeeze(
            0
        )

        top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))

        results = []

        for rank, (score, idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist())
        ):
            doc = self.index["documents"][idx]

            results.append(
                RetrievalResult(
                    document_id=self.passage_ids[idx],
                    document=doc.get("text", doc.get("document", "")),
                    score=score,
                    rank=rank,
                    title=doc.get("title"),
                    passage=doc.get("text", doc.get("document", ""))[:200],
                )
            )

        return results


class HybridRetriever(DenseRetriever):
    """Hybrid Retrieval System.

    Combines dense and sparse retrieval methods.
    """

    def __init__(
        self,
        config: Optional[QAConfig] = None,
        hidden_size: int = 768,
        alpha: float = 0.5,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            config: Optional QA configuration
            hidden_size: Hidden dimension size
            alpha: Weight for dense retrieval (1-alpha for sparse)
        """
        super().__init__(config, hidden_size)
        self.alpha = alpha

        self.vocabulary: Dict[str, int] = {}
        self.document_freq: Dict[str, int] = {}
        self.num_documents = 0

    def build_inverted_index(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """Build inverted index for sparse retrieval.

        Args:
            documents: List of documents
        """
        self.num_documents = len(documents)

        for doc in documents:
            text = doc.get("text", doc.get("document", "")).lower()
            tokens = text.split()
            unique_tokens = set(tokens)

            for token in unique_tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                    self.document_freq[token] = 0
                self.document_freq[token] += 1

    def compute_tfidf(
        self,
        query: str,
        document: str,
    ) -> float:
        """Compute TF-IDF score.

        Args:
            query: Query string
            document: Document string

        Returns:
            TF-IDF score
        """
        query_tokens = query.lower().split()
        doc_tokens = document.lower().split()

        doc_freq = {}
        for token in doc_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            if token in doc_freq:
                tf = doc_freq[token] / len(doc_tokens)
                idf = torch.log(
                    torch.tensor(self.num_documents)
                    / torch.tensor(self.document_freq.get(token, 1) + 1)
                )
                score += (tf * idf).item()

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid approach.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of retrieval results
        """
        if self.index is None:
            return []

        query_emb = self.encode_query(query)
        query_vec = query_emb.mean(dim=1)

        doc_embs = self.index["embeddings"]

        dense_scores = self.similarity(
            query_vec.unsqueeze(0), doc_embs.unsqueeze(0)
        ).squeeze(0)

        sparse_scores = []
        for doc in self.index["documents"]:
            text = doc.get("text", doc.get("document", ""))
            tfidf = self.compute_tfidf(query, text)
            sparse_scores.append(tfidf)

        sparse_scores = torch.tensor(sparse_scores, dtype=torch.float32)

        dense_norm = (dense_scores - dense_scores.min()) / (
            dense_scores.max() - dense_scores.min() + 1e-8
        )
        sparse_norm = (sparse_scores - sparse_scores.min()) / (
            sparse_scores.max() - sparse_scores.min() + 1e-8
        )

        combined_scores = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        top_scores, top_indices = torch.topk(
            combined_scores, min(top_k, len(combined_scores))
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(top_scores.tolist(), top_indices.tolist())
        ):
            doc = self.index["documents"][idx]

            results.append(
                RetrievalResult(
                    document_id=self.passage_ids[idx],
                    document=doc.get("text", doc.get("document", "")),
                    score=score,
                    rank=rank,
                    title=doc.get("title"),
                    passage=doc.get("text", doc.get("document", ""))[:200],
                    metadata={
                        "dense_score": dense_scores[idx].item(),
                        "sparse_score": sparse_scores[idx].item(),
                    },
                )
            )

        return results


class KnowledgeAugmentedQA(nn.Module):
    """Knowledge Augmented QA.

    Augments QA with external knowledge.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """Initialize KnowledgeAugmentedQA.

        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.knowledge_encoder = nn.LSTM(
            hidden_size,
            hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.knowledge_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def encode_knowledge(
        self,
        knowledge: List[str],
    ) -> Tensor:
        """Encode knowledge items.

        Args:
            knowledge: List of knowledge items

        Returns:
            Knowledge representations
        """
        embeddings = []

        for item in knowledge:
            tokens = item.split()
            emb = torch.randn(1, min(len(tokens), 100), self.hidden_size)
            embeddings.append(emb)

        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.zeros(1, 1, self.hidden_size)

    def augment_with_knowledge(
        self,
        question_hidden: Tensor,
        knowledge_hidden: Tensor,
    ) -> Tensor:
        """Augment question with knowledge.

        Args:
            question_hidden: [batch, q_len, hidden]
            knowledge_hidden: [num_knowledge, k_len, hidden]

        Returns:
            Augmented representations
        """
        knowledge_flat = knowledge_hidden.reshape(
            -1, knowledge_hidden.size(-1)
        ).unsqueeze(0)
        question_expanded = question_hidden.unsqueeze(1).expand(
            -1, knowledge_hidden.size(0), -1, -1
        )

        attended, _ = self.knowledge_attention(
            question_expanded.reshape(-1, question_hidden.size(1), self.hidden_size),
            knowledge_flat,
            knowledge_flat,
        )

        attended = attended.reshape(
            question_hidden.size(0), knowledge_hidden.size(0), -1, self.hidden_size
        )
        attended = attended.mean(dim=2)

        combined = torch.cat([question_hidden, attended], dim=-1)

        return self.fusion_layer(combined)


class RAGIntegration(QAPipeline):
    """Retrieval-Augmented Generation Pipeline.

    Combines retrieval with generative QA.
    """

    def __init__(
        self,
        config: QAConfig,
        retriever: Optional[DenseRetriever] = None,
    ) -> None:
        """Initialize RAG Integration.

        Args:
            config: QA configuration
            retriever: Retriever instance
        """
        super().__init__(config)

        self.retriever = retriever or DenseRetriever(config)

        from fishstick.question_answering.generative import T5GenerativeQA

        self.generator = T5GenerativeQA(config)

    def retrieve_and_answer(
        self,
        question: str,
        top_k_docs: int = 5,
    ) -> QAPrediction:
        """Retrieve documents and answer question.

        Args:
            question: Question to answer
            top_k_docs: Number of documents to retrieve

        Returns:
            Prediction with answer
        """
        retrieved = self.retriever.retrieve(question, top_k=top_k_docs)

        if not retrieved:
            answer = Answer(
                text="No relevant documents found.",
                type=AnswerType.FREE_FORM,
                confidence=0.0,
            )
            return QAPrediction(
                id="unknown",
                question=question,
                answer=answer,
                context_used="",
                retrieved_documents=[],
            )

        context = " ".join([r.document for r in retrieved[:3]])

        from fishstick.question_answering.types import Question, Context

        answer = self.generator.forward(
            Question(text=question),
            Context(text=context),
        )

        return QAPrediction(
            id="unknown",
            question=question,
            answer=answer,
            context_used=context,
            retrieved_documents=[r.document_id for r in retrieved],
            metadata={"retrieved_scores": [r.score for r in retrieved]},
        )

    def batch_retrieve_and_answer(
        self,
        questions: List[str],
        top_k_docs: int = 5,
    ) -> List[QAPrediction]:
        """Batch retrieve and answer.

        Args:
            questions: List of questions
            top_k_docs: Number of documents per question

        Returns:
            List of predictions
        """
        return [self.retrieve_and_answer(q, top_k_docs) for q in questions]


class CompleteQAPipeline(nn.Module):
    """Complete QA Pipeline.

    End-to-end QA pipeline with retrieval, processing, and generation.
    """

    def __init__(
        self,
        config: QAConfig,
    ) -> None:
        """Initialize Complete QA Pipeline.

        Args:
            config: QA configuration
        """
        super().__init__()

        self.config = config
        self.device = torch.device(config.device)

        self.retriever = DenseRetriever(config)

        from fishstick.question_answering.extractive import BERTExtractiveQA
        from fishstick.question_answering.generative import T5GenerativeQA

        if config.task_type == QATaskType.EXTRACTIVE:
            self.qa_model = BERTExtractiveQA(config)
        else:
            self.qa_model = T5GenerativeQA(config)

    def forward(
        self,
        question: str,
        top_k: int = 5,
    ) -> QAPrediction:
        """Run complete pipeline.

        Args:
            question: Question to answer
            top_k: Number of documents to retrieve

        Returns:
            Prediction with answer
        """
        retrieved = self.retriever.retrieve(question, top_k=top_k)

        if not retrieved:
            answer = Answer(
                text="No relevant documents found.",
                type=AnswerType.FREE_FORM,
                confidence=0.0,
            )
            return QAPrediction(
                id="unknown",
                question=question,
                answer=answer,
                context_used="",
                retrieved_documents=[],
            )

        context = retrieved[0].document

        from fishstick.question_answering.types import (
            Question as QAQuestion,
            Context as QAContext,
        )

        answer = self.qa_model.forward(
            QAQuestion(text=question),
            QAContext(text=context),
        )

        return QAPrediction(
            id="unknown",
            question=question,
            answer=answer,
            context_used=context,
            retrieved_documents=[r.document_id for r in retrieved],
        )
