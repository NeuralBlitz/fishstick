"""
Extractive Summarization Methods
================================

Implements various extractive summarization algorithms:
- TF-IDF based summarization
- TextRank (graph-based)
- LexRank (cosine similarity)
- Maximal Marginal Relevance (MMR)
- Latent Semantic Analysis (LSA)
- Cluster-based extraction

Author: Fishstick Team
"""

from __future__ import annotations

from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import heapq
import math

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .base import (
    SummarizerBase,
    SummaryResult,
    Document,
    SummaryConfig,
    SummarizationMethod,
)
from .utils import (
    SentenceTokenizer,
    WordTokenizer,
    StopwordFilter,
    TextPreprocessor,
)


class TFIDFSummarizer(SummarizerBase):
    """TF-IDF based extractive summarization.

    Ranks sentences by their TF-IDF scores to identify the most
    important sentences in a document.
    """

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        """Initialize TF-IDF summarizer.

        Args:
            config: Summarization configuration
            vectorizer: Pre-configured TF-IDF vectorizer
        """
        super().__init__(config)
        self.vectorizer = vectorizer or TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_df=0.95,
            min_df=2,
        )
        self.sentence_tokenizer = SentenceTokenizer()
        self.word_tokenizer = WordTokenizer()

    def summarize(
        self,
        text: str | Document,
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate summary using TF-IDF scores.

        Args:
            text: Input text or Document
            config: Optional configuration override

        Returns:
            SummaryResult with the generated summary
        """
        config = config or self.config
        text_str = text if isinstance(text, str) else text.text

        sentences = self.sentence_tokenizer.tokenize(text_str)
        if len(sentences) <= config.num_sentences:
            return self._create_result(
                summary=text_str,
                original_text=text_str,
                method=SummarizationMethod.EXTRACTIVE,
            )

        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        top_indices = self._get_top_sentences(sentence_scores, config.num_sentences)
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        selected_scores = [sentence_scores[i] for i in top_indices]

        summary = " ".join(selected_sentences)

        return self._create_result(
            summary=summary,
            original_text=text_str,
            method=SummarizationMethod.EXTRACTIVE,
            sentences=selected_sentences,
            scores=selected_scores,
            algorithm="tfidf",
        )

    def _get_top_sentences(
        self, scores: NDArray[np.float64], num_sentences: int
    ) -> List[int]:
        """Get indices of top-scoring sentences.

        Args:
            scores: Sentence scores
            num_sentences: Number of sentences to select

        Returns:
            List of sentence indices
        """
        if len(scores) <= num_sentences:
            return list(range(len(scores)))

        top_indices = np.argsort(scores)[::-1][:num_sentences]
        return top_indices.tolist()


class TextRankSummarizer(SummarizerBase):
    """TextRank algorithm for extractive summarization.

    Uses a graph-based ranking algorithm similar to PageRank to
    identify important sentences.
    """

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        damping: float = 0.85,
        max_iterations: int = 100,
        convergence_threshold: float = 0.0001,
    ):
        """Initialize TextRank summarizer.

        Args:
            config: Summarization configuration
            damping: Damping factor for PageRank
            max_iterations: Maximum iterations for convergence
            convergence_threshold: Convergence threshold
        """
        super().__init__(config)
        self.damping = damping
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
        )
        self.sentence_tokenizer = SentenceTokenizer()

    def summarize(
        self,
        text: str | Document,
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate summary using TextRank algorithm.

        Args:
            text: Input text or Document
            config: Optional configuration override

        Returns:
            SummaryResult with the generated summary
        """
        config = config or self.config
        text_str = text if isinstance(text, str) else text.text

        sentences = self.sentence_tokenizer.tokenize(text_str)
        if len(sentences) <= config.num_sentences:
            return self._create_result(
                summary=text_str,
                original_text=text_str,
                method=SummarizationMethod.EXTRACTIVE,
            )

        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        scores = self._pagerank(similarity_matrix)

        top_indices = self._get_top_sentences(scores, config.num_sentences)
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]

        summary = " ".join(selected_sentences)

        return self._create_result(
            summary=summary,
            original_text=text_str,
            method=SummarizationMethod.EXTRACTIVE,
            sentences=selected_sentences,
            scores=selected_scores,
            algorithm="textrank",
        )

    def _pagerank(self, similarity_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute PageRank scores for sentences.

        Args:
            similarity_matrix: Sentence similarity matrix

        Returns:
            PageRank scores for each sentence
        """
        num_sentences = similarity_matrix.shape[0]
        scores = np.ones(num_sentences) / num_sentences

        normalized = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)
        normalized = np.nan_to_num(normalized, 0)

        for _ in range(self.max_iterations):
            new_scores = (1 - self.damping) / num_sentences + self.damping * (
                normalized.T @ scores
            )

            diff = np.abs(new_scores - scores).sum()
            scores = new_scores

            if diff < self.convergence_threshold:
                break

        return scores

    def _get_top_sentences(
        self, scores: NDArray[np.float64], num_sentences: int
    ) -> List[int]:
        """Get indices of top-scoring sentences."""
        if len(scores) <= num_sentences:
            return list(range(len(scores)))

        top_indices = np.argsort(scores)[::-1][:num_sentences]
        return top_indices.tolist()


class LexRankSummarizer(SummarizerBase):
    """LexRank algorithm for extractive summarization.

    Uses cosine similarity with IDF weighting to identify
    sentences that are most similar to others.
    """

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        similarity_threshold: float = 0.1,
        use_idf: bool = True,
    ):
        """Initialize LexRank summarizer.

        Args:
            config: Summarization configuration
            similarity_threshold: Threshold for considering sentences similar
            use_idf: Whether to use IDF weighting
        """
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self.use_idf = use_idf
        self.sentence_tokenizer = SentenceTokenizer()

    def summarize(
        self,
        text: str | Document,
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate summary using LexRank algorithm.

        Args:
            text: Input text or Document
            config: Optional configuration override

        Returns:
            SummaryResult with the generated summary
        """
        config = config or self.config
        text_str = text if isinstance(text, str) else text.text

        sentences = self.sentence_tokenizer.tokenize(text_str)
        if len(sentences) <= config.num_sentences:
            return self._create_result(
                summary=text_str,
                original_text=text_str,
                method=SummarizationMethod.EXTRACTIVE,
            )

        if self.use_idf:
            vectorizer = TfidfVectorizer(stop_words="english")
        else:
            vectorizer = CountVectorizer(stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        scores = self._lexrank_scores(similarity_matrix)

        top_indices = self._get_top_sentences(scores, config.num_sentences)
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        selected_scores = [scores[i] for i in top_indices]

        summary = " ".join(selected_sentences)

        return self._create_result(
            summary=summary,
            original_text=text_str,
            method=SummarizationMethod.EXTRACTIVE,
            sentences=selected_sentences,
            scores=selected_scores,
            algorithm="lexrank",
        )

    def _lexrank_scores(
        self, similarity_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute LexRank scores.

        Args:
            similarity_matrix: Sentence similarity matrix

        Returns:
            LexRank scores for each sentence
        """
        adjacency = (similarity_matrix >= self.similarity_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)

        degrees = adjacency.sum(axis=1, keepdims=True)
        degrees[degrees == 0] = 1
        transition = adjacency / degrees

        num_sentences = similarity_matrix.shape[0]
        scores = np.ones(num_sentences) / num_sentences
        damping = 0.1

        for _ in range(100):
            new_scores = (1 - damping) / num_sentences + damping * (
                transition.T @ scores
            )

            if np.abs(new_scores - scores).sum() < 1e-6:
                break

            scores = new_scores

        return scores

    def _get_top_sentences(
        self, scores: NDArray[np.float64], num_sentences: int
    ) -> List[int]:
        """Get indices of top-scoring sentences."""
        if len(scores) <= num_sentences:
            return list(range(len(scores)))

        top_indices = np.argsort(scores)[::-1][:num_sentences]
        return top_indices.tolist()


class MMRSummarizer(SummarizerBase):
    """Maximal Marginal Relevance (MMR) based summarization.

    Balances relevance to the query with diversity among selected sentences.
    """

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        lambda_factor: float = 0.5,
    ):
        """Initialize MMR summarizer.

        Args:
            config: Summarization configuration
            lambda_factor: Balance between relevance (1) and diversity (0)
        """
        super().__init__(config)
        self.lambda_factor = lambda_factor
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.sentence_tokenizer = SentenceTokenizer()

    def summarize(
        self,
        text: str | Document,
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate summary using MMR algorithm.

        Args:
            text: Input text or Document
            config: Optional configuration override

        Returns:
            SummaryResult with the generated summary
        """
        config = config or self.config
        text_str = text if isinstance(text, str) else text.text

        sentences = self.sentence_tokenizer.tokenize(text_str)
        if len(sentences) <= config.num_sentences:
            return self._create_result(
                summary=text_str,
                original_text=text_str,
                method=SummarizationMethod.EXTRACTIVE,
            )

        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        selected_indices = self._mmr_select(similarity_matrix, config.num_sentences)
        selected_indices = sorted(selected_indices)

        selected_sentences = [sentences[i] for i in selected_indices]
        selected_scores = [similarity_matrix[i, i] for i in selected_indices]

        summary = " ".join(selected_sentences)

        return self._create_result(
            summary=summary,
            original_text=text_str,
            method=SummarizationMethod.EXTRACTIVE,
            sentences=selected_sentences,
            scores=selected_scores,
            algorithm="mmr",
        )

    def _mmr_select(
        self,
        similarity_matrix: NDArray[np.float64],
        num_sentences: int,
    ) -> List[int]:
        """Select sentences using MMR algorithm.

        Args:
            similarity_matrix: Sentence similarity matrix
            num_sentences: Number of sentences to select

        Returns:
            List of selected sentence indices
        """
        num_sentences_total = similarity_matrix.shape[0]
        selected: List[int] = []
        remaining = set(range(num_sentences_total))

        first_idx = np.argmax(similarity_matrix.diagonal())
        selected.append(first_idx)
        remaining.remove(first_idx)

        while len(selected) < num_sentences and remaining:
            best_score = -float("inf")
            best_idx = -1

            for idx in remaining:
                relevance = similarity_matrix[idx, idx]

                max_similarity = (
                    max(similarity_matrix[idx, s] for s in selected) if selected else 0
                )

                mmr_score = (
                    self.lambda_factor * relevance
                    - (1 - self.lambda_factor) * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected


class LSAExtractor(SummarizerBase):
    """Latent Semantic Analysis based extractive summarization.

    Uses SVD to find the most important sentences in the
    latent semantic space.
    """

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        num_topics: int = 10,
    ):
        """Initialize LSA extractor.

        Args:
            config: Summarization configuration
            num_topics: Number of topics for SVD
        """
        super().__init__(config)
        self.num_topics = num_topics
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.sentence_tokenizer = SentenceTokenizer()

    def summarize(
        self,
        text: str | Document,
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate summary using LSA.

        Args:
            text: Input text or Document
            config: Optional configuration override

        Returns:
            SummaryResult with the generated summary
        """
        config = config or self.config
        text_str = text if isinstance(text, str) else text.text

        sentences = self.sentence_tokenizer.tokenize(text_str)
        if len(sentences) <= config.num_sentences:
            return self._create_result(
                summary=text_str,
                original_text=text_str,
                method=SummarizationMethod.EXTRACTIVE,
            )

        tfidf_matrix = self.vectorizer.fit_transform(sentences)

        num_topics = min(self.num_topics, len(sentences) - 1)
        svd = TruncatedSVD(n_components=num_topics)
        topic_matrix = svd.fit_transform(tfidf_matrix)

        sentence_scores = np.linalg.norm(topic_matrix, axis=1)

        top_indices = np.argsort(sentence_scores)[::-1][: config.num_sentences]
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        selected_scores = [sentence_scores[i] for i in top_indices]

        summary = " ".join(selected_sentences)

        return self._create_result(
            summary=summary,
            original_text=text_str,
            method=SummarizationMethod.EXTRACTIVE,
            sentences=selected_sentences,
            scores=selected_scores,
            algorithm="lsa",
        )


class ClusterExtractor(SummarizerBase):
    """Cluster-based extractive summarization.

    Clusters sentences and selects representative sentences from each cluster.
    """

    def __init__(
        self,
        config: Optional[SummaryConfig] = None,
        num_clusters: Optional[int] = None,
    ):
        """Initialize Cluster extractor.

        Args:
            config: Summarization configuration
            num_clusters: Number of clusters (defaults to num_sentences)
        """
        super().__init__(config)
        self.num_clusters = num_clusters
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.sentence_tokenizer = SentenceTokenizer()

    def summarize(
        self,
        text: str | Document,
        config: Optional[SummaryConfig] = None,
    ) -> SummaryResult:
        """Generate summary using cluster-based extraction.

        Args:
            text: Input text or Document
            config: Optional configuration override

        Returns:
            SummaryResult with the generated summary
        """
        config = config or self.config
        text_str = text if isinstance(text, str) else text.text

        sentences = self.sentence_tokenizer.tokenize(text_str)
        if len(sentences) <= config.num_sentences:
            return self._create_result(
                summary=text_str,
                original_text=text_str,
                method=SummarizationMethod.EXTRACTIVE,
            )

        tfidf_matrix = self.vectorizer.fit_transform(sentences)

        num_clusters = self.num_clusters or min(
            config.num_sentences, len(sentences) // 2
        )
        num_clusters = min(num_clusters, len(sentences))

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)

        selected_indices = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_vectors = tfidf_matrix[cluster_indices].toarray()
            cluster_center = cluster_vectors.mean(axis=0)

            distances = np.linalg.norm(cluster_vectors - cluster_center, axis=1)
            closest = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest)

        selected_indices = sorted(selected_indices)[: config.num_sentences]

        selected_sentences = [sentences[i] for i in selected_indices]
        selected_scores = [1.0] * len(selected_indices)

        summary = " ".join(selected_sentences)

        return self._create_result(
            summary=summary,
            original_text=text_str,
            method=SummarizationMethod.EXTRACTIVE,
            sentences=selected_sentences,
            scores=selected_scores,
            algorithm="cluster",
        )
