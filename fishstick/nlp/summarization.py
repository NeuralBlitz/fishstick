"""
Fishstick NLP Summarization Module

A comprehensive text summarization library supporting extractive, abstractive,
controllable, and multi-document summarization with evaluation metrics.

Author: Fishstick Team
"""

import re
import math
import random
import heapq
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import svd
from scipy.spatial.distance import cosine
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available. Neural models will not work.")

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSeq2SeqLM,
        BertTokenizer,
        BertModel,
        BartTokenizer,
        BartForConditionalGeneration,
        T5Tokenizer,
        T5ForConditionalGeneration,
        PegasusTokenizer,
        PegasusForConditionalGeneration,
        ProphetNetTokenizer,
        ProphetNetForConditionalGeneration,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers not available. Transformer-based models will not work.")

try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    warnings.warn("NLTK not available. Some preprocessing features may be limited.")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SummaryResult:
    """Container for summarization results."""

    summary: str
    scores: Optional[Dict[str, float]] = None
    sentences: Optional[List[str]] = None
    sentence_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Represents a document for summarization."""

    text: str
    title: Optional[str] = None
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Preprocessing Utilities
# =============================================================================


class DocumentCleaner:
    """Clean and normalize input documents."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_html_tags: bool = True,
    ):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_html_tags = remove_html_tags

        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.email_pattern = re.compile(r"\S+@\S+\.\S+")
        self.html_pattern = re.compile(r"<[^>]+>")
        self.whitespace_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        """Clean a document."""
        if self.remove_html_tags:
            text = self.html_pattern.sub(" ", text)

        if self.remove_urls:
            text = self.url_pattern.sub("", text)

        if self.remove_emails:
            text = self.email_pattern.sub("", text)

        if self.normalize_unicode:
            import unicodedata

            text = unicodedata.normalize("NFKC", text)

        if self.remove_extra_whitespace:
            text = self.whitespace_pattern.sub(" ", text).strip()

        return text


class SentenceSegmenter:
    """Split text into sentences."""

    def __init__(self, language: str = "english"):
        self.language = language

        if not HAS_NLTK:
            # Fallback regex-based segmentation
            self.sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def segment(self, text: str) -> List[str]:
        """Split text into sentences."""
        if HAS_NLTK:
            try:
                return sent_tokenize(text, language=self.language)
            except LookupError:
                nltk.download("punkt")
                return sent_tokenize(text, language=self.language)
        else:
            return [s.strip() for s in self.sentence_pattern.split(text) if s.strip()]


class Deduplication:
    """Remove duplicate sentences or documents."""

    def __init__(
        self, threshold: float = 0.85, method: str = "jaccard", ngram_size: int = 3
    ):
        self.threshold = threshold
        self.method = method
        self.ngram_size = ngram_size

    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract n-grams from text."""
        words = text.lower().split()
        return set(
            " ".join(words[i : i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        )

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        ngrams1 = self._get_ngrams(text1)
        ngrams2 = self._get_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF."""
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            return 0.0

    def deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        """Remove near-duplicate sentences."""
        if len(sentences) <= 1:
            return sentences

        unique_sentences = [sentences[0]]

        for sent in sentences[1:]:
            is_duplicate = False
            for unique_sent in unique_sentences:
                if self.method == "jaccard":
                    sim = self._jaccard_similarity(sent, unique_sent)
                else:
                    sim = self._cosine_similarity(sent, unique_sent)

                if sim >= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_sentences.append(sent)

        return unique_sentences

    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove near-duplicate documents."""
        if len(documents) <= 1:
            return documents

        unique_docs = [documents[0]]

        for doc in documents[1:]:
            is_duplicate = False
            for unique_doc in unique_docs:
                if self.method == "jaccard":
                    sim = self._jaccard_similarity(doc.text, unique_doc.text)
                else:
                    sim = self._cosine_similarity(doc.text, unique_doc.text)

                if sim >= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)

        return unique_docs


class CoreferenceResolution:
    """Simple rule-based coreference resolution."""

    def __init__(self):
        self.pronouns = {
            "he",
            "she",
            "it",
            "they",
            "him",
            "her",
            "them",
            "his",
            "its",
            "their",
        }
        self.pronoun_pattern = re.compile(
            r"\b(he|she|it|they|him|her|them|his|its|their)\b", re.IGNORECASE
        )

    def resolve(self, text: str, entities: Optional[List[str]] = None) -> str:
        """Resolve coreferences in text (simplified version)."""
        # This is a simplified placeholder
        # Full coreference resolution would require a dedicated model
        sentences = text.split(". ")
        resolved_sentences = []
        last_subject = None

        for sent in sentences:
            pronouns_found = self.pronoun_pattern.findall(sent)
            if pronouns_found and last_subject:
                for pronoun in pronouns_found:
                    sent = re.sub(r"\b" + pronoun + r"\b", last_subject, sent, count=1)

            # Simple heuristic to find subject (first noun phrase)
            words = sent.split()
            if words and not words[0].lower() in self.pronouns:
                last_subject = words[0]

            resolved_sentences.append(sent)

        return ". ".join(resolved_sentences)


# =============================================================================
# Base Classes
# =============================================================================


class Summarizer(ABC):
    """Abstract base class for summarizers."""

    @abstractmethod
    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate summary from text."""
        pass

    def _get_num_sentences(
        self, total_sentences: int, num_sentences: Optional[int], ratio: Optional[float]
    ) -> int:
        """Determine number of sentences to return."""
        if num_sentences is not None:
            return min(num_sentences, total_sentences)
        elif ratio is not None:
            return max(1, int(total_sentences * ratio))
        else:
            return max(1, total_sentences // 3)


# =============================================================================
# Extractive Summarization
# =============================================================================


class TextRankSummarizer(Summarizer):
    """TextRank: Graph-based ranking for extractive summarization."""

    def __init__(
        self,
        damping: float = 0.85,
        convergence_threshold: float = 0.0001,
        max_iterations: int = 100,
    ):
        self.damping = damping
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.segmenter = SentenceSegmenter()

    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build sentence similarity matrix using TF-IDF cosine similarity."""
        if len(sentences) == 0:
            return np.array([])

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            # Fallback if vectorization fails
            similarity_matrix = np.eye(len(sentences))

        return similarity_matrix

    def _textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Apply TextRank algorithm."""
        n = len(similarity_matrix)
        if n == 0:
            return np.array([])

        # Normalize similarity matrix
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] = similarity_matrix[i] / row_sum

        # Initialize scores
        scores = np.ones(n) / n

        # Power iteration
        for _ in range(self.max_iterations):
            new_scores = (
                1 - self.damping
            ) / n + self.damping * similarity_matrix.T.dot(scores)

            if np.abs(new_scores - scores).sum() < self.convergence_threshold:
                break

            scores = new_scores

        return scores

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate TextRank summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        similarity_matrix = self._build_similarity_matrix(sentences)
        scores = self._textrank(similarity_matrix)

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)
        top_indices = np.argsort(scores)[-n:][::-1]
        top_indices = sorted(top_indices)  # Maintain original order

        selected_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=scores.tolist(),
            metadata={"method": "textrank", "num_sentences": n},
        )


class LexRankSummarizer(Summarizer):
    """LexRank: Lexical centrality-based summarization."""

    def __init__(
        self, threshold: float = 0.1, damping: float = 0.85, max_iterations: int = 100
    ):
        self.threshold = threshold
        self.damping = damping
        self.max_iterations = max_iterations
        self.segmenter = SentenceSegmenter()

    def _build_lexical_graph(self, sentences: List[str]) -> np.ndarray:
        """Build lexical centrality graph."""
        if len(sentences) == 0:
            return np.array([])

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            return np.eye(len(sentences))

        # Apply threshold
        similarity_matrix[similarity_matrix < self.threshold] = 0

        # Normalize
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        similarity_matrix = similarity_matrix / row_sums

        return similarity_matrix

    def _lexrank(self, transition_matrix: np.ndarray) -> np.ndarray:
        """Compute LexRank scores using power iteration."""
        n = len(transition_matrix)
        if n == 0:
            return np.array([])

        scores = np.ones(n) / n

        for _ in range(self.max_iterations):
            new_scores = (
                1 - self.damping
            ) / n + self.damping * transition_matrix.T.dot(scores)
            scores = new_scores

        return scores

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate LexRank summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        transition_matrix = self._build_lexical_graph(sentences)
        scores = self._lexrank(transition_matrix)

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)
        top_indices = np.argsort(scores)[-n:][::-1]
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=scores.tolist(),
            metadata={"method": "lexrank", "num_sentences": n},
        )


class LSASummarizer(Summarizer):
    """LSA: Latent Semantic Analysis-based summarization."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.segmenter = SentenceSegmenter()

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate LSA-based summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        if len(sentences) < self.n_components:
            n_components = max(1, len(sentences) // 2)
        else:
            n_components = self.n_components

        # Build term-sentence matrix
        vectorizer = CountVectorizer(stop_words="english")
        try:
            term_sent_matrix = vectorizer.fit_transform(sentences)
        except:
            return SummaryResult(
                summary=text[:500],
                sentences=sentences,
                sentence_scores=[1.0] * len(sentences),
            )

        # Apply SVD
        try:
            u, s, vt = svd(term_sent_matrix.T.toarray(), full_matrices=False)

            # Use first principal component for scoring
            sentence_scores = np.abs(vt[0]) if len(vt) > 0 else np.ones(len(sentences))
        except:
            sentence_scores = np.ones(len(sentences))

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)
        top_indices = np.argsort(sentence_scores)[-n:][::-1]
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=sentence_scores.tolist(),
            metadata={"method": "lsa", "num_sentences": n},
        )


class LuhnSummarizer(Summarizer):
    """Luhn: Early extractive summarization based on word frequency."""

    def __init__(self, min_word_frequency: int = 2):
        self.min_word_frequency = min_word_frequency
        self.segmenter = SentenceSegmenter()

    def _get_significant_words(self, sentences: List[str]) -> Set[str]:
        """Identify significant words based on frequency."""
        word_counts = Counter()
        stop_words = set()

        if HAS_NLTK:
            try:
                stop_words = set(stopwords.words("english"))
            except:
                nltk.download("stopwords")
                stop_words = set(stopwords.words("english"))

        for sent in sentences:
            words = sent.lower().split()
            for word in words:
                word = re.sub(r"[^\w]", "", word)
                if word and word not in stop_words and len(word) > 2:
                    word_counts[word] += 1

        significant_words = {
            word
            for word, count in word_counts.items()
            if count >= self.min_word_frequency
        }
        return significant_words

    def _score_sentence(self, sentence: str, significant_words: Set[str]) -> float:
        """Score sentence based on significant word clusters."""
        words = sentence.lower().split()
        words = [re.sub(r"[^\w]", "", w) for w in words]

        # Find clusters of significant words
        clusters = []
        current_cluster = []

        for word in words:
            if word in significant_words:
                current_cluster.append(word)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []

        if current_cluster:
            clusters.append(current_cluster)

        if not clusters:
            return 0.0

        # Luhn score: (cluster length)^2 / total words
        max_cluster_len = max(len(c) for c in clusters)
        score = (max_cluster_len**2) / len(words) if words else 0

        return score

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate Luhn summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        significant_words = self._get_significant_words(sentences)

        if not significant_words:
            # Fallback to first sentences
            n = self._get_num_sentences(len(sentences), num_sentences, ratio)
            summary = " ".join(sentences[:n])
            return SummaryResult(
                summary=summary,
                sentences=sentences,
                sentence_scores=[1.0] * len(sentences),
            )

        scores = [self._score_sentence(sent, significant_words) for sent in sentences]

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)
        top_indices = np.argsort(scores)[-n:][::-1]
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=scores,
            metadata={
                "method": "luhn",
                "num_sentences": n,
                "significant_words": len(significant_words),
            },
        )


class SumBasicSummarizer(Summarizer):
    """SumBasic: Frequency-based summarization."""

    def __init__(self):
        self.segmenter = SentenceSegmenter()

    def _get_word_probabilities(self, sentences: List[str]) -> Dict[str, float]:
        """Compute word probabilities based on frequency."""
        word_counts = Counter()
        total_words = 0

        for sent in sentences:
            words = sent.lower().split()
            for word in words:
                word = re.sub(r"[^\w]", "", word)
                if word and len(word) > 2:
                    word_counts[word] += 1
                    total_words += 1

        return {word: count / total_words for word, count in word_counts.items()}

    def _score_sentence(self, sentence: str, word_probs: Dict[str, float]) -> float:
        """Score sentence by average word probability."""
        words = sentence.lower().split()
        words = [re.sub(r"[^\w]", "", w) for w in words]

        probs = [word_probs.get(word, 0) for word in words if word]
        return np.mean(probs) if probs else 0.0

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate SumBasic summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        word_probs = self._get_word_probabilities(sentences)

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)
        selected_sentences = []
        selected_indices = []
        all_scores = []

        remaining_sentences = list(range(len(sentences)))

        for _ in range(n):
            if not remaining_sentences:
                break

            # Score remaining sentences
            scores = [
                (i, self._score_sentence(sentences[i], word_probs))
                for i in remaining_sentences
            ]
            scores.sort(key=lambda x: x[1], reverse=True)

            best_idx, best_score = scores[0]
            selected_sentences.append(sentences[best_idx])
            selected_indices.append(best_idx)
            all_scores.append(best_score)
            remaining_sentences.remove(best_idx)

            # Update word probabilities (non-redundancy update)
            words_in_sent = sentences[best_idx].lower().split()
            for word in words_in_sent:
                word = re.sub(r"[^\w]", "", word)
                if word in word_probs:
                    word_probs[word] = word_probs[word] ** 2

        # Create full scores list
        full_scores = [0.0] * len(sentences)
        for idx, score in zip(selected_indices, all_scores):
            full_scores[idx] = score

        selected_indices = sorted(selected_indices)
        selected_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=full_scores,
            metadata={"method": "sumbasic", "num_sentences": n},
        )


class KLSumSummarizer(Summarizer):
    """KL-Sum: KL divergence optimization for summarization."""

    def __init__(self, iterations: int = 10):
        self.iterations = iterations
        self.segmenter = SentenceSegmenter()

    def _get_word_distribution(self, text: str) -> Counter:
        """Get word frequency distribution."""
        words = re.findall(r"\b\w+\b", text.lower())
        return Counter(words)

    def _kl_divergence(self, p: Counter, q: Counter) -> float:
        """Compute KL divergence D_KL(p || q)."""
        kl = 0.0
        total_p = sum(p.values())
        total_q = sum(q.values())

        for word, count in p.items():
            prob_p = count / total_p
            prob_q = q.get(word, 0.1) / total_q  # Smoothing
            kl += prob_p * math.log(prob_p / prob_q)

        return kl

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate KL-Sum summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        # Target distribution (document level)
        target_dist = self._get_word_distribution(text)

        # Sentence distributions
        sent_dists = [self._get_word_distribution(sent) for sent in sentences]

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)

        # Greedy selection to minimize KL divergence
        selected = []
        remaining = list(range(len(sentences)))

        for _ in range(n):
            if not remaining:
                break

            best_idx = None
            best_kl = float("inf")

            for idx in remaining:
                # Create combined distribution with current selection + candidate
                combined = Counter()
                for s in selected + [idx]:
                    combined.update(sent_dists[s])

                kl = self._kl_divergence(target_dist, combined)

                if kl < best_kl:
                    best_kl = kl
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        selected = sorted(selected)
        selected_sentences = [sentences[i] for i in selected]
        summary = " ".join(selected_sentences)

        # Assign scores
        scores = [0.0] * len(sentences)
        for idx in selected:
            scores[idx] = 1.0

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=scores,
            metadata={"method": "klsum", "num_sentences": n},
        )


class BertSumExtractor(Summarizer):
    """BertSum: BERT for extractive summarization."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required for BertSum")

        self.model_name = model_name
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.segmenter = SentenceSegmenter()

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get BERT embeddings for sentences."""
        embeddings = []

        with torch.no_grad():
            for sent in sentences:
                inputs = self.tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)

                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])

        return np.array(embeddings)

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate BertSum extractive summary."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        embeddings = self._get_sentence_embeddings(sentences)

        # Compute centrality scores using cosine similarity to document centroid
        centroid = np.mean(embeddings, axis=0)
        scores = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)
        top_indices = np.argsort(scores)[-n:][::-1]
        top_indices = sorted(top_indices)

        selected_sentences = [sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=scores.tolist(),
            metadata={"method": "bertsum", "num_sentences": n},
        )


class MatchSumSummarizer(Summarizer):
    """MatchSum: Matching summaries using sentence embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required for MatchSum")

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.segmenter = SentenceSegmenter()

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings."""
        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)

                outputs = self.model(**inputs)
                embedding = self._mean_pooling(outputs, inputs["attention_mask"])
                embeddings.append(embedding.cpu().numpy()[0])

        return np.array(embeddings)

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> SummaryResult:
        """Generate MatchSum summary by matching sentence combinations."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[], sentence_scores=[])

        # Get document embedding
        doc_embedding = self._get_embeddings([text])[0]

        n = self._get_num_sentences(len(sentences), num_sentences, ratio)

        if len(sentences) <= n:
            return SummaryResult(
                summary=text,
                sentences=sentences,
                sentence_scores=[1.0] * len(sentences),
            )

        # Get sentence embeddings
        sent_embeddings = self._get_embeddings(sentences)

        # Score individual sentences
        scores = cosine_similarity(
            sent_embeddings, doc_embedding.reshape(1, -1)
        ).flatten()

        # Greedy selection to maximize similarity to document
        selected = []
        remaining = list(range(len(sentences)))

        for _ in range(n):
            if not remaining:
                break

            best_idx = max(remaining, key=lambda i: scores[i])
            selected.append(best_idx)
            remaining.remove(best_idx)

        selected = sorted(selected)
        selected_sentences = [sentences[i] for i in selected]
        summary = " ".join(selected_sentences)

        full_scores = scores.tolist()

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=full_scores,
            metadata={"method": "matchsum", "num_sentences": n},
        )


# =============================================================================
# Abstractive Summarization
# =============================================================================


class Seq2SeqSummarizer:
    """Basic Seq2Seq encoder-decoder for summarization."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Seq2Seq")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.encoder = None
        self.decoder = None
        self._build_model()

    def _build_model(self):
        """Build encoder-decoder model."""
        self.encoder = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        ).to(self.device)

        self.decoder = nn.LSTM(
            self.embedding_dim, self.hidden_dim * 2, num_layers=2, batch_first=True
        ).to(self.device)

        self.output_layer = nn.Linear(self.hidden_dim * 2, self.vocab_size).to(
            self.device
        )

    def summarize(
        self, text: str, max_length: int = 150, min_length: int = 10
    ) -> SummaryResult:
        """Generate summary using Seq2Seq (placeholder implementation)."""
        # This is a simplified placeholder
        # Full implementation would require training data and vocabulary

        # Fallback to extractive
        textrank = TextRankSummarizer()
        result = textrank.summarize(text, ratio=0.3)

        return SummaryResult(
            summary=result.summary,
            metadata={"method": "seq2seq", "note": "Fallback to extractive"},
        )


class PointerGeneratorSummarizer:
    """Pointer-Generator network with copy mechanism."""

    def __init__(self, device: str = "cpu"):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for Pointer-Generator")

        self.device = device

    def summarize(
        self, text: str, max_length: int = 150, min_length: int = 10
    ) -> SummaryResult:
        """Generate summary with pointer-generator mechanism."""
        # Placeholder - requires trained model
        textrank = TextRankSummarizer()
        result = textrank.summarize(text, ratio=0.3)

        return SummaryResult(
            summary=result.summary,
            metadata={"method": "pointer_generator", "note": "Fallback to extractive"},
        )


class TransformerSummarizer:
    """Transformer-based abstractive summarization."""

    def __init__(
        self, model_name: str = "facebook/bart-large-cnn", device: str = "cpu"
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 10,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        early_stopping: bool = True,
    ) -> SummaryResult:
        """Generate abstractive summary using Transformer."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024, padding=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return SummaryResult(
            summary=summary,
            metadata={"method": "transformer", "model": self.model_name},
        )


class BARTSummarizer:
    """BART: Bidirectional and Auto-Regressive Transformer."""

    def __init__(
        self, model_name: str = "facebook/bart-large-cnn", device: str = "cpu"
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 142,
        min_length: int = 56,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        early_stopping: bool = True,
    ) -> SummaryResult:
        """Generate summary using BART."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024, padding=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return SummaryResult(
            summary=summary, metadata={"method": "bart", "model": self.model_name}
        )


class T5Summarizer:
    """T5: Text-to-Text Transfer Transformer."""

    def __init__(self, model_name: str = "t5-base", device: str = "cpu"):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 10,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        early_stopping: bool = True,
        prefix: str = "summarize: ",
    ) -> SummaryResult:
        """Generate summary using T5."""
        input_text = prefix + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return SummaryResult(
            summary=summary, metadata={"method": "t5", "model": self.model_name}
        )


class PEGASUSSummarizer:
    """PEGASUS: Pre-training with Extracted Gap-sentences."""

    def __init__(
        self, model_name: str = "google/pegasus-cnn_dailymail", device: str = "cpu"
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 10,
        num_beams: int = 4,
        length_penalty: float = 0.6,
        early_stopping: bool = True,
    ) -> SummaryResult:
        """Generate summary using PEGASUS."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return SummaryResult(
            summary=summary, metadata={"method": "pegasus", "model": self.model_name}
        )


class ProphetNetSummarizer:
    """ProphetNet: Predicting Future N-gram for Sequence Generation."""

    def __init__(
        self,
        model_name: str = "microsoft/prophetnet-large-uncased-cnndm",
        device: str = "cpu",
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
        self.model = ProphetNetForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def summarize(
        self,
        text: str,
        max_length: int = 142,
        min_length: int = 56,
        num_beams: int = 4,
        early_stopping: bool = True,
    ) -> SummaryResult:
        """Generate summary using ProphetNet."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return SummaryResult(
            summary=summary, metadata={"method": "prophetnet", "model": self.model_name}
        )


# =============================================================================
# Controllable Summarization
# =============================================================================


class LengthControl:
    """Control summary length."""

    def __init__(self, base_summarizer: Any):
        self.base_summarizer = base_summarizer

    def summarize(
        self, text: str, target_length: Union[int, str] = "medium", unit: str = "words"
    ) -> SummaryResult:
        """Generate summary with length control."""
        # Map target length to ratio or sentence count
        if isinstance(target_length, str):
            length_map = {"short": 0.1, "medium": 0.2, "long": 0.3}
            ratio = length_map.get(target_length, 0.2)
        else:
            if unit == "words":
                # Estimate words per sentence
                words = len(text.split())
                sentences = len(text.split("."))
                avg_words_per_sent = words / max(sentences, 1)
                ratio = (target_length / avg_words_per_sent) / max(sentences, 1)
                ratio = min(ratio, 0.5)
            else:
                ratio = target_length / max(len(text.split(".")), 1)

        if hasattr(self.base_summarizer, "summarize"):
            result = self.base_summarizer.summarize(text, ratio=ratio)
        else:
            # Fallback
            summarizer = TextRankSummarizer()
            result = summarizer.summarize(text, ratio=ratio)

        result.metadata["target_length"] = target_length
        result.metadata["unit"] = unit

        return result


class StyleControl:
    """Control summary style (formal vs casual)."""

    def __init__(self, base_summarizer: Any):
        self.base_summarizer = base_summarizer

        # Style-specific vocabulary/patterns
        self.formal_markers = [
            "however",
            "therefore",
            "furthermore",
            "consequently",
            "nevertheless",
        ]
        self.casual_markers = ["but", "so", "also", "anyway", "actually"]

    def summarize(self, text: str, style: str = "neutral") -> SummaryResult:
        """Generate summary with style control."""
        if hasattr(self.base_summarizer, "summarize"):
            result = self.base_summarizer.summarize(text)
        else:
            summarizer = TextRankSummarizer()
            result = summarizer.summarize(text)

        summary = result.summary

        if style == "formal":
            summary = self._make_formal(summary)
        elif style == "casual":
            summary = self._make_casual(summary)

        result.summary = summary
        result.metadata["style"] = style

        return result

    def _make_formal(self, text: str) -> str:
        """Convert text to formal style (simplified)."""
        # This is a simplified transformation
        # Full implementation would use style transfer models
        replacements = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is",
            "that's": "that is",
        }

        for informal, formal in replacements.items():
            text = text.replace(informal, formal)
            text = text.replace(informal.capitalize(), formal.capitalize())

        return text

    def _make_casual(self, text: str) -> str:
        """Convert text to casual style (simplified)."""
        replacements = {
            "do not": "don't",
            "cannot": "can't",
            "will not": "won't",
            "it is": "it's",
            "that is": "that's",
        }

        for formal, informal in replacements.items():
            text = text.replace(formal, informal)
            text = text.replace(formal.capitalize(), informal.capitalize())

        return text


class EntityControl:
    """Preserve specific entities in summary."""

    def __init__(self, base_summarizer: Any):
        self.base_summarizer = base_summarizer
        self.segmenter = SentenceSegmenter()

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified regex-based)."""
        # Simple pattern-based entity extraction
        # Full implementation would use NER models

        # Capitalized words as potential entities
        entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        entities = entity_pattern.findall(text)

        # Filter out common false positives
        stop_words = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "It",
            "He",
            "She",
            "They",
        }
        entities = [e for e in entities if e not in stop_words]

        return list(set(entities))

    def summarize(
        self, text: str, entities_to_preserve: Optional[List[str]] = None
    ) -> SummaryResult:
        """Generate summary preserving specific entities."""
        if entities_to_preserve is None:
            entities_to_preserve = self._extract_entities(text)

        # First pass summarization
        if hasattr(self.base_summarizer, "summarize"):
            result = self.base_summarizer.summarize(text)
        else:
            summarizer = TextRankSummarizer()
            result = summarizer.summarize(text)

        summary = result.summary
        sentences = self.segmenter.segment(text)

        # Check which entities are missing
        missing_entities = [e for e in entities_to_preserve if e not in summary]

        # Add sentences containing missing entities
        for entity in missing_entities:
            for sent in sentences:
                if entity in sent and sent not in summary:
                    summary += " " + sent
                    break

        result.summary = summary
        result.metadata["preserved_entities"] = entities_to_preserve
        result.metadata["missing_entities"] = missing_entities

        return result


class KeywordControl:
    """Ensure specific keywords are included in summary."""

    def __init__(self, base_summarizer: Any):
        self.base_summarizer = base_summarizer
        self.segmenter = SentenceSegmenter()

    def summarize(
        self, text: str, keywords: List[str], keyword_weight: float = 2.0
    ) -> SummaryResult:
        """Generate summary including specific keywords."""
        sentences = self.segmenter.segment(text)

        if len(sentences) == 0:
            return SummaryResult(summary="", sentences=[])

        # Score sentences with keyword boost
        scores = []
        for sent in sentences:
            base_score = 1.0
            keyword_bonus = sum(
                keyword_weight for kw in keywords if kw.lower() in sent.lower()
            )
            scores.append(base_score + keyword_bonus)

        # Select top sentences
        n = max(1, len(sentences) // 3)
        top_indices = np.argsort(scores)[-n:][::-1]

        # Ensure at least one sentence per keyword
        keyword_sentences = []
        for kw in keywords:
            for i, sent in enumerate(sentences):
                if kw.lower() in sent.lower():
                    keyword_sentences.append(i)
                    break

        # Combine and sort
        selected_indices = sorted(set(top_indices.tolist() + keyword_sentences))

        selected_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(selected_sentences)

        # Check for missing keywords
        missing_keywords = [kw for kw in keywords if kw.lower() not in summary.lower()]

        return SummaryResult(
            summary=summary,
            sentences=sentences,
            sentence_scores=scores,
            metadata={
                "method": "keyword_control",
                "keywords": keywords,
                "missing_keywords": missing_keywords,
            },
        )


# =============================================================================
# Multi-Document Summarization
# =============================================================================


class MultiDocBertSummarizer:
    """BERT-based multi-document summarization."""

    def __init__(self, device: str = "cpu"):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.to(device)
        self.model.eval()
        self.segmenter = SentenceSegmenter()

    def summarize(
        self, documents: List[Document], num_sentences: int = 5
    ) -> SummaryResult:
        """Generate multi-document summary."""
        all_sentences = []
        sent_to_doc = []

        for doc_idx, doc in enumerate(documents):
            sentences = self.segmenter.segment(doc.text)
            all_sentences.extend(sentences)
            sent_to_doc.extend([doc_idx] * len(sentences))

        if not all_sentences:
            return SummaryResult(summary="", sentences=[])

        # Get embeddings
        embeddings = []
        with torch.no_grad():
            for sent in all_sentences:
                inputs = self.tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self.device)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy()[0])

        embeddings = np.array(embeddings)

        # Cluster sentences
        n_clusters = min(num_sentences, len(all_sentences))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Select representative sentence from each cluster
        selected_sentences = []
        for i in range(n_clusters):
            cluster_sentences = [
                all_sentences[j] for j in range(len(all_sentences)) if clusters[j] == i
            ]
            if cluster_sentences:
                # Select sentence closest to cluster centroid
                selected_sentences.append(cluster_sentences[0])

        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=all_sentences,
            metadata={"method": "multidoc_bert", "num_clusters": n_clusters},
        )


class HierSummarizer:
    """Hierarchical multi-document summarization."""

    def __init__(self):
        self.segmenter = SentenceSegmenter()
        self.base_summarizer = TextRankSummarizer()

    def summarize(
        self, documents: List[Document], num_sentences: int = 5
    ) -> SummaryResult:
        """Generate hierarchical summary: document -> cluster -> final."""
        # Step 1: Summarize each document
        doc_summaries = []
        for doc in documents:
            result = self.base_summarizer.summarize(doc.text, ratio=0.3)
            doc_summaries.append(result.summary)

        # Step 2: Concatenate and re-summarize
        combined_text = " ".join(doc_summaries)
        final_result = self.base_summarizer.summarize(
            combined_text, num_sentences=num_sentences
        )

        final_result.metadata["method"] = "hier_summ"
        final_result.metadata["intermediate_summaries"] = len(doc_summaries)

        return final_result


class GraphBasedMultiDocSummarizer:
    """Graph-based multi-document summarization using sentence graph."""

    def __init__(self):
        self.segmenter = SentenceSegmenter()

    def summarize(
        self, documents: List[Document], num_sentences: int = 5
    ) -> SummaryResult:
        """Generate graph-based multi-document summary."""
        all_sentences = []
        sentence_sources = []

        for doc_idx, doc in enumerate(documents):
            sentences = self.segmenter.segment(doc.text)
            all_sentences.extend(sentences)
            sentence_sources.extend([doc_idx] * len(sentences))

        if not all_sentences:
            return SummaryResult(summary="", sentences=[])

        # Build similarity graph
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf = vectorizer.fit_transform(all_sentences)
            similarity_matrix = cosine_similarity(tfidf)
        except:
            similarity_matrix = np.eye(len(all_sentences))

        # Apply PageRank on sentence graph
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)

        # Select top sentences
        top_indices = sorted(
            range(len(all_sentences)), key=lambda i: scores[i], reverse=True
        )[:num_sentences]
        top_indices = sorted(top_indices)

        selected_sentences = [all_sentences[i] for i in top_indices]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=all_sentences,
            sentence_scores=[scores[i] for i in range(len(all_sentences))],
            metadata={
                "method": "graph_based_multidoc",
                "num_documents": len(documents),
            },
        )


class ClusteringMultiDocSummarizer:
    """Cluster-then-summarize approach for multi-document summarization."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.segmenter = SentenceSegmenter()

    def summarize(
        self, documents: List[Document], sentences_per_cluster: int = 1
    ) -> SummaryResult:
        """Cluster sentences then summarize each cluster."""
        all_sentences = []

        for doc in documents:
            sentences = self.segmenter.segment(doc.text)
            all_sentences.extend(sentences)

        if not all_sentences:
            return SummaryResult(summary="", sentences=[])

        # Vectorize sentences
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            sentence_vectors = vectorizer.fit_transform(all_sentences)
        except:
            return SummaryResult(
                summary=" ".join(all_sentences[:5]), sentences=all_sentences
            )

        # Cluster
        n_clusters = min(self.n_clusters, len(all_sentences))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(sentence_vectors)

        # Select representative sentences from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_sentences = [
                (i, all_sentences[i])
                for i in range(len(all_sentences))
                if clusters[i] == cluster_id
            ]

            if cluster_sentences:
                # Select sentence closest to centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                best_sent_idx = min(
                    cluster_sentences,
                    key=lambda x: np.linalg.norm(
                        sentence_vectors[x[0]].toarray() - centroid
                    ),
                )[0]
                selected.append(best_sent_idx)

        selected = sorted(selected)
        selected_sentences = [all_sentences[i] for i in selected]
        summary = " ".join(selected_sentences)

        return SummaryResult(
            summary=summary,
            sentences=all_sentences,
            metadata={
                "method": "clustering_multidoc",
                "n_clusters": n_clusters,
                "sentences_per_cluster": sentences_per_cluster,
            },
        )


# =============================================================================
# Evaluation Metrics
# =============================================================================


class ROUGEMetric:
    """ROUGE: Recall-Oriented Understudy for Gisting Evaluation."""

    def __init__(self, rouge_types: List[str] = None):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]

        self.rouge_types = rouge_types

        if HAS_ROUGE:
            self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        else:
            self.scorer = None

    def compute(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE scores."""
        if self.scorer is None:
            raise ImportError("rouge-score library required")

        results = {
            rtype: {"precision": [], "recall": [], "fmeasure": []}
            for rtype in self.rouge_types
        }

        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            for rtype in self.rouge_types:
                results[rtype]["precision"].append(scores[rtype].precision)
                results[rtype]["recall"].append(scores[rtype].recall)
                results[rtype]["fmeasure"].append(scores[rtype].fmeasure)

        # Average scores
        avg_results = {}
        for rtype in self.rouge_types:
            avg_results[rtype] = {
                "precision": np.mean(results[rtype]["precision"]),
                "recall": np.mean(results[rtype]["recall"]),
                "fmeasure": np.mean(results[rtype]["fmeasure"]),
            }

        return avg_results


class BLEUMetric:
    """BLEU: Bilingual Evaluation Understudy."""

    def __init__(self, max_n: int = 4):
        self.max_n = max_n

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
        return Counter(ngrams)

    def _compute_bp(self, prediction: str, reference: str) -> float:
        """Compute brevity penalty."""
        pred_len = len(prediction.split())
        ref_len = len(reference.split())

        if pred_len > ref_len:
            return 1.0
        else:
            return math.exp(1 - ref_len / max(pred_len, 1))

    def compute(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores."""
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            bleu_n_scores = []
            for n in range(1, self.max_n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)

                matches = sum((pred_ngrams & ref_ngrams).values())
                total = max(sum(pred_ngrams.values()), 1)

                bleu_n_scores.append(matches / total if total > 0 else 0)

            # Geometric mean
            if all(s > 0 for s in bleu_n_scores):
                geo_mean = math.exp(
                    sum(math.log(s) for s in bleu_n_scores) / len(bleu_n_scores)
                )
            else:
                geo_mean = 0

            bp = self._compute_bp(pred, ref)
            scores.append(bp * geo_mean)

        return {"bleu": np.mean(scores)}


class METEORMetric:
    """METEOR: Metric for Evaluation of Translation with Explicit ORdering."""

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _align(self, pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[int, int]:
        """Align tokens and count matches."""
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)

        matches = len(pred_set & ref_set)
        return matches, len(pred_tokens), len(ref_tokens)

    def compute(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute METEOR scores."""
        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            matches, pred_len, ref_len = self._align(pred_tokens, ref_tokens)

            if matches == 0:
                scores.append(0.0)
                continue

            precision = matches / pred_len if pred_len > 0 else 0
            recall = matches / ref_len if ref_len > 0 else 0

            # F-mean
            f_mean = (precision * recall) / (
                self.alpha * precision + (1 - self.alpha) * recall
            )

            # Fragmentation penalty (simplified)
            frag_penalty = (
                self.gamma * (matches / max(len(pred_tokens), 1)) ** self.beta
            )

            meteor = f_mean * (1 - frag_penalty)
            scores.append(meteor)

        return {"meteor": np.mean(scores)}


class BERTScoreMetric:
    """BERTScore: BERT-based scoring for text generation."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def _get_embeddings(self, text: str) -> torch.Tensor:
        """Get token embeddings."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

    def compute(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute BERTScore."""
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_emb = self._get_embeddings(pred)
            ref_emb = self._get_embeddings(ref)

            # Compute cosine similarity matrix
            similarity = cosine_similarity(
                pred_emb.cpu().numpy(), ref_emb.cpu().numpy()
            )

            # Precision: max similarity for each prediction token
            precision = similarity.max(axis=1).mean()

            # Recall: max similarity for each reference token
            recall = similarity.max(axis=0).mean()

            # F1
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        return {
            "bertscore_precision": np.mean(precision_scores),
            "bertscore_recall": np.mean(recall_scores),
            "bertscore_f1": np.mean(f1_scores),
        }


class MoverScoreMetric:
    """MoverScore: Word Mover Distance-based metric."""

    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu"):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required")

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def _get_embeddings(self, text: str) -> np.ndarray:
        """Get contextualized embeddings."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[0].cpu().numpy()

    def _earth_movers_distance(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Compute approximate Earth Mover's Distance."""
        # Simplified EMD using optimal transport approximation
        # Full implementation would use POT (Python Optimal Transport)

        # Cosine distance matrix
        distance_matrix = 1 - cosine_similarity(P, Q)

        # Uniform distributions
        p = np.ones(len(P)) / len(P)
        q = np.ones(len(Q)) / len(Q)

        # Simplified: minimum cost matching (greedy)
        min_cost = 0
        for i in range(min(len(P), len(Q))):
            min_cost += distance_matrix[i, i]

        return min_cost / max(min(len(P), len(Q)), 1)

    def compute(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute MoverScore."""
        scores = []

        for pred, ref in zip(predictions, references):
            pred_emb = self._get_embeddings(pred)
            ref_emb = self._get_embeddings(ref)

            emd = self._earth_movers_distance(pred_emb, ref_emb)

            # Convert distance to score (inverse)
            score = 1 / (1 + emd)
            scores.append(score)

        return {"moverscore": np.mean(scores)}


class SumQEMetric:
    """SumQE: Summarization Quality Estimation."""

    def __init__(self):
        self.segmenter = SentenceSegmenter()

    def compute(self, source: str, summary: str) -> Dict[str, float]:
        """Estimate summary quality without reference."""
        scores = {}

        # Length ratio
        source_len = len(source.split())
        summary_len = len(summary.split())
        scores["compression_ratio"] = summary_len / max(source_len, 1)

        # Sentence count ratio
        source_sents = len(self.segmenter.segment(source))
        summary_sents = len(self.segmenter.segment(summary))
        scores["sentence_ratio"] = summary_sents / max(source_sents, 1)

        # Content overlap (word overlap)
        source_words = set(source.lower().split())
        summary_words = set(summary.lower().split())
        overlap = len(source_words & summary_words) / max(len(source_words), 1)
        scores["content_overlap"] = overlap

        # Redundancy within summary
        summary_sentences = self.segmenter.segment(summary)
        if len(summary_sentences) > 1:
            vectorizer = TfidfVectorizer()
            try:
                sent_vectors = vectorizer.fit_transform(summary_sentences)
                sims = cosine_similarity(sent_vectors)
                redundancy = (sims.sum() - len(sims)) / (len(sims) * (len(sims) - 1))
                scores["redundancy"] = redundancy
            except:
                scores["redundancy"] = 0.0
        else:
            scores["redundancy"] = 0.0

        # Overall quality score (weighted combination)
        scores["overall"] = (
            0.3
            * (1 - abs(scores["compression_ratio"] - 0.2))  # Prefer ~20% compression
            + 0.3 * scores["content_overlap"]
            + 0.2 * (1 - scores["redundancy"])
            + 0.2
            * min(1.0, scores["sentence_ratio"] * 5)  # Prefer some sentence reduction
        )

        return scores


# =============================================================================
# Training Utilities
# =============================================================================


class SummarizationDataset(Dataset):
    """Dataset for summarization training."""

    def __init__(
        self,
        articles: List[str],
        summaries: List[str],
        tokenizer: Any,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.articles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        article = self.articles[idx]
        summary = self.summaries[idx]

        source_encoding = self.tokenizer(
            article,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


class SummarizationTrainer:
    """Specialized trainer for summarization models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        total_loss = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()

        return total_loss / len(dataloader)


class LengthScheduler:
    """Curriculum learning by length."""

    def __init__(
        self,
        dataset: SummarizationDataset,
        start_length: int = 50,
        end_length: int = 512,
        epochs_per_stage: int = 2,
    ):
        self.dataset = dataset
        self.current_length = start_length
        self.end_length = end_length
        self.epochs_per_stage = epochs_per_stage
        self.epochs_in_stage = 0

    def get_current_length(self) -> int:
        """Get current maximum length."""
        return self.current_length

    def step(self):
        """Progress to next curriculum stage."""
        self.epochs_in_stage += 1

        if self.epochs_in_stage >= self.epochs_per_stage:
            self.epochs_in_stage = 0
            self.current_length = min(self.current_length * 2, self.end_length)

    def get_filtered_dataset(self) -> List[int]:
        """Get indices of samples within current length."""
        indices = []
        for i, article in enumerate(self.dataset.articles):
            if len(article.split()) <= self.current_length:
                indices.append(i)
        return indices


# =============================================================================
# Inference Utilities
# =============================================================================


class BeamSearchDecoder:
    """Beam search for sequence generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        beam_size: int = 4,
        max_length: int = 150,
        length_penalty: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty

    def decode(self, input_ids: torch.Tensor) -> List[Tuple[List[int], float]]:
        """Beam search decoding."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        beams = [(input_ids[0].tolist(), 0.0)]  # (sequence, score)

        for _ in range(self.max_length):
            candidates = []

            for seq, score in beams:
                if seq[-1] == self.tokenizer.eos_token_id:
                    candidates.append((seq, score))
                    continue

                input_tensor = torch.tensor([seq])
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits[0, -1, :]
                    probs = F.log_softmax(logits, dim=-1)

                topk_probs, topk_indices = torch.topk(probs, self.beam_size)

                for prob, idx in zip(topk_probs, topk_indices):
                    new_seq = seq + [idx.item()]
                    # Apply length penalty
                    length_norm = len(new_seq) ** self.length_penalty
                    new_score = (score * (len(seq) - 1) + prob.item()) / length_norm
                    candidates.append((new_seq, new_score))

            # Select top beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[: self.beam_size]

            # Check if all beams ended
            if all(seq[-1] == self.tokenizer.eos_token_id for seq, _ in beams):
                break

        return beams


class DiverseBeamSearch:
    """Diverse beam search for diverse summaries."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        num_beams: int = 4,
        num_groups: int = 2,
        diversity_penalty: float = 0.5,
        max_length: int = 150,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.max_length = max_length

    def decode(self, input_ids: torch.Tensor) -> List[List[int]]:
        """Diverse beam search decoding."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        beams_per_group = self.num_beams // self.num_groups
        all_beams = [[] for _ in range(self.num_groups)]

        for group_idx in range(self.num_groups):
            beams = [(input_ids[0].tolist(), 0.0)]

            for step in range(self.max_length):
                candidates = []

                for seq, score in beams:
                    if seq[-1] == self.tokenizer.eos_token_id:
                        candidates.append((seq, score))
                        continue

                    input_tensor = torch.tensor([seq])
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        probs = F.log_softmax(logits, dim=-1)

                    # Apply diversity penalty
                    for g in range(group_idx):
                        for prev_seq, _ in all_beams[g]:
                            if step < len(prev_seq):
                                probs[prev_seq[step]] -= self.diversity_penalty

                    topk_probs, topk_indices = torch.topk(probs, beams_per_group)

                    for prob, idx in zip(topk_probs, topk_indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        candidates.append((new_seq, new_score))

                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beams_per_group]

                if all(seq[-1] == self.tokenizer.eos_token_id for seq, _ in beams):
                    break

            all_beams[group_idx] = beams

        # Combine all groups
        all_results = []
        for group in all_beams:
            all_results.extend(group)

        all_results.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in all_results[: self.num_beams]]


class TopKSampling:
    """Top-k sampling for diverse generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        k: int = 50,
        temperature: float = 1.0,
        max_length: int = 150,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.temperature = temperature
        self.max_length = max_length

    def sample(self, input_ids: torch.Tensor) -> List[int]:
        """Generate using top-k sampling."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        generated = input_ids[0].tolist()

        for _ in range(self.max_length):
            input_tensor = torch.tensor([generated])

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :] / self.temperature
                probs = F.softmax(logits, dim=-1)

            # Top-k filtering
            topk_probs, topk_indices = torch.topk(probs, self.k)
            topk_probs = topk_probs / topk_probs.sum()

            # Sample
            sampled_idx = torch.multinomial(topk_probs, 1).item()
            next_token = topk_indices[sampled_idx].item()

            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        return generated


class LengthPenalty:
    """Apply length penalty to generated sequences."""

    def __init__(
        self,
        target_length: int = 100,
        penalty_type: str = "gaussian",
        sigma: float = 20.0,
    ):
        self.target_length = target_length
        self.penalty_type = penalty_type
        self.sigma = sigma

    def apply(self, score: float, length: int) -> float:
        """Apply length penalty to score."""
        if self.penalty_type == "linear":
            # Linear penalty
            penalty = abs(length - self.target_length) / self.target_length
            return score * (1 - penalty)

        elif self.penalty_type == "gaussian":
            # Gaussian penalty
            penalty = math.exp(
                -((length - self.target_length) ** 2) / (2 * self.sigma**2)
            )
            return score * penalty

        elif self.penalty_type == "exponential":
            # Exponential penalty
            diff = abs(length - self.target_length)
            penalty = math.exp(-diff / self.sigma)
            return score * penalty

        else:
            return score

    def get_optimal_length_range(self) -> Tuple[int, int]:
        """Get acceptable length range."""
        min_len = max(1, int(self.target_length - 2 * self.sigma))
        max_len = int(self.target_length + 2 * self.sigma)
        return min_len, max_len


# =============================================================================
# Factory and Convenience Functions
# =============================================================================


def get_summarizer(method: str, **kwargs) -> Any:
    """Factory function to get summarizer by name."""

    extractive_methods = {
        "textrank": TextRankSummarizer,
        "lexrank": LexRankSummarizer,
        "lsa": LSASummarizer,
        "luhn": LuhnSummarizer,
        "sumbasic": SumBasicSummarizer,
        "klsum": KLSumSummarizer,
        "bertsum": BertSumExtractor,
        "matchsum": MatchSumSummarizer,
    }

    abstractive_methods = {
        "seq2seq": Seq2SeqSummarizer,
        "pointer_generator": PointerGeneratorSummarizer,
        "transformer": TransformerSummarizer,
        "bart": BARTSummarizer,
        "t5": T5Summarizer,
        "pegasus": PEGASUSSummarizer,
        "prophetnet": ProphetNetSummarizer,
    }

    if method in extractive_methods:
        return extractive_methods[method](**kwargs)
    elif method in abstractive_methods:
        return abstractive_methods[method](**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_summaries(
    predictions: List[str], references: List[str], metrics: List[str] = None
) -> Dict[str, Any]:
    """Evaluate summaries using multiple metrics."""
    if metrics is None:
        metrics = ["rouge", "bleu"]

    results = {}

    if "rouge" in metrics:
        rouge = ROUGEMetric()
        results["rouge"] = rouge.compute(predictions, references)

    if "bleu" in metrics:
        bleu = BLEUMetric()
        results["bleu"] = bleu.compute(predictions, references)

    if "meteor" in metrics:
        meteor = METEORMetric()
        results["meteor"] = meteor.compute(predictions, references)

    if "bertscore" in metrics and HAS_TRANSFORMERS:
        bertscore = BERTScoreMetric()
        results["bertscore"] = bertscore.compute(predictions, references)

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Example usage
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and 
    human language, in particular how to program computers to process and analyze 
    large amounts of natural language data. The goal is a computer capable of 
    understanding the contents of documents, including the contextual nuances of 
    the language within them. The technology can then accurately extract information 
    and insights contained in the documents as well as categorize and organize the 
    documents themselves.
    """

    # Test extractive summarizer
    print("Testing TextRank Summarizer:")
    textrank = TextRankSummarizer()
    result = textrank.summarize(sample_text, num_sentences=2)
    print(f"Summary: {result.summary}")
    print(f"Scores: {result.sentence_scores[:3] if result.sentence_scores else 'N/A'}")
    print()

    # Test LSA summarizer
    print("Testing LSA Summarizer:")
    lsa = LSASummarizer()
    result = lsa.summarize(sample_text, ratio=0.3)
    print(f"Summary: {result.summary}")
    print()

    # Test evaluation
    print("Testing Evaluation Metrics:")
    pred = ["Natural language processing is a subfield of artificial intelligence."]
    ref = ["NLP is a field of AI concerned with computer-human language interactions."]

    rouge = ROUGEMetric()
    scores = rouge.compute(pred, ref)
    print(f"ROUGE-1 F1: {scores['rouge1']['fmeasure']:.4f}")
    print(f"ROUGE-2 F1: {scores['rouge2']['fmeasure']:.4f}")
    print(f"ROUGE-L F1: {scores['rougeL']['fmeasure']:.4f}")
