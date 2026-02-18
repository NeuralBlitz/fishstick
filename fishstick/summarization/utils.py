"""
Text Preprocessing Utilities for Summarization
==============================================

Provides text preprocessing, tokenization, and utility functions
for the summarization module.

Author: Fishstick Team
"""

from __future__ import annotations

import re
import string
from typing import List, Set, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from numpy.typing import NDArray


class SentenceTokenizer:
    """Tokenizer for splitting text into sentences."""

    def __init__(
        self,
        language: str = "en",
        preserve_whitespace: bool = False,
    ):
        """Initialize sentence tokenizer.

        Args:
            language: Language code for tokenization
            preserve_whitespace: Whether to preserve whitespace in sentences
        """
        self.language = language
        self.preserve_whitespace = preserve_whitespace
        self._nltk_available = self._check_nltk()

    def _check_nltk(self) -> bool:
        """Check if NLTK is available."""
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                try:
                    nltk.download("punkt_tab", quiet=True)
                except:
                    pass
            return True
        except ImportError:
            return False

    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text to tokenize

        Returns:
            List of sentences
        """
        if self._nltk_available:
            try:
                import nltk

                return nltk.tokenize.sent_tokenize(text)
            except:
                pass

        return self._simple_sentence_tokenize(text)

    def _simple_sentence_tokenize(self, text: str) -> List[str]:
        """Simple sentence tokenization without NLTK.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences and text.strip():
            sentences = [text.strip()]

        if self.preserve_whitespace:
            return sentences

        return [s.strip() for s in sentences]


class WordTokenizer:
    """Tokenizer for splitting text into words."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
    ):
        """Initialize word tokenizer.

        Args:
            lowercase: Whether to convert to lowercase
            remove_punctuation: Whether to remove punctuation
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def tokenize(self, text: str) -> List[str]:
        """Split text into words.

        Args:
            text: Input text to tokenize

        Returns:
            List of words
        """
        if self.lowercase:
            text = text.lower()

        words = re.findall(r"\b\w+\b", text)

        if self.remove_punctuation:
            words = [w for w in words if w not in string.punctuation]

        return words

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class StopwordFilter:
    """Filter for removing stopwords from text."""

    DEFAULT_STOPWORDS: Set[str] = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "just",
        "don",
        "now",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
    }

    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        language: str = "en",
    ):
        """Initialize stopword filter.

        Args:
            stopwords: Custom set of stopwords (uses default if None)
            language: Language for default stopwords
        """
        self.stopwords = stopwords or self.DEFAULT_STOPWORDS.copy()
        self.language = language
        self._load_language_stopwords()

    def _load_language_stopwords(self) -> None:
        """Load language-specific stopwords from NLTK if available."""
        if self.language == "en":
            try:
                import nltk

                try:
                    nltk.data.find("corpora/stopwords")
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords

                nltk_stopwords = set(stopwords.words("english"))
                self.stopwords.update(nltk_stopwords)
            except ImportError:
                pass

    def filter(self, words: List[str]) -> List[str]:
        """Remove stopwords from word list.

        Args:
            words: List of words to filter

        Returns:
            List of words with stopwords removed
        """
        return [w for w in words if w.lower() not in self.stopwords]

    def __call__(self, words: List[str]) -> List[str]:
        return self.filter(words)


@dataclass
class TextPreprocessor:
    """Comprehensive text preprocessor for summarization.

    Attributes:
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numbers
        remove_extra_whitespace: Whether to normalize whitespace
        remove_urls: Whether to remove URLs
        remove_emails: Whether to remove email addresses
        stem: Whether to apply stemming
        stopword_filter: Whether to filter stopwords
    """

    lowercase: bool = True
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    stem: bool = False
    stopword_filter: bool = False
    custom_filters: List[Callable[[str], str]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize additional components."""
        self.sentence_tokenizer = SentenceTokenizer()
        self.word_tokenizer = WordTokenizer(lowercase=self.lowercase)
        self._stemmer = None
        if self.stem:
            self._init_stemmer()
        if self.stopword_filter:
            self.stopword_filter_instance = StopwordFilter()
        else:
            self.stopword_filter_instance = None

    def _init_stemmer(self) -> None:
        """Initialize stemmer."""
        try:
            from nltk.stem import PorterStemmer

            self._stemmer = PorterStemmer()
        except ImportError:
            self.stem = False

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to text.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        processed = text

        for custom_filter in self.custom_filters:
            processed = custom_filter(processed)

        if self.remove_urls:
            processed = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                processed,
            )

        if self.remove_emails:
            processed = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "",
                processed,
            )

        if self.remove_extra_whitespace:
            processed = re.sub(r"\s+", " ", processed)
            processed = processed.strip()

        if self.remove_numbers:
            processed = re.sub(r"\d+", "", processed)

        if self.remove_punctuation:
            processed = processed.translate(str.maketrans("", "", string.punctuation))

        if self.lowercase:
            processed = processed.lower()

        return processed

    def preprocess_sentences(
        self, text: str, return_tokens: bool = False
    ) -> List[Dict[str, Any]]:
        """Preprocess text at sentence level.

        Args:
            text: Input text
            return_tokens: Whether to return tokenized words

        Returns:
            List of processed sentences with metadata
        """
        sentences = self.sentence_tokenizer.tokenize(text)
        processed_sentences = []

        for sent in sentences:
            processed = self.preprocess(sent)
            words = self.word_tokenizer.tokenize(processed)

            if self.stopword_filter_instance:
                words = self.stopword_filter_instance.filter(words)

            if self.stem and self._stemmer:
                words = [self._stemmer.stem(w) for w in words]

            result = {
                "original": sent,
                "processed": processed,
                "word_count": len(words),
            }

            if return_tokens:
                result["tokens"] = words

            processed_sentences.append(result)

        return processed_sentences

    def get_word_frequencies(
        self, text: str, normalize: bool = True
    ) -> Dict[str, float]:
        """Calculate word frequencies from text.

        Args:
            text: Input text
            normalize: Whether to normalize frequencies

        Returns:
            Dictionary of word frequencies
        """
        processed = self.preprocess(text)
        words = self.word_tokenizer.tokenize(processed)

        if self.stopword_filter_instance:
            words = self.stopword_filter_instance.filter(words)

        counter = Counter(words)
        total = sum(counter.values())

        if normalize and total > 0:
            return {w: count / total for w, count in counter.items()}

        return dict(counter)

    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        method: str = "frequency",
    ) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text
            top_k: Number of keywords to extract
            method: Extraction method ("frequency" or "tfidf")

        Returns:
            List of top keywords
        """
        if method == "frequency":
            freqs = self.get_word_frequencies(text)
            sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_words[:top_k]]
        elif method == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            sentences = self.sentence_tokenizer.tokenize(text)
            if not sentences:
                return []
            vectorizer = TfidfVectorizer(
                stop_words=list(StopwordFilter.DEFAULT_STOPWORDS),
                lowercase=self.lowercase,
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
                sorted_indices = mean_tfidf.argsort()[::-1]
                return [feature_names[i] for i in sorted_indices[:top_k]]
            except:
                return self.extract_keywords(text, top_k, "frequency")
        else:
            raise ValueError(f"Unknown keyword extraction method: {method}")


def normalize_text(text: str) -> str:
    """Basic text normalization.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def compute_text_statistics(text: str) -> Dict[str, Any]:
    """Compute basic statistics about text.

    Args:
        text: Input text

    Returns:
        Dictionary of text statistics
    """
    tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()

    sentences = tokenizer.tokenize(text)
    words = word_tokenizer.tokenize(text)

    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "avg_sentence_length": np.mean([len(s.split()) for s in sentences])
        if sentences
        else 0,
    }
