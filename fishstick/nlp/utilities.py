"""
Comprehensive NLP Utilities for Fishstick
=========================================

A complete suite of NLP utilities including tokenization, text preprocessing,
emebddings, transformer components, language models, generation utilities,
and task-specific implementations.

Author: Fishstick Team
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# =============================================================================
# Tokenization
# =============================================================================


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword units."""
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        ...


class CharacterTokenizer(Tokenizer):
    """Simple character-level tokenizer."""

    def __init__(
        self,
        special_tokens: Optional[Dict[str, int]] = None,
        vocab: Optional[Dict[str, int]] = None,
    ):
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self.vocab = vocab or {}
        self.reverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self._update_reverse_vocab()

    def _update_reverse_vocab(self) -> None:
        """Update reverse vocabulary mapping."""
        self.reverse_vocab = {
            v: k for k, v in {**self.special_tokens, **self.vocab}.items()
        }

    def build_vocab(self, texts: List[str], min_freq: int = 1) -> None:
        """Build vocabulary from texts."""
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        self.vocab = {
            char: idx + len(self.special_tokens)
            for char, count in char_counts.items()
            if count >= min_freq
        }
        self._update_reverse_vocab()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        full_vocab = {**self.special_tokens, **self.vocab}
        tokens = [full_vocab.get(char, self.special_tokens["<unk>"]) for char in text]

        if add_special_tokens:
            tokens = (
                [self.special_tokens["<s>"]] + tokens + [self.special_tokens["</s>"]]
            )

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        special_ids = (
            set(self.special_tokens.values()) if skip_special_tokens else set()
        )
        chars = [
            self.reverse_vocab.get(tid, "<unk>")
            for tid in token_ids
            if tid not in special_ids
        ]
        return "".join(chars)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        return list(text)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.special_tokens) + len(self.vocab)


class BytePairEncoder(Tokenizer):
    """Byte Pair Encoding (BPE) tokenizer."""

    def __init__(
        self,
        vocab_size: int = 10000,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self._update_reverse_vocab()

    def _update_reverse_vocab(self) -> None:
        """Update reverse vocabulary mapping."""
        self.reverse_vocab = {
            v: k for k, v in {**self.special_tokens, **self.vocab}.items()
        }

    def _get_word_tokens(self, word: str) -> List[str]:
        """Get initial character tokens for a word."""
        return list(word) + ["</w>"]

    def _get_pairs(self, word_tokens: List[str]) -> set:
        """Get all adjacent pairs in word tokens."""
        return set(zip(word_tokens[:-1], word_tokens[1:]))

    def train(self, texts: List[str], num_merges: Optional[int] = None) -> None:
        """Train BPE on texts."""
        if num_merges is None:
            num_merges = self.target_vocab_size - len(self.special_tokens)

        # Build word frequency dictionary
        word_freqs: Dict[str, int] = Counter()
        for text in texts:
            word_freqs.update(text.split())

        # Initialize vocab with characters
        char_vocab: set = set()
        for word in word_freqs:
            char_vocab.update(word)

        self.vocab = {
            char: i + len(self.special_tokens)
            for i, char in enumerate(sorted(char_vocab))
        }
        self.vocab["</w>"] = len(self.vocab) + len(self.special_tokens)

        # Convert words to token sequences
        word_tokens: Dict[Tuple[str, ...], int] = {
            tuple(self._get_word_tokens(word)): freq
            for word, freq in word_freqs.items()
        }

        # Perform BPE merges
        for i in range(num_merges):
            pairs: Dict[Tuple[str, str], int] = Counter()
            for tokens, freq in word_tokens.items():
                for pair in self._get_pairs(list(tokens)):
                    pairs[pair] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            new_token = "".join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab) + len(self.special_tokens)

            # Apply merge to all word tokens
            new_word_tokens: Dict[Tuple[str, ...], int] = {}
            for tokens, freq in word_tokens.items():
                new_tokens = self._apply_merge(list(tokens), best_pair)
                new_word_tokens[tuple(new_tokens)] = freq

            word_tokens = new_word_tokens

        self._update_reverse_vocab()

    def _apply_merge(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Apply a merge operation to token sequence."""
        new_tokens: List[str] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append("".join(pair))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = text.split()
        tokens: List[str] = []

        for word in words:
            word_tokens = self._get_word_tokens(word)
            for merge in self.merges:
                word_tokens = self._apply_merge(word_tokens, merge)
            tokens.extend(word_tokens)

        full_vocab = {**self.special_tokens, **self.vocab}
        token_ids = [
            full_vocab.get(token, self.special_tokens["<unk>"]) for token in tokens
        ]

        if add_special_tokens:
            token_ids = (
                [self.special_tokens["<s>"]] + token_ids + [self.special_tokens["</s>"]]
            )

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        special_ids = (
            set(self.special_tokens.values()) if skip_special_tokens else set()
        )
        tokens = [
            self.reverse_vocab.get(tid, "<unk>")
            for tid in token_ids
            if tid not in special_ids
        ]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into BPE units."""
        words = text.split()
        all_tokens: List[str] = []

        for word in words:
            word_tokens = self._get_word_tokens(word)
            for merge in self.merges:
                word_tokens = self._apply_merge(word_tokens, merge)
            all_tokens.extend(word_tokens)

        return all_tokens

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.special_tokens) + len(self.vocab)


class WordPieceTokenizer(Tokenizer):
    """WordPiece tokenizer (used in BERT)."""

    def __init__(
        self,
        vocab_size: int = 30000,
        special_tokens: Optional[Dict[str, int]] = None,
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
    ):
        self.target_vocab_size = vocab_size
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.special_tokens = special_tokens or {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }
        self.vocab: Dict[str, int] = {}
        self._update_reverse_vocab()

    def _update_reverse_vocab(self) -> None:
        """Update reverse vocabulary mapping."""
        self.reverse_vocab = {
            v: k for k, v in {**self.special_tokens, **self.vocab}.items()
        }

    def train(self, texts: List[str]) -> None:
        """Train WordPiece on texts."""
        # Count word frequencies
        word_freqs: Dict[str, int] = Counter()
        for text in texts:
            word_freqs.update(text.split())

        # Initialize vocab with characters (prefix ## for subwords)
        char_vocab: set = set()
        for word in word_freqs:
            if len(word) <= self.max_input_chars_per_word:
                char_vocab.update(word)

        # Add characters without ## prefix initially
        self.vocab = {
            char: i + len(self.special_tokens)
            for i, char in enumerate(sorted(char_vocab))
        }

        # Build training corpus
        word_tokens: Dict[str, Tuple[str, ...]] = {}
        for word in word_freqs:
            if len(word) <= self.max_input_chars_per_word:
                tokens = tuple(word)
                word_tokens[word] = tokens

        # Greedy WordPiece training
        while len(self.vocab) + len(self.special_tokens) < self.target_vocab_size:
            # Count pair frequencies
            pair_scores: Dict[Tuple[str, str], float] = {}
            for word, tokens in word_tokens.items():
                freq = word_freqs[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair not in pair_scores:
                        pair_scores[pair] = 0.0
                    pair_scores[pair] += freq

            if not pair_scores:
                break

            # Select best pair
            best_pair = max(pair_scores, key=lambda p: pair_scores[p])
            new_token = "".join(best_pair)

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab) + len(self.special_tokens)

            # Update word tokens
            new_word_tokens: Dict[str, Tuple[str, ...]] = {}
            for word, tokens in word_tokens.items():
                new_tokens = self._merge_pair(list(tokens), best_pair)
                new_word_tokens[word] = tuple(new_tokens)

            word_tokens = new_word_tokens

        self._update_reverse_vocab()

    def _merge_pair(self, tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge a pair in token sequence."""
        new_tokens: List[str] = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append("".join(pair))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def _wordpiece_tokenize(self, word: str) -> List[str]:
        """Tokenize a single word using WordPiece."""
        if word in self.vocab:
            return [word]

        tokens: List[str] = []
        remaining = word

        while remaining:
            # Try to find longest matching prefix
            found = False
            for i in range(len(remaining), 0, -1):
                substr = remaining[:i]
                if substr in self.vocab:
                    tokens.append(substr)
                    remaining = remaining[i:]
                    found = True
                    break

            if not found:
                return [self.unk_token]

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = text.split()
        tokens: List[str] = []

        for word in words:
            word_tokens = self._wordpiece_tokenize(word)
            tokens.extend(word_tokens)

        full_vocab = {**self.special_tokens, **self.vocab}
        token_ids = [
            full_vocab.get(token, self.special_tokens[self.unk_token])
            for token in tokens
        ]

        if add_special_tokens:
            token_ids = (
                [self.special_tokens["[CLS]"]]
                + token_ids
                + [self.special_tokens["[SEP]"]]
            )

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        special_ids = (
            set(self.special_tokens.values()) if skip_special_tokens else set()
        )
        tokens = [
            self.reverse_vocab.get(tid, self.unk_token)
            for tid in token_ids
            if tid not in special_ids
        ]
        return "".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into WordPiece units."""
        words = text.split()
        all_tokens: List[str] = []

        for word in words:
            all_tokens.extend(self._wordpiece_tokenize(word))

        return all_tokens

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.special_tokens) + len(self.vocab)


class SentencePieceTokenizer(Tokenizer):
    """Unigram Language Model-based SentencePiece tokenizer."""

    def __init__(
        self,
        vocab_size: int = 32000,
        special_tokens: Optional[Dict[str, int]] = None,
        character_coverage: float = 0.9995,
    ):
        self.target_vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        self.vocab: Dict[str, int] = {}
        self.scores: Dict[str, float] = {}
        self._update_reverse_vocab()

    def _update_reverse_vocab(self) -> None:
        """Update reverse vocabulary mapping."""
        self.reverse_vocab = {
            v: k for k, v in {**self.special_tokens, **self.vocab}.items()
        }

    def _get_substrings(self, text: str) -> List[str]:
        """Get all possible substrings of text."""
        substrings = set()
        for i in range(len(text)):
            for j in range(i + 1, min(i + 20, len(text) + 1)):
                substrings.add(text[i:j])
        return list(substrings)

    def _viterbi_segment(
        self, text: str, token_probs: Dict[str, float]
    ) -> Tuple[List[str], float]:
        """Viterbi algorithm for optimal segmentation."""
        n = len(text)
        # best_score[i] = best log probability for text[:i]
        best_score: List[float] = [float("-inf")] * (n + 1)
        best_edge: List[int] = [0] * (n + 1)
        best_score[0] = 0.0

        for i in range(n):
            if best_score[i] == float("-inf"):
                continue
            for j in range(i + 1, min(i + 20, n + 1)):
                substr = text[i:j]
                if substr in token_probs:
                    score = best_score[i] + math.log(token_probs[substr])
                    if score > best_score[j]:
                        best_score[j] = score
                        best_edge[j] = i

        # Backtrack to get segmentation
        tokens: List[str] = []
        i = n
        while i > 0:
            j = best_edge[i]
            tokens.append(text[j:i])
            i = j

        tokens.reverse()
        return tokens, best_score[n]

    def train(self, texts: List[str], num_iterations: int = 10) -> None:
        """Train Unigram model using EM algorithm."""
        # Build seed vocabulary from character n-grams
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)

        # Keep characters covering character_coverage
        total_chars = sum(char_counts.values())
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        cumulative = 0
        seed_tokens: set = set()
        for char, count in sorted_chars:
            cumulative += count
            seed_tokens.add(char)
            if cumulative / total_chars >= self.character_coverage:
                break

        # Add common substrings
        substring_counts: Counter = Counter()
        for text in texts[:10000]:  # Sample for efficiency
            for substr in self._get_substrings(text):
                if len(substr) > 1:
                    substring_counts[substr] += 1

        # Add top frequent substrings to seed vocab
        for substr, _ in substring_counts.most_common(
            self.target_vocab_size - len(seed_tokens) - 100
        ):
            seed_tokens.add(substr)
            if len(seed_tokens) >= self.target_vocab_size - len(self.special_tokens):
                break

        # EM algorithm
        token_probs: Dict[str, float] = {
            token: 1.0 / len(seed_tokens) for token in seed_tokens
        }

        for iteration in range(num_iterations):
            # E-step: compute expected counts
            expected_counts: Counter = Counter()
            for text in texts:
                tokens, _ = self._viterbi_segment(text, token_probs)
                for token in tokens:
                    expected_counts[token] += 1

            # M-step: update probabilities
            total_count = sum(expected_counts.values())
            if total_count > 0:
                for token in token_probs:
                    token_probs[token] = expected_counts.get(token, 0.1) / total_count

            # Prune vocabulary
            sorted_tokens = sorted(
                token_probs.items(),
                key=lambda x: (expected_counts.get(x[0], 0), x[1]),
                reverse=True,
            )
            top_tokens = dict(
                sorted_tokens[: self.target_vocab_size - len(self.special_tokens)]
            )
            token_probs = {k: v for k, v in token_probs.items() if k in top_tokens}

        # Normalize final probabilities
        total_prob = sum(token_probs.values())
        if total_prob > 0:
            token_probs = {k: v / total_prob for k, v in token_probs.items()}

        self.vocab = {
            token: i + len(self.special_tokens)
            for i, token in enumerate(token_probs.keys())
        }
        self.scores = token_probs
        self._update_reverse_vocab()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens, _ = self._viterbi_segment(text, self.scores)
        full_vocab = {**self.special_tokens, **self.vocab}
        token_ids = [
            full_vocab.get(token, self.special_tokens["<unk>"]) for token in tokens
        ]

        if add_special_tokens:
            token_ids = (
                [self.special_tokens["<s>"]] + token_ids + [self.special_tokens["</s>"]]
            )

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        special_ids = (
            set(self.special_tokens.values()) if skip_special_tokens else set()
        )
        tokens = [
            self.reverse_vocab.get(tid, "<unk>")
            for tid in token_ids
            if tid not in special_ids
        ]
        return "".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into SentencePiece units."""
        tokens, _ = self._viterbi_segment(text, self.scores)
        return tokens

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.special_tokens) + len(self.vocab)


# =============================================================================
# Text Preprocessing
# =============================================================================


def normalize_text(text: str, form: str = "NFC", lowercase: bool = True) -> str:
    """
    Normalize Unicode text.

    Args:
        text: Input text
        form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
        lowercase: Whether to lowercase the text

    Returns:
        Normalized text
    """
    import unicodedata

    text = unicodedata.normalize(form, text)
    if lowercase:
        text = text.lower()
    return text


def remove_urls(text: str, replacement: str = "") -> str:
    """
    Remove URLs from text.

    Args:
        text: Input text
        replacement: String to replace URLs with

    Returns:
        Text with URLs removed
    """
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.sub(replacement, text)


def remove_emails(text: str, replacement: str = "") -> str:
    """
    Remove email addresses from text.

    Args:
        text: Input text
        replacement: String to replace emails with

    Returns:
        Text with emails removed
    """
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    return email_pattern.sub(replacement, text)


def clean_html(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Input text with HTML

    Returns:
        Text with HTML tags removed
    """
    # Remove script and style elements
    text = re.sub(r"<(script|style)[^>]*>[^<]*</\1>", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TextAugmenter:
    """Text augmentation utilities."""

    def __init__(
        self,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        random_state: Optional[int] = None,
    ):
        self.synonym_dict = synonym_dict or {}
        self.rng = __import__("random").Random(random_state)

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        Replace n words with synonyms.

        Args:
            text: Input text
            n: Number of words to replace

        Returns:
            Augmented text
        """
        words = text.split()
        new_words = words.copy()

        # Find words with synonyms
        candidates = [
            (i, word)
            for i, word in enumerate(words)
            if word.lower() in self.synonym_dict
        ]

        self.rng.shuffle(candidates)
        num_replaced = 0

        for i, word in candidates:
            if num_replaced >= n:
                break

            synonyms = self.synonym_dict.get(word.lower(), [])
            if synonyms:
                synonym = self.rng.choice(synonyms)
                new_words[i] = synonym
                num_replaced += 1

        return " ".join(new_words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p.

        Args:
            text: Input text
            p: Deletion probability

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) == 1:
            return text

        new_words = [word for word in words if self.rng.random() > p]

        # Ensure at least one word remains
        if not new_words:
            new_words = [self.rng.choice(words)]

        return " ".join(new_words)

    def random_swap(self, text: str, n: int = 1) -> str:
        """
        Randomly swap n pairs of words.

        Args:
            text: Input text
            n: Number of swaps

        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text

        new_words = words.copy()

        for _ in range(n):
            idx1, idx2 = self.rng.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

        return " ".join(new_words)

    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n synonyms.

        Args:
            text: Input text
            n: Number of insertions

        Returns:
            Augmented text
        """
        words = text.split()
        new_words = words.copy()

        for _ in range(n):
            # Find a word with synonyms
            candidates = [word for word in words if word.lower() in self.synonym_dict]

            if candidates:
                word = self.rng.choice(candidates)
                synonyms = self.synonym_dict[word.lower()]
                synonym = self.rng.choice(synonyms)
                insert_pos = self.rng.randint(0, len(new_words))
                new_words.insert(insert_pos, synonym)

        return " ".join(new_words)

    def augment(
        self,
        text: str,
        operations: Optional[List[str]] = None,
    ) -> str:
        """
        Apply multiple augmentation operations.

        Args:
            text: Input text
            operations: List of operations to apply

        Returns:
            Augmented text
        """
        operations = operations or ["synonym_replacement", "random_deletion"]

        for op in operations:
            if hasattr(self, op):
                text = getattr(self, op)(text)

        return text


# =============================================================================
# Embeddings
# =============================================================================


class TokenEmbedding(nn.Module):
    """Base token embedding layer."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int = 0,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(input_ids)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from Attention Is All You Need)."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output with positional encoding [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings."""

    def __init__(
        self,
        max_seq_len: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output with learned positional embedding
        """
        batch_size, seq_len, _ = x.shape
        positions = (
            torch.arange(0, seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        position_embeds = self.position_embeddings(positions)
        return self.dropout(x + position_embeds)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    From RoFormer: Enhanced Transformer with Rotary Position Embedding.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin cache
        self._update_cos_sin_cache(max_seq_len, torch.get_default_dtype())

    def _update_cos_sin_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        """Update cos and sin cache for given sequence length."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but uses a different permutation to match GLM/LLaMA
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]

        self.register_buffer("cos_cached", cos_cached.to(dtype))
        self.register_buffer("sin_cached", sin_cached.to(dtype))

    def rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embedding to query and key tensors."""
        cos = self.cos_cached[:, :, : q.size(2), :]
        sin = self.sin_cached[:, :, : q.size(2), :]

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length (for cache update)

        Returns:
            Rotary-encoded q and k tensors
        """
        if seq_len is not None and seq_len > self.cos_cached.size(2):
            self._update_cos_sin_cache(seq_len, q.dtype)

        return self.apply_rotary_pos_emb(q, k)


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    From Train Short, Test Long: Attention with Linear Biases.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

        # Create linear biases slopes
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

    def _get_slopes(self, n: int) -> Tensor:
        """Get ALiBi slopes for n heads."""

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

        return torch.tensor(slopes)

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Generate ALiBi bias matrix.

        Args:
            seq_len: Sequence length
            device: Target device

        Returns:
            Bias matrix [num_heads, seq_len, seq_len]
        """
        # Create distance matrix
        distance = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(
            seq_len, device=device
        ).unsqueeze(1)
        distance = distance.abs().unsqueeze(0).expand(self.num_heads, -1, -1)

        # Apply slopes
        bias = -distance * self.slopes.view(-1, 1, 1).to(device)

        return bias


# =============================================================================
# Transformer Components
# =============================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional Flash Attention support."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_flash_attention = use_flash_attention and hasattr(
            F, "scaled_dot_product_attention"
        )

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor [batch_size, tgt_len, d_model]
            key: Key tensor [batch_size, src_len, d_model]
            value: Value tensor [batch_size, src_len, d_model]
            attention_mask: Attention mask [tgt_len, src_len]
            key_padding_mask: Key padding mask [batch_size, src_len]
            need_weights: Whether to return attention weights
            is_causal: Whether to apply causal mask

        Returns:
            Output tensor and optional attention weights
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)

        # Project and reshape
        q = (
            self.q_proj(query)
            .view(batch_size, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Use Flash Attention if available and no custom mask needed
        if self.use_flash_attention and not need_weights:
            # Convert masks to attention_bias format if needed
            attn_bias = None
            if attention_mask is not None:
                attn_bias = attention_mask.unsqueeze(0).unsqueeze(0)
            if key_padding_mask is not None:
                # Expand to [batch_size, num_heads, tgt_len, src_len]
                padding_bias = (
                    key_padding_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, self.num_heads, tgt_len, -1)
                )
                if attn_bias is None:
                    attn_bias = padding_bias.float().masked_fill(
                        padding_bias, float("-inf")
                    )
                else:
                    attn_bias = attn_bias + padding_bias.float().masked_fill(
                        padding_bias, float("-inf")
                    )

            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, tgt_len, self.d_model)
            )
            output = self.out_proj(attn_output)
            return output, None

        # Manual attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(tgt_len, src_len, device=query.device), diagonal=1
            ).bool()
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.d_model)
        )

        output = self.out_proj(attn_output)

        if need_weights:
            # Average attention weights across heads
            attn_weights = attn_weights.mean(dim=1)
            return output, attn_weights

        return output, None


class FeedForward(nn.Module):
    """Feed-forward network with multiple activation variants."""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.activation_name = activation

        # SwiGLU variant uses different structure
        if activation == "swiglu":
            self.gate_proj = nn.Linear(d_model, self.d_ff, bias=bias)
            self.up_proj = nn.Linear(d_model, self.d_ff, bias=bias)
            self.down_proj = nn.Linear(self.d_ff, d_model, bias=bias)
        else:
            self.fc1 = nn.Linear(d_model, self.d_ff, bias=bias)
            self.fc2 = nn.Linear(self.d_ff, d_model, bias=bias)

            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            elif activation == "silu":
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if self.activation_name == "swiglu":
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
            x = self.dropout(x)
            return self.down_proj(x)
        else:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            return self.fc2(x)


class LayerNorm(nn.Module):
    """Layer normalization with optional pre-norm and post-norm."""

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_first = norm_first

        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
        )

        # Layer norms
        norm_class = RMSNorm if use_rms_norm else LayerNorm
        self.norm1 = norm_class(d_model)
        self.norm2 = norm_class(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            src: Source tensor [batch_size, seq_len, d_model]
            src_mask: Source mask
            src_key_padding_mask: Source key padding mask

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention block
        if self.norm_first:
            src2 = self.norm1(src)
            src2, _ = self.self_attn(
                src2,
                src2,
                src2,
                attention_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = src + self.dropout1(src2)

            src2 = self.norm2(src)
            src2 = self.ffn(src2)
            src = src + self.dropout2(src2)
        else:
            src2, _ = self.self_attn(
                src,
                src,
                src,
                attention_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )
            src = self.norm1(src + self.dropout1(src2))

            src2 = self.ffn(src)
            src = self.norm2(src + self.dropout2(src2))

        return src


class TransformerDecoderLayer(nn.Module):
    """Standard Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_first = norm_first

        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
        )

        # Layer norms
        norm_class = RMSNorm if use_rms_norm else LayerNorm
        self.norm1 = norm_class(d_model)
        self.norm2 = norm_class(d_model)
        self.norm3 = norm_class(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            tgt: Target tensor [batch_size, tgt_len, d_model]
            memory: Memory tensor from encoder [batch_size, src_len, d_model]
            tgt_mask: Target mask (causal)
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask

        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # Self-attention block
        if self.norm_first:
            tgt2 = self.norm1(tgt)
            tgt2, _ = self.self_attn(
                tgt2,
                tgt2,
                tgt2,
                attention_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=True,
            )
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention block
            tgt2 = self.norm2(tgt)
            tgt2, _ = self.cross_attn(
                tgt2,
                memory,
                memory,
                attention_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = tgt + self.dropout2(tgt2)

            # Feed-forward block
            tgt2 = self.norm3(tgt)
            tgt2 = self.ffn(tgt2)
            tgt = tgt + self.dropout3(tgt2)
        else:
            tgt2, _ = self.self_attn(
                tgt,
                tgt,
                tgt,
                attention_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=True,
            )
            tgt = self.norm1(tgt + self.dropout1(tgt2))

            tgt2, _ = self.cross_attn(
                tgt,
                memory,
                memory,
                attention_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )
            tgt = self.norm2(tgt + self.dropout2(tgt2))

            tgt2 = self.ffn(tgt)
            tgt = self.norm3(tgt + self.dropout3(tgt2))

        return tgt


# =============================================================================
# Language Models
# =============================================================================


class CausalLM(nn.Module):
    """Causal (autoregressive) Language Model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        tie_weights: bool = True,
        use_rope: bool = False,
        use_alibi: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(d_model // num_heads, max_seq_len)
        else:
            self.rope = None
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # ALiBi
        self.alibi = ALiBi(num_heads) if use_alibi else None

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        if tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation

        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)
        if self.rope is None:
            x = self.pos_encoding(x)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()

        # Transformer layers
        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        x = self.norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Get predictions
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :]  # Last token

            # Apply temperature
            logits = logits / temperature

            # Apply top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class MaskedLM(nn.Module):
    """BERT-style Masked Language Model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: MLM labels (-100 for unmasked tokens)

        Returns:
            Dictionary with logits and optionally loss
        """
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(
                x,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        x = self.norm(x)
        logits = self.mlm_head(x)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


class Seq2SeqLM(nn.Module):
    """Sequence-to-Sequence Language Model (encoder-decoder)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.encoder_norm = LayerNorm(d_model)

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.decoder_norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode input sequence.

        Args:
            input_ids: Input token IDs [batch_size, src_len]
            attention_mask: Source attention mask

        Returns:
            Encoder output [batch_size, src_len, d_model]
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(
                x,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        return self.encoder_norm(x)

    def decode(
        self,
        tgt_ids: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode target sequence.

        Args:
            tgt_ids: Target token IDs [batch_size, tgt_len]
            memory: Encoder output
            tgt_mask: Target causal mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask

        Returns:
            Decoder output [batch_size, tgt_len, d_model]
        """
        x = self.token_embedding(tgt_ids)
        x = self.pos_encoding(x)

        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.decoder_norm(x)

    def forward(
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Encoder input token IDs
            decoder_input_ids: Decoder input token IDs
            attention_mask: Encoder attention mask
            decoder_attention_mask: Decoder attention mask
            labels: Target labels for loss

        Returns:
            Dictionary with logits and optionally loss
        """
        memory = self.encode(input_ids, attention_mask)
        decoder_output = self.decode(
            decoder_input_ids,
            memory,
            tgt_key_padding_mask=~decoder_attention_mask
            if decoder_attention_mask is not None
            else None,
            memory_key_padding_mask=~attention_mask
            if attention_mask is not None
            else None,
        )

        logits = self.lm_head(decoder_output)
        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_length: int = 50,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
    ) -> Tensor:
        """
        Generate sequence using greedy decoding.

        Args:
            input_ids: Source input token IDs
            attention_mask: Source attention mask
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs
        """
        self.eval()

        memory = self.encode(input_ids, attention_mask)

        batch_size = input_ids.size(0)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        for _ in range(max_length):
            decoder_output = self.decode(decoder_input_ids, memory)
            logits = self.lm_head(decoder_output[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return decoder_input_ids


class PrefixLM(nn.Module):
    """T5-style Prefix Language Model (encoder-decoder shared)."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embeddings
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Shared encoder-decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (prefix + target)
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        # Causal mask for prefix LM
        # Prefix tokens can attend to all prefix tokens
        # Target tokens can attend to all prefix tokens and previous target tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1,
        ).bool()

        # Transformer layers
        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        x = self.norm(x)
        logits = self.lm_head(x)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


# =============================================================================
# Generation Utilities
# =============================================================================


@torch.no_grad()
def greedy_search(
    model: nn.Module,
    input_ids: Tensor,
    max_length: int = 50,
    eos_token_id: int = 1,
) -> Tensor:
    """
    Greedy decoding.

    Args:
        model: Language model
        input_ids: Initial token IDs
        max_length: Maximum generation length
        eos_token_id: End of sequence token ID

    Returns:
        Generated token IDs
    """
    model.eval()

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if (next_token == eos_token_id).all():
            break

    return input_ids


@dataclass
class BeamSearchOutput:
    """Output from beam search."""

    sequences: Tensor
    scores: Tensor


@torch.no_grad()
def beam_search(
    model: nn.Module,
    input_ids: Tensor,
    beam_width: int = 5,
    max_length: int = 50,
    length_penalty: float = 1.0,
    eos_token_id: int = 1,
    pad_token_id: int = 0,
) -> BeamSearchOutput:
    """
    Beam search with length penalty.

    Args:
        model: Language model
        input_ids: Initial token IDs [batch_size, seq_len]
        beam_width: Number of beams
        max_length: Maximum generation length
        length_penalty: Length penalty factor
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID

    Returns:
        BeamSearchOutput with sequences and scores
    """
    model.eval()
    batch_size = input_ids.size(0)
    device = input_ids.device

    # Expand input for each beam
    input_ids = input_ids.unsqueeze(1).expand(batch_size, beam_width, -1)
    input_ids = input_ids.contiguous().view(batch_size * beam_width, -1)

    # Initialize beam scores
    beam_scores = torch.zeros(batch_size, beam_width, device=device)
    beam_scores[:, 1:] = -1e9  # Only first beam has score 0

    done = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)
    sequences = input_ids.clone()
    finished_sequences = []
    finished_scores = []

    for step in range(max_length):
        outputs = model(sequences)
        logits = outputs["logits"][:, -1, :]  # [batch_size * beam_width, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        # Add scores
        log_probs = log_probs.view(batch_size, beam_width, -1)
        scores = beam_scores.unsqueeze(-1) + log_probs

        # Reshape for top-k
        vocab_size = scores.size(-1)
        scores = scores.view(batch_size, -1)

        # Get top-k
        topk_scores, topk_indices = torch.topk(scores, 2 * beam_width, dim=-1)

        # Convert to beam and token indices
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size

        # Process each batch
        next_beam_scores = []
        next_beam_tokens = []
        next_beam_indices = []

        for batch_idx in range(batch_size):
            batch_beam_scores = []
            batch_beam_tokens = []
            batch_beam_indices = []

            for j in range(2 * beam_width):
                beam_idx = beam_indices[batch_idx, j]
                token_idx = token_indices[batch_idx, j]
                score = topk_scores[batch_idx, j]

                if token_idx == eos_token_id:
                    # Add to finished sequences
                    length = (
                        sequences[batch_idx * beam_width + beam_idx] != pad_token_id
                    ).sum()
                    length_penalized_score = score / (length**length_penalty)
                    finished_sequences.append(
                        sequences[batch_idx * beam_width + beam_idx]
                    )
                    finished_scores.append(length_penalized_score)
                else:
                    batch_beam_scores.append(score)
                    batch_beam_tokens.append(token_idx)
                    batch_beam_indices.append(beam_idx)

                if len(batch_beam_scores) == beam_width:
                    break

            # Pad if needed
            while len(batch_beam_scores) < beam_width:
                batch_beam_scores.append(torch.tensor(-1e9, device=device))
                batch_beam_tokens.append(torch.tensor(pad_token_id, device=device))
                batch_beam_indices.append(0)

            next_beam_scores.append(torch.stack(batch_beam_scores))
            next_beam_tokens.append(torch.stack(batch_beam_tokens))
            next_beam_indices.append(torch.stack(batch_beam_indices))

        beam_scores = torch.stack(next_beam_scores)
        next_tokens = torch.stack(next_beam_tokens).view(batch_size * beam_width, 1)
        beam_indices = torch.stack(next_beam_indices)

        # Gather sequences
        sequences = sequences.view(batch_size, beam_width, -1)
        sequences = torch.gather(
            sequences, 1, beam_indices.unsqueeze(-1).expand(-1, -1, sequences.size(-1))
        )
        sequences = sequences.view(batch_size * beam_width, -1)
        sequences = torch.cat([sequences, next_tokens], dim=1)

    # Return best sequences
    if finished_sequences:
        # Sort finished sequences by score
        finished_scores_tensor = torch.stack(finished_scores)
        best_idx = torch.argmax(finished_scores_tensor)
        best_sequence = finished_sequences[best_idx].unsqueeze(0)
        best_score = finished_scores_tensor[best_idx].unsqueeze(0)
    else:
        # Return current best beam
        best_sequence = sequences[:beam_width]
        best_score = beam_scores[0]

    return BeamSearchOutput(sequences=best_sequence, scores=best_score)


@torch.no_grad()
def top_k_sampling(
    model: nn.Module,
    input_ids: Tensor,
    max_length: int = 50,
    top_k: int = 50,
    temperature: float = 1.0,
    eos_token_id: int = 1,
) -> Tensor:
    """
    Top-k sampling.

    Args:
        model: Language model
        input_ids: Initial token IDs
        max_length: Maximum generation length
        top_k: Number of top tokens to sample from
        temperature: Sampling temperature
        eos_token_id: End of sequence token ID

    Returns:
        Generated token IDs
    """
    model.eval()

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if (next_token == eos_token_id).all():
            break

    return input_ids


@torch.no_grad()
def top_p_sampling(
    model: nn.Module,
    input_ids: Tensor,
    max_length: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    eos_token_id: int = 1,
) -> Tensor:
    """
    Top-p (nucleus) sampling.

    Args:
        model: Language model
        input_ids: Initial token IDs
        max_length: Maximum generation length
        top_p: Nucleus probability threshold
        temperature: Sampling temperature
        eos_token_id: End of sequence token ID

    Returns:
        Generated token IDs
    """
    model.eval()

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :] / temperature

        # Top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if (next_token == eos_token_id).all():
            break

    return input_ids


@torch.no_grad()
def contrastive_search(
    model: nn.Module,
    input_ids: Tensor,
    max_length: int = 50,
    top_k: int = 5,
    penalty_alpha: float = 0.6,
    eos_token_id: int = 1,
) -> Tensor:
    """
    Contrastive search decoding.

    From: A Contrastive Framework for Neural Text Generation

    Args:
        model: Language model
        input_ids: Initial token IDs
        max_length: Maximum generation length
        top_k: Number of candidates to consider
        penalty_alpha: Degeneration penalty (0-1)
        eos_token_id: End of sequence token ID

    Returns:
        Generated token IDs
    """
    model.eval()

    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs["logits"][:, -1, :]

        # Get top-k candidates
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Get hidden states for similarity computation
        # For simplicity, we use token embeddings as proxy for hidden states
        candidates = top_k_indices  # [batch_size, top_k]

        # Compute similarity with previous tokens
        # This is a simplified version - in practice you'd use the model's hidden states
        prev_tokens = input_ids[:, -1:].expand(-1, top_k)
        similarity = (
            candidates == prev_tokens
        ).float()  # Simple token match similarity

        # Compute contrastive scores
        contrastive_scores = (
            1 - penalty_alpha
        ) * top_k_probs - penalty_alpha * similarity

        # Select best candidate
        best_idx = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
        next_token = torch.gather(candidates, 1, best_idx)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if (next_token == eos_token_id).all():
            break

    return input_ids


# =============================================================================
# NLP Tasks
# =============================================================================


class TextClassifier(nn.Module):
    """Text classification model."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pooling: str = "cls",
        classifier_dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = pooling

        # Backbone
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)

        # Classification head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def _pool(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Pool hidden states."""
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
                return sum_embeddings / mask_expanded.sum(dim=1).clamp(min=1e-9)
            return hidden_states.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    ~attention_mask.unsqueeze(-1), float("-inf")
                )
            return hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Classification labels

        Returns:
            Dictionary with logits and optionally loss
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(
                x,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        x = self.norm(x)
        pooled = self._pool(x, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output


class NER(nn.Module):
    """Named Entity Recognition model."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        classifier_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Backbone
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)

        # Token classification head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(d_model, num_labels)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Token-level labels

        Returns:
            Dictionary with logits and optionally loss
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(
                x,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output


class QuestionAnswering(nn.Module):
    """Extractive Question Answering model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Backbone
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNorm(d_model)

        # QA heads (start and end positions)
        self.qa_outputs = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.qa_outputs.weight, std=0.02)
        nn.init.zeros_(self.qa_outputs.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            start_positions: Start position labels
            end_positions: End position labels

        Returns:
            Dictionary with start/end logits and optionally loss
        """
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(
                x,
                src_key_padding_mask=~attention_mask
                if attention_mask is not None
                else None,
            )

        x = self.norm(x)
        logits = self.qa_outputs(x)  # [batch_size, seq_len, 2]

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)  # [batch_size, seq_len]

        output = {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

        if start_positions is not None and end_positions is not None:
            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output["loss"] = total_loss

        return output


# =============================================================================
# Utility Functions
# =============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> str:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return f"{size_mb:.2f} MB"


__all__ = [
    # Tokenizers
    "Tokenizer",
    "CharacterTokenizer",
    "BytePairEncoder",
    "WordPieceTokenizer",
    "SentencePieceTokenizer",
    # Text Preprocessing
    "normalize_text",
    "remove_urls",
    "remove_emails",
    "clean_html",
    "TextAugmenter",
    # Embeddings
    "TokenEmbedding",
    "PositionalEncoding",
    "LearnedPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "ALiBi",
    # Transformer Components
    "MultiHeadAttention",
    "FeedForward",
    "LayerNorm",
    "RMSNorm",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    # Language Models
    "CausalLM",
    "MaskedLM",
    "Seq2SeqLM",
    "PrefixLM",
    # Generation Utilities
    "greedy_search",
    "beam_search",
    "BeamSearchOutput",
    "top_k_sampling",
    "top_p_sampling",
    "contrastive_search",
    # NLP Tasks
    "TextClassifier",
    "NER",
    "QuestionAnswering",
    # Utilities
    "count_parameters",
    "get_model_size",
]
