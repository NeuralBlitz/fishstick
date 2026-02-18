"""
Token Utilities
===============

Utilities for token manipulation, streaming, and penalty calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterator, Sequence
import re

import torch
from torch import Tensor


class Tokenizer:
    """Simple tokenizer interface for text generation."""

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
        reverse_vocab: Optional[dict[int, str]] = None,
        unk_token_id: int = 0,
        pad_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ):
        self.vocab = vocab or {}
        self.reverse_vocab = reverse_vocab or {}
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        tokens = text.split()

        if self.vocab:
            token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        else:
            token_ids = [hash(token) % 50000 for token in tokens]

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = []

        for token_id in token_ids:
            if skip_special_tokens and token_id in [
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ]:
                continue

            if self.reverse_vocab and token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append(f"<unk_{token_id}>")

        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return max(max(self.vocab.values()) if self.vocab else 0, 50000)


class TokenStream:
    """Token stream for streaming generation."""

    def __init__(self):
        self._tokens: list[int] = []
        self._log_probs: list[float] = []

    def add_token(self, token_id: int, log_prob: float = 0.0) -> None:
        """Add a token to the stream."""
        self._tokens.append(token_id)
        self._log_probs.append(log_prob)

    def get_tokens(self) -> list[int]:
        """Get all tokens."""
        return self._tokens.copy()

    def get_log_probs(self) -> list[float]:
        """Get all log probabilities."""
        return self._log_probs.copy()

    def get_text(self, tokenizer: Tokenizer) -> str:
        """Get text from tokens."""
        return tokenizer.decode(self._tokens)

    def __iter__(self) -> Iterator[int]:
        """Iterate over tokens."""
        return iter(self._tokens)

    def __len__(self) -> int:
        """Get stream length."""
        return len(self._tokens)


class RepetitionPenalty:
    """Applies repetition penalty to prevent repetitive text generation."""

    def __init__(self, penalty: float = 1.0):
        self.penalty = penalty

    def apply(self, logits: Tensor, input_ids: Tensor) -> Tensor:
        """
        Apply repetition penalty to logits.

        Args:
            logits: Model output logits (batch_size, vocab_size)
            input_ids: Previously generated tokens

        Returns:
            Modified logits with repetition penalty applied
        """
        if self.penalty == 1.0:
            return logits

        for batch_idx in range(input_ids.size(0)):
            for token_id in input_ids[batch_idx]:
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= self.penalty
                else:
                    logits[batch_idx, token_id] /= self.penalty

        return logits


class LengthPenalty:
    """Applies length penalty to favor longer or shorter sequences."""

    def __init__(self, length_penalty: float = 1.0, min_length: int = 0):
        self.length_penalty = length_penalty
        self.min_length = min_length

    def apply(self, score: float, length: int) -> float:
        """
        Apply length penalty to a score.

        Args:
            score: Current score
            length: Current sequence length

        Returns:
            Penalized score
        """
        if length < self.min_length:
            return score

        return score / ((length + 1) ** self.length_penalty)

    def normalize_score(
        self,
        score: float,
        length: int,
        alpha: float = 0.6,
    ) -> float:
        """
        Apply exponential length penalty (used in beam search).

        Args:
            score: Current score
            length: Current sequence length
            alpha: Penalty parameter

        Returns:
            Normalized score
        """
        length_penalty = ((5.0 + length) ** alpha) / ((5.0 + 1) ** alpha)
        return score / length_penalty


class NoRepeatNGram:
    """Prevents repeating n-grams in generation."""

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size
        self._ngram_cache: dict[int, set[tuple]] = {}

    def apply(self, logits: Tensor, input_ids: Tensor, batch_idx: int = 0) -> Tensor:
        """
        Apply no-repeat-ngram penalty.

        Args:
            logits: Model output logits
            input_ids: Previously generated tokens
            batch_idx: Batch index to apply penalty to

        Returns:
            Modified logits with n-grams masked
        """
        if input_ids.size(1) < self.ngram_size:
            return logits

        ngrams = self._get_ngrams(input_ids[batch_idx])
        self._ngram_cache[batch_idx] = ngrams

        for i in range(logits.size(0)):
            if input_ids.size(1) >= self.ngram_size:
                current_ngram = tuple(
                    input_ids[batch_idx][-self.ngram_size + 1 :].tolist()
                )

                for token_id in range(logits.size(1)):
                    if current_ngram + (token_id,) in ngrams:
                        logits[i, token_id] = float("-inf")

        return logits

    def _get_ngrams(self, tokens: Tensor) -> set[tuple]:
        """Extract n-grams from tokens."""
        ngrams = set()
        tokens_list = tokens.tolist()

        for i in range(len(tokens_list) - self.ngram_size + 1):
            ngram = tuple(tokens_list[i : i + self.ngram_size])
            ngrams.add(ngram)

        return ngrams

    def clear_cache(self) -> None:
        """Clear the ngram cache."""
        self._ngram_cache.clear()


class TokenFilter:
    """Filters tokens based on various criteria."""

    @staticmethod
    def filter_by_ids(logits: Tensor, allowed_ids: Tensor) -> Tensor:
        """Keep only allowed token IDs."""
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, allowed_ids, False)
        logits[mask] = float("-inf")
        return logits

    @staticmethod
    def filter_by_threshold(logits: Tensor, threshold: float) -> Tensor:
        """Remove tokens below probability threshold."""
        probs = torch.softmax(logits, dim=-1)
        mask = probs < threshold
        logits[mask] = float("-inf")
        return logits

    @staticmethod
    def keep_top_k(logits: Tensor, k: int) -> Tensor:
        """Keep only top-k tokens."""
        if k <= 0:
            return logits

        top_k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits


class TokenScorer:
    """Scores tokens for ranking/selection."""

    @staticmethod
    def score_by_frequency(tokens: Tensor) -> Tensor:
        """Score tokens by frequency in input."""
        unique, counts = torch.unique(tokens, return_counts=True)
        scores = torch.zeros(tokens.size(0), tokens.max().item() + 1)

        for token_id, count in zip(unique.tolist(), counts.tolist()):
            scores[:, token_id] = count.float()

        return scores

    @staticmethod
    def score_by_position(
        tokens: Tensor,
        decay: float = 0.9,
    ) -> Tensor:
        """Score tokens based on recency."""
        seq_len = tokens.size(0)
        positions = torch.arange(seq_len, 0, -1, dtype=torch.float)
        position_weights = decay**positions

        scores = torch.zeros(tokens.size(0), tokens.max().item() + 1)

        for pos, weight in enumerate(position_weights):
            token_id = tokens[pos].item()
            scores[:, token_id] += weight

        return scores


def batch_decode(
    batch_tokens: list[list[int]],
    tokenizer: Tokenizer,
    skip_special_tokens: bool = True,
) -> list[str]:
    """Decode a batch of token sequences."""
    return [tokenizer.decode(tokens, skip_special_tokens) for tokens in batch_tokens]


def batch_encode(
    texts: list[str],
    tokenizer: Tokenizer,
    add_special_tokens: bool = True,
    padding: bool = True,
    max_length: Optional[int] = None,
) -> Tensor:
    """Encode a batch of texts."""
    encoded = [tokenizer.encode(text, add_special_tokens) for text in texts]

    if padding:
        max_len = max(len(e) for e in encoded) if max_length is None else max_length

        padded = []
        for e in encoded:
            if len(e) < max_len:
                e = e + [tokenizer.pad_token_id] * (max_len - len(e))
            padded.append(e[:max_len])

        encoded = padded

    return torch.tensor(encoded, dtype=torch.long)


def create_bpe_tokenizer(
    text: str,
    vocab_size: int = 10000,
) -> Tokenizer:
    """Create a simple BPE-style tokenizer."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    unique_tokens = list(set(tokens))

    vocab = {token: idx + 4 for idx, token in enumerate(unique_tokens)}
    vocab.update(
        {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
    )

    reverse_vocab = {v: k for k, v in vocab.items()}

    return Tokenizer(
        vocab=vocab,
        reverse_vocab=reverse_vocab,
    )
