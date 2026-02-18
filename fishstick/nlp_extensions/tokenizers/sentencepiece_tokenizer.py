"""
Advanced SentencePiece Tokenizer

A complete implementation of SentencePiece tokenization with:
- UNIGRAM language model training
- BPE mode support
- Character-based fallback
- Efficient vocabulary building
"""

from typing import List, Dict, Set, Optional, Tuple
import re
from collections import defaultdict
import random
import math
import pickle
from pathlib import Path


class SentencePieceTokenizer:
    """Advanced SentencePiece tokenizer with UNIGRAM and BPE support.

    SentencePiece is an unsupervised text tokenizer and detokenizer designed
    for neural network-based text processing.

    Attributes:
        vocab_size: Target vocabulary size
        model_type: Either 'unigram' or 'bpe'
        character_coverage: Coverage of characters in the vocabulary
        unk_id: ID for unknown token
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        model_type: str = "unigram",
        character_coverage: float = 0.9995,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.special_tokens = {
            unk_token: 0,
            pad_token: 1,
            bos_token: 2,
            eos_token: 3,
        }

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.char_frequencies: Dict[str, int] = {}
        self.trie: Dict = {}

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for training."""
        text = text.lower()
        return [text]

    def _build_char_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        """Build character frequency map from corpus."""
        freq = defaultdict(int)

        for text in corpus:
            for char in text:
                freq[char] += 1

        total = sum(freq.values())
        sorted_chars = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        covered = 0
        result = {}
        for char, count in sorted_chars:
            result[char] = count
            covered += count
            if covered / total >= self.character_coverage:
                break

        return result

    def _add_characters_to_vocab(self, char_freqs: Dict[str, int]) -> None:
        """Add characters to vocabulary."""
        sorted_chars = sorted(char_freqs.keys())

        for idx, char in enumerate(sorted_chars, start=len(self.special_tokens)):
            self.token_to_id[char] = idx

    def _add_pieces_to_vocab(self, pieces: List[str]) -> None:
        """Add pieces to vocabulary."""
        next_id = len(self.token_to_id)

        for piece in pieces:
            if piece not in self.token_to_id:
                self.token_to_id[piece] = next_id
                next_id += 1

    def _make_trie(self) -> None:
        """Build trie for efficient tokenization."""
        self.trie = {}

        for token, idx in self.token_to_id.items():
            node = self.trie

            for char in token:
                if char not in node:
                    node[char] = {}
                node = node[char]

            node["\0"] = idx

    def _tokenize_trie(self, text: str) -> List[Tuple[str, int]]:
        """Tokenize using trie, returning (token, id) pairs."""
        if not text:
            return []

        tokens = []
        start = 0

        while start < len(text):
            end = len(text)
            match = None

            node = self.trie
            pos = start

            while pos < len(text):
                char = text[pos]

                if char not in node:
                    break

                node = node[char]

                if "\0" in node:
                    match = (text[start : pos + 1], node["\0"])

                pos += 1

            if match is None:
                tokens.append((self.unk_token, self.special_tokens[self.unk_token]))
                start += 1
            else:
                tokens.append(match)
                start += len(match[0])

        return tokens

    def train(self, corpus: List[str]) -> None:
        """Train SentencePiece tokenizer on corpus.

        Args:
            corpus: List of text strings to train on
        """
        self.token_to_id = dict(self.special_tokens)

        char_freqs = self._build_char_frequencies(corpus)
        self._add_characters_to_vocab(char_freqs)

        pieces = self._extract_pieces(corpus)

        while len(self.token_to_id) < self.vocab_size and pieces:
            piece_scores = self._calculate_piece_scores(pieces)

            if not piece_scores:
                break

            best_piece = max(piece_scores, key=piece_scores.get)

            self.token_to_id[best_piece] = len(self.token_to_id)

            pieces = [p for p in pieces if p != best_piece]

            if len(self.token_to_id) >= self.vocab_size:
                break

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self._make_trie()

    def _extract_pieces(self, corpus: List[str]) -> List[str]:
        """Extract candidate pieces from corpus."""
        pieces = set()

        for text in corpus:
            for length in range(2, min(16, len(text) + 1)):
                for start in range(len(text) - length + 1):
                    piece = text[start : start + length]
                    if piece.isprintable():
                        pieces.add(piece)

        return list(pieces)

    def _calculate_piece_scores(self, pieces: List[str]) -> Dict[str, float]:
        """Calculate scores for candidate pieces using UNIGRAM model."""
        scores = {}

        for piece in pieces:
            freq = sum(
                1 for text in corpus for _ in re.finditer(re.escape(piece), text)
            )

            if freq > 0:
                scores[piece] = freq * (len(piece) ** 0.5)

        return scores

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens and self.bos_token:
            tokens.append(self.token_to_id.get(self.bos_token, 2))

        text_lower = text.lower()
        token_pairs = self._tokenize_trie(text_lower)

        for token, idx in token_pairs:
            tokens.append(idx)

        if add_special_tokens and self.eos_token:
            tokens.append(self.token_to_id.get(self.eos_token, 3))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = [self.id_to_token.get(tid, self.unk_token) for tid in token_ids]

        text = ""
        for token in tokens:
            if token in [self.bos_token, self.eos_token, self.pad_token]:
                continue
            text += token

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to token strings.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        text_lower = text.lower()
        token_pairs = self._tokenize_trie(text_lower)

        return [token for token, _ in token_pairs]

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "special_tokens": self.special_tokens,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.vocab_size = data["vocab_size"]
        self.model_type = data["model_type"]
        self.character_coverage = data["character_coverage"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = data["id_to_token"]
        self.special_tokens = data["special_tokens"]
        self._make_trie()

    @property
    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size."""
        return len(self.token_to_id)


class UnigramSentencePiece(SentencePieceTokenizer):
    """SentencePiece with UNIGRAM language model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_type="unigram", **kwargs)
        self.piece_scores: Dict[str, float] = {}

    def train(self, corpus: List[str]) -> None:
        """Train UNIGRAM model."""
        super().train(corpus)

        for token, idx in self.token_to_id.items():
            self.piece_scores[token] = 1.0 / (idx + 1)


class BPESentencePiece(SentencePieceTokenizer):
    """SentencePiece with BPE model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, model_type="bpe", **kwargs)
        self.merges: List[Tuple[str, str]] = []

    def _extract_pieces(self, corpus: List[str]) -> List[str]:
        """Extract pieces for BPE training."""
        pieces = set()

        for text in corpus:
            words = text.split()
            for word in words:
                chars = list(word) + ["</w>"]
                for i in range(len(chars) - 1):
                    pieces.add("".join(chars[i : i + 2]))

        return list(pieces)


# Global corpus reference for piece scoring
corpus: List[str] = []
