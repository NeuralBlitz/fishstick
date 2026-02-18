"""
Advanced WordPiece Tokenizer

A complete implementation of WordPiece tokenization with:
- Vocabulary training from corpus
- greedy and maximum matching subword segmentation
- Support for special tokens and out-of-vocabulary handling
"""

from typing import List, Dict, Set, Optional, Tuple
import re
from collections import defaultdict
import pickle
from pathlib import Path


class WordPieceTokenizer:
    """Advanced WordPiece tokenizer with full implementation.

    WordPiece builds a vocabulary of subword units, preferring longer
    subwords when they appear frequently in the training corpus.

    Attributes:
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency threshold for vocabulary
        max_input_chars_per_word: Maximum characters per word for tokenization
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        max_input_chars_per_word: int = 100,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_input_chars_per_word = max_input_chars_per_word

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

        self.special_tokens = {
            unk_token: 0,
            pad_token: 1,
            cls_token: 2,
            sep_token: 3,
            mask_token: 4,
        }

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.word_frequencies: Dict[str, int] = {}

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for training."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()
        return words

    def train(self, corpus: List[str]) -> None:
        """Train WordPiece tokenizer on corpus.

        Args:
            corpus: List of text strings to train on
        """
        self.word_frequencies = defaultdict(int)

        for text in corpus:
            words = self._preprocess_text(text)
            for word in words:
                self.word_frequencies[word] += 1

        self._build_vocab()
        self._build_trie()

    def _build_vocab(self) -> None:
        """Build vocabulary from word frequencies."""
        vocab = set()

        for word, freq in self.word_frequencies.items():
            if freq >= self.min_frequency:
                chars = list(word)
                for i in range(len(chars)):
                    for j in range(i + 1, len(chars) + 1):
                        vocab.add("".join(chars[i:j]))

        sorted_vocab = sorted(vocab)

        self.token_to_id = dict(self.special_tokens)

        for idx, token in enumerate(sorted_vocab, start=len(self.special_tokens)):
            if len(self.token_to_id) >= self.vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = idx

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def _build_trie(self) -> None:
        """Build a trie for efficient tokenization."""
        self.trie = {}

        for token, idx in self.token_to_id.items():
            if token.startswith("[") and token.endswith("]"):
                continue

            node = self.trie
            for char in token:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node["\0"] = idx

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using WordPiece algorithm.

        Args:
            word: Input word to tokenize

        Returns:
            List of subword tokens
        """
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]

        word = word.lower()

        start = 0
        tokens = []

        while start < len(word):
            end = len(word)
            found = False

            while end > start:
                substr = word[start:end]

                if start > 0:
                    candidate = substr
                else:
                    candidate = substr

                if candidate in self.token_to_id:
                    tokens.append(candidate)
                    found = True
                    break

                end -= 1

            if not found:
                tokens.append(self.unk_token)
                break

            start = end

        if not tokens:
            tokens.append(self.unk_token)

        return tokens

    def _encode_word_max_match(self, word: str) -> List[str]:
        """Encode word using maximum matching algorithm."""
        if not word:
            return []

        if word in self.token_to_id:
            return [word]

        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            longest_match = None

            while end > start:
                substr = word[start:end]

                if substr in self.token_to_id:
                    longest_match = substr
                    break
                end -= 1

            if longest_match is None:
                tokens.append(self.unk_token)
                break

            tokens.append(longest_match)
            start += len(longest_match)

        return tokens if tokens else [self.unk_token]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens and self.cls_token:
            tokens.append(self.token_to_id.get(self.cls_token, 2))

        words = re.findall(r"\S+", text.lower())

        for word in words:
            word_tokens = self._encode_word_max_match(word)
            for token in word_tokens:
                tokens.append(
                    self.token_to_id.get(token, self.token_to_id.get(self.unk_token, 0))
                )

        if add_special_tokens and self.sep_token:
            tokens.append(self.token_to_id.get(self.sep_token, 3))

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
            if token in [self.cls_token, self.sep_token, self.pad_token]:
                continue
            text += token.replace("##", "")

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text to token strings.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        words = re.findall(r"\S+", text.lower())
        tokens = []

        for word in words:
            word_tokens = self._encode_word_max_match(word)
            tokens.extend(word_tokens)

        return tokens

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "word_frequencies": dict(self.word_frequencies),
            "special_tokens": self.special_tokens,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load tokenizer from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.vocab_size = data["vocab_size"]
        self.min_frequency = data["min_frequency"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = data["id_to_token"]
        self.word_frequencies = data["word_frequencies"]
        self.special_tokens = data["special_tokens"]
        self._build_trie()

    @property
    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size."""
        return len(self.token_to_id)


class FastWordPieceTokenizer(WordPieceTokenizer):
    """Optimized WordPiece tokenizer using trie for fast lookups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _encode_word_fast(self, word: str) -> List[str]:
        """Fast tokenization using trie traversal."""
        if not word:
            return []

        word_lower = word.lower()

        if word_lower in self.token_to_id:
            return [word_lower]

        tokens = []
        start = 0

        while start < len(word_lower):
            node = self.trie
            end = len(word_lower)
            longest_match = None

            for pos in range(start, len(word_lower)):
                char = word_lower[pos]

                if char not in node:
                    break

                node = node[char]

                if "\0" in node:
                    longest_match = (word_lower[start : pos + 1], node["\0"])

            if longest_match is None:
                tokens.append(self.unk_token)
                break

            tokens.append(longest_match[0])
            start += len(longest_match[0])

        return tokens if tokens else [self.unk_token]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using fast trie-based algorithm."""
        words = re.findall(r"\S+", text.lower())
        tokens = []

        for word in words:
            word_tokens = self._encode_word_fast(word)
            tokens.extend(word_tokens)

        return tokens


class BertWordPieceTokenizer(WordPieceTokenizer):
    """WordPiece tokenizer compatible with BERT."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        self.special_tokens = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.cls_token: 2,
            self.sep_token: 3,
            self.mask_token: 4,
        }

    def _encode_word_bert(self, word: str) -> List[str]:
        """BERT-style encoding with ## prefix for subwords."""
        if not word:
            return []

        if word in self.token_to_id:
            return [word]

        chars = list(word)
        tokens = []
        start = 0

        while start < len(chars):
            end = len(chars)
            found = False

            while end > start:
                substr = "".join(chars[start:end])

                if start > 0:
                    candidate = "##" + substr
                else:
                    candidate = substr

                if candidate in self.token_to_id:
                    tokens.append(candidate)
                    found = True
                    break

                end -= 1

            if not found:
                return [self.unk_token]

            start = end

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with BERT-style subword marking."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()

        tokens = []
        for word in words:
            word_tokens = self._encode_word_bert(word)
            tokens.extend(word_tokens)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode tokens, removing ## prefixes."""
        tokens = [self.id_to_token.get(tid, self.unk_token) for tid in token_ids]

        text = ""
        for token in tokens:
            if token.startswith("##"):
                text += token[2:]
            elif token not in [self.cls_token, self.sep_token, self.pad_token]:
                text += token

        return text
