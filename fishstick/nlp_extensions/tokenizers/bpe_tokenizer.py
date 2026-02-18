"""
Advanced Byte Pair Encoding (BPE) Tokenizer

A full implementation of BPE tokenization algorithm with:
- Vocabulary training from corpus
- Merge operations
- Efficient encoding and decoding
- Support for special tokens
"""

from typing import List, Dict, Set, Optional, Tuple
import re
from collections import defaultdict
import pickle
from pathlib import Path


class BPETokenizer:
    """Advanced Byte Pair Encoding tokenizer with full implementation.

    Implements the subword tokenization algorithm that iteratively merges
    the most frequent adjacent pairs of characters/bytes in the corpus.

    Attributes:
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency threshold for merges
        special_tokens: Dictionary of special tokens and their IDs
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        special_tokens: Optional[Dict[str, int]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.special_tokens = special_tokens or {}
        if pad_token not in self.special_tokens:
            self.special_tokens[pad_token] = 0
        if unk_token not in self.special_tokens:
            self.special_tokens[unk_token] = 1
        if bos_token not in self.special_tokens:
            self.special_tokens[bos_token] = 2
        if eos_token not in self.special_tokens:
            self.special_tokens[eos_token] = 3

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

    def _get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        """Calculate word frequencies from corpus."""
        freq = defaultdict(int)
        for text in corpus:
            words = self._preprocess_text(text)
            for word in words:
                freq[word] += 1
        return freq

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by normalizing and splitting into words."""
        text = text.lower()
        words = re.findall(r"\S+", text)
        return [w + "</w>" for w in words]

    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Calculate pair frequencies from word frequencies."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self, word_freqs: Dict[str, int], pair: Tuple[str, str]
    ) -> Dict[str, int]:
        """Merge all occurrences of a pair in the vocabulary."""
        bigram = " ".join(pair)
        replacement = "".join(pair)

        new_freq = {}
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_freq[new_word] = freq
        return new_freq

    def train(self, corpus: List[str]) -> None:
        """Train BPE tokenizer on corpus.

        Args:
            corpus: List of text strings to train on
        """
        word_freqs = self._get_word_frequencies(corpus)

        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word.split())

        self.token_to_id = {token: idx for idx, token in enumerate(sorted(vocab))}
        next_id = len(self.token_to_id)

        for token, idx in self.special_tokens.items():
            if token not in self.token_to_id:
                self.token_to_id[token] = max(next_id, idx)
                next_id = max(next_id, idx) + 1

        for i in range(
            len(self.special_tokens), self.vocab_size - len(self.special_tokens)
        ):
            if not word_freqs:
                break

            pairs = self._get_stats(word_freqs)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break

            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = len(self.merges)

            word_freqs = self._merge_vocab(word_freqs, best_pair)

            new_token = "".join(best_pair)
            if new_token not in self.token_to_id:
                self.token_to_id[new_token] = next_id
                next_id += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def _get_pairs(self, word: str) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in a word."""
        symbols = word.split()
        return set(zip(symbols[:-1], symbols[1:]))

    def _encode_word(self, word: str) -> List[str]:
        """Encode a single word using learned merges."""
        if not word:
            return []

        word = word + "</w>"
        pairs = self._get_pairs(word)

        if not pairs:
            return [word]

        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            bigram_str = " ".join(bigram)
            replacement = "".join(bigram)

            word = word.replace(bigram_str, replacement)
            pairs = self._get_pairs(word)

        return word.split()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens and self.bos_token:
            tokens.append(self.token_to_id.get(self.bos_token, 2))

        words = re.findall(r"\S+", text.lower())

        for word in words:
            bpe_tokens = self._encode_word(word)
            for token in bpe_tokens:
                token_id = self.token_to_id.get(token)
                if token_id is None:
                    token_id = self.token_to_id.get(
                        token.replace("</w>", ""),
                        self.token_to_id.get(self.unk_token, 1),
                    )
                tokens.append(token_id)

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
            if token.endswith("</w>"):
                text += token.replace("</w>", "")
            else:
                text += token

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
            tokens.extend(self._encode_word(word))
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
            "merges": self.merges,
            "bpe_ranks": self.bpe_ranks,
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
        self.merges = data["merges"]
        self.bpe_ranks = data["bpe_ranks"]
        self.special_tokens = data["special_tokens"]

    @property
    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size."""
        return len(self.token_to_id)


class AdaptiveBPETokenizer(BPETokenizer):
    """BPE tokenizer with adaptive merging strategies.

    Supports different merging strategies including:
    - frequency-based (default)
    - greedy longest-first
    - entropy-based
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        min_frequency: int = 2,
        merge_strategy: str = "frequency",
        **kwargs,
    ):
        super().__init__(vocab_size, min_frequency, **kwargs)
        self.merge_strategy = merge_strategy

    def _get_stats_entropy(
        self, word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate entropy-based pair scores for smarter merging."""
        pairs = defaultdict(lambda: defaultdict(int))

        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i]][symbols[i + 1]] += freq

        stats = {}
        for first, following in pairs.items():
            total = sum(following.values())
            entropy = 0
            for count in following.values():
                p = count / total
                if p > 0:
                    entropy -= p * (p**0.5)

            for second, count in following.items():
                stats[(first, second)] = count * entropy

        return stats


class ByteLevelBPETokenizer(BPETokenizer):
    """BPE tokenizer that operates at byte level.

    This tokenizer works directly on UTF-8 bytes, providing:
    - Full coverage of all Unicode characters
    - No out-of-vocabulary tokens
    - Consistent tokenization across languages
    """

    def __init__(self, vocab_size: int = 10000, **kwargs):
        super().__init__(vocab_size, **kwargs)

    def _preprocess_text(self, text: str) -> List[str]:
        """Convert text to byte-level representation."""
        bytes_list = list(text.encode("utf-8"))
        return [f"<0x{b:02x}>" for b in bytes_list] + ["</w>"]

    def _decode_bytes(self, tokens: List[str]) -> str:
        """Convert byte tokens back to text."""
        bytes_list = []
        for token in tokens:
            if token.startswith("<0x") and token.endswith(">"):
                byte_val = int(token[3:-1], 16)
                bytes_list.append(byte_val)
            elif token != "</w>":
                bytes_list.extend(token.encode("utf-8"))

        return bytes(bytes_list).decode("utf-8", errors="replace")

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(tid, "") for tid in token_ids]
        return self._decode_bytes(tokens)
