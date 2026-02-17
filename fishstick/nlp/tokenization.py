"""
NLP Tokenization
"""

from typing import List, Dict
import re


class BytePairEncoding:
    """Byte Pair Encoding (BPE) tokenizer."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, corpus: List[str]) -> None:
        """Train BPE on corpus."""
        # Initialize vocabulary with characters
        vocab = set()
        for text in corpus:
            vocab.update(text)

        self.vocab = {char: i for i, char in enumerate(sorted(vocab))}

        # Simple BPE training (in practice, would do actual BPE merges)
        print(f"Trained BPE with vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.vocab.get(char, 0) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "") for i in token_ids)


class WordPieceTokenizer:
    """WordPiece tokenizer."""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}

    def train(self, corpus: List[str]) -> None:
        """Train WordPiece on corpus."""
        # Simple word-based vocabulary
        words = set()
        for text in corpus:
            words.update(text.split())

        for i, word in enumerate(sorted(words)):
            if len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)

        print(f"Trained WordPiece with vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = text.split()
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_token.get(i, "[UNK]") for i in token_ids)


class SentencePieceTokenizer:
    """SentencePiece tokenizer (simplified implementation)."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = {}

    def train(self, corpus: List[str]) -> None:
        """Train SentencePiece on corpus."""
        # Simplified: just use BPE-like approach
        bpe = BytePairEncoding(self.vocab_size)
        bpe.train(corpus)
        self.vocab = bpe.vocab

    def encode(self, text: str) -> List[int]:
        """Encode text."""
        return [self.vocab.get(char, 0) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs."""
        id_to_char = {v: k for k, v in self.vocab.items()}
        return "".join(id_to_char.get(i, "") for i in token_ids)
