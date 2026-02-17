# Natural Language Processing

NLP utilities including tokenizers, embeddings, and text models.

## Installation

```bash
pip install fishstick[nlp]
```

## Overview

The `nlp` module provides natural language processing utilities including tokenizers, embeddings, and text classification models.

## Usage

```python
from fishstick.nlp import BytePairEncoding, WordPieceTokenizer, TextClassifier

# Tokenizers
bpe = BytePairEncoding(vocab_size=30000)
tokens = bpe.encode("Hello world")

# Text classifier
classifier = TextClassifier(
    vocab_size=30000,
    embed_dim=256,
    num_classes=2
)
predictions = classifier(text_ids)
```

## Tokenizers

| Tokenizer | Description |
|-----------|-------------|
| `BytePairEncoding` | BPE tokenization |
| `WordPieceTokenizer` | WordPiece tokenization |

## Models

| Model | Description |
|-------|-------------|
| `WordEmbedding` | Word embeddings |
| `PositionalEncoding` | Positional encoding |
| `TextClassifier` | Text classification model |
| `SequenceTagger` | Sequence tagging model |
| `LanguageModel` | Language model |

## Examples

See `examples/nlp/` for complete examples.
