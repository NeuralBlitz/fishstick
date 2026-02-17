"""
NLP Models
"""

from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TextClassifier(nn.Module):
    """Text classification model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        classifier_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        embedded = self.embedding(x)

        # Pack sequence if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return self.classifier(hidden)


class SequenceTagger(nn.Module):
    """Sequence tagging model (for NER, POS tagging, etc.)."""

    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(output_dim, num_tags)
        self.crf = None  # Could add CRF layer

    def forward(self, x: Tensor) -> Tensor:
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits


class LanguageModel(nn.Module):
    """Language model for next token prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedded = self.embedding(x)
        embedded = self.pos_encoding(embedded)

        # Create causal mask
        mask = (
            torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        )

        output = self.transformer(embedded, mask=mask)
        return self.fc(output)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class TextGenerator:
    """Text generator using a language model."""

    def __init__(self, model: LanguageModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """Generate text from prompt."""
        self.model.eval()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids])

        generated = input_ids.copy()

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_tensor)
                logits = outputs[0, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                generated.append(next_token)
                input_tensor = torch.tensor([generated])

        return self.tokenizer.decode(generated)


class SentimentAnalyzer(TextClassifier):
    """Sentiment analysis model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 3,  # Negative, Neutral, Positive
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            **kwargs,
        )


class NamedEntityRecognizer(SequenceTagger):
    """Named entity recognition model."""

    def __init__(
        self,
        vocab_size: int,
        num_entities: int = 9,  # B-/I- for PER, ORG, LOC, MISC + O
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_tags=num_entities,
            **kwargs,
        )
