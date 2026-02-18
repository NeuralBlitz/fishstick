"""
Generative Question Answering Implementation

This module provides implementations for generative QA systems including
T5, BART, and Fusion-in-Decoder models.

Author: Fishstick AI Framework
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fishstick.question_answering.types import (
    QAExample,
    QAPrediction,
    Answer,
    AnswerType,
    Context,
    Question,
    QAConfig,
    QATaskType,
)
from fishstick.question_answering.base import GenerativeQABase


class AnswerGenerator(nn.Module):
    """Answer Generation Head for Generative QA.

    Generates free-form answers using a language modeling head.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize AnswerGenerator.

        Args:
            hidden_size: Hidden dimension size
            vocab_size: Vocabulary size
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        hidden_states: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate answer logits.

        Args:
            hidden_states: [batch, seq_len, hidden]
            temperature: Sampling temperature

        Returns:
            Logits over vocabulary [batch, seq_len, vocab]
        """
        hidden = self.dropout(hidden_states)
        logits = self.lm_head(hidden)

        if temperature != 1.0:
            logits = logits / temperature

        return self.softmax(logits)

    def generate(
        self,
        hidden_states: Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tensor:
        """Generate tokens using nucleus sampling.

        Args:
            hidden_states: [batch, seq_len, hidden]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter

        Returns:
            Generated token IDs [batch, max_length]
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)

        for step in range(max_length):
            logits = self.forward(hidden_states, temperature)
            next_token_logits = logits[:, -1, :]

            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
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
                next_token_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            generated[:, step] = next_tokens.squeeze(-1)

            hidden_states = torch.cat([hidden_states, hidden_states[:, -1:, :]], dim=1)

        return generated


class CopyMechanism(nn.Module):
    """Copy Mechanism for Generative QA.

    Allows the model to copy tokens from the input context.
    """

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        """Initialize CopyMechanism.

        Args:
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.copy_attention = nn.Linear(hidden_size * 2, 1, bias=False)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        hidden_states: Tensor,
        context_hidden: Tensor,
        encoder_output: Tensor,
        copy_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute copy probabilities.

        Args:
            hidden_states: [batch, gen_len, hidden]
            context_hidden: [batch, ctx_len, hidden]
            encoder_output: [batch, ctx_len, hidden]
            copy_mask: [batch, ctx_len]

        Returns:
            Copy probabilities [batch, gen_len, ctx_len]
        """
        gen_len = hidden_states.size(1)
        ctx_len = context_hidden.size(1)

        hidden_expanded = hidden_states.unsqueeze(2).expand(-1, -1, ctx_len, -1)
        ctx_expanded = context_hidden.unsqueeze(1).expand(-1, gen_len, -1, -1)

        combined = torch.cat([hidden_expanded, ctx_expanded], dim=-1)

        copy_scores = self.copy_attention(combined).squeeze(-1)

        if copy_mask is not None:
            mask_expanded = copy_mask.unsqueeze(1).expand(-1, gen_len, -1)
            copy_scores = copy_scores.masked_fill(mask_expanded == 0, float("-inf"))

        copy_probs = F.softmax(copy_scores, dim=-1)

        gate = torch.sigmoid(self.gate(combined))

        return copy_probs * gate


class GenerativeQAModel(GenerativeQABase[nn.Module]):
    """Abstract Generative QA Model.

    Base class for generative QA models.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize Generative QA Model.

        Args:
            config: QA configuration
        """
        super().__init__(config)
        self.hidden_size = config.metadata.get("hidden_size", 768)
        self.vocab_size = config.metadata.get("vocab_size", 30522)

        self.max_new_tokens = config.max_answer_length
        self.num_beams = config.metadata.get("num_beams", 4)

    @abstractmethod
    def encode(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Tensor]:
        """Encode question and context.

        Args:
            question: Question string
            context: Context string

        Returns:
            Dictionary with input_ids, attention_mask
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        pass

    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        max_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Tensor:
        """Generate answer tokens.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search

        Returns:
            Generated token IDs
        """
        max_len = max_new_tokens or self.max_new_tokens
        beams = num_beams or self.num_beams

        if beams > 1:
            return self._beam_search(input_ids, attention_mask, max_len, beams)
        else:
            return self._greedy_decode(input_ids, attention_mask, max_len)

    def _greedy_decode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        max_length: int,
    ) -> Tensor:
        """Greedy decoding.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length

        Returns:
            Generated token IDs
        """
        generated = input_ids

        for _ in range(max_length):
            outputs = self.model(generated, attention_mask)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.eos_token_id:
                break

        return generated

    def _beam_search(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        max_length: int,
        num_beams: int,
    ) -> Tensor:
        """Beam search decoding.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length
            num_beams: Number of beams

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.size(0)

        beam_scores = torch.zeros(batch_size, num_beams, device=input_ids.device)
        beam_scores[:, 1:] = -1e9

        beam_tokens = torch.zeros(
            batch_size,
            num_beams,
            max_length + input_ids.size(1),
            dtype=torch.long,
            device=input_ids.device,
        )
        beam_tokens[:, :, : input_ids.size(1)] = input_ids.unsqueeze(1).expand(
            -1, num_beams, -1
        )

        for step in range(max_length):
            outputs = self.model(beam_tokens.view(-1, beam_tokens.size(2)), None)
            next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams, -1)

            next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

            next_token_scores = next_token_scores.view(batch_size, -1)
            topk_scores, topk_tokens = torch.topk(next_token_scores, num_beams * 2)

            beam_idx = topk_tokens // self.vocab_size
            token_idx = topk_tokens % self.vocab_size

            beam_scores = topk_scores

        return beam_tokens[:, 0, :]


class T5GenerativeQA(GenerativeQAModel):
    """T5-based Generative QA Model.

    Implements generative QA using T5 architecture.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize T5 Generative QA.

        Args:
            config: QA configuration
        """
        super().__init__(config)

        try:
            from transformers import T5ForConditionalGeneration, T5Config

            model_cfg = T5Config.from_pretrained(config.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(config.model_name)
            self.hidden_size = model_cfg.d_model
        except ImportError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        self.eos_token_id = 1
        self.pad_token_id = 0

    def encode(
        self,
        question: str,
        context: str,
    ) -> Dict[str, Tensor]:
        """Encode question and context.

        Args:
            question: Question string
            context: Context string

        Returns:
            Dictionary with input_ids, attention_mask
        """
        if hasattr(self, "model"):
            input_text = f"question: {question} context: {context}"
            encoding = self.model.tokenizer(
                input_text, return_tensors="pt", padding=True
            )
            return encoding

        return {"input_ids": torch.zeros(1, 10), "attention_mask": torch.ones(1, 10)}

    def decode(
        self,
        token_ids: Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if hasattr(self, "model"):
            return self.model.tokenizer.decode(
                token_ids, skip_special_tokens=skip_special_tokens
            )
        return " ".join([str(t) for t in token_ids.tolist()])

    def forward(
        self,
        question: Union[str, Question],
        context: Union[str, Context],
    ) -> Answer:
        """Forward pass to generate answer.

        Args:
            question: The question to answer
            context: The context to generate answer from

        Returns:
            Answer object with the predicted answer
        """
        q_text = question.text if isinstance(question, Question) else question
        c_text = context.text if isinstance(context, Context) else context

        encoding = self.encode(q_text, c_text)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        generated_ids = self.generate(input_ids, attention_mask)
        answer_text = self.decode(generated_ids[0])

        return Answer(
            text=answer_text,
            type=AnswerType.FREE_FORM,
            confidence=0.9,
        )

    def predict(
        self,
        examples: List[QAExample],
    ) -> List[QAPrediction]:
        """Generate predictions for a batch of examples.

        Args:
            examples: List of QA examples to predict

        Returns:
            List of predictions
        """
        predictions = []

        for example in examples:
            answer = self.forward(example.question, example.context)

            pred = QAPrediction(
                id=example.id,
                question=example.question.text
                if isinstance(example.question, Question)
                else example.question,
                answer=answer,
                context_used=example.context.text
                if isinstance(example.context, Context)
                else example.context,
            )
            predictions.append(pred)

        return predictions

    def train_model(
        self,
        train_examples: List[QAExample],
        eval_examples: Optional[List[QAExample]] = None,
    ) -> Dict[str, Any]:
        """Train the QA model.

        Args:
            train_examples: Training examples
            eval_examples: Optional evaluation examples

        Returns:
            Training history dictionary
        """
        raise NotImplementedError("Training not implemented. Use QATrainer.")

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        if hasattr(self, "model"):
            self.model.save_pretrained(path)
        else:
            torch.save(
                {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "lm_head": self.lm_head.state_dict(),
                    "config": self.config.to_dict(),
                },
                path,
            )

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load the model from
        """
        if hasattr(self, "model"):
            self.model = self.model.from_pretrained(path)
        else:
            checkpoint = torch.load(path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.lm_head.load_state_dict(checkpoint["lm_head"])


class BARTGenerativeQA(T5GenerativeQA):
    """BART-based Generative QA Model.

    Implements generative QA using BART architecture.
    """

    def __init__(self, config: QAConfig) -> None:
        """Initialize BART Generative QA.

        Args:
            config: QA configuration
        """
        super().__init__(config)

        try:
            from transformers import BartForConditionalGeneration, BartConfig

            model_cfg = BartConfig.from_pretrained(config.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(config.model_name)
            self.hidden_size = model_cfg.d_model
        except ImportError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        self.eos_token_id = 2
        self.pad_token_id = 1


class FusionInDecoder(nn.Module):
    """Fusion-in-Decoder (FiD) Architecture.

    Processes multiple retrieved passages independently then fuses them
    in the decoder for open-domain QA.
    """

    def __init__(
        self,
        encoder_hidden_size: int = 768,
        decoder_hidden_size: int = 768,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize Fusion-in-Decoder.

        Args:
            encoder_hidden_size: Encoder hidden size
            decoder_hidden_size: Decoder hidden size
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.passage_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_hidden_size,
                nhead=num_heads,
                dim_feedforward=encoder_hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        self.passage_projection = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=decoder_hidden_size,
                nhead=num_heads,
                dim_feedforward=decoder_hidden_size * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )

        self.lm_head = nn.Linear(decoder_hidden_size, decoder_hidden_size)

    def encode_passages(
        self,
        passage_ids: Tensor,
        passage_mask: Tensor,
    ) -> Tensor:
        """Encode multiple passages.

        Args:
            passage_ids: [batch, num_passages, passage_len]
            passage_mask: [batch, num_passages, passage_len]

        Returns:
            Encoded passages [batch, num_passages, passage_len, hidden]
        """
        batch_size, num_passages, passage_len = passage_ids.size()

        passage_ids_flat = passage_ids.view(-1, passage_len)
        passage_mask_flat = passage_mask.view(-1, passage_len)

        encoded_flat = self.passage_encoder(
            passage_ids_flat, src_key_padding_mask=passage_mask_flat == 0
        )

        encoded = encoded_flat.view(batch_size, num_passages, passage_len, -1)

        return self.passage_projection(encoded)

    def forward(
        self,
        passage_hidden: Tensor,
        question_hidden: Tensor,
        target_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of FiD.

        Args:
            passage_hidden: [batch, num_passages, passage_len, hidden]
            question_hidden: [batch, query_len, hidden]
            target_ids: [batch, target_len] for training

        Returns:
            Logits [batch, target_len, vocab_size]
        """
        batch_size, num_passages, passage_len, hidden = passage_hidden.size()

        fused = passage_hidden.sum(dim=1)

        if target_ids is not None:
            tgt_emb = self.decoder(fused, question_hidden)
            output = self.lm_head(tgt_emb)
        else:
            output = self.lm_head(fused)

        return output

    def generate(
        self,
        passage_hidden: Tensor,
        question_hidden: Tensor,
        max_length: int = 50,
    ) -> Tensor:
        """Generate answer.

        Args:
            passage_hidden: [batch, num_passages, passage_len, hidden]
            question_hidden: [batch, query_len, hidden]
            max_length: Maximum length to generate

        Returns:
            Generated token IDs
        """
        batch_size = passage_hidden.size(0)
        device = passage_hidden.device

        generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)

        for step in range(max_length):
            output = self.forward(
                passage_hidden, question_hidden, generated[:, : step + 1]
            )
            next_token = output[:, -1, :].argmax(dim=-1)
            generated[:, step] = next_token

        return generated
