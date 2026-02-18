"""
Text Decoding Algorithms
=========================

Implements various text decoding strategies for language model generation.
Includes greedy search, beam search, top-k, nucleus (top-p), and temperature sampling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DecodingResult:
    """Container for decoding results."""

    sequences: Tensor
    scores: Optional[Tensor] = None
    log_probs: Optional[Tensor] = None
    attention_mask: Optional[Tensor] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DecoderBase(ABC):
    """Abstract base class for text decoders."""

    @abstractmethod
    def decode(
        self,
        logits: Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> DecodingResult:
        """
        Decode logits to generate text tokens.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size)
            max_length: Maximum sequence length to generate
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID

        Returns:
            DecodingResult containing generated sequences and metadata
        """
        pass


class GreedyDecoder(DecoderBase):
    """Greedy decoding strategy - always picks highest probability token."""

    def decode(
        self,
        logits: Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> DecodingResult:
        batch_size = logits.size(0)
        device = logits.device

        generated = torch.full(
            (batch_size, 1),
            logits.argmax(dim=-1).squeeze().item() if logits.size(1) == 1 else 0,
            dtype=torch.long,
            device=device,
        )

        scores = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits_current = (
                logits[:, -1, :] if logits.size(1) > 1 else logits.squeeze(1)
            )
            next_tokens = logits_current.argmax(dim=-1)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_tokens == eos_token_id)

            if finished.all():
                break

            logits = self._get_next_logits(generated, logits, logits_current)

        final_scores = torch.zeros(batch_size, device=device)
        return DecodingResult(sequences=generated, scores=final_scores)

    def _get_next_logits(
        self, generated: Tensor, old_logits: Tensor, current_logits: Tensor
    ) -> Tensor:
        """Update logits for next iteration."""
        return current_logits.unsqueeze(1)


class BeamSearchDecoder(DecoderBase):
    """Beam search decoding - maintains top-k candidates."""

    def __init__(self, num_beams: int = 5, length_penalty: float = 1.0):
        self.num_beams = num_beams
        self.length_penalty = length_penalty

    def decode(
        self,
        logits: Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> DecodingResult:
        batch_size = logits.size(0)
        vocab_size = logits.size(-1)
        device = logits.device

        beam_scores = torch.zeros(batch_size, self.num_beams, device=device)
        beam_scores[:, 1:] = -1e9

        beam_tokens = torch.zeros(
            batch_size, self.num_beams, 1, dtype=torch.long, device=device
        )

        beam_scores = beam_scores.view(-1)
        done = [False] * batch_size

        for step in range(max_length - 1):
            inputs = (
                beam_tokens.view(-1, step + 1) if step > 0 else beam_tokens[:, :, 0]
            )

            logits_current = logits[:, -1, :] if step > 0 else logits.squeeze(1)
            log_probs = F.log_softmax(logits_current, dim=-1)

            vocab_size = log_probs.size(-1)
            beam_scores = beam_scores.unsqueeze(-1).expand(-1, vocab_size)

            beam_scores = beam_scores + log_probs

            beam_scores_flat = beam_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(
                beam_scores_flat, self.num_beams, dim=-1
            )

            beam_ids = top_indices // vocab_size
            token_ids = top_indices % vocab_size

            beam_tokens_new = []
            for b in range(batch_size):
                prev = beam_tokens[b, beam_ids[b]]
                beam_tokens_new.append(
                    torch.cat([prev, token_ids[b].unsqueeze(1)], dim=1)
                )

            beam_tokens = torch.stack(beam_tokens_new)

            if eos_token_id is not None:
                for b in range(batch_size):
                    if (beam_tokens[b] == eos_token_id).any():
                        done[b] = True

            if all(done):
                break

            beam_scores = top_scores

        best_sequences = beam_tokens[:, 0]
        best_scores = beam_scores[:, 0] / (
            best_sequences.size(1) ** self.length_penalty
        )

        return DecodingResult(sequences=best_sequences, scores=best_scores)


class TopKDecoder(DecoderBase):
    """Top-k sampling - samples from top-k most likely tokens."""

    def __init__(self, top_k: int = 50, temperature: float = 1.0):
        self.top_k = top_k
        self.temperature = temperature

    def decode(
        self,
        logits: Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> DecodingResult:
        batch_size = logits.size(0)
        device = logits.device
        vocab_size = logits.size(-1)

        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            logits_current = logits[:, -1, :] if step > 0 else logits.squeeze(1)

            if self.temperature != 1.0:
                logits_current = logits_current / self.temperature

            probs = F.softmax(logits_current, dim=-1)

            top_k = min(self.top_k, vocab_size)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

            top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

            next_tokens = torch.multinomial(top_probs, num_samples=1)
            next_tokens = top_indices.gather(-1, next_tokens)

            generated = torch.cat([generated, next_tokens], dim=1)

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

            logits = torch.cat(
                [logits, logits_current.unsqueeze(1).expand(-1, 1, -1)], dim=1
            )

        return DecodingResult(sequences=generated)


class NucleusDecoder(DecoderBase):
    """Nucleus (top-p) sampling - samples from smallest set of tokens with cumulative prob > p."""

    def __init__(self, top_p: float = 0.9, temperature: float = 1.0, top_k: int = 0):
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k

    def decode(
        self,
        logits: Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> DecodingResult:
        batch_size = logits.size(0)
        device = logits.device

        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            logits_current = logits[:, -1, :] if step > 0 else logits.squeeze(1)

            if self.temperature != 1.0:
                logits_current = logits_current / self.temperature

            probs = F.softmax(logits_current, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            nucleus_mask = cumsum_probs <= self.top_p
            nucleus_mask = torch.cat(
                [
                    torch.ones_like(nucleus_mask[:, :1]),
                    nucleus_mask[:, :-1] & ~nucleus_mask[:, 1:],
                ],
                dim=-1,
            )

            if self.top_k > 0:
                top_k_mask = torch.zeros_like(probs, dtype=torch.bool)
                top_k_mask.scatter_(1, sorted_indices[:, : self.top_k], True)
                nucleus_mask = nucleus_mask | top_k_mask

            probs_filtered = probs.clone()
            probs_filtered[~nucleus_mask] = 0
            probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

            next_tokens = torch.multinomial(probs_filtered, num_samples=1)

            generated = torch.cat([generated, next_tokens], dim=1)

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

            logits = torch.cat(
                [logits, logits_current.unsqueeze(1).expand(-1, 1, -1)], dim=1
            )

        return DecodingResult(sequences=generated)


class TemperatureDecoder(DecoderBase):
    """Temperature sampling - applies temperature scaling to logits."""

    def __init__(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def decode(
        self,
        logits: Tensor,
        max_length: int,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> DecodingResult:
        batch_size = logits.size(0)
        device = logits.device
        vocab_size = logits.size(-1)

        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            logits_current = logits[:, -1, :] if step > 0 else logits.squeeze(1)
            logits_current = logits_current / self.temperature

            probs = F.softmax(logits_current, dim=-1)

            if self.top_k > 0:
                top_k = min(self.top_k, vocab_size)
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if self.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumsum_probs > self.top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_tokens = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tokens], dim=1)

            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break

            logits = torch.cat(
                [logits, logits_current.unsqueeze(1).expand(-1, 1, -1)], dim=1
            )

        return DecodingResult(sequences=generated)


def create_decoder(
    method: str,
    **kwargs,
) -> DecoderBase:
    """Factory function to create decoder by name."""
    decoders = {
        "greedy": GreedyDecoder,
        "beam": BeamSearchDecoder,
        "beam_search": BeamSearchDecoder,
        "top_k": TopKDecoder,
        "topk": TopKDecoder,
        "nucleus": NucleusDecoder,
        "top_p": NucleusDecoder,
        "topp": NucleusDecoder,
        "temperature": TemperatureDecoder,
    }

    if method.lower() not in decoders:
        raise ValueError(
            f"Unknown decoder: {method}. Available: {list(decoders.keys())}"
        )

    return decoders[method.lower()](**kwargs)
