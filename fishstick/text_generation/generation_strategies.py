"""
Generation Strategies
=====================

Advanced text generation strategies including:
- Greedy search
- Beam search with various configurations
- Sampling strategies
- Contrastive decoding
- Guided generation with constraints
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_length: int = 100
    min_length: int = 0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False
    num_return_sequences: int = 1


class GenerationStrategy(ABC):
    """Abstract base class for generation strategies."""

    @abstractmethod
    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        """
        Generate text given input IDs and a logits function.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            logits_fn: Function that returns logits for given input
            config: Generation configuration

        Returns:
            List of generated token sequences
        """
        pass

    def _apply_repetition_penalty(
        self,
        logits: Tensor,
        input_ids: Tensor,
        penalty: float,
    ) -> Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits

        for b in range(input_ids.size(0)):
            for token_id in input_ids[b]:
                if logits[b, token_id] < 0:
                    logits[b, token_id] *= penalty
                else:
                    logits[b, token_id] /= penalty

        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: Tensor,
        input_ids: Tensor,
        ngram_size: int,
    ) -> Tensor:
        """Prevent repeating n-grams."""
        if ngram_size <= 0 or input_ids.size(1) < ngram_size:
            return logits

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        if seq_len < ngram_size:
            return logits

        for b in range(batch_size):
            ngram = input_ids[b, -ngram_size:].tolist()
            for i in range(self._vocab_size):
                if input_ids[b, -(ngram_size - 1) :].tolist() + [i] == ngram + [
                    input_ids[b, -1].item()
                ]:
                    logits[b, i] = float("-inf")

        return logits

    @property
    def _vocab_size(self) -> int:
        """Get vocabulary size from the strategy config."""
        return 50000


class GreedyStrategy(GenerationStrategy):
    """Greedy generation strategy - always selects highest probability token."""

    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        generated = input_ids.clone()
        max_length = config.max_length

        for _ in range(max_length - input_ids.size(1)):
            logits = logits_fn(generated)

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, config.repetition_penalty
                )

            next_tokens = logits.argmax(dim=-1)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            if (
                next_tokens == config.eos_token_id
                if hasattr(config, "eos_token_id") and config.eos_token_id
                else False
            ):
                break

        return [generated]


class BeamSearchStrategy(GenerationStrategy):
    """Beam search generation strategy."""

    def __init__(self, vocab_size: int = 50000):
        self._vocab_size = vocab_size

    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        num_beams = config.num_beams
        batch_size = input_ids.size(0)
        device = input_ids.device

        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        beam_tokens = input_ids.unsqueeze(1).expand(-1, num_beams, -1).clone()
        beam_tokens = beam_tokens.view(-1, input_ids.size(1))

        completed_sequences = []
        completed_scores = []

        for step in range(config.max_length - input_ids.size(1)):
            logits = logits_fn(beam_tokens[:, -1:].clone())

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits.view(-1, logits.size(-1)),
                    beam_tokens,
                    config.repetition_penalty,
                )
                logits = logits.view(beam_tokens.size(0), 1, -1)

            log_probs = F.log_softmax(logits.squeeze(1), dim=-1)

            log_probs = log_probs + beam_scores.unsqueeze(-1)

            beam_scores_flat = log_probs.view(batch_size, -1)
            top_scores, top_indices = torch.topk(beam_scores_flat, num_beams, dim=-1)

            beam_ids = top_indices // log_probs.size(-1)
            token_ids = top_indices % log_probs.size(-1)

            new_sequences = []
            for b in range(batch_size):
                prev = beam_tokens[beam_ids[b] + b * num_beams]
                new_sequences.append(
                    torch.cat([prev, token_ids[b].unsqueeze(1)], dim=1)
                )

            beam_tokens = torch.cat(new_sequences, dim=0)
            beam_scores = top_scores.view(-1)

        best_sequences = beam_tokens.view(batch_size, num_beams, -1)[:, 0, :]

        return [best_sequences]


class SamplingStrategy(GenerationStrategy):
    """Sampling-based generation strategy with temperature, top-k, and top-p."""

    def __init__(self, vocab_size: int = 50000):
        self._vocab_size = vocab_size

    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        batch_size = input_ids.size(0)
        device = input_ids.device

        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(config.max_length - input_ids.size(1)):
            logits = logits_fn(generated)

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated, config.repetition_penalty
                )

            if config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram(
                    logits, generated, config.no_repeat_ngram_size
                )

            if config.temperature != 1.0:
                logits = logits / config.temperature

            probs = F.softmax(logits, dim=-1)

            if config.top_k > 0:
                top_k = min(config.top_k, probs.size(-1))
                indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if config.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumsum_probs > config.top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

            finished = finished | (
                next_tokens == config.eos_token_id
                if hasattr(config, "eos_token_id")
                else False
            )

            if finished.all():
                break

        return [generated]


class ContrastiveDecoding(GenerationStrategy):
    """Contrastive decoding for improved generation quality."""

    def __init__(
        self,
        alpha: float = 0.5,
        num_samples: int = 64,
        vocab_size: int = 50000,
    ):
        self.alpha = alpha
        self.num_samples = num_samples
        self._vocab_size = vocab_size

    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        generated = input_ids.clone()
        batch_size = input_ids.size(0)

        for step in range(config.max_length - input_ids.size(1)):
            main_logits = logits_fn(generated).squeeze(1)

            sampled_input = generated.repeat(self.num_samples, 1)
            sampled_logits = logits_fn(sampled_input)

            sampled_probs = F.softmax(sampled_logits.squeeze(1), dim=-1)
            contrastive_logits = sampled_probs.mean(dim=0, keepdim=True).log()

            combined_logits = main_logits - self.alpha * contrastive_logits

            if config.temperature != 1.0:
                combined_logits = combined_logits / config.temperature

            probs = F.softmax(combined_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

        return [generated]


class GuidedGeneration(GenerationStrategy):
    """Guided generation with constraints (e.g., keywords, grammar)."""

    def __init__(
        self,
        required_tokens: Optional[list[list[int]]] = None,
        banned_tokens: Optional[list[int]] = None,
        force_words: Optional[list[list[int]]] = None,
        vocab_size: int = 50000,
    ):
        self.required_tokens = required_tokens or []
        self.banned_tokens = banned_tokens or []
        self.force_words = force_words or []
        self._vocab_size = vocab_size

    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        generated = input_ids.clone()

        for step in range(config.max_length - input_ids.size(1)):
            logits = logits_fn(generated).squeeze(1)

            for banned in self.banned_tokens:
                logits[:, banned] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

        return [generated]

    def add_required_token(self, token_id: int) -> None:
        """Add a required token."""
        self.required_tokens.append([token_id])

    def add_banned_token(self, token_id: int) -> None:
        """Add a banned token."""
        self.banned_tokens.append(token_id)

    def add_force_word(self, token_ids: list[int]) -> None:
        """Add a word that should be forced in generation."""
        self.force_words.append(token_ids)


class DiverseBeamSearch(GenerationStrategy):
    """Diverse beam search for diverse outputs."""

    def __init__(
        self,
        num_beams: int = 5,
        diversity_penalty: float = 0.5,
        vocab_size: int = 50000,
    ):
        self.num_beams = num_beams
        self.diversity_penalty = diversity_penalty
        self._vocab_size = vocab_size

    def generate(
        self,
        input_ids: Tensor,
        logits_fn: Callable[[Tensor], Tensor],
        config: GenerationConfig,
    ) -> list[Tensor]:
        batch_size = input_ids.size(0)
        device = input_ids.device
        vocab_size = self._vocab_size

        beam_scores = torch.zeros(batch_size, self.num_beams, device=device)
        beam_scores[:, 1:] = -1e9

        beam_tokens = input_ids.unsqueeze(1).expand(-1, self.num_beams, -1).clone()

        for step in range(config.max_length - input_ids.size(1)):
            logits = logits_fn(beam_tokens[:, :, -1].clone().view(-1, 1))
            logits = logits.view(batch_size, self.num_beams, -1)

            diversity_scores = torch.zeros_like(logits)
            for b in range(batch_size):
                for beam_idx in range(self.num_beams):
                    prev_tokens = set(beam_tokens[b, beam_idx].tolist())
                    for v in range(vocab_size):
                        if v in prev_tokens:
                            diversity_scores[b, beam_idx, v] = self.diversity_penalty

            logits = logits - diversity_scores

            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs + beam_scores.unsqueeze(-1)

            log_probs_flat = log_probs.view(batch_size, -1)
            top_scores, top_indices = torch.topk(log_probs_flat, self.num_beams, dim=-1)

            beam_ids = top_indices // vocab_size
            token_ids = top_indices % vocab_size

            beam_tokens_new = []
            for b in range(batch_size):
                for beam_idx in range(self.num_beams):
                    prev = beam_tokens[b, beam_ids[b, beam_idx]]
                    beam_tokens_new.append(
                        torch.cat([prev, token_ids[b, beam_idx].unsqueeze(0)])
                    )

            beam_tokens = torch.stack(beam_tokens_new).view(
                batch_size, self.num_beams, -1
            )
            beam_scores = top_scores

        return [beam_tokens[:, 0, :]]


def create_strategy(
    strategy_name: str,
    vocab_size: int = 50000,
    **kwargs,
) -> GenerationStrategy:
    """Factory function to create generation strategy."""
    strategies = {
        "greedy": GreedyStrategy,
        "beam": lambda: BeamSearchStrategy(vocab_size),
        "beam_search": lambda: BeamSearchStrategy(vocab_size),
        "sampling": lambda: SamplingStrategy(vocab_size),
        "contrastive": lambda: ContrastiveDecoding(vocab_size=vocab_size, **kwargs),
        "guided": lambda: GuidedGeneration(vocab_size=vocab_size),
        "diverse_beam": lambda: DiverseBeamSearch(vocab_size=vocab_size, **kwargs),
    }

    if strategy_name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategies[strategy_name.lower()]()
