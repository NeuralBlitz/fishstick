"""
Beam Search Implementations

Beam search decoding strategies for sequence generation including
length penalty, diverse beam search, and iterative refinement.
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BeamSearchScorer:
    """Beam search scorer for managing beam hypotheses.

    Handles beam hypothesis tracking, scoring, and selection.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.device = device

        self.beam_scores = torch.zeros(batch_size, num_beams, device=device)
        self.beam_scores[:, 1:] = -1e9
        self.beam_indices: List[List[int]] = []
        self.gen_sequences: List[Tensor] = []
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def process(
        self,
        logits: Tensor,
        curr_length: int,
    ) -> Tuple[Tensor, Tensor]:
        vocab_size = logits.size(-1)

        scores = F.log_softmax(logits, dim=-1)

        scores = self.beam_scores.unsqueeze(-1) + scores

        scores = scores.view(self.batch_size, self.num_beams * vocab_size)

        topk_scores, topk_indices = torch.topk(scores, self.num_beams, dim=-1)

        self.beam_scores = topk_scores

        beam_indices = topk_indices // vocab_size
        next_token_indices = topk_indices % vocab_size

        self.beam_indices.append(beam_indices)

        return next_token_indices, beam_indices

    def finalize(
        self,
        last_tokens: Tensor,
        final_beam_indices: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        sequences = torch.zeros(
            self.batch_size,
            self.num_beams,
            self.max_length + 1,
            dtype=torch.long,
            device=self.device,
        )

        beam_indices = (
            torch.arange(self.num_beams, device=self.device)
            .unsqueeze(0)
            .repeat(self.batch_size, 1)
        )

        for beam_idx, (beam_idx_layer, next_token) in enumerate(
            zip(
                reversed(self.beam_indices),
                [last_tokens.view(-1)] + [None] * len(self.beam_indices),
            )
        ):
            beam_indices = torch.gather(beam_indices, 1, beam_idx_layer)
            sequences[:, :, beam_idx] = torch.gather(
                sequences[:, :, beam_idx],
                1,
                beam_indices.unsqueeze(-1).expand(-1, -1, self.max_length + 1),
            ).squeeze(1)

        final_scores = self._length_penalty(sequences)

        return sequences, final_scores

    def _length_penalty(self, sequences: Tensor) -> Tensor:
        length = sequences.ne(0).sum(dim=-1).float()

        if self.length_penalty == 0:
            return self.beam_scores
        elif self.length_penalty < 0:
            return self.beam_scores / (length ** (-1.0 / self.length_penalty))
        else:
            return self.beam_scores / ((length + 1) ** self.length_penalty)


class BeamSearchDecoder(nn.Module):
    """Beam search decoder for sequence generation.

    Integrates with encoder-decoder models for beam search decoding.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        max_length: int = 50,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        no_repeat_ngram_size: int = 0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def forward(
        self,
        src: Tensor,
        src_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = src.size(0)
        device = src.device

        encoder_output, encoder_hidden = self.encoder(src, src_lengths)

        decoder_hidden = (
            encoder_hidden[0][-2:],
            encoder_hidden[1][-2:],
        )

        beam_scores = torch.zeros(batch_size, self.num_beams, device=device)
        beam_scores[:, 1:] = -1e9

        beam_indices = (
            torch.arange(self.num_beams, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        seq_len = encoder_output.size(1)
        encoder_output = encoder_output.unsqueeze(1).repeat(1, self.num_beams, 1, 1)
        encoder_output = encoder_output.view(batch_size * self.num_beams, seq_len, -1)

        decoder_hidden = (
            decoder_hidden[0]
            .repeat(1, 1, self.num_beams, 1)
            .view(2, batch_size * self.num_beams, -1),
            decoder_hidden[1]
            .repeat(1, 1, self.num_beams, 1)
            .view(2, batch_size * self.num_beams, -1),
        )

        sequences = torch.zeros(
            batch_size, self.num_beams, self.max_length, dtype=torch.long, device=device
        )
        sequences[:, :, 0] = 1

        for t in range(1, self.max_length):
            curr_len = t
            tokens = sequences[:, :, t - 1].reshape(-1)

            logits, decoder_hidden = self.decoder(
                tokens.unsqueeze(-1),
                decoder_hidden,
                encoder_output,
            )

            next_token_logits = logits.squeeze(1)

            if self.no_repeat_ngram_size > 0 and t >= self.no_repeat_ngram_size:
                for batch_idx in range(batch_size):
                    for beam_idx in range(self.num_beams):
                        ngram_sequences = sequences[
                            batch_idx, beam_idx, t - self.no_repeat_ngram_size : t
                        ]
                        for ngram in ngram_sequences.unique_consecutive():
                            if (ngram_sequences == ngram).all():
                                next_token_logits[
                                    batch_idx * self.num_beams + beam_idx, ngram
                                ] = -1e9

            scores = F.log_softmax(next_token_logits, dim=-1)
            scores = beam_scores.unsqueeze(-1) + scores
            scores = scores.view(batch_size, self.num_beams * self.vocab_size)

            topk_scores, topk_indices = torch.topk(scores, self.num_beams, dim=-1)

            beam_indices = topk_indices // self.vocab_size
            next_tokens = topk_indices % self.vocab_size

            beam_scores = topk_scores

            sequences = sequences.view(batch_size, self.num_beams, -1)
            sequences = torch.gather(
                sequences, 1, beam_indices.unsqueeze(-1).expand(-1, -1, t)
            )
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

            sequences = sequences.view(batch_size, self.num_beams, -1)

            if self.early_stopping and (next_tokens == 0).all():
                break

        if self.length_penalty != 0:
            lengths = sequences.ne(0).sum(dim=-1).float()
            penalties = ((lengths + 1) ** self.length_penalty) / (
                2**self.length_penalty
            )
            final_scores = beam_scores / penalties
        else:
            final_scores = beam_scores

        best_sequences = sequences[:, 0, :]
        best_scores = final_scores[:, 0]

        return best_sequences, best_scores


class DiverseBeamSearchDecoder(nn.Module):
    """Diverse beam search decoder.

    Implements diverse beam search from "Diverse Beam Search for
    Describing Images" (Vijayasel et al., 2016).
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        num_groups: int = 2,
        max_length: int = 50,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        diversity_strength: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.num_groups = num_groups
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.diversity_strength = diversity_strength

    def forward(
        self,
        src: Tensor,
        src_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = src.size(0)
        device = src.device

        encoder_output, encoder_hidden = self.encoder(src, src_lengths)

        all_sequences = []
        all_scores = []

        for group_idx in range(self.num_groups):
            group_beams = self.num_beams

            group_scores = torch.zeros(batch_size, group_beams, device=device)
            group_scores[:, 1:] = -1e9

            sequences = torch.zeros(
                batch_size,
                group_beams,
                self.max_length,
                dtype=torch.long,
                device=device,
            )
            sequences[:, :, 0] = 1

            for t in range(1, self.max_length):
                tokens = sequences[:, :, t - 1].reshape(-1)

                logits, encoder_hidden = self.decoder(
                    tokens.unsqueeze(-1), encoder_output
                )

                scores = F.log_softmax(logits.squeeze(1), dim=-1)

                if group_idx > 0:
                    for prev_group_idx in range(group_idx):
                        prev_sequences = all_sequences[prev_group_idx]
                        for b in range(group_beams):
                            prev_seq = prev_sequences[:, b, :t]
                            for token in prev_seq.unique():
                                scores[:, b, token] -= self.diversity_strength

                scores = group_scores.unsqueeze(-1) + scores
                scores = scores.view(batch_size, group_beams * self.vocab_size)

                topk_scores, topk_indices = torch.topk(scores, group_beams, dim=-1)

                beam_indices = topk_indices // self.vocab_size
                next_tokens = topk_indices % self.vocab_size

                group_scores = topk_scores

                sequences = sequences.view(batch_size, group_beams, -1)
                sequences = torch.gather(
                    sequences, 1, beam_indices.unsqueeze(-1).expand(-1, -1, t)
                )
                sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

                if (next_tokens == 0).all():
                    break

            all_sequences.append(sequences)
            all_scores.append(group_scores)

        final_sequences = torch.cat(all_sequences, dim=1)
        final_scores = torch.cat(all_scores, dim=1)

        return final_sequences, final_scores


class LengthPenalty:
    """Length penalty implementations for beam search.

    Various length penalty strategies to prevent偏向短序列或长序列。
    """

    @staticmethod
    def exponential(sequence_length: Tensor, penalty: float) -> Tensor:
        return (penalty + sequence_length) / (penalty + 1)

    @staticmethod
    def cubic(sequence_length: Tensor, penalty: float) -> Tensor:
        return ((sequence_length + penalty) ** penalty) / ((1 + penalty) ** penalty)

    @staticmethod
    def coverage(
        sequence_length: Tensor, coverage: Tensor, penalty: float = 0.5
    ) -> Tensor:
        return (sequence_length + coverage.sum(-1) * penalty) / (sequence_length + 1)

    @staticmethod
    def eos_penalty(sequences: Tensor, eos_id: int, penalty: float = 0.5) -> Tensor:
        lengths = (sequences == eos_id).float().cumsum(dim=-1)
        return lengths + penalty


class IterativeRefinementDecoder(nn.Module):
    """Iterative refinement decoder (like in diffusion models).

    Implements iterative refinement for sequence generation,
    similar to language model approaches like AR-LM.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        vocab_size: int,
        num_iterations: int = 10,
        max_length: int = 50,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.num_iterations = num_iterations
        self.max_length = max_length

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        src_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = src.size(0)
        device = src.device

        encoder_output, encoder_hidden = self.encoder(src, src_lengths)

        if tgt is not None:
            return encoder_output

        sequences = torch.zeros(
            batch_size, self.max_length, dtype=torch.long, device=device
        )

        for iteration in range(self.num_iterations):
            for t in range(self.max_length):
                tokens = sequences[:, : t + 1]

                logits, _ = self.decoder(tokens, encoder_output)

                next_token = logits[:, -1, :].argmax(dim=-1)
                sequences[:, t] = next_token

        return sequences
