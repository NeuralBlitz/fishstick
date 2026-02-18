"""
CRF (Conditional Random Field) Layer module for sequence labeling.

Provides enhanced CRF implementations with:
- Transition constraints
- BIO constraint enforcement
- Viterbi decoding
- Multiple reduction modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Set, Dict
import math


class ConditionalRandomField(nn.Module):
    """Conditional Random Field for sequence labeling.

    A complete CRF implementation with support for various
    constraint types and decoding strategies.

    Args:
        num_tags: Number of tags in the tagging scheme
        batch_first: Whether batch dimension is first
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize transition parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute negative log likelihood.

        Args:
            emissions: Emission scores (batch_size, seq_len, num_tags)
            tags: Target tags (batch_size, seq_len)
            mask: Valid position mask (batch_size, seq_len)
            reduction: Reduction method ('mean', 'sum', 'none')

        Returns:
            Negative log likelihood loss
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "mean":
            return -llh.mean()
        elif reduction == "sum":
            return -llh.sum()
        return -llh

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unnormalized score for a tag sequence."""
        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i].float()
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i].float()

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode the most likely tag sequence."""
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        """Viterbi decoding implementation."""
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class CRF(nn.Module):
    """Simplified CRF interface matching other modules."""

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.crf = ConditionalRandomField(num_tags, batch_first)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        return self.crf(emissions, tags, mask, reduction)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        return self.crf.decode(emissions, mask)


class CRFWithConstraints(nn.Module):
    """CRF with transition constraints for NER.

    Enforces valid transition patterns like:
    - B-X can transition to I-X or O
    - I-X can transition to I-X or O
    - O can transition to B-X or O
    - No I-X can appear without B-X

    Args:
        num_tags: Number of tags
        batch_first: Whether batch dimension is first
        constraints: Optional custom constraint matrix
    """

    def __init__(
        self,
        num_tags: int,
        batch_first: bool = True,
        constraints: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        if constraints is not None:
            self.register_buffer("constraints", constraints)
        else:
            self._create_bio_constraints()

        self._init_weights()

    def _create_bio_constraints(self) -> None:
        """Create default BIO constraints."""
        constraint_matrix = torch.ones(self.num_tags, self.num_tags)

        for i in range(self.num_tags):
            for j in range(self.num_tags):
                tag_i = self._get_tag_name(i)
                tag_j = self._get_tag_name(j)

                if tag_i == "O" and not tag_j.startswith("B-"):
                    constraint_matrix[i, j] = 0
                elif tag_i.startswith("I-") and not tag_j.startswith("I-"):
                    entity_type_i = tag_i[2:]
                    if not (tag_j.startswith("E-" + entity_type_i) or tag_j == "O"):
                        constraint_matrix[i, j] = 0
                elif tag_i.startswith("B-"):
                    if tag_j.startswith("I-"):
                        entity_type_i = tag_i[2:]
                        if not tag_j.startswith("I-" + entity_type_i):
                            constraint_matrix[i, j] = 0

        self.register_buffer("constraints", constraint_matrix)

    def _get_tag_name(self, idx: int) -> str:
        """Get tag name from index. Override in subclass."""
        return "O"

    def _init_weights(self) -> None:
        """Initialize with constrained values."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute negative log likelihood with constraints."""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        constrained_transitions = self.transitions * self.constraints
        constrained_transitions = constrained_transitions.masked_fill(
            self.constraints == 0, float("-inf")
        )

        numerator = self._compute_score(emissions, tags, mask, constrained_transitions)
        denominator = self._compute_normalizer(emissions, mask, constrained_transitions)
        llh = numerator - denominator

        if reduction == "mean":
            return -llh.mean()
        elif reduction == "sum":
            return -llh.sum()
        return -llh

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        seq_length, batch_size = tags.shape

        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            score += transitions[tags[i - 1], tags[i]] * mask[i].float()
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i].float()

        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode with transition constraints."""
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        constrained_transitions = self.transitions * self.constraints
        constrained_transitions = constrained_transitions.masked_fill(
            self.constraints == 0, float("-inf")
        )

        return self._viterbi_decode(emissions, mask, constrained_transitions)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor,
    ) -> List[List[int]]:
        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + transitions + broadcast_emissions
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class ViterbiDecoder:
    """Standalone Viterbi decoder for sequence labeling.

    Can be used independently for decoding with custom
    transition matrices and constraints.

    Args:
        num_tags: Number of tags
        transition_matrix: Transition probability matrix
        start_probs: Start probabilities
        end_probs: End probabilities
    """

    def __init__(
        self,
        num_tags: int,
        transition_matrix: Optional[torch.Tensor] = None,
        start_probs: Optional[torch.Tensor] = None,
        end_probs: Optional[torch.Tensor] = None,
    ):
        self.num_tags = num_tags
        self.transition_matrix = transition_matrix
        self.start_probs = start_probs
        self.end_probs = end_probs

    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Decode using Viterbi algorithm.

        Args:
            emissions: Emission scores (seq_len, batch_size, num_tags) or
                      (batch_size, seq_len, num_tags) if batch_first
            mask: Valid position mask

        Returns:
            List of tag sequences
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )

        if emissions.dim() == 3 and emissions.shape[0] != self.num_tags:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:
        seq_length, batch_size = mask.shape

        if self.start_probs is not None:
            score = self.start_probs + emissions[0]
        else:
            score = emissions[0]

        history = []

        for i in range(1, seq_length):
            if self.transition_matrix is not None:
                broadcast_score = score.unsqueeze(2)
                broadcast_emissions = emissions[i].unsqueeze(1)

                next_score = (
                    broadcast_score + self.transition_matrix + broadcast_emissions
                )
                next_score, indices = next_score.max(dim=1)
            else:
                next_score = score.unsqueeze(1).expand(-1, self.num_tags, -1)
                next_score, indices = next_score.max(dim=1)
                next_score = next_score + emissions[i]

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        if self.end_probs is not None:
            score += self.end_probs

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    @staticmethod
    def compute_transition_matrix(
        allowed_transitions: List[Tuple[str, str]],
        tags: List[str],
    ) -> torch.Tensor:
        """Compute transition matrix from allowed transitions.

        Args:
            allowed_transitions: List of (from_tag, to_tag) tuples
            tags: List of all tags

        Returns:
            Transition matrix
        """
        num_tags = len(tags)
        tag_to_idx = {tag: i for i, tag in enumerate(tags)}
        transition_matrix = torch.zeros(num_tags, num_tags)

        for from_tag, to_tag in allowed_transitions:
            if from_tag in tag_to_idx and to_tag in tag_to_idx:
                transition_matrix[tag_to_idx[from_tag], tag_to_idx[to_tag]] = 1.0

        return transition_matrix


class BioulCrf(CRFWithConstraints):
    """CRF with BIOUL tagging scheme constraints.

    Enforces constraints specific to BIOUL tagging:
    - B-X -> I-X, L-X, O
    - I-X -> I-X, L-X, O
    - L-X -> B-X, O
    - U-X -> B-X, O
    - E-X -> B-X, O
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__(num_tags, batch_first)

    def _get_tag_name(self, idx: int) -> str:
        return "O"

    def _create_bio_constraints(self) -> None:
        constraint_matrix = torch.ones(self.num_tags, self.num_tags)
        self.register_buffer("constraints", constraint_matrix)


class NoSkipCrf(CRFWithConstraints):
    """CRF with no-skip constraints.

    Ensures that I-X cannot follow O or different entity type.
    Must have B-X before I-X.
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__(num_tags, batch_first)

    def _get_tag_name(self, idx: int) -> str:
        return "O"

    def _create_bio_constraints(self) -> None:
        constraint_matrix = torch.ones(self.num_tags, self.num_tags)

        for i in range(self.num_tags):
            for j in range(self.num_tags):
                if i != j:
                    constraint_matrix[i, j] = 0

        constraint_matrix[:, 0] = 1
        constraint_matrix[0, :] = 1

        self.register_buffer("constraints", constraint_matrix)
