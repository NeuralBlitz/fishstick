"""
Voting mechanisms.

Implements various voting rules and analyzes their properties
such as Condorcet winners, strategyproofness, and manipulability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from itertools import permutations
from collections import Counter

from fishstick.gametheory.core_types import Player


@dataclass
class Vote:
    """A vote (ranking) over candidates.

    Attributes:
        voter_id: ID of the voter
        ranking: List of candidates in order of preference
    """

    voter_id: int
    ranking: List[str]

    def get_position(self, candidate: str) -> int:
        """Get the position of a candidate in the ranking."""
        return self.ranking.index(candidate)

    def prefers(self, candidate1: str, candidate2: str) -> bool:
        """Check if voter prefers candidate1 over candidate2."""
        return self.get_position(candidate1) < self.get_position(candidate2)


@dataclass
class VotingRule(ABC):
    """Abstract base class for voting rules."""

    candidates: Set[str]

    @abstractmethod
    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute the winner given votes."""
        pass

    def is_condorcet_winner(self, votes: List[Vote], candidate: str) -> bool:
        """Check if candidate is Condorcet winner."""
        for other in self.candidates:
            if other == candidate:
                continue

            wins = sum(1 for vote in votes if vote.prefers(candidate, other))

            if wins <= len(votes) / 2:
                return False

        return True

    def get_condorcet_winner(self, votes: List[Vote]) -> Optional[str]:
        """Find Condorcet winner if one exists."""
        for candidate in self.candidates:
            if self.is_condorcet_winner(votes, candidate):
                return candidate
        return None


@dataclass
class PluralityVoting(VotingRule):
    """Plurality voting (first-past-the-post).

    Candidate with most first-place votes wins.
    """

    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute winner using plurality rule."""
        if not votes:
            return None

        first_place_counts = Counter()

        for vote in votes:
            if vote.ranking:
                first_place_counts[vote.ranking[0]] += 1

        if not first_place_counts:
            return None

        return first_place_counts.most_common(1)[0][0]

    def get_scores(self, votes: List[Vote]) -> Dict[str, float]:
        """Get scores for all candidates."""
        scores = {c: 0.0 for c in self.candidates}

        for vote in votes:
            if vote.ranking:
                scores[vote.ranking[0]] += 1

        return scores


@dataclass
class BordaCount(VotingRule):
    """Borda count voting.

    Candidates receive points based on their position in each vote.
    """

    def __init__(self, candidates: Set[str], num_winners: int = 1):
        super().__init__(candidates)
        self.num_winners = num_winners

    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute winner using Borda count."""
        scores = self.get_scores(votes)

        if not scores:
            return None

        return max(scores.items(), key=lambda x: x[1])[0]

    def get_scores(self, votes: List[Vote]) -> Dict[str, float]:
        """Get Borda scores for all candidates."""
        n = len(self.candidates)
        scores = {c: 0.0 for c in self.candidates}

        for vote in votes:
            for position, candidate in enumerate(vote.ranking):
                scores[candidate] += n - position - 1

        return scores

    def get_winners(self, votes: List[Vote]) -> List[str]:
        """Get top candidates by Borda score."""
        scores = self.get_scores(votes)
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_candidates[: self.num_winners]]


@dataclass
class ApprovalVoting(VotingRule):
    """Approval voting.

    Voters can approve of multiple candidates; candidate with
    most approvals wins.
    """

    def __init__(self, candidates: Set[str], approvals: Dict[int, Set[str]]):
        super().__init__(candidates)
        self.approvals = approvals

    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute winner using approval voting."""
        approval_counts = Counter()

        for vote in votes:
            for candidate in vote.ranking[:3]:
                approval_counts[candidate] += 1

        if not approval_counts:
            return None

        return approval_counts.most_common(1)[0][0]

    def get_scores(self, votes: List[Vote]) -> Dict[str, int]:
        """Get approval counts."""
        counts = Counter()

        for vote in votes:
            for candidate in vote.ranking[:3]:
                counts[candidate] += 1

        return dict(counts)


@dataclass
class CondorcetWinner(VotingRule):
    """Condorcet method: find Condorcet winner if one exists.

    A Condorcet winner beats every other candidate in head-to-head
    comparisons.
    """

    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute Condorcet winner."""
        return self.get_condorcet_winner(votes)

    def get_pairwise_matrix(self, votes: List[Vote]) -> NDArray[np.float64]:
        """Get pairwise comparison matrix."""
        candidate_list = sorted(self.candidates)
        n = len(candidate_list)
        matrix = np.zeros((n, n))

        for i, c1 in enumerate(candidate_list):
            for j, c2 in enumerate(candidate_list):
                if i == j:
                    continue

                wins = sum(1 for vote in votes if vote.prefers(c1, c2))
                matrix[i, j] = wins

        return matrix


@dataclass
class RankedPairs(VotingRule):
    """Ranked pairs voting method.

    A Condorcet-compliant method that resolves conflicts
    by looking at strongest pairwise defeats first.
    """

    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute winner using ranked pairs."""
        candidate_list = sorted(self.candidates)
        n = len(candidate_list)

        margins = {}

        for i, c1 in enumerate(candidate_list):
            for j, c2 in enumerate(candidate_list):
                if i == j:
                    continue

                wins_for = sum(1 for vote in votes if vote.prefers(c1, c2))
                wins_against = sum(1 for vote in votes if vote.prefers(c2, c1))

                margin = wins_for - wins_against
                if margin > 0:
                    margins[(c1, c2)] = margin

        sorted_pairs = sorted(margins.items(), key=lambda x: x[1], reverse=True)

        lock = {}
        for (winner, loser), margin in sorted_pairs:
            if self._creates_cycle(lock, winner, loser):
                continue

            if winner not in lock:
                lock[winner] = set()
            lock[winner].add(loser)

        rankings = self._get_rankings_from_lock(lock, candidate_list)

        return rankings[0] if rankings else None

    def _creates_cycle(
        self, lock: Dict[str, Set[str]], winner: str, loser: str
    ) -> bool:
        """Check if adding pair creates cycle."""
        if winner not in lock:
            return False

        def has_path(start: str, target: str, visited: Set[str]) -> bool:
            if start == target:
                return True
            if start in visited:
                return False

            visited.add(start)

            if start in lock:
                for neighbor in lock[start]:
                    if has_path(neighbor, target, visited):
                        return True

            return False

        return has_path(loser, winner, set())

    def _get_rankings_from_lock(
        self, lock: Dict[str, Set[str]], candidates: List[str]
    ) -> List[str]:
        """Get final rankings from lock structure."""
        ranked = []
        remaining = set(candidates)

        while remaining:
            for candidate in list(remaining):
                beaten = set()

                for winner, losers in lock.items():
                    if candidate in losers:
                        beaten.add(winner)

                unbeaten = remaining - beaten

                if unbeaten:
                    ranked.append(list(unbeaten)[0])
                    remaining.remove(list(unbeaten)[0])
                    break
            else:
                ranked.append(list(remaining)[0])
                remaining.remove(list(remaining)[0])

        return ranked


@dataclass
class CopelandMethod(VotingRule):
    """Copeland's method.

    Each candidate gets 1 point for each head-to-head win,
    0.5 for ties, and 0 for losses.
    """

    def compute_winner(self, votes: List[Vote]) -> Optional[str]:
        """Compute winner using Copeland's method."""
        scores = self.get_scores(votes)

        if not scores:
            return None

        return max(scores.items(), key=lambda x: x[1])[0]

    def get_scores(self, votes: List[Vote]) -> Dict[str, float]:
        """Get Copeland scores."""
        scores = {c: 0.0 for c in self.candidates}

        candidate_list = sorted(self.candidates)

        for i, c1 in enumerate(candidate_list):
            for j, c2 in enumerate(candidate_list):
                if i == j:
                    continue

                wins = sum(1 for vote in votes if vote.prefers(c1, c2))
                losses = sum(1 for vote in votes if vote.prefers(c2, c1))

                if wins > losses:
                    scores[c1] += 1
                elif wins == losses:
                    scores[c1] += 0.5

        return scores


def create_voting_rule(rule_type: str, candidates: Set[str], **kwargs) -> VotingRule:
    """Factory function to create voting rule instances."""

    if rule_type == "plurality":
        return PluralityVoting(candidates)
    elif rule_type == "borda":
        return BordaCount(candidates, kwargs.get("num_winners", 1))
    elif rule_type == "approval":
        return ApprovalVoting(candidates, kwargs.get("approvals", {}))
    elif rule_type == "condorcet":
        return CondorcetWinner(candidates)
    elif rule_type == "ranked_pairs":
        return RankedPairs(candidates)
    elif rule_type == "copeland":
        return CopelandMethod(candidates)
    else:
        raise ValueError(f"Unknown voting rule: {rule_type}")


def analyze_manipulability(
    votes: List[Vote], rule: VotingRule
) -> Tuple[bool, Optional[Vote]]:
    """Check if the voting rule is manipulable.

    Returns:
        Tuple of (is_manipulable, manipulating_vote)
    """
    original_winner = rule.compute_winner(votes)

    for vote in votes:
        for new_ranking in permutations(vote.ranking):
            if new_ranking == tuple(vote.ranking):
                continue

            modified_vote = Vote(voter_id=vote.voter_id, ranking=list(new_ranking))

            modified_votes = [
                modified_vote if v.voter_id == vote.voter_id else v for v in votes
            ]

            new_winner = rule.compute_winner(modified_votes)

            if new_winner != original_winner:
                return True, modified_vote

    return False, None
