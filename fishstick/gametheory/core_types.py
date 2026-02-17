"""
Core types for game theory module.

Defines fundamental game theory data structures including strategies,
payoffs, and game outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor


@dataclass
class Player:
    """Represents a player in a game.

    Attributes:
        id: Unique identifier for the player
        name: Optional human-readable name
        num_actions: Number of actions available to the player
    """

    id: int
    name: Optional[str] = None
    num_actions: int = 1

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"Player_{self.id}"


@dataclass
class Strategy:
    """Represents a pure strategy in a game.

    Attributes:
        player_id: ID of the player who owns this strategy
        action: The specific action/strategy index
    """

    player_id: int
    action: int

    def __hash__(self) -> int:
        return hash((self.player_id, self.action))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Strategy):
            return False
        return self.player_id == other.player_id and self.action == other.action


@dataclass
class MixedStrategy:
    """Represents a mixed strategy (probability distribution over actions).

    Attributes:
        player_id: ID of the player who owns this strategy
        probabilities: Probability distribution over actions
    """

    player_id: int
    probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not np.isclose(self.probabilities.sum(), 1.0):
            self.probabilities = self.probabilities / self.probabilities.sum()

    @property
    def entropy(self) -> float:
        """Compute entropy of the mixed strategy."""
        p = self.probabilities
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def sample(self, rng: Optional[np.random.Generator] = None) -> int:
        """Sample a pure strategy from this mixed strategy."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(len(self.probabilities), p=self.probabilities)

    def expected_payoff(self, payoff_vector: NDArray[np.float64]) -> float:
        """Compute expected payoff given a payoff vector."""
        return np.dot(self.probabilities, payoff_vector)


@dataclass
class PayoffMatrix:
    """Represents the payoff matrix for a normal-form game.

    Attributes:
        matrices: Dictionary mapping player IDs to their payoff matrices
        players: List of players in the game
    """

    matrices: Dict[int, NDArray[np.float64]]
    players: List[Player] = field(default_factory=list)

    def get_payoff(self, player_id: int, action_profile: Tuple[int, ...]) -> float:
        """Get payoff for a player given an action profile.

        Args:
            player_id: The player whose payoff to retrieve
            action_profile: Tuple of actions (one per player)

        Returns:
            The payoff value
        """
        matrix = self.matrices[player_id]
        return float(np.ravel_multi_index(action_profile, matrix.shape) % matrix.size)

    def get_payoff_tensor(self, player_id: int) -> NDArray[np.float64]:
        """Get the full payoff tensor for a player."""
        return self.matrices[player_id]

    def num_players(self) -> int:
        """Get number of players."""
        return len(self.matrices)

    def num_actions(self, player_id: int) -> int:
        """Get number of actions for a player."""
        return self.matrices[player_id].shape[0]


@dataclass
class GameOutcome:
    """Represents an outcome of a game.

    Attributes:
        action_profile: The actions chosen by each player
        payoffs: Payoff for each player
        is_equilibrium: Whether this outcome is a Nash equilibrium
    """

    action_profile: Tuple[int, ...]
    payoffs: Dict[int, float]
    is_equilibrium: bool = False

    def __str__(self) -> str:
        actions = ", ".join(f"Player{i}={a}" for i, a in enumerate(self.action_profile))
        payoffs_str = ", ".join(f"Player{i}={p:.3f}" for i, p in self.payoffs.items())
        eq_str = " [EQUILIBRIUM]" if self.is_equilibrium else ""
        return f"Outcome({actions}) -> {payoffs_str}{eq_str}"


@dataclass
class StrategyProfile:
    """A profile of strategies, one for each player.

    Attributes:
        strategies: Dictionary mapping player IDs to their strategies
    """

    strategies: Dict[int, Union[Strategy, MixedStrategy]]

    def get_action_profile(self) -> Tuple[int, ...]:
        """Get the action profile as a tuple of action indices."""
        return tuple(
            s.action if isinstance(s, Strategy) else int(np.argmax(s.probabilities))
            for s in self.strategies.values()
        )

    def to_mixed_profile(self, num_actions: Dict[int, int]) -> Dict[int, MixedStrategy]:
        """Convert to a mixed strategy profile."""
        result = {}
        for pid, strategy in self.strategies.items():
            if isinstance(strategy, Strategy):
                probs = np.zeros(num_actions[pid])
                probs[strategy.action] = 1.0
                result[pid] = MixedStrategy(player_id=pid, probabilities=probs)
            else:
                result[pid] = strategy
        return result


class Game(ABC):
    """Abstract base class for all game types."""

    @abstractmethod
    def num_players(self) -> int:
        """Return the number of players."""
        pass

    @abstractmethod
    def num_actions(self, player_id: int) -> int:
        """Return the number of actions for a player."""
        pass

    @abstractmethod
    def get_payoffs(self, action_profile: Tuple[int, ...]) -> Dict[int, float]:
        """Return payoffs for all players given an action profile."""
        pass

    def get_payoff_matrix(self, player_id: int) -> NDArray[np.float64]:
        """Return the payoff matrix for a player."""
        num_act = self.num_actions(player_id)
        matrix = np.zeros([self.num_actions(i) for i in range(self.num_players())])

        indices = np.ndindex(tuple(matrix.shape))
        for idx in indices:
            profile = tuple(int(i) for i in idx)
            payoffs = self.get_payoffs(profile)
            matrix[idx] = payoffs[player_id]

        return matrix


@dataclass
class UtilityFunction:
    """Represents a utility function for a player.

    Attributes:
        player_id: The player whose utility this represents
        payoff_tensor: The tensor of payoffs for all action profiles
    """

    player_id: int
    payoff_tensor: NDArray[np.float64]

    def __call__(self, action_profile: Tuple[int, ...]) -> float:
        """Evaluate utility at a given action profile."""
        return float(self.payoff_tensor[action_profile])

    def best_response(self, opponent_strategies: Dict[int, MixedStrategy]) -> int:
        """Find the best response to opponent strategies."""
        expected_payoffs = self._compute_expected_payoffs(opponent_strategies)
        return int(np.argmax(expected_payoffs))

    def _compute_expected_payoffs(
        self, opponent_strategies: Dict[int, MixedStrategy]
    ) -> NDArray[np.float64]:
        """Compute expected payoffs for each action."""
        num_actions = self.payoff_tensor.shape[self.player_id]
        expected = np.zeros(num_actions)

        all_opponent_ids = [
            i for i in opponent_strategies.keys() if i != self.player_id
        ]

        for action in range(num_actions):
            profile = [action]
            for opp_id in all_opponent_ids:
                profile.append(opp_id)

            marginals = np.ones(num_actions)
            for opp_id in all_opponent_ids:
                opp_probs = opponent_strategies[opp_id].probabilities
                marginals = np.tensordot(marginals, opp_probs, axes=0)

            expected[action] = np.sum(self.payoff_tensor * marginals)

        return expected


@dataclass
class NashEquilibrium:
    """Represents a Nash equilibrium.

    Attributes:
        strategy_profile: The equilibrium strategy profile
        is_mixed: Whether this is a mixed strategy equilibrium
        is_pareto_optimal: Whether the equilibrium is Pareto optimal
    """

    strategy_profile: Dict[int, Union[Strategy, MixedStrategy]]
    is_mixed: bool = False
    is_pareto_optimal: bool = False

    def get_payoffs(self, game: Game) -> Dict[int, float]:
        """Get the payoffs at this equilibrium."""
        if self.is_mixed:
            profile = tuple(s.probabilities for s in self.strategy_profile.values())
        else:
            profile = tuple(s.action for s in self.strategy_profile.values())

        return game.get_payoffs(profile)


@dataclass
class StabilityAnalysis:
    """Results of stability analysis for a game outcome.

    Attributes:
        outcome: The outcome being analyzed
        is_stable: Whether the outcome is stable
        deviations: List of profitable deviations
        deviation_payoffs: Payoffs from deviations
    """

    outcome: GameOutcome
    is_stable: bool
    deviations: List[Strategy] = field(default_factory=list)
    deviation_payoffs: Dict[Strategy, float] = field(default_factory=dict)

    def add_deviation(self, strategy: Strategy, payoff: float) -> None:
        """Add a deviation to the analysis."""
        self.deviations.append(strategy)
        self.deviation_payoffs[strategy] = payoff
        self.is_stable = False
