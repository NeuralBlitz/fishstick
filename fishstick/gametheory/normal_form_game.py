"""
Normal-form game representations and classic game examples.

Provides classes for representing normal-form (strategic) games,
including two-player zero-sum and general-sum games, plus common
game theory examples like Prisoner's Dilemma and Battle of Sexes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import numpy as np
from numpy.typing import NDArray

from fishstick.gametheory.core_types import (
    Game,
    Player,
    Strategy,
    MixedStrategy,
    PayoffMatrix,
    GameOutcome,
    NashEquilibrium,
)


class GameType(Enum):
    """Types of normal-form games."""

    GENERAL_SUM = "general_sum"
    ZERO_SUM = "zero_sum"
    COORDINATION = "coordination"
    ANTICOORDINATION = "anticoordination"
    PRISONERS_DILEMMA = "prisoners_dilemma"
    CHICKEN = "chicken"
    BATTLE_OF_SEXES = "battle_of_sexes"


@dataclass
class NormalFormGame(Game):
    """A normal-form (strategic) game.

    Attributes:
        players: List of players in the game
        payoff_matrices: Payoff matrices for each player
        game_type: Type of the game
    """

    players: List[Player]
    payoff_matrices: Dict[int, NDArray[np.float64]]
    game_type: GameType = GameType.GENERAL_SUM

    def __post_init__(self) -> None:
        self._validate_payoffs()

    def _validate_payoffs(self) -> None:
        """Validate that payoff matrices have consistent dimensions."""
        shapes = [m.shape for m in self.payoff_matrices.values()]
        if len(set(shapes)) > 1:
            raise ValueError("All payoff matrices must have the same shape")

    def num_players(self) -> int:
        return len(self.players)

    def num_actions(self, player_id: int) -> int:
        return self.payoff_matrices[player_id].shape[0]

    def get_payoffs(self, action_profile: Tuple[int, ...]) -> Dict[int, float]:
        """Get payoffs for all players given an action profile."""
        payoffs = {}
        for pid, matrix in self.payoff_matrices.items():
            payoffs[pid] = float(matrix[action_profile])
        return payoffs

    def get_action_space_size(self) -> int:
        """Get the total size of the action space."""
        shape = next(iter(self.payoff_matrices.values())).shape
        return int(np.prod(shape))

    def get_payoff_matrix(self, player_id: int) -> NDArray[np.float64]:
        """Get the payoff matrix for a specific player."""
        return self.payoff_matrices[player_id].copy()

    def get_row_player_payoffs(self) -> NDArray[np.float64]:
        """Get the row player's payoff matrix."""
        return self.payoff_matrices[self.players[0].id]

    def get_col_player_payoffs(self) -> NDArray[np.float64]:
        """Get the column player's payoff matrix."""
        return self.payoff_matrices[self.players[1].id]

    def is_zero_sum(self) -> bool:
        """Check if this is a zero-sum game."""
        if self.num_players() != 2:
            return False
        p1_payoffs = self.payoff_matrices[self.players[0].id]
        p2_payoffs = self.payoff_matrices[self.players[1].id]
        return np.allclose(p1_payoffs + p2_payoffs, 0.0)

    def is_coordination_game(self) -> bool:
        """Check if this is a coordination game."""
        if self.num_players() != 2:
            return False
        matrix = self.get_row_player_payoffs()
        return matrix[0, 0] > matrix[0, 1] and matrix[1, 1] > matrix[1, 0]

    def is_anticoordination_game(self) -> bool:
        """Check if this is an anti-coordination game (like Chicken)."""
        if self.num_players() != 2:
            return False
        matrix = self.get_row_player_payoffs()
        return matrix[0, 0] > matrix[0, 1] and matrix[1, 0] > matrix[1, 1]

    def get_dominant_strategy(self, player_id: int) -> Optional[int]:
        """Find a dominant strategy for a player if it exists.

        A dominant strategy gives a higher payoff regardless of what
        other players do.
        """
        matrix = self.payoff_matrices[player_id]

        if self.num_players() == 2:
            n_actions = matrix.shape[0]
            for a in range(n_actions):
                is_dominant = True
                for a_other in range(matrix.shape[1]):
                    row_payoffs = matrix[a, a_other]
                    for a_prime in range(n_actions):
                        if a_prime != a:
                            if matrix[a_prime, a_other] >= row_payoffs:
                                is_dominant = False
                                break
                    if not is_dominant:
                        break
                if is_dominant:
                    return a
        return None

    def get_pareto_frontier(self) -> List[GameOutcome]:
        """Find Pareto-optimal outcomes."""
        outcomes = []
        shape = next(iter(self.payoff_matrices.values())).shape

        for idx in np.ndindex(shape):
            profile = tuple(int(i) for i in idx)
            payoffs = self.get_payoffs(profile)
            outcome = GameOutcome(action_profile=profile, payoffs=payoffs)

            is_pareto = True
            for idx2 in np.ndindex(shape):
                profile2 = tuple(int(i) for i in idx2)
                payoffs2 = self.get_payoffs(profile2)

                if self._dominates(payoffs2, payoffs, profile != profile2):
                    is_pareto = False
                    break

            outcome.is_pareto_optimal = is_pareto
            if is_pareto:
                outcomes.append(outcome)

        return outcomes

    def _dominates(
        self,
        payoffs1: Dict[int, float],
        payoffs2: Dict[int, float],
        strict: bool = True,
    ) -> bool:
        """Check if payoffs1 dominate payoffs2 for all players."""
        for pid in payoffs1:
            if strict:
                if payoffs1[pid] <= payoffs2[pid]:
                    return False
            else:
                if payoffs1[pid] < payoffs2[pid]:
                    return False
        return True


@dataclass
class TwoPlayerGame(NormalFormGame):
    """A two-player normal-form game."""

    def __init__(
        self,
        row_payoffs: NDArray[np.float64],
        col_payoffs: NDArray[np.float64],
        row_player_name: str = "Row Player",
        col_player_name: str = "Col Player",
    ):
        players = [
            Player(id=0, name=row_player_name, num_actions=row_payoffs.shape[0]),
            Player(id=1, name=col_player_name, num_actions=col_payoffs.shape[1]),
        ]
        payoff_matrices = {0: row_payoffs, 1: col_payoffs}

        if np.allclose(row_payoffs + col_payoffs, 0.0):
            game_type = GameType.ZERO_SUM
        elif (
            row_payoffs[0, 0] > row_payoffs[0, 1]
            and row_payoffs[1, 1] > row_payoffs[1, 0]
        ):
            game_type = GameType.COORDINATION
        else:
            game_type = GameType.GENERAL_SUM

        super().__init__(players, payoff_matrices, game_type)

        self.row_payoffs = row_payoffs
        self.col_payoffs = col_payoffs

    @property
    def row_player(self) -> Player:
        return self.players[0]

    @property
    def col_player(self) -> Player:
        return self.players[1]


@dataclass
class ZeroSumGame(TwoPlayerGame):
    """A two-player zero-sum game."""

    def __init__(
        self,
        payoff_matrix: NDArray[np.float64],
        row_player_name: str = "Row Player",
        col_player_name: str = "Col Player",
    ):
        col_payoffs = -payoff_matrix
        super().__init__(payoff_matrix, col_payoffs, row_player_name, col_player_name)
        self.game_type = GameType.ZERO_SUM


class CoordinationGame(TwoPlayerGame):
    """A coordination game where players want to take the same action.

    Classic example: Players want to meet at the same location.
    """

    def __init__(
        self,
        high_reward: float = 3.0,
        low_reward: float = 1.0,
        mismatch_penalty: float = 0.0,
    ):
        row_payoffs = np.array(
            [
                [high_reward, mismatch_penalty],
                [mismatch_penalty, low_reward],
            ]
        )
        col_payoffs = row_payoffs.copy()
        super().__init__(row_payoffs, col_payoffs)
        self.game_type = GameType.COORDINATION


class PrisonersDilemma(TwoPlayerGame):
    """The classic Prisoner's Dilemma game.

    Both players defecting is the dominant strategy but worse
    for both than mutual cooperation.
    """

    def __init__(
        self,
        temptation: float = 5.0,
        reward: float = 3.0,
        punishment: float = 1.0,
        sucker: float = 0.0,
    ):
        row_payoffs = np.array(
            [
                [reward, sucker],
                [temptation, punishment],
            ]
        )
        col_payoffs = np.array(
            [
                [reward, temptation],
                [sucker, punishment],
            ]
        )
        super().__init__(row_payoffs, col_payoffs)
        self.game_type = GameType.PRISONERS_DILEMMA

        self.temptation = temptation
        self.reward = reward
        self.punishment = punishment
        self.sucker = sucker


class ChickenGame(TwoPlayerGame):
    """The game of Chicken (also called Hawk-Dove).

    Anti-coordination game where players want to take opposite actions.
    """

    def __init__(
        self,
        reward: float = 3.0,
        temptation: float = 5.0,
        loss: float = 1.0,
        catastrophe: float = 0.0,
    ):
        row_payoffs = np.array(
            [
                [reward, catastrophe],
                [temptation, loss],
            ]
        )
        col_payoffs = np.array(
            [
                [reward, temptation],
                [catastrophe, loss],
            ]
        )
        super().__init__(row_payoffs, col_payoffs)
        self.game_type = GameType.CHICKEN


class BattleOfSexes(TwoPlayerGame):
    """Battle of the Sexes game.

    Coordination game with different preferences - both want
    to coordinate but on different outcomes.
    """

    def __init__(
        self,
        football: float = 2.0,
        opera: float = 1.0,
        mismatch: float = 0.0,
    ):
        row_payoffs = np.array(
            [
                [football, mismatch],
                [mismatch, opera],
            ]
        )
        col_payoffs = np.array(
            [
                [opera, mismatch],
                [mismatch, football],
            ]
        )
        super().__init__(row_payoffs, col_payoffs, "He", "She")
        self.game_type = GameType.BATTLE_OF_SEXES


def create_game_from_payoffs(
    player_payoffs: List[NDArray[np.float64]],
    player_names: Optional[List[str]] = None,
) -> NormalFormGame:
    """Create a normal-form game from payoff arrays.

    Args:
        player_payoffs: List of payoff matrices, one per player
        player_names: Optional names for players

    Returns:
        A NormalFormGame instance
    """
    if player_names is None:
        player_names = [f"Player_{i}" for i in range(len(player_payoffs))]

    players = [
        Player(id=i, name=player_names[i], num_actions=player_payoffs[i].shape[0])
        for i in range(len(player_payoffs))
    ]

    payoff_matrices = {i: player_payoffs[i] for i in range(len(player_payoffs))}

    return NormalFormGame(players, payoff_matrices)


def create_matching_pennies() -> ZeroSumGame:
    """Create the matching pennies game."""
    payoff_matrix = np.array(
        [
            [1, -1],
            [-1, 1],
        ]
    )
    return ZeroSumGame(payoff_matrix, "Heads", "Tails")


def create_rock_paper_scissors() -> ZeroSumGame:
    """Create the rock-paper-scissors game."""
    payoff_matrix = np.array(
        [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ]
    )
    return ZeroSumGame(payoff_matrix, "Player 1", "Player 2")
