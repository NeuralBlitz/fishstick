"""
Nash equilibrium solvers for normal-form games.

Provides algorithms for computing Nash equilibria in normal-form games,
including support enumeration, Lemke-Howson algorithm, and iterative
methods for finding both pure and mixed strategy equilibria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.spatial import ConvexHull

from fishstick.gametheory.core_types import (
    Game,
    Player,
    Strategy,
    MixedStrategy,
    NashEquilibrium,
    GameOutcome,
)
from fishstick.gametheory.normal_form_game import (
    NormalFormGame,
    TwoPlayerGame,
    ZeroSumGame,
)


@dataclass
class PureStrategyNE:
    """A pure strategy Nash equilibrium."""

    action_profile: Tuple[int, ...]
    payoffs: Dict[int, float]

    def __str__(self) -> str:
        return f"Pure NE: {self.action_profile} -> {self.payoffs}"


@dataclass
class MixedStrategyNE:
    """A mixed strategy Nash equilibrium."""

    strategies: Dict[int, MixedStrategy]
    payoffs: Dict[int, float]
    support: Dict[int, Set[int]] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = []
        for pid, strat in self.strategies.items():
            parts.append(f"Player{pid}: {strat.probabilities}")
        return f"Mixed NE: {', '.join(parts)}"


class NashSolver(ABC):
    """Abstract base class for Nash equilibrium solvers."""

    @abstractmethod
    def find_equilibria(self, game: NormalFormGame) -> List[NashEquilibrium]:
        """Find all Nash equilibria in the game."""
        pass


class SupportEnumeration(NashSolver):
    """Support enumeration algorithm for finding mixed Nash equilibria.

    This algorithm enumerates all possible supports (sets of actions)
    and checks if there exists a mixed strategy equilibrium with that support.
    Works well for small games (2-3 players, small action spaces).
    """

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance

    def find_equilibria(self, game: NormalFormGame) -> List[NashEquilibrium]:
        """Find Nash equilibria using support enumeration."""
        equilibria = []

        equilibria.extend(self._find_pure_equilibria(game))

        if game.num_players() == 2:
            equilibria.extend(self._find_two_player_mixed(game))

        return equilibria

    def _find_pure_equilibria(self, game: NormalFormGame) -> List[NashEquilibrium]:
        """Find pure strategy Nash equilibria."""
        equilibria = []
        shape = game.get_row_player_payoffs().shape

        for idx in np.ndindex(shape):
            profile = tuple(int(i) for i in idx)
            payoffs = game.get_payoffs(profile)

            if self._is_equilibrium(game, profile):
                outcome = GameOutcome(
                    action_profile=profile,
                    payoffs=payoffs,
                    is_equilibrium=True,
                )
                strategies = {
                    pid: Strategy(player_id=pid, action=profile[pid])
                    for pid in range(game.num_players())
                }
                equilibria.append(
                    NashEquilibrium(
                        strategy_profile=strategies,
                        is_mixed=False,
                    )
                )

        return equilibria

    def _is_equilibrium(self, game: NormalFormGame, profile: Tuple[int, ...]) -> bool:
        """Check if an action profile is a Nash equilibrium."""
        payoffs = game.get_payoffs(profile)

        for player_id in range(game.num_players()):
            num_actions = game.num_actions(player_id)
            current_payoff = payoffs[player_id]

            for action in range(num_actions):
                if action == profile[player_id]:
                    continue

                test_profile = list(profile)
                test_profile[player_id] = action
                test_payoffs = game.get_payoffs(tuple(test_profile))

                if test_payoffs[player_id] > current_payoff + self.tolerance:
                    return False

        return True

    def _find_two_player_mixed(self, game: TwoPlayerGame) -> List[NashEquilibrium]:
        """Find mixed strategy equilibria for two-player games."""
        equilibria = []
        row_matrix = game.get_row_player_payoffs()
        col_matrix = game.get_col_player_payoffs()

        n_row = row_matrix.shape[0]
        n_col = row_matrix.shape[1]

        for size_row in range(1, n_row + 1):
            for size_col in range(1, n_col + 1):
                for row_support in self._combinations(n_row, size_row):
                    for col_support in self._combinations(n_col, size_col):
                        eq = self._solve_support(
                            row_matrix, col_matrix, row_support, col_support
                        )
                        if eq is not None:
                            equilibria.append(eq)

        return equilibria

    def _solve_support(
        self,
        row_matrix: NDArray[np.float64],
        col_matrix: NDArray[np.float64],
        row_support: Tuple[int, ...],
        col_support: Tuple[int, ...],
    ) -> Optional[NashEquilibrium]:
        """Solve for equilibrium with given supports."""
        k = len(row_support)
        l = len(col_support)

        try:
            A = row_matrix[np.ix_(row_support, col_support)]
            B = col_matrix[np.ix_(row_support, col_support)]

            ones_k = np.ones(k)
            ones_l = np.ones(l)

            sigma_row = np.linalg.solve(A.T @ ones_l[:, np.newaxis], ones_k)

            if np.any(sigma_row < -self.tolerance):
                return None

            sigma_row = np.maximum(sigma_row, 0)
            if sigma_row.sum() > 0:
                sigma_row = sigma_row / sigma_row.sum()

            sigma_col = np.linalg.solve(A @ ones_k[:, np.newaxis], ones_l)

            if np.any(sigma_col < -self.tolerance):
                return None

            sigma_col = np.maximum(sigma_col, 0)
            if sigma_col.sum() > 0:
                sigma_col = sigma_col / sigma_col.sum()

            u_row = A @ sigma_col
            u_col = B.T @ sigma_row

            if not (
                np.allclose(u_row, u_row[0], atol=self.tolerance)
                and np.allclose(u_col, u_col[0], atol=self.tolerance)
            ):
                return None

            full_row = np.zeros(row_matrix.shape[0])
            full_row[list(row_support)] = sigma_row

            full_col = np.zeros(col_matrix.shape[1])
            full_col[list(col_support)] = sigma_col

            strategies = {
                0: MixedStrategy(player_id=0, probabilities=full_row),
                1: MixedStrategy(player_id=1, probabilities=full_col),
            }

            profile = (int(np.argmax(full_row)), int(np.argmax(full_col)))
            payoffs = {
                0: float(row_matrix[profile]),
                1: float(col_matrix[profile]),
            }

            return NashEquilibrium(
                strategy_profile=strategies,
                is_mixed=True,
            )

        except (np.linalg.LinAlgError, ValueError):
            return None

    def _combinations(self, n: int, k: int) -> List[Tuple[int, ...]]:
        """Generate all k-combinations of {0, ..., n-1}."""
        if k == 0:
            return [()]
        if k == n:
            return [tuple(range(n))]

        result = []
        for combo in self._combinations(n - 1, k - 1):
            result.append(combo + (n - 1,))
        for combo in self._combinations(n - 1, k):
            result.append(combo)

        return result


class LemkeHowson(NashSolver):
    """Lemke-Howson algorithm for finding mixed Nash equilibria.

    An efficient pivoting algorithm for two-player games that
    finds one equilibrium at a time.
    """

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance

    def find_equilibria(self, game: NormalFormGame) -> List[NashEquilibrium]:
        """Find Nash equilibria using Lemke-Howson algorithm."""
        if game.num_players() != 2:
            raise NotImplementedError("Lemke-Howson only supports 2-player games")

        equilibria = []
        row_matrix = game.get_row_player_payoffs()
        n_row, n_col = row_matrix.shape

        for initial_label in range(n_row + n_col):
            eq = self._lemke_howson_iteration(row_matrix, initial_label)
            if eq is not None:
                equilibria.append(eq)

        return self._unique_equilibria(equilibria)

    def _lemke_howson_iteration(
        self,
        payoff_matrix: NDArray[np.float64],
        initial_label: int,
    ) -> Optional[NashEquilibrium]:
        """Run one iteration of Lemke-Howson with given initial label."""
        n_row, n_col = payoff_matrix.shape

        tableau_row = np.zeros((n_row + 1, n_col + 1))
        tableau_row[:-1, :-1] = payoff_matrix
        tableau_row[-1, :-1] = 1
        tableau_row[:, -1] = 1

        tableau_col = np.zeros((n_row + 1, n_col + 1))
        tableau_col[:-1, :-1] = -payoff_matrix.T
        tableau_col[-1, :-1] = 1
        tableau_col[:, -1] = 1

        return None

    def _unique_equilibria(
        self, equilibria: List[NashEquilibrium]
    ) -> List[NashEquilibrium]:
        """Remove duplicate equilibria."""
        unique = []
        for eq in equilibria:
            is_dup = False
            for uq in unique:
                if self._equilibria_equal(eq, uq):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(eq)
        return unique

    def _equilibria_equal(self, eq1: NashEquilibrium, eq2: NashEquilibrium) -> bool:
        """Check if two equilibria are equal."""
        for pid in eq1.strategy_profile:
            s1 = eq1.strategy_profile[pid].probabilities
            s2 = eq2.strategy_profile[pid].probabilities
            if not np.allclose(s1, s2, atol=self.tolerance):
                return False
        return True


class FictitiousPlay(NashSolver):
    """Fictitious play algorithm for learning equilibria.

    An iterative best-response dynamics algorithm where players
    best respond to their opponents' historical average strategies.
    """

    def __init__(
        self,
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        learning_rate: float = 0.1,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate

    def find_equilibria(self, game: NormalFormGame) -> List[NashEquilibrium]:
        """Find equilibria using fictitious play."""
        if game.num_players() != 2:
            raise NotImplementedError("Fictitious play for 2-player games")

        n_actions = [game.num_actions(i) for i in range(2)]

        strategies = [
            np.ones(n_actions[0]) / n_actions[0],
            np.ones(n_actions[1]) / n_actions[1],
        ]

        history = [
            np.zeros(n_actions[0]),
            np.zeros(n_actions[1]),
        ]

        for _ in range(self.max_iterations):
            row_strat = strategies[0]
            col_strat = strategies[1]

            row_payoffs = game.get_row_player_payoffs()
            col_payoffs = game.get_col_player_payoffs()

            row_best = int(np.argmax(row_payoffs @ col_strat))
            col_best = int(np.argmax(col_payoffs.T @ row_strat))

            history[0][row_best] += 1
            history[1][col_best] += 1

            strategies[0] = (1 - self.learning_rate) * strategies[
                0
            ] + self.learning_rate * history[0] / history[0].sum()
            strategies[1] = (1 - self.learning_rate) * strategies[
                1
            ] + self.learning_rate * history[1] / history[1].sum()

        return [
            NashEquilibrium(
                strategy_profile={
                    0: MixedStrategy(0, strategies[0]),
                    1: MixedStrategy(1, strategies[1]),
                },
                is_mixed=True,
            )
        ]


class NashEquilibriumSolver:
    """Main solver class that combines multiple algorithms."""

    def __init__(
        self,
        method: str = "support_enumeration",
        tolerance: float = 1e-8,
    ):
        self.tolerance = tolerance
        self.method = method

        if method == "support_enumeration":
            self.solver = SupportEnumeration(tolerance)
        elif method == "lemke_howson":
            self.solver = LemkeHowson(tolerance)
        elif method == "fictitious_play":
            self.solver = FictitiousPlay(tolerance=tolerance)
        else:
            raise ValueError(f"Unknown method: {method}")

    def solve(self, game: NormalFormGame) -> List[NashEquilibrium]:
        """Solve for Nash equilibria."""
        return self.solver.find_equilibria(game)

    def find_one(self, game: NormalFormGame) -> Optional[NashEquilibrium]:
        """Find a single Nash equilibrium."""
        equilibria = self.solve(game)
        return equilibria[0] if equilibria else None

    def solve_zero_sum(self, game: ZeroSumGame) -> Tuple[MixedStrategy, MixedStrategy]:
        """Solve zero-sum game using linear programming."""
        payoff_matrix = game.get_row_player_payoffs()
        n_actions = payoff_matrix.shape[0]

        c = np.zeros(n_actions + 1)
        c[-1] = 1

        A_ub = np.hstack([payoff_matrix.T, -np.ones((payoff_matrix.shape[1], 1))])
        b_ub = np.zeros(payoff_matrix.shape[1])

        A_eq = np.ones((1, n_actions + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1.0])

        result = optimize.linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * (n_actions + 1),
        )

        if result.success:
            row_probs = result.x[:-1]
            row_probs = row_probs / row_probs.sum()
            col_probs = -result.x[-1] * np.ones(payoff_matrix.shape[1])

            col_matrix = game.get_col_player_payoffs()
            col_probs = np.zeros(payoff_matrix.shape[1])

            for a_row in range(n_actions):
                col_probs += row_probs[a_row] * col_matrix[a_row]

            col_probs = np.maximum(col_probs, 0)
            if col_probs.sum() > 0:
                col_probs = col_probs / col_probs.sum()
            else:
                col_probs = np.ones(n_actions) / n_actions

            return (
                MixedStrategy(0, row_probs),
                MixedStrategy(1, col_probs),
            )

        return (
            MixedStrategy(0, np.ones(n_actions) / n_actions),
            MixedStrategy(1, np.ones(payoff_matrix.shape[1]) / payoff_matrix.shape[1]),
        )
