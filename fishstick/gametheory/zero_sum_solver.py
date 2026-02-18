"""
Zero-sum game solvers.

Specialized algorithms for solving two-player zero-sum games,
including linear programming, fictitious play, and gradient-based
methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from fishstick.gametheory.core_types import MixedStrategy
from fishstick.gametheory.normal_form_game import ZeroSumGame, TwoPlayerGame


@dataclass
class ZeroSumSolution:
    """Solution to a zero-sum game.

    Attributes:
        row_strategy: Optimal mixed strategy for row player
        col_strategy: Optimal mixed strategy for column player
        value: Value of the game (row player's expected payoff)
        row_value: Value for row player
        col_value: Value for column player
    """

    row_strategy: MixedStrategy
    col_strategy: MixedStrategy
    value: float
    row_value: float
    col_value: float

    def __str__(self) -> str:
        return (
            f"ZeroSumSolution(value={self.value:.4f}, "
            f"row_strategy={self.row_strategy.probabilities}, "
            f"col_strategy={self.col_strategy.probabilities})"
        )


class ZeroSumSolver(ABC):
    """Abstract base class for zero-sum game solvers."""

    @abstractmethod
    def solve(self, game: ZeroSumGame) -> ZeroSumSolution:
        """Solve the zero-sum game."""
        pass


class LinearProgrammingSolver(ZeroSumSolver):
    """Solve zero-sum games using linear programming.

    For zero-sum games, finding the Nash equilibrium is equivalent
    to solving two linear programs - one for each player.
    """

    def __init__(self, player: int = 0):
        """
        Args:
            player: Which player to optimize for (0 = row, 1 = column)
        """
        self.player = player

    def solve(self, game: ZeroSumGame) -> ZeroSumSolution:
        """Solve using linear programming."""
        payoff = game.get_row_player_payoffs()
        n_row, n_col = payoff.shape

        if self.player == 0:
            return self._solve_row_player(payoff, n_row, n_col)
        else:
            return self._solve_col_player(payoff, n_row, n_col)

    def _solve_row_player(
        self,
        payoff: NDArray[np.float64],
        n_row: int,
        n_col: int,
    ) -> ZeroSumSolution:
        """Solve from row player's perspective."""
        c = np.zeros(n_row + 1)
        c[-1] = -1

        A_ub = np.hstack([-(payoff.T - 1e-6), np.ones((n_col, 1))])
        b_ub = np.zeros(n_col)

        A_eq = np.ones((1, n_row + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1.0])

        result = optimize.linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * (n_row + 1),
        )

        if not result.success:
            row_strat = np.ones(n_row) / n_row
            col_strat = np.ones(n_col) / n_col
            value = float(np.sum(row_strat @ payoff @ col_strat))
            return ZeroSumSolution(
                row_strategy=MixedStrategy(0, row_strat),
                col_strategy=MixedStrategy(1, col_strat),
                value=value,
                row_value=value,
                col_value=-value,
            )

        row_probs = result.x[:-1]
        row_probs = np.maximum(row_probs, 0)
        row_probs = (
            row_probs / row_probs.sum()
            if row_probs.sum() > 0
            else np.ones(n_row) / n_row
        )

        value = -result.fun

        expected_col = np.zeros(n_col)
        for a in range(n_row):
            expected_col += row_probs[a] * payoff[a, :]

        col_probs = np.zeros(n_col)
        col_probs = np.ones(n_col) / n_col

        return ZeroSumSolution(
            row_strategy=MixedStrategy(0, row_probs),
            col_strategy=MixedStrategy(1, col_probs),
            value=value,
            row_value=value,
            col_value=-value,
        )

    def _solve_col_player(
        self,
        payoff: NDArray[np.float64],
        n_row: int,
        n_col: int,
    ) -> ZeroSumSolution:
        """Solve from column player's perspective."""
        c = np.zeros(n_col + 1)
        c[-1] = 1

        A_ub = np.hstack([payoff - 1e-6, np.ones((n_row, 1))])
        b_ub = np.zeros(n_row)

        A_eq = np.ones((1, n_col + 1))
        A_eq[0, -1] = 0
        b_eq = np.array([1.0])

        result = optimize.linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=[(0, None)] * (n_col + 1),
        )

        if not result.success:
            col_strat = np.ones(n_col) / n_col
            row_strat = np.ones(n_row) / n_row
            value = float(np.sum(row_strat @ payoff @ col_strat))
            return ZeroSumSolution(
                row_strategy=MixedStrategy(0, row_strat),
                col_strategy=MixedStrategy(1, col_strat),
                value=value,
                row_value=value,
                col_value=-value,
            )

        col_probs = result.x[:-1]
        col_probs = np.maximum(col_probs, 0)
        col_probs = (
            col_probs / col_probs.sum()
            if col_probs.sum() > 0
            else np.ones(n_col) / n_col
        )

        value = result.fun

        row_probs = np.zeros(n_row)
        row_probs = np.ones(n_row) / n_row

        return ZeroSumSolution(
            row_strategy=MixedStrategy(0, row_probs),
            col_strategy=MixedStrategy(1, col_probs),
            value=-value,
            row_value=-value,
            col_value=value,
        )


class FictitiousPlay(ZeroSumSolver):
    """Fictitious play for zero-sum games.

    Players best respond to opponent's historical average strategy.
    """

    def __init__(
        self,
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        learning_rate: float = 0.01,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate

    def solve(self, game: ZeroSumGame) -> ZeroSumSolution:
        """Solve using fictitious play."""
        payoff = game.get_row_player_payoffs()
        n_row, n_col = payoff.shape

        row_strat = np.ones(n_row) / n_row
        col_strat = np.ones(n_col) / n_col

        row_history = np.zeros(n_row)
        col_history = np.zeros(n_col)

        values_history = []

        for iteration in range(self.max_iterations):
            row_payoffs = payoff @ col_strat
            col_payoffs = payoff.T @ row_strat

            row_best = np.argmax(row_payoffs)
            col_best = np.argmax(col_payoffs)

            row_history[row_best] += 1
            col_history[col_best] += 1

            row_strat = (
                1 - self.learning_rate
            ) * row_strat + self.learning_rate * row_history / row_history.sum()
            col_strat = (
                1 - self.learning_rate
            ) * col_strat + self.learning_rate * col_history / col_history.sum()

            value = row_strat @ payoff @ col_strat
            values_history.append(value)

            if len(values_history) > 100:
                recent = values_history[-100:]
                if np.std(recent) < self.tolerance:
                    break

        value = float(row_strat @ payoff @ col_strat)

        return ZeroSumSolution(
            row_strategy=MixedStrategy(0, row_strat),
            col_strategy=MixedStrategy(1, col_strat),
            value=value,
            row_value=value,
            col_value=-value,
        )


class GradientDescentSolver(ZeroSumSolver):
    """Gradient descent for finding mixed strategy equilibrium.

    Uses gradient-based optimization to find equilibrium strategies.
    """

    def __init__(
        self,
        max_iterations: int = 5000,
        learning_rate: float = 0.01,
        tolerance: float = 1e-6,
    ):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def solve(self, game: ZeroSumGame) -> ZeroSumSolution:
        """Solve using gradient descent."""
        payoff = game.get_row_player_payoffs()
        n_row, n_col = payoff.shape

        row_logit = np.random.randn(n_row) * 0.1
        col_logit = np.random.randn(n_col) * 0.1

        for iteration in range(self.max_iterations):
            row_probs = self._softmax(row_logit)
            col_probs = self._softmax(col_logit)

            row_grad = payoff @ col_probs - np.dot(row_probs, payoff @ col_probs)
            col_grad = -(payoff.T @ row_probs - np.dot(col_grad, payoff.T @ row_probs))

            row_logit += self.learning_rate * row_grad
            col_logit += self.learning_rate * col_grad

            row_logit = row_logit - row_logit.max()
            col_logit = col_logit - col_logit.max()

        row_probs = self._softmax(row_logit)
        col_probs = self._softmax(col_logit)

        value = float(row_probs @ payoff @ col_probs)

        return ZeroSumSolution(
            row_strategy=MixedStrategy(0, row_probs),
            col_strategy=MixedStrategy(1, col_probs),
            value=value,
            row_value=value,
            col_value=-value,
        )

    def _softmax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute softmax with numerical stability."""
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()


class MultiplicativeWeights(ZeroSumSolver):
    """Multiplicative weights algorithm for zero-sum games.

    A no-regret learning algorithm that can find approximate
    equilibria in zero-sum games.
    """

    def __init__(
        self,
        max_iterations: int = 10000,
        learning_rate: float = 0.1,
    ):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def solve(self, game: ZeroSumGame) -> ZeroSumSolution:
        """Solve using multiplicative weights."""
        payoff = game.get_row_player_payoffs()
        n_row, n_col = payoff.shape

        row_weights = np.ones(n_row)
        col_weights = np.ones(n_col)

        row_history = []
        col_history = []

        for _ in range(self.max_iterations):
            row_probs = row_weights / row_weights.sum()
            col_probs = col_weights / col_weights.sum()

            row_payoffs = payoff @ col_probs
            col_payoffs = payoff.T @ row_probs

            row_loss = 1 - row_payoffs / row_payoffs.max()
            col_loss = 1 - col_payoffs / col_payoffs.max()

            row_weights *= np.exp(-self.learning_rate * row_loss)
            col_weights *= np.exp(-self.learning_rate * col_loss)

            row_history.append(row_probs.copy())
            col_history.append(col_probs.copy())

        row_avg = np.mean(row_history[1000:], axis=0)
        col_avg = np.mean(col_history[1000:], axis=0)

        row_avg = row_avg / row_avg.sum()
        col_avg = col_avg / col_avg.sum()

        value = float(row_avg @ payoff @ col_avg)

        return ZeroSumSolution(
            row_strategy=MixedStrategy(0, row_avg),
            col_strategy=MixedStrategy(1, col_avg),
            value=value,
            row_value=value,
            col_value=-value,
        )


def solve_zero_sum_game(
    game: ZeroSumGame,
    method: str = "linear_programming",
) -> ZeroSumSolution:
    """Convenience function to solve a zero-sum game.

    Args:
        game: The zero-sum game to solve
        method: One of 'linear_programming', 'fictitious_play',
                'gradient_descent', 'multiplicative_weights'

    Returns:
        ZeroSumSolution with equilibrium strategies
    """
    if method == "linear_programming":
        solver = LinearProgrammingSolver()
    elif method == "fictitious_play":
        solver = FictitiousPlay()
    elif method == "gradient_descent":
        solver = GradientDescentSolver()
    elif method == "multiplicative_weights":
        solver = MultiplicativeWeights()
    else:
        raise ValueError(f"Unknown method: {method}")

    return solver.solve(game)
