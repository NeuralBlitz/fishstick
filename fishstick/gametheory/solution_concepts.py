"""
Solution concepts for cooperative game theory.

Implements Shapley value, Nucleolus, Core, Kernel, and other
solution concepts for TU games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from itertools import permutations
from scipy.optimize import linprog, minimize

from fishstick.gametheory.cooperative_game import (
    TUGame,
    Coalition,
    SimpleGame,
    WeightedVotingGame,
)
from fishstick.gametheory.core_types import Player


@dataclass
class SolutionConcept:
    """Base class for solution concepts."""

    game: TUGame

    @abstractmethod
    def compute(self) -> Dict[int, float]:
        """Compute the solution."""
        pass


@dataclass
class ShapleyValue(SolutionConcept):
    """Shapley value: fair distribution of coalitional gains.

    The Shapley value distributes the value of the grand coalition
    based on each player's average marginal contribution.
    """

    num_samples: Optional[int] = None
    random_seed: Optional[int] = None

    def compute(self) -> Dict[int, float]:
        """Compute Shapley value using exact computation or sampling."""
        if self.num_samples is not None:
            return self._compute_sampling()
        return self._compute_exact()

    def _compute_exact(self) -> Dict[int, float]:
        """Exact computation using all permutations."""
        n = self.game.num_players
        players_list = sorted(self.game.players)

        shapley = {p: 0.0 for p in players_list}

        grand = Coalition(frozenset(players_list))
        grand_value = self.game.value(grand)

        for perm in permutations(players_list):
            prev_coalition = Coalition(frozenset())
            prev_value = self.game.value(prev_coalition)

            for i, player in enumerate(perm):
                coalition = Coalition(frozenset(perm[:i]))
                marginal = self.game.value(coalition) - prev_value
                shapley[player] += marginal

                prev_coalition = coalition
                prev_value = self.game.value(coalition)

        for player in players_list:
            shapley[player] /= np.math.factorial(n)

        return shapley

    def _compute_sampling(self) -> Dict[int, float]:
        """Monte Carlo approximation of Shapley value."""
        np.random.seed(self.random_seed)

        n = self.game.num_players
        players_list = sorted(self.game.players)

        shapley = {p: 0.0 for p in players_list}

        for _ in range(self.num_samples):
            perm = np.random.permutation(players_list)

            prev_value = self.game.value(Coalition(frozenset()))

            for i in range(n):
                coalition = Coalition(frozenset(perm[:i]))
                marginal = self.game.value(coalition) - prev_value
                shapley[perm[i]] += marginal

                prev_value = self.game.value(coalition)

        for player in players_list:
            shapley[player] /= self.num_samples

        return shapley


@dataclass
class Core(TUGame):
    """The Core: set of imputations that are stable against deviation.

    An imputation is in the Core if no coalition can deviate and
    get a better outcome for all its members.
    """

    def __init__(self, game: TUGame):
        self.game = game
        self._imputations: Optional[List[Dict[int, float]]] = None

    def compute(self) -> List[Dict[int, float]]:
        """Find all imputations in the Core."""
        if self._imputations is not None:
            return self._imputations

        self._imputations = []

        grand = Coalition(frozenset(self.game.players))
        grand_value = self.game.value(grand)

        n = self.game.num_players

        A_ub = []
        b_ub = []

        coalitions = self.game.get_all_coalitions()

        for coalition in coalitions:
            if coalition.is_empty or coalition.is_grand:
                continue

            row = np.zeros(n)
            for i, player in enumerate(sorted(self.game.players)):
                if player in coalition.members:
                    row[i] = 1

            A_ub.append(row)
            b_ub.append(self.game.value(coalition))

        A_eq = np.ones((1, n))
        b_eq = np.array([grand_value])

        bounds = [(0, None) for _ in range(n)]

        result = linprog(
            np.zeros(n),
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
        )

        if result.success:
            imputation = {
                p: result.x[i] for i, p in enumerate(sorted(self.game.players))
            }
            self._imputations.append(imputation)

        return self._imputations

    def is_non_empty(self) -> bool:
        """Check if the Core is non-empty."""
        return len(self.compute()) > 0


@dataclass
class Nucleolus(SolutionConcept):
    """Nucleolus: the "most stable" imputation.

    The nucleolus minimizes the maximum excess (dissatisfaction)
    of any coalition.
    """

    tolerance: float = 1e-6

    def compute(self) -> Dict[int, float]:
        """Compute the nucleolus."""
        grand = Coalition(frozenset(self.game.players))
        grand_value = self.game.value(grand)

        n = self.game.num_players
        players_list = sorted(self.game.players)

        def excess(imputation: NDArray[np.float64], coalition: Coalition) -> float:
            value = self.game.value(coalition)
            sum_imputation = sum(
                imputation[i]
                for i, p in enumerate(players_list)
                if p in coalition.members
            )
            return sum_imputation - value

        def find_lex_min(violations: List[Tuple[float, int]]) -> NDArray[np.float64]:
            """Find lexicographically minimal excess vector."""

            def objective(x):
                sorted_excess = sorted(
                    [
                        (excess(x, c), c)
                        for c in self.game.get_all_coalitions()
                        if not c.is_empty and not c.is_grand
                    ],
                    key=lambda t: t[0],
                    reverse=True,
                )
                return sum(e * (i + 1) for i, (e, _) in enumerate(sorted_excess))

            x0 = np.ones(n) * grand_value / n

            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - grand_value}]

            bounds = [(0, None) for _ in range(n)]

            result = minimize(
                objective, x0, method="SLSQP", constraints=constraints, bounds=bounds
            )

            return result.x if result.success else x0

        imputation = find_lex_min([])

        return {p: imputation[i] for i, p in enumerate(players_list)}

    def compute_excesses(self, imputation: Dict[int, float]) -> Dict[Coalition, float]:
        """Compute excess for all coalitions."""
        arr = np.array([imputation[p] for p in sorted(self.game.players)])

        return {
            coalition: excess(arr, coalition)
            for coalition in self.game.get_all_coalitions()
            if not coalition.is_empty and not coalition.is_grand
        }


@dataclass
class Kernel(TUGame):
    """The Kernel: set of imputations where no player is essential.

    A player is inessential if they can be removed without reducing
    the coalition's value.
    """

    def compute(self) -> List[Dict[int, float]]:
        """Find all imputations in the Kernel."""
        grand = Coalition(frozenset(self.game.players))
        grand_value = self.game.value(grand)

        n = self.game.num_players
        players_list = sorted(self.game.players)

        imputations = []

        def get_surplus(imputation: Dict[int, float], coalition: Coalition) -> float:
            coalition_value = self.game.value(coalition)
            members_sum = sum(imputation[p] for p in coalition.members)
            return max(0, members_sum - coalition_value)

        def can_remove(imputation: Dict[int, float], i: int, j: int) -> bool:
            i_coalition = Coalition(frozenset(self.game.players - {i, j}))
            j_coalition = Coalition(frozenset(self.game.players - {i, j}))

            surplus_i = get_surplus(imputation, i_coalition)
            surplus_j = get_surplus(imputation, j_coalition)

            return surplus_i <= surplus_j

        n_solutions = 100

        for _ in range(n_solutions):
            x0 = np.random.dirichlet(np.ones(n))
            x0 = x0 * grand_value

            def objective(x):
                return 0.0

            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - grand_value}]

            bounds = [(0, None) for _ in range(n)]

            result = minimize(
                objective, x0, method="SLSQP", constraints=constraints, bounds=bounds
            )

            if result.success:
                imputation = {p: result.x[i] for i, p in enumerate(players_list)}
                imputations.append(imputation)

        return imputations


@dataclass
class BanzhafIndex(SolutionConcept):
    """Banzhaf power index for simple games.

    Measures the number of coalitions where a player is critical.
    """

    def compute(self) -> Dict[int, float]:
        """Compute Banzhaf index."""
        if not isinstance(self.game, SimpleGame):
            raise ValueError("Banzhaf index requires a SimpleGame")

        simple_game = self.game
        players_list = sorted(self.game.players)

        banzhaf = {p: 0 for p in players_list}

        for coalition in self.game.get_all_coalitions():
            if coalition.is_empty:
                continue

            for player in coalition.members:
                without = coalition.members - {player}

                if simple_game.is_winning(Coalition(without)):
                    if not simple_game.is_winning(coalition):
                        banzhaf[player] += 1

        total = sum(banzhaf.values())
        if total > 0:
            return {p: v / total for p, v in banzhaf.items()}
        return banzhaf


@dataclass
class OwenValue(SolutionConcept):
    """Owen value for games with coalition structure.

    Extends Shapley value when players can form pre-commitments
    (coalitions) before the game.

    Attributes:
        coalition_structure: Partition of players into coalitions
    """

    coalition_structure: List[Set[int]]

    def compute(self) -> Dict[int, float]:
        """Compute Owen value."""
        grand = Coalition(frozenset(self.game.players))
        grand_value = self.game.value(grand)

        owen = {p: 0.0 for p in self.game.players}

        for perm in permutations(self.coalition_structure):
            prev_coalitions = set()
            prev_value = 0.0

            for coalition_set in perm:
                coalition = Coalition(frozenset(coalition_set))

                for player in coalition_set:
                    marginal = self.game.value(coalition) - prev_value
                    owen[player] += marginal / np.math.factorial(
                        len(self.coalition_structure)
                    )

                prev_value = self.game.value(coalition)
                prev_coalitions = coalition_set

            rest = self.game.players - set().union(*perm[:-1]) if perm else set()
            if rest:
                rest_coalition = Coalition(frozenset(rest))
                marginal = self.game.value(rest_coalition) - prev_value
                for player in rest:
                    owen[player] += marginal / np.math.factorial(
                        len(self.coalition_structure)
                    )

        return owen


def compute_solutions(
    game: TUGame, methods: List[str] = ["shapley", "core", "nucleolus"]
) -> Dict[str, Dict[int, float]]:
    """Convenience function to compute multiple solution concepts.

    Args:
        game: The TU game
        methods: List of methods to compute

    Returns:
        Dictionary mapping method names to their solutions
    """
    results = {}

    if "shapley" in methods:
        results["shapley"] = ShapleyValue(game).compute()

    if "core" in methods:
        core = Core(game)
        if core.is_non_empty():
            results["core"] = core.compute()[0]

    if "nucleolus" in methods:
        results["nucleolus"] = Nucleolus(game).compute()

    return results
