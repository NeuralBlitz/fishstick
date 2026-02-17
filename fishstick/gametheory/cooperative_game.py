"""
Cooperative game theory (TU games).

Implements transferable utility games, including characteristic
functions, player sets, and coalition values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable, FrozenSet
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from itertools import combinations

from fishstick.gametheory.core_types import Player


@dataclass
class Coalition:
    """A coalition of players in a cooperative game.

    Attributes:
        members: Set of player IDs in the coalition
    """

    members: FrozenSet[int]

    def __hash__(self) -> int:
        return hash(self.members)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coalition):
            return False
        return self.members == other.members

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def __str__(self) -> str:
        return f"{{{', '.join(str(m) for m in sorted(self.members))}}}"

    @property
    def is_empty(self) -> bool:
        return len(self.members) == 0

    @property
    def is_singleton(self) -> bool:
        return len(self.members) == 1

    @property
    def is_grand(self) -> bool:
        return False


class TUGame(ABC):
    """Transferable Utility (TU) cooperative game.

    A cooperative game where players can transfer utility between
    each other to achieve better outcomes.

    Attributes:
        players: Set of players in the game
        characteristic_function: Maps coalitions to their value
    """

    def __init__(
        self,
        players: Set[int],
        characteristic_function: Optional[Callable[[Coalition], float]] = None,
    ):
        self.players = players
        self.num_players = len(players)

        if characteristic_function is not None:
            self.characteristic_function = characteristic_function
        else:
            self.characteristic_function = self._default_characteristic

    @abstractmethod
    def _default_characteristic(self, coalition: Coalition) -> float:
        """Default characteristic function."""
        pass

    def value(self, coalition: Coalition) -> float:
        """Get the value of a coalition."""
        return self.characteristic_function(coalition)

    def __call__(self, coalition: Coalition) -> float:
        """Syntactic sugar for getting coalition value."""
        return self.value(coalition)

    def get_all_coalitions(self) -> List[Coalition]:
        """Get all possible coalitions."""
        coalitions = []
        for size in range(self.num_players + 1):
            for combo in combinations(sorted(self.players), size):
                coalitions.append(Coalition(frozenset(combo)))
        return coalitions

    def get_coalitions_of_size(self, size: int) -> List[Coalition]:
        """Get all coalitions of a specific size."""
        return [
            Coalition(frozenset(combo))
            for combo in combinations(sorted(self.players), size)
        ]

    def is_superadditive(self) -> bool:
        """Check if the game is superadditive.

        A game is superadditive if the value of the union of two
        disjoint coalitions is at least the sum of their values.
        """
        coalitions = self.get_all_coalitions()

        for c1 in coalitions:
            for c2 in coalitions:
                if c1.is_empty or c2.is_empty:
                    continue
                if c1.members.isdisjoint(c2.members):
                    union = Coalition(c1.members | c2.members)
                    if self.value(union) < self.value(c1) + self.value(c2):
                        return False
        return True

    def is_convex(self) -> bool:
        """Check if the game is convex.

        A game is convex if adding a player to a coalition never
        decreases the marginal contribution.
        """
        coalitions = self.get_all_coalitions()

        for c1 in coalitions:
            for c2 in coalitions:
                if c1.members.issubset(c2.members):
                    diff = c2.members - c1.members
                    for player in diff:
                        c1_plus = Coalition(c1.members | {player})
                        c2_plus = Coalition(c2.members | {player})

                        if (self.value(c2_plus) - self.value(c2)) < (
                            self.value(c1_plus) - self.value(c1)
                        ):
                            return False
        return True


class SimpleGame(TUGame):
    """A simple game where coalitions are either winning or losing.

    Attributes:
        players: Set of players
        winning_coalitions: Set of winning coalitions
    """

    def __init__(
        self,
        players: Set[int],
        winning_coalitions: Optional[Set[FrozenSet[int]]] = None,
    ):
        self.winning_coalitions = winning_coalitions or set()

        def char_function(coalition: Coalition) -> float:
            return 1.0 if coalition.members in self.winning_coalitions else 0.0

        super().__init__(players, char_function)

    def is_winning(self, coalition: Coalition) -> bool:
        """Check if a coalition is winning."""
        return coalition.members in self.winning_coalitions

    def is_losing(self, coalition: Coalition) -> bool:
        """Check if a coalition is losing."""
        return not self.is_winning(coalition)

    def _default_characteristic(self, coalition: Coalition) -> float:
        return 1.0 if self.is_winning(coalition) else 0.0

    def get_minimal_winning(self) -> List[Coalition]:
        """Get all minimal winning coalitions."""
        minimal = []

        for wc in self.winning_coalitions:
            is_minimal = True
            for other in self.winning_coalitions:
                if other != wc and other.issubset(wc):
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(Coalition(wc))

        return minimal


class MajorityGame(SimpleGame):
    """Simple majority voting game.

    A coalition is winning if it has more than half the players.
    """

    def __init__(self, num_players: int):
        players = set(range(num_players))

        winning = set()
        majority = (num_players + 1) // 2

        for size in range(majority, num_players + 1):
            for combo in combinations(players, size):
                winning.add(frozenset(combo))

        super().__init__(players, winning)


class WeightedVotingGame(TUGame):
    """Weighted voting game.

    Players have weights and a quota. A coalition is winning if
    its total weight exceeds the quota.

    Attributes:
        weights: Dictionary mapping player IDs to their weights
        quota: Quota that must be met to win
    """

    def __init__(self, weights: Dict[int, float], quota: Optional[float] = None):
        self.weights = weights
        self.quota = quota if quota is not None else sum(weights.values()) / 2

        players = set(weights.keys())

        def char_function(coalition: Coalition) -> float:
            total_weight = sum(self.weights.get(p, 0) for p in coalition.members)
            return 1.0 if total_weight >= self.quota else 0.0

        super().__init__(players, char_function)

    def _default_characteristic(self, coalition: Coalition) -> float:
        total_weight = sum(self.weights.get(p, 0) for p in coalition.members)
        return 1.0 if total_weight >= self.quota else 0.0

    def get_critical_players(self) -> Set[int]:
        """Find players whose removal would cause the grand coalition to lose."""
        critical = set()
        grand = frozenset(self.players)

        for player in self.players:
            without = grand - {player}
            without_weight = sum(self.weights.get(p, 0) for p in without)

            if without_weight < self.quota:
                critical.add(player)

        return critical

    def get_shapley_value(self) -> Dict[int, float]:
        """Compute Shapley value for weighted voting game."""
        from fishstick.gametheory.solution_concepts import ShapleyValue

        solver = ShapleyValue(self)
        return solver.compute()


class CostGame(TUGame):
    """A cost game where the characteristic function represents costs.

    Lower values are better for the coalition.
    """

    def __init__(
        self,
        players: Set[int],
        cost_function: Callable[[Coalition], float],
    ):
        self.cost_function = cost_function

        def char_function(coalition: Coalition) -> float:
            return -self.cost_function(coalition)

        super().__init__(players, char_function)

    def _default_characteristic(self, coalition: Coalition) -> float:
        return -self.cost_function(coalition)


class MarketGame(TUGame):
    """A market game derived from an exchange economy.

    Attributes:
        endowments: Initial endowments for each player
        utilities: Utility functions for each player
    """

    def __init__(
        self,
        endowments: Dict[int, NDArray[np.float64]],
        utilities: Dict[int, Callable[[NDArray[np.float64]], float]],
    ):
        self.endowments = endowments
        self.utilities = utilities
        players = set(endowments.keys())

        def char_function(coalition: Coalition) -> float:
            if coalition.is_empty:
                return 0.0

            prices = self._compute_equilibrium_prices(coalition)
            if prices is None:
                return 0.0

            total_value = sum(
                sum(p * e for p, e in zip(prices, self.endowments[p]))
                for p in coalition.members
            )
            return total_value

        super().__init__(players, char_function)

    def _default_characteristic(self, coalition: Coalition) -> float:
        return self.characteristic_function(coalition)

    def _compute_equilibrium_prices(
        self, coalition: Coalition
    ) -> Optional[NDArray[np.float64]]:
        """Compute competitive equilibrium prices."""
        if coalition.is_empty:
            return None

        num_goods = len(next(iter(self.endowments.values())))
        return np.ones(num_goods)


class NetworkGame(TUGame):
    """A game formed from a network structure.

    The value of a coalition depends on the network connections
    between its members.

    Attributes:
        network: Adjacency matrix or edge list
        threshold: Minimum connectivity threshold
    """

    def __init__(
        self,
        players: Set[int],
        edges: List[Tuple[int, int]],
        value_per_edge: float = 1.0,
    ):
        self.edges = edges
        self.value_per_edge = value_per_edge
        self.num_players = len(players)

        def char_function(coalition: Coalition) -> float:
            if len(coalition.members) < 2:
                return 0.0

            member_list = sorted(coalition.members)
            member_set = set(member_list)

            count = 0
            for i, j in edges:
                if i in member_set and j in member_set:
                    count += 1

            return count * value_per_edge

        super().__init__(players, char_function)

    def _default_characteristic(self, coalition: Coalition) -> float:
        return self.characteristic_function(coalition)


def create_minimum_spanning_tree_game(
    num_players: int,
    edge_costs: Dict[Tuple[int, int], float],
) -> CostGame:
    """Create a game where coalition value is the cost savings
    from building a minimum spanning tree.
    """
    players = set(range(num_players))

    def cost_function(coalition: Coalition) -> float:
        if len(coalition.members) < 2:
            return 0.0

        member_list = sorted(coalition.members)
        subgraph_edges = [
            (i, j, c)
            for (i, j), c in edge_costs.items()
            if i in coalition.members and j in coalition.members
        ]

        if not subgraph_edges:
            return 0.0

        edges_sorted = sorted(subgraph_edges, key=lambda x: x[2])

        parent = {p: p for p in coalition.members}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        mst_cost = 0
        edges_used = 0

        for i, j, cost in edges_sorted:
            if union(i, j):
                mst_cost += cost
                edges_used += 1
                if edges_used == len(coalition.members) - 1:
                    break

        return mst_cost

    return CostGame(players, cost_function)
