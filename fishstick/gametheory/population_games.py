"""
Population games for evolutionary game theory.

Implements population-level games and analysis of evolutionary
stability in large populations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, linear_sum_assignment

from fishstick.gametheory.core_types import MixedStrategy
from fishstick.gametheory.normal_form_game import TwoPlayerGame


@dataclass
class PopulationGame:
    """A population game where strategies interact.

    Attributes:
        num_strategies: Number of available strategies
        payoff_matrix: Payoff matrix for the game
    """

    num_strategies: int
    payoff_matrix: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.payoff_matrix.shape[0] != self.payoff_matrix.shape[1]:
            if len(self.payoff_matrix.shape) == 2:
                self.num_strategies = self.payoff_matrix.shape[0]

    def get_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute fitness of each strategy given population composition."""
        return population @ self.payoff_matrix

    def get_average_fitness(self, population: NDArray[np.float64]) -> float:
        """Compute average population fitness."""
        fitness = self.get_fitness(population)
        return float(population @ fitness)

    def get_best_response(self, population: NDArray[np.float64]) -> int:
        """Get the best response to the population."""
        fitness = self.get_fitness(population)
        return int(np.argmax(fitness))

    def get_best_responses(self, population: NDArray[np.float64]) -> List[int]:
        """Get all best responses (may be multiple)."""
        fitness = self.get_fitness(population)
        max_fitness = fitness.max()

        return [i for i, f in enumerate(fitness) if abs(f - max_fitness) < 1e-9]

    def is_potential_game(self) -> bool:
        """Check if the game is a potential game."""
        n = self.num_strategies

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        if i == j or k == l:
                            continue

                        lhs = self.payoff_matrix[i, j] - self.payoff_matrix[j, j]
                        rhs = self.payoff_matrix[i, l] - self.payoff_matrix[l, l]

                        if abs(lhs - rhs) > 1e-6:
                            return False

        return True

    def find_potential_function(
        self, initial_guess: Optional[NDArray[np.float64]] = None
    ) -> Optional[NDArray[np.float64]]:
        """Find potential function if it exists."""
        if not self.is_potential_game():
            return None

        n = self.num_strategies
        potential = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == 0 and j == 0:
                    potential[i, j] = self.payoff_matrix[i, j]
                elif i == 0:
                    potential[i, j] = potential[i, j - 1] + (
                        self.payoff_matrix[i, j] - self.payoff_matrix[i, j - 1]
                    )
                elif j == 0:
                    potential[i, j] = potential[i - 1, j] + (
                        self.payoff_matrix[i, j] - self.payoff_matrix[i - 1, j]
                    )
                else:
                    potential[i, j] = (potential[i - 1, j] + potential[i, j - 1]) / 2

        return potential


@dataclass
class StablePopulation:
    """A population state with stability analysis."""

    population: NDArray[np.float64]
    is_nash: bool
    is_evolutionarily_stable: bool

    def __str__(self) -> str:
        return (
            f"StablePopulation(population={self.population}, "
            f"is_Nash={self.is_nash}, is_ES={self.is_evolutionarily_stable})"
        )


def find_nash_equilibrium_population(
    game: PopulationGame,
    method: str = "brute_force",
    grid_size: int = 10,
) -> Optional[StablePopulation]:
    """Find Nash equilibrium in population game.

    Args:
        game: The population game
        method: Method to use ('brute_force', 'optimization')
        grid_size: Size of grid for discretization

    Returns:
        StablePopulation if found
    """
    if method == "brute_force":
        return _find_nash_brute_force(game, grid_size)
    elif method == "optimization":
        return _find_nash_optimization(game)
    else:
        raise ValueError(f"Unknown method: {method}")


def _find_nash_brute_force(
    game: PopulationGame,
    grid_size: int,
) -> Optional[StablePopulation]:
    """Find Nash equilibrium using brute force search."""
    n = game.num_strategies

    points = np.linspace(0, 1, grid_size)

    best_responses = set()

    for _ in range(n - 1):
        fractions = []
        remaining = 1.0

        for _ in range(n - 2):
            f = np.random.random() * remaining
            fractions.append(f)
            remaining -= f

        fractions.append(remaining)

        population = np.array(fractions + [1.0 - sum(fractions)])
        population = np.maximum(population, 0)
        population = population / population.sum()

        br = game.get_best_response(population)
        best_responses.add(br)

    for br in best_responses:
        population = np.zeros(n)
        population[br] = 1.0

        return StablePopulation(
            population=population,
            is_nash=True,
            is_evolutionarily_stable=True,
        )

    return None


def _find_nash_optimization(
    game: PopulationGame,
) -> Optional[StablePopulation]:
    """Find Nash equilibrium using optimization."""
    n = game.num_strategies

    def objective(pop):
        fitness = game.get_fitness(pop)
        return -np.min(fitness)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]

    bounds = [(0, 1) for _ in range(n)]

    best_result = None
    best_min_fitness = -float("inf")

    for _ in range(100):
        x0 = np.random.dirichlet(np.ones(n))

        result = minimize(
            objective, x0, method="SLSQP", constraints=constraints, bounds=bounds
        )

        if result.success:
            min_fitness = game.get_fitness(result.x).min()

            if min_fitness > best_min_fitness:
                best_min_fitness = min_fitness
                best_result = result.x

    if best_result is not None:
        return StablePopulation(
            population=best_result,
            is_nash=True,
            is_evolutionarily_stable=False,
        )

    return None


def compute_invasion_fitness(
    resident: NDArray[np.float64],
    mutant: NDArray[np.float64],
    game: PopulationGame,
    epsilon: float = 0.01,
) -> float:
    """Compute fitness of a mutant strategy in a resident population.

    Args:
        resident: Resident population composition
        mutant: Mutant strategy (unit vector)
        game: The population game
        epsilon: Mutation frequency

    Returns:
        Invasion fitness of the mutant
    """
    mixed_population = (1 - epsilon) * resident + epsilon * mutant

    fitness_mutant = mutant @ game.payoff_matrix @ mixed_population
    fitness_resident = resident @ game.payoff_matrix @ mixed_population

    return fitness_mutant - fitness_resident


def check_evolutionary_stability(
    population: NDArray[np.float64],
    game: PopulationGame,
) -> bool:
    """Check if a population is evolutionarily stable.

    Args:
        population: Population composition
        game: The population game

    Returns:
        True if the population is evolutionarily stable
    """
    n = len(population)

    for mutant_idx in range(n):
        if population[mutant_idx] > 0:
            continue

        mutant = np.zeros(n)
        mutant[mutant_idx] = 1.0

        invasion_fitness = compute_invasion_fitness(
            population, mutant, game, epsilon=0.01
        )

        if invasion_fitness > 0:
            return False

    return True


def find_limit_cycles(
    game: PopulationGame,
    initial_populations: List[NDArray[np.float64]],
    num_steps: int = 1000,
) -> List[List[NDArray[np.float64]]]:
    """Find limit cycles in population dynamics.

    Args:
        game: The population game
        initial_populations: List of initial populations to simulate
        num_steps: Number of simulation steps

    Returns:
        List of population trajectories that may contain limit cycles
    """
    from fishstick.gametheory.evolutionary_dynamics import ReplicatorDynamics

    dynamics = ReplicatorDynamics(
        TwoPlayerGame(
            game.payoff_matrix,
            game.payoff_matrix,
        )
    )

    cycles = []

    for init_pop in initial_populations:
        pop = init_pop.copy()

        trajectory = [pop.copy()]

        for _ in range(num_steps):
            pop = dynamics.step(pop)
            trajectory.append(pop.copy())

        if _detect_cycle(trajectory):
            cycles.append(trajectory)

    return cycles


def _detect_cycle(trajectory: List[NDArray[np.float64]]) -> bool:
    """Detect if trajectory contains a cycle."""
    if len(trajectory) < 10:
        return False

    recent = trajectory[-100:]

    if len(recent) < 10:
        return False

    std = np.std(recent, axis=0)

    return std.max() > 0.01


def compute_stationary_distribution(
    game: PopulationGame,
    num_iterations: int = 10000,
    sample_interval: int = 100,
) -> Optional[NDArray[np.float64]]:
    """Compute stationary distribution using long-run simulation.

    Args:
        game: The population game
        num_iterations: Total simulation steps
        sample_interval: Interval for sampling

    Returns:
        Stationary distribution if found
    """
    from fishstick.gametheory.evolutionary_dynamics import ReplicatorDynamics

    dynamics = ReplicatorDynamics(
        TwoPlayerGame(
            game.payoff_matrix,
            game.payoff_matrix,
        )
    )

    population = np.ones(game.num_strategies) / game.num_strategies

    samples = []

    for step in range(num_iterations):
        population = dynamics.step(population)

        if step % sample_interval == 0:
            samples.append(population.copy())

    if samples:
        return np.mean(samples, axis=0)

    return None
