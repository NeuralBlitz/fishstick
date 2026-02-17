"""
Evolutionary game theory dynamics.

Implements replicator dynamics, best response dynamics,
logit dynamics, and the Moran process for evolutionary game theory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from fishstick.gametheory.core_types import MixedStrategy
from fishstick.gametheory.normal_form_game import (
    NormalFormGame,
    TwoPlayerGame,
    ZeroSumGame,
)


@dataclass
class EvolutionaryDynamics(ABC):
    """Abstract base class for evolutionary dynamics."""

    game: TwoPlayerGame

    @abstractmethod
    def step(
        self, population: NDArray[np.float64], dt: float = 0.1
    ) -> NDArray[np.float64]:
        """Take one step of the dynamics."""
        pass

    @abstractmethod
    def compute_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute fitness of each strategy."""
        pass


@dataclass
class ReplicatorDynamics(EvolutionaryDynamics):
    """Replicator dynamics for evolutionary games.

    The growth rate of a strategy is proportional to its
    current fitness relative to average fitness.
    """

    def __init__(self, game: TwoPlayerGame):
        self.game = game

    def compute_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute expected fitness of each strategy."""
        payoff_matrix = self.game.get_row_player_payoffs()

        fitness = population @ payoff_matrix

        return fitness

    def step(
        self, population: NDArray[np.float64], dt: float = 0.1
    ) -> NDArray[np.float64]:
        """Take one replicator dynamics step."""
        fitness = self.compute_fitness(population)

        avg_fitness = population @ fitness

        growth_rates = fitness - avg_fitness

        new_population = population + dt * population * growth_rates

        new_population = np.maximum(new_population, 0)

        if new_population.sum() > 0:
            new_population = new_population / new_population.sum()
        else:
            new_population = np.ones(len(population)) / len(population)

        return new_population

    def simulate(
        self,
        initial_population: NDArray[np.float64],
        num_steps: int = 1000,
        dt: float = 0.1,
        convergence_threshold: float = 1e-6,
    ) -> Tuple[List[NDArray[np.float64]], List[float]]:
        """Simulate replicator dynamics from initial population.

        Returns:
            Tuple of (population history, fitness history)
        """
        population = initial_population.copy()

        history = [population.copy()]
        fitness_history = [float(population @ self.compute_fitness(population))]

        for _ in range(num_steps):
            population = self.step(population, dt)

            history.append(population.copy())
            fitness_history.append(float(population @ self.compute_fitness(population)))

            if len(history) > 2:
                diff = np.abs(history[-1] - history[-2]).max()
                if diff < convergence_threshold:
                    break

        return history, fitness_history


@dataclass
class BestResponseDynamics(EvolutionaryDynamics):
    """Best response dynamics.

    At each step, a fraction of the population switches to
    a best response against the current population.
    """

    def __init__(
        self,
        game: TwoPlayerGame,
        mutation_rate: float = 0.01,
    ):
        super().__init__(game)
        self.mutation_rate = mutation_rate

    def compute_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fitness is max over best responses."""
        payoff_matrix = self.game.get_row_player_payoffs()
        return population @ payoff_matrix

    def step(
        self, population: NDArray[np.float64], dt: float = 0.1
    ) -> NDArray[np.float64]:
        """Take one step of best response dynamics."""
        fitness = self.compute_fitness(population)

        best_responses = np.zeros_like(population)
        best_response_value = fitness.max()

        for i, f in enumerate(fitness):
            if abs(f - best_response_value) < 1e-9:
                best_responses[i] = 1.0

        if best_responses.sum() > 0:
            best_responses = best_responses / best_responses.sum()

        new_population = (1 - dt) * population + dt * best_responses

        new_population = (
            1 - self.mutation_rate
        ) * new_population + self.mutation_rate * (1.0 / len(population))

        new_population = np.maximum(new_population, 0)
        new_population = new_population / new_population.sum()

        return new_population

    def find_equilibrium(
        self,
        initial_population: NDArray[np.float64],
        num_steps: int = 1000,
    ) -> Optional[NDArray[np.float64]]:
        """Find equilibrium using best response dynamics."""
        population = initial_population.copy()

        for _ in range(num_steps):
            new_population = self.step(population)

            if np.allclose(population, new_population, atol=1e-6):
                return new_population

            population = new_population

        return None


@dataclass
class LogitDynamics(EvolutionaryDynamics):
    """Logit dynamics (softmax replicator).

    Strategies are chosen according to a softmax function
    based on their fitness, with a temperature parameter.
    """

    def __init__(
        self,
        game: TwoPlayerGame,
        temperature: float = 1.0,
    ):
        super().__init__(game)
        self.temperature = temperature

    def compute_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute fitness with temperature-scaled noise."""
        payoff_matrix = self.game.get_row_player_payoffs()

        raw_fitness = population @ payoff_matrix

        exp_fitness = np.exp(raw_fitness / self.temperature)
        expected_fitness = exp_fitness / exp_fitness.sum()

        return expected_fitness

    def step(
        self, population: NDArray[np.float64], dt: float = 0.1
    ) -> NDArray[np.float64]:
        """Take one logit dynamics step."""
        fitness = self.compute_fitness(population)

        new_population = population + dt * (fitness - population)

        new_population = np.maximum(new_population, 0)
        new_population = new_population / new_population.sum()

        return new_population


@dataclass
class MoranProcess(EvolutionaryDynamics):
    """Moran process for evolutionary game theory.

    A population process where individuals reproduce in proportion
    to fitness, and one individual is replaced each generation.
    """

    def __init__(
        self,
        game: TwoPlayerGame,
        population_size: int = 100,
        mutation_rate: float = 0.0,
    ):
        super().__init__(game)
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def compute_fitness(self, population: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute fitness of each strategy type."""
        payoff_matrix = self.game.get_row_player_payoffs()

        fitness = population * (payoff_matrix @ population)

        return fitness

    def step(
        self, population: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        """Take one Moran process step."""
        fitness = self.compute_fitness(population)

        fitness = np.maximum(fitness, 0)

        total_fitness = fitness.sum()
        if total_fitness > 0:
            reproduction_probs = fitness / total_fitness
        else:
            reproduction_probs = np.ones(len(population)) / len(population)

        new_population = population.copy()

        reproducing_idx = np.random.choice(len(population), p=reproduction_probs)

        if self.mutation_rate > 0 and np.random.random() < self.mutation_rate:
            strategy_to_add = np.random.randint(len(population))
        else:
            strategy_to_add = reproducing_idx

        dying_idx = np.random.randint(len(population))

        new_population[dying_idx] = 0
        new_population[strategy_to_add] += 1

        new_population = new_population / new_population.sum()

        return new_population

    def simulate(
        self,
        initial_population: NDArray[np.float64],
        num_steps: int = 1000,
    ) -> Tuple[List[NDArray[np.float64]], List[float]]:
        """Simulate Moran process.

        Returns:
            Tuple of (population history, average fitness history)
        """
        population = initial_population.copy()

        history = [population.copy()]
        fitness_history = [float(population @ self.compute_fitness(population))]

        for _ in range(num_steps):
            population = self.step(population)

            history.append(population.copy())
            fitness_history.append(float(population @ self.compute_fitness(population)))

        return history, fitness_history


@dataclass
class StableState:
    """An evolutionarily stable state."""

    population: NDArray[np.float64]
    is_stable: bool

    def __str__(self) -> str:
        return f"StableState(population={self.population}, stable={self.is_stable})"


@dataclass
class EvolutionarilyStableStrategy:
    """Evolutionarily Stable Strategy (ESS).

    A strategy is evolutionarily stable if, when adopted by
    a population, cannot be invaded by a rare mutant strategy.
    """

    game: TwoPlayerGame

    def find_ess(self) -> Optional[NDArray[np.float64]]:
        """Find evolutionarily stable strategies."""
        payoff_matrix = self.game.get_row_player_payoffs()
        n = payoff_matrix.shape[0]

        possible_ess = []

        for i in range(n):
            strategy = np.zeros(n)
            strategy[i] = 1.0

            if self._is_ess(strategy, payoff_matrix):
                possible_ess.append(strategy)

        mixed_ess = self._find_mixed_ess(payoff_matrix)

        if mixed_ess is not None:
            possible_ess.append(mixed_ess)

        if possible_ess:
            return possible_ess[0]
        return None

    def _is_ess(
        self, strategy: NDArray[np.float64], payoff_matrix: NDArray[np.float64]
    ) -> bool:
        """Check if strategy is evolutionarily stable."""
        mutant = np.ones(len(strategy)) / len(strategy)

        payoff_strategy = strategy @ payoff_matrix @ strategy
        payoff_mutant = mutant @ payoff_matrix @ strategy
        payoff_mutant_against_mutant = mutant @ payoff_matrix @ mutant

        if payoff_strategy > payoff_mutant:
            return True

        if payoff_strategy == payoff_mutant:
            return payoff_strategy > payoff_mutant_against_mutant

        return False

    def _find_mixed_ess(
        self, payoff_matrix: NDArray[np.float64]
    ) -> Optional[NDArray[np.float64]]:
        """Find mixed ESS using linear programming approach."""
        n = payoff_matrix.shape[0]

        from scipy.optimize import linprog

        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])

        bounds = [(0, 1) for _ in range(n)]

        c = np.zeros(n)

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        if result.success:
            strategy = result.x

            for i in range(n):
                for j in range(n):
                    if i != j:
                        mutant = np.zeros(n)
                        mutant[j] = 1.0

                        payoff_s = strategy @ payoff_matrix @ strategy
                        payoff_m = mutant @ payoff_matrix @ strategy
                        payoff_mm = mutant @ payoff_matrix @ mutant

                        if payoff_s < payoff_m:
                            return None

                        if payoff_s == payoff_m and payoff_s <= payoff_mm:
                            return None

            return strategy

        return None


def compute_equilibrium_stability(
    population: NDArray[np.float64],
    dynamics: EvolutionaryDynamics,
) -> StableState:
    """Compute stability of an equilibrium population.

    Args:
        population: The population state to check
        dynamics: The evolutionary dynamics

    Returns:
        StableState with stability analysis
    """
    fitness = dynamics.compute_fitness(population)

    avg_fitness = population @ fitness

    growth_rates = fitness - avg_fitness

    is_stable = np.allclose(growth_rates, 0, atol=1e-6) or (growth_rates <= 0).all()

    return StableState(
        population=population,
        is_stable=bool(is_stable),
    )


def find_lyapunov_stability(
    population: NDArray[np.float64],
    dynamics: EvolutionaryDynamics,
    num_steps: int = 100,
    perturbation: float = 1e-4,
) -> bool:
    """Check Lyapunov stability of a population state.

    A state is Lyapunov stable if small perturbations do not
    grow significantly.
    """
    base_fitness = dynamics.compute_fitness(population)

    for strategy_idx in range(len(population)):
        perturbed = population.copy()
        perturbed[strategy_idx] += perturbation
        perturbed = perturbed / perturbed.sum()

        perturbed_fitness = dynamics.compute_fitness(perturbed)

        fitness_diff = np.abs(perturbed_fitness - base_fitness).max()

        if fitness_diff > perturbation * 10:
            return False

    return True
