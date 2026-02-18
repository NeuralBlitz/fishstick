"""
Genetic Algorithm Feature Selector for fishstick

Evolutionary wrapper method for feature selection.
"""

from typing import Optional, Union, Callable, List, Tuple
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from . import SupervisedSelector, SelectionResult


class GeneticAlgorithmSelector(SupervisedSelector):
    """
    Genetic Algorithm feature selector.

    Uses evolutionary algorithms to find optimal feature subsets.
    Evaluates subsets using a fitness function (typically CV score).

    Args:
        estimator: Base estimator for fitness evaluation
        n_features_to_select: Number of features to select
        population_size: Size of population (default 20)
        n_generations: Number of generations (default 30)
        crossover_rate: Probability of crossover (default 0.8)
        mutation_rate: Probability of mutation (default 0.1)
        tournament_size: Tournament size for selection (default 3)
        cv: Cross-validation folds
        scoring: Scoring metric
        random_state: Random seed
        n_jobs: Parallel jobs

    Example:
        >>> selector = GeneticAlgorithmSelector(
        ...     estimator=RandomForestClassifier(),
        ...     n_features_to_select=10,
        ...     n_generations=20
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        n_features_to_select: Optional[Union[int, float]] = None,
        population_size: int = 20,
        n_generations: int = 30,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        cv: int = 5,
        scoring: str = "accuracy",
        random_state: Optional[int] = 42,
        n_jobs: int = 1,
    ):
        super().__init__(n_features_to_select=n_features_to_select, cv=cv)
        self.estimator = estimator
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_fitness_history_: List[float] = []

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
    ) -> "GeneticAlgorithmSelector":
        """
        Run genetic algorithm for feature selection.

        Args:
            X: Input features (n_samples, n_features)
            y: Target labels

        Returns:
            self
        """
        X_np, is_torch = self._to_numpy(X)
        y_np, _ = self._to_numpy(y) if not isinstance(y, np.ndarray) else (y, False)

        self.n_features_in_ = X_np.shape[1]
        n_target = self._parse_n_features(self.n_features_in_)

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(n_estimators=50, random_state=42)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.best_fitness_history_ = []

        population = self._initialize_population(n_target)

        best_individual = None
        best_fitness = -np.inf

        for gen in range(self.n_generations):
            fitness_scores = self._evaluate_population(population, X_np, y_np)

            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()

            self.best_fitness_history_.append(best_fitness)

            parents = self._select_parents(population, fitness_scores)

            offspring = self._crossover(parents)

            offspring = self._mutate(offspring)

            population = self._replace_population(population, offspring, fitness_scores)

        self.selected_features_ = np.where(best_individual)[0]
        self.scores_ = np.ones(self.n_features_in_) * best_fitness
        self.best_individual_ = best_individual

        return self

    def _initialize_population(self, n_target_features: int) -> np.ndarray:
        """Initialize random population."""
        n_features = self.n_features_in_
        population = np.zeros((self.population_size, n_features), dtype=int)

        for i in range(self.population_size):
            indices = np.random.choice(n_features, n_target_features, replace=False)
            population[i, indices] = 1

        return population

    def _evaluate_population(
        self, population: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Evaluate fitness for each individual."""
        fitness = np.zeros(len(population))

        for i, individual in enumerate(population):
            selected = np.where(individual)[0]

            if len(selected) == 0:
                fitness[i] = -np.inf
                continue

            try:
                scores = cross_val_score(
                    self.estimator,
                    X[:, selected],
                    y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                )
                fitness[i] = scores.mean()
            except Exception:
                fitness[i] = -np.inf

        return fitness

    def _select_parents(
        self, population: np.ndarray, fitness: np.ndarray
    ) -> List[np.ndarray]:
        """Tournament selection."""
        parents = []

        for _ in range(self.population_size):
            tournament_idx = np.random.choice(
                len(population), self.tournament_size, replace=False
            )
            tournament_fitness = fitness[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())

        return parents

    def _crossover(self, parents: List[np.ndarray]) -> List[np.ndarray]:
        """Single-point crossover."""
        offspring = []

        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

            if np.random.random() < self.crossover_rate:
                point = np.random.randint(1, len(parent1))
                child1 = np.concatenate([parent1[:point], parent2[point:]])
                child2 = np.concatenate([parent2[:point], parent1[point:]])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            offspring.extend([child1, child2])

        return offspring[: self.population_size]

    def _mutate(self, offspring: List[np.ndarray]) -> List[np.ndarray]:
        """Bit flip mutation."""
        for individual in offspring:
            for i in range(len(individual)):
                if np.random.random() < self.mutation_rate:
                    individual[i] = 1 - individual[i]

        return offspring

    def _replace_population(
        self,
        population: np.ndarray,
        offspring: List[np.ndarray],
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Replace population with offspring, keeping elites."""
        new_population = np.zeros(
            (self.population_size, self.n_features_in_), dtype=int
        )

        n_elites = max(1, self.population_size // 5)
        elite_indices = np.argsort(fitness)[-n_elites:]

        for i, idx in enumerate(elite_indices):
            new_population[i] = population[idx]

        for i in range(n_elites, self.population_size):
            if i - n_elites < len(offspring):
                new_population[i] = offspring[i - n_elites]

        return new_population


def genetic_algorithm_selector(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    estimator: Optional[BaseEstimator] = None,
    n_features: Optional[int] = None,
) -> SelectionResult:
    """
    Functional interface for genetic algorithm feature selection.

    Args:
        X: Input features
        y: Target labels
        estimator: Base estimator
        n_features: Number of features to select

    Returns:
        SelectionResult
    """
    selector = GeneticAlgorithmSelector(
        estimator=estimator,
        n_features_to_select=n_features,
    )
    selector.fit(X, y)

    return SelectionResult(
        selected_features=selector.selected_features_,
        scores=selector.scores_,
        n_features=X.shape[1],
        method="genetic_algorithm",
        metadata={"fitness_history": selector.best_fitness_history_},
    )
