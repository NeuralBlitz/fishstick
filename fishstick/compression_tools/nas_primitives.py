"""
Neural Architecture Search Primitives

Search space definitions, supernet implementations, architecture sampling,
performance estimation, and evolutionary search primitives.

References:
- https://arxiv.org/abs/1806.09055 (AutoML)
- https://arxiv.org/abs/1909.10836 (Once for All)
- https://arxiv.org/abs/1810.04014 (Progressive NAS)
- https://arxiv.org.org/abs/1802.03268 (ENAS)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Callable, Tuple, Union, Any, Literal
from enum import Enum
import random
import copy
import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer


class SearchSpaceType(Enum):
    """Types of search spaces."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


class LayerChoice(Enum):
    """Layer types for architecture search."""

    IDENTITY = "identity"
    DILATED_CONV_3X3 = "dconv3"
    DILATED_CONV_5X5 = "dconv5"
    SEP_CONV_3X3 = "sep3"
    SEP_CONV_5X5 = "sep5"
    POOL_3X3 = "pool3"
    POOL_5X5 = "pool5"
    ATTENTION = "attention"


class SearchSpace:
    """Neural architecture search space definition.

    Defines the search space for NAS including layer types,
    connections, and hyperparameters.

    Args:
        num_layers: Number of layers in the search space
        layer_choices: Available layer types
        max_channels: Maximum number of channels
        depth_range: Range of depths to search

    Example:
        >>> space = SearchSpace(num_layers=5, layer_choices=['sep3', 'sep5', 'pool3'])
        >>> config = space.sample_random()
    """

    def __init__(
        self,
        num_layers: int = 5,
        layer_choices: Optional[List[str]] = None,
        max_channels: int = 256,
        depth_range: Tuple[int, int] = (1, 4),
    ):
        self.num_layers = num_layers
        self.layer_choices = layer_choices or [
            "identity",
            "dconv3",
            "dconv5",
            "sep3",
            "sep5",
            "pool3",
        ]
        self.max_channels = max_channels
        self.depth_range = depth_range

        self._validate_space()

    def _validate_space(self):
        """Validate search space configuration."""
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not self.layer_choices:
            raise ValueError("At least one layer choice required")

    def sample_random(self) -> Dict[str, Any]:
        """Sample random architecture configuration.

        Returns:
            Dict with architecture configuration
        """
        config = {
            "layers": [],
            "channels": [],
            "skip_connections": [],
        }

        for i in range(self.num_layers):
            config["layers"].append(random.choice(self.layer_choices))
            config["channels"].append(
                random.choice(
                    [self.max_channels // 4, self.max_channels // 2, self.max_channels]
                )
            )
            config["skip_connections"].append(
                random.choice([True, False]) if i > 0 else False
            )

        return config

    def sample_uniform(self) -> Dict[str, Any]:
        """Sample architecture with uniform distribution."""
        config = {
            "layers": [],
            "channels": [],
            "skip_connections": [],
        }

        num_choices = len(self.layer_choices)
        for i in range(self.num_layers):
            layer_idx = (i * len(self.layer_choices) // self.num_layers) % num_choices
            config["layers"].append(self.layer_choices[layer_idx])

            channel_idx = i * 3 // self.num_layers
            channels = [
                self.max_channels // 4,
                self.max_channels // 2,
                self.max_channels,
            ]
            config["channels"].append(channels[min(channel_idx, 2)])

            config["skip_connections"].append(False)

        return config

    def mutate(
        self,
        config: Dict[str, Any],
        mutation_rate: float = 0.1,
    ) -> Dict[str, Any]:
        """Mutate an architecture configuration.

        Args:
            config: Original configuration
            mutation_rate: Probability of mutation per element

        Returns:
            Mutated configuration
        """
        new_config = copy.deepcopy(config)

        for i in range(len(new_config["layers"])):
            if random.random() < mutation_rate:
                new_config["layers"][i] = random.choice(self.layer_choices)

            if random.random() < mutation_rate:
                channels = [
                    self.max_channels // 4,
                    self.max_channels // 2,
                    self.max_channels,
                ]
                new_config["channels"][i] = random.choice(channels)

            if i > 0 and random.random() < mutation_rate:
                new_config["skip_connections"][i] = not new_config["skip_connections"][
                    i
                ]

        return new_config

    def crossover(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform crossover between two configurations.

        Args:
            config1: First parent configuration
            config2: Second parent configuration

        Returns:
            Child configuration
        """
        child = {"layers": [], "channels": [], "skip_connections": []}

        for i in range(self.num_layers):
            if random.random() < 0.5:
                child["layers"].append(config1["layers"][i])
                child["channels"].append(config1["channels"][i])
                child["skip_connections"].append(config1["skip_connections"][i])
            else:
                child["layers"].append(config2["layers"][i])
                child["channels"].append(config2["channels"][i])
                child["skip_connections"].append(config2["skip_connections"][i])

        return child


class SuperNet(nn.Module):
    """One-shot supernet for efficient architecture search.

    A supernet that contains all possible architectures as subgraphs,
    enabling efficient weight sharing during search.

    Args:
        search_space: Search space definition
        in_channels: Input channels
        num_classes: Number of output classes

    Example:
        >>> space = SearchSpace(num_layers=5)
        >>> supernet = SuperNet(space, in_channels=3, num_classes=10)
        >>> config = space.sample_random()
        >>> output = supernet(input, config)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.search_space = search_space
        self.in_channels = in_channels

        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        self.layer_ops = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(search_space.num_layers):
            ops = nn.ModuleDict()
            for choice in search_space.layer_choices:
                ops[choice] = self._create_layer(
                    choice,
                    64 if i == 0 else search_space.max_channels // 4,
                    search_space.max_channels // 4,
                )
            self.layer_ops.append(ops)
            self.layer_norms.append(nn.BatchNorm2d(search_space.max_channels // 4))

        self.classifier = nn.Linear(search_space.max_channels // 4, num_classes)

    def _create_layer(
        self,
        layer_type: str,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """Create a layer based on type."""
        if layer_type == "identity":
            return nn.Identity()
        elif layer_type == "dconv3":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif layer_type == "dconv5":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, padding=4, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif layer_type == "sep3":
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif layer_type == "sep5":
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif layer_type == "pool3":
            return nn.Sequential(
                nn.AvgPool2d(3, padding=1),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        elif layer_type == "pool5":
            return nn.Sequential(
                nn.AvgPool2d(5, padding=2),
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            return nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        x: Tensor,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """Forward pass with optional architecture config.

        Args:
            x: Input tensor
            config: Architecture configuration (if None, uses random)

        Returns:
            Output logits
        """
        if config is None:
            config = self.search_space.sample_random()

        x = self.stem(x)

        for i, ops in enumerate(self.layer_ops):
            layer_choice = config["layers"][i]

            if layer_choice in ops:
                x = ops[layer_choice](x)
            else:
                x = ops["identity"](x)

            x = self.layer_norms[i](x)

            if i > 0 and config["skip_connections"][i]:
                if x.shape == x.shape:
                    pass

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        x = self.classifier(x)

        return x

    def get_weighted_forward(
        self,
        x: Tensor,
        weights: Dict[str, Tensor],
    ) -> Tensor:
        """Forward pass with architectural weights.

        Args:
            x: Input tensor
            weights: Architecture weights for each layer

        Returns:
            Output logits
        """
        x = self.stem(x)

        for i, ops in enumerate(self.layer_ops):
            layer_weights = weights.get(f"layer_{i}", None)

            if layer_weights is None:
                x = ops["identity"](x)
            else:
                outputs = []
                for choice in self.search_space.layer_choices:
                    if choice in ops:
                        outputs.append(ops[choice](x) * layer_weights[i])

                x = sum(outputs) if outputs else x

            x = self.layer_norms[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        x = self.classifier(x)

        return x


class ArchitectureSampler:
    """Sampler for architecture configurations.

    Provides various sampling strategies for architecture search.

    Args:
        search_space: Search space definition
        sampling_strategy: Sampling strategy to use

    Example:
        >>> sampler = ArchitectureSampler(space, strategy='random')
        >>> config = sampler.sample()
    """

    def __init__(
        self,
        search_space: SearchSpace,
        sampling_strategy: str = "random",
    ):
        self.search_space = search_space
        self.sampling_strategy = sampling_strategy
        self.sample_history: List[Dict[str, Any]] = []

    def sample(self) -> Dict[str, Any]:
        """Sample an architecture configuration."""
        if self.sampling_strategy == "random":
            config = self.search_space.sample_random()
        elif self.sampling_strategy == "uniform":
            config = self.search_space.sample_uniform()
        elif self.sampling_strategy == "gradient":
            config = self.search_space.sample_random()
        else:
            config = self.search_space.sample_random()

        self.sample_history.append(config)
        return config

    def sample_diverse(
        self,
        num_samples: int = 10,
    ) -> List[Dict[str, Any]]:
        """Sample diverse architectures.

        Args:
            num_samples: Number of samples

        Returns:
            List of diverse configurations
        """
        samples = []

        for _ in range(num_samples * 3):
            if len(samples) >= num_samples:
                break

            config = self.sample()

            is_diverse = True
            for existing in samples:
                if self._config_similarity(config, existing) > 0.7:
                    is_diverse = False
                    break

            if is_diverse:
                samples.append(config)

        return samples

    def _config_similarity(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two configs."""
        layer_sim = sum(
            a == b for a, b in zip(config1["layers"], config2["layers"])
        ) / len(config1["layers"])

        channel_sim = sum(
            a == b for a, b in zip(config1["channels"], config2["channels"])
        ) / len(config1["channels"])

        return (layer_sim + channel_sim) / 2


class PerformanceEstimator:
    """Performance estimator for architecture configurations.

    Estimates latency, accuracy, and resource usage without
    full training/inference.

    Args:
        search_space: Search space definition
        device: Device for profiling

    Example:
        >>> estimator = PerformanceEstimator(space)
        >>> metrics = estimator.estimate(config, input_shape)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        device: str = "cpu",
    ):
        self.search_space = search_space
        self.device = device
        self.layer_latencies: Dict[str, float] = self._calibrate_latencies()
        self.param_counts: Dict[str, int] = self._count_parameters()

    def _calibrate_latencies(self) -> Dict[str, float]:
        """Calibrate layer latencies through profiling."""
        latencies = {}

        for layer_type in self.search_space.layer_choices:
            dummy_input = torch.randn(1, 64, 32, 32).to(self.device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            if torch.cuda.is_available():
                start.record()
                _ = torch.nn.functional.conv2d(
                    dummy_input,
                    torch.randn(64, 64, 3, 3).to(self.device),
                    padding=1,
                )
                end.record()
                torch.cuda.synchronize()
            else:
                latencies[layer_type] = 0.001

        return latencies

    def _count_parameters(self) -> Dict[str, int]:
        """Count parameters for each layer type."""
        counts = {}

        for layer_type in self.search_space.layer_choices:
            layer = nn.Conv2d(64, 64, 3, padding=1)
            counts[layer_type] = sum(p.numel() for p in layer.parameters())

        return counts

    def estimate_latency(
        self,
        config: Dict[str, Any],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> float:
        """Estimate inference latency.

        Args:
            config: Architecture configuration
            input_shape: Input tensor shape

        Returns:
            Estimated latency in milliseconds
        """
        total_latency = 0.0

        h, w = input_shape[2], input_shape[3]

        for i, layer_type in enumerate(config["layers"]):
            channels = config["channels"][i]
            scale_factor = channels / 64.0

            base_latency = self.layer_latencies.get(layer_type, 0.001)
            total_latency += base_latency * scale_factor

            h = max(1, h // 2)
            w = max(1, w // 2)

        return total_latency * 1000

    def estimate_parameters(
        self,
        config: Dict[str, Any],
    ) -> int:
        """Estimate total parameter count.

        Args:
            config: Architecture configuration

        Returns:
            Estimated parameter count
        """
        total_params = 0

        total_params += 3 * 3 * 3 * 64

        for i, layer_type in enumerate(config["layers"]):
            channels = config["channels"][i]
            total_params += self.param_counts.get(layer_type, 1000) * (channels // 64)

        total_params += config["channels"][-1] * 10

        return total_params

    def estimate_accuracy(
        self,
        config: Dict[str, Any],
        proxy_accuracy: float = 0.7,
    ) -> float:
        """Estimate expected accuracy based on complexity.

        Args:
            config: Architecture configuration
            proxy_accuracy: Known accuracy from proxy model

        Returns:
            Estimated accuracy
        """
        num_layers = len(config["layers"])
        avg_channels = sum(config["channels"]) / len(config["channels"])
        complexity = num_layers * avg_channels / self.search_space.max_channels

        complexity_penalty = math.log(complexity + 1) * 0.1

        estimated = proxy_accuracy - complexity_penalty

        return max(0.1, min(1.0, estimated))

    def estimate(
        self,
        config: Dict[str, Any],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, float]:
        """Get all performance estimates.

        Args:
            config: Architecture configuration
            input_shape: Input shape

        Returns:
            Dict of performance metrics
        """
        return {
            "latency_ms": self.estimate_latency(config, input_shape),
            "parameters": self.estimate_parameters(config),
            "accuracy": self.estimate_accuracy(config),
            "gflops": self.estimate_latency(config, input_shape) * 1e-3,
        }


class EvolutionSearch:
    """Evolutionary architecture search.

    Performs neural architecture search using evolutionary algorithms.

    Args:
        search_space: Search space definition
        population_size: Number of architectures in population
        generations: Number of generations
        tournament_size: Tournament size for selection
        mutation_rate: Probability of mutation

    Example:
        >>> searcher = EvolutionSearch(space, population_size=20, generations=50)
        >>> best_config = searcher.search(evaluate_fn)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 20,
        generations: int = 50,
        tournament_size: int = 3,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.3,
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population: List[Dict[str, Any]] = []
        self.fitness_history: List[float] = []

    def _initialize_population(self):
        """Initialize random population."""
        self.population = [
            self.search_space.sample_random() for _ in range(self.population_size)
        ]

    def _evaluate_population(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
    ):
        """Evaluate fitness for all individuals."""
        for config in self.population:
            if "fitness" not in config:
                config["fitness"] = evaluate_fn(config)

    def _select_parent(self) -> Dict[str, Any]:
        """Tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.get("fitness", 0.0))

    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Crossover two parents."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)

        return self.search_space.crossover(parent1, parent2)

    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate configuration."""
        return self.search_space.mutate(config, self.mutation_rate)

    def _evolve_population(self) -> List[Dict[str, Any]]:
        """Create next generation."""
        new_population = []

        sorted_pop = sorted(
            self.population,
            key=lambda x: x.get("fitness", 0.0),
            reverse=True,
        )

        new_population.extend(sorted_pop[:2])

        while len(new_population) < self.population_size:
            parent1 = self._select_parent()
            parent2 = self._select_parent()

            child = self._crossover(parent1, parent2)
            child = self._mutate(child)

            new_population.append(child)

        return new_population

    def search(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run evolutionary search.

        Args:
            evaluate_fn: Function to evaluate architecture fitness
            verbose: Print progress

        Returns:
            Best architecture configuration
        """
        self._initialize_population()

        best_config = None
        best_fitness = float("-inf")

        for gen in range(self.generations):
            self._evaluate_population(evaluate_fn)

            gen_best = max(self.population, key=lambda x: x.get("fitness", 0.0))
            if gen_best.get("fitness", 0) > best_fitness:
                best_fitness = gen_best.get("fitness", 0)
                best_config = copy.deepcopy(gen_best)

            self.fitness_history.append(best_fitness)

            if verbose:
                print(
                    f"Generation {gen + 1}/{self.generations}: Best fitness = {best_fitness:.4f}"
                )

            self.population = self._evolve_population()

        return best_config

    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            "generations": self.generations,
            "population_size": self.population_size,
            "best_fitness": max(self.fitness_history) if self.fitness_history else 0.0,
            "final_fitness": self.fitness_history[-1] if self.fitness_history else 0.0,
            "fitness_history": self.fitness_history,
        }


class LatencyAwareSearch:
    """Latency-aware architecture search.

    Optimizes for both accuracy and latency constraints.

    Args:
        search_space: Search space definition
        latency_constraint: Maximum latency in ms
        weight_accuracy: Weight for accuracy in objective
    """

    def __init__(
        self,
        search_space: SearchSpace,
        latency_constraint: float = 10.0,
        weight_accuracy: float = 0.5,
    ):
        self.search_space = search_space
        self.latency_constraint = latency_constraint
        self.weight_accuracy = weight_accuracy
        self.weight_latency = 1.0 - weight_accuracy

    def objective(
        self,
        config: Dict[str, Any],
        estimator: PerformanceEstimator,
    ) -> float:
        """Compute optimization objective.

        Args:
            config: Architecture configuration
            estimator: Performance estimator

        Returns:
            Objective score (higher is better)
        """
        metrics = estimator.estimate(config)

        latency = metrics["latency_ms"]
        accuracy = metrics["accuracy"]

        if latency > self.latency_constraint:
            latency_penalty = (latency - self.latency_constraint) * 10
            return accuracy - latency_penalty

        acc_score = self.weight_accuracy * accuracy
        lat_score = self.weight_latency * (1 - latency / self.latency_constraint)

        return acc_score + lat_score

    def search(
        self,
        estimator: PerformanceEstimator,
        num_samples: int = 100,
    ) -> Dict[str, Any]:
        """Search under latency constraint.

        Args:
            estimator: Performance estimator
            num_samples: Number of samples to try

        Returns:
            Best configuration within constraint
        """
        best_config = None
        best_score = float("-inf")

        for _ in range(num_samples):
            config = self.search_space.sample_random()

            score = self.objective(config, estimator)

            if score > best_score:
                best_score = score
                best_config = config

        return best_config
