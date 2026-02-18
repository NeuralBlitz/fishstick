"""
Comprehensive Motion Planning Module for Fishstick Robotics

This module provides implementations of various motion planning algorithms including:
- Sampling-based planners (RRT, RRT*, RRT-Connect, PRM, EST)
- Search-based planners (A*, Dijkstra, D*, ARA*, Theta*)
- Optimization-based planners (CHOMP, TrajOpt, STOMP, ITOMP)
- Potential field methods (APF, Harmonic, Navigation Functions)
- Constraint handling and path smoothing utilities

Author: Fishstick Robotics Team
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Dict, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import heapq
import random
import math
import numpy as np
from collections import defaultdict
import copy
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.spatial import KDTree, distance
from scipy.optimize import minimize, differential_evolution
import warnings

# ============================================================================
# Configuration Space and Basic Types
# ============================================================================


@dataclass
class Configuration:
    """Represents a configuration in n-dimensional space."""

    values: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values)

    def __hash__(self):
        return hash(tuple(self.values.round(6)))

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return False
        return np.allclose(self.values, other.values, atol=1e-6)

    def distance_to(self, other: "Configuration") -> float:
        """Euclidean distance to another configuration."""
        return np.linalg.norm(self.values - other.values)

    def copy(self) -> "Configuration":
        return Configuration(self.values.copy())

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass
class Path:
    """Represents a path as a sequence of configurations."""

    configurations: List[Configuration] = field(default_factory=list)
    cost: float = float("inf")

    def __len__(self) -> int:
        return len(self.configurations)

    def __getitem__(self, idx: int) -> Configuration:
        return self.configurations[idx]

    def length(self) -> float:
        """Calculate the total path length."""
        if len(self.configurations) < 2:
            return 0.0
        total = 0.0
        for i in range(len(self.configurations) - 1):
            total += self.configurations[i].distance_to(self.configurations[i + 1])
        return total

    def smooth(self) -> "Path":
        """Apply simple smoothing to the path."""
        if len(self.configurations) < 3:
            return self

        smoothed = [self.configurations[0]]
        i = 1
        while i < len(self.configurations) - 1:
            # Check if we can skip this point
            if np.random.random() > 0.3:
                smoothed.append(self.configurations[i])
            i += 1
        smoothed.append(self.configurations[-1])
        return Path(smoothed, self.cost)


@dataclass
class Trajectory:
    """Represents a trajectory with time parameterization."""

    path: Path
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None

    def __post_init__(self):
        if len(self.timestamps) == 0 and len(self.path) > 0:
            self.timestamps = np.linspace(0, 1, len(self.path))

    def get_state_at_time(
        self, t: float
    ) -> Tuple[Configuration, np.ndarray, np.ndarray]:
        """Get position, velocity, acceleration at time t."""
        idx = np.searchsorted(self.timestamps, t)
        if idx >= len(self.path):
            idx = len(self.path) - 1
        pos = self.path[idx]
        vel = (
            self.velocities[idx]
            if self.velocities is not None
            else np.zeros(pos.dimension)
        )
        acc = (
            self.accelerations[idx]
            if self.accelerations is not None
            else np.zeros(pos.dimension)
        )
        return pos, vel, acc


class ConfigurationSpace:
    """Represents the configuration space for motion planning."""

    def __init__(
        self,
        dim: int,
        bounds: List[Tuple[float, float]],
        collision_checker: Optional["CollisionChecker"] = None,
    ):
        self.dim = dim
        self.bounds = bounds
        self.collision_checker = collision_checker

        if len(bounds) != dim:
            raise ValueError(f"Expected {dim} bounds, got {len(bounds)}")

    def sample_random(self) -> Configuration:
        """Sample a random configuration within bounds."""
        values = np.array([random.uniform(low, high) for low, high in self.bounds])
        return Configuration(values)

    def sample_random_near(self, center: Configuration, radius: float) -> Configuration:
        """Sample a random configuration near the center."""
        direction = np.random.randn(self.dim)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        distance = random.uniform(0, radius)
        values = center.values + direction * distance
        values = np.clip(
            values, [b[0] for b in self.bounds], [b[1] for b in self.bounds]
        )
        return Configuration(values)

    def is_valid(self, config: Configuration) -> bool:
        """Check if configuration is valid (within bounds and collision-free)."""
        # Check bounds
        for i, (low, high) in enumerate(self.bounds):
            if not (low <= config.values[i] <= high):
                return False

        # Check collision
        if self.collision_checker is not None:
            return not self.collision_checker.check_collision(config)

        return True

    def interpolate(
        self, start: Configuration, goal: Configuration, alpha: float
    ) -> Configuration:
        """Linear interpolation between configurations."""
        values = start.values + alpha * (goal.values - start.values)
        return Configuration(values)

    def local_planner(
        self, start: Configuration, goal: Configuration, step_size: float = 0.1
    ) -> List[Configuration]:
        """Generate a local path from start to goal with collision checking."""
        distance = start.distance_to(goal)
        if distance < step_size:
            return [start, goal]

        num_steps = int(np.ceil(distance / step_size))
        path = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            config = self.interpolate(start, goal, alpha)
            if self.is_valid(config):
                path.append(config)
            else:
                return []  # Collision detected
        return path


# ============================================================================
# Collision Checking and Constraints
# ============================================================================


class CollisionChecker(ABC):
    """Abstract base class for collision checking."""

    @abstractmethod
    def check_collision(self, config: Configuration) -> bool:
        """Check if configuration is in collision. Returns True if in collision."""
        pass

    @abstractmethod
    def check_edge_collision(
        self, config1: Configuration, config2: Configuration, num_checks: int = 10
    ) -> bool:
        """Check if edge between two configurations is collision-free."""
        pass


class SphereCollisionChecker(CollisionChecker):
    """Simple sphere-based collision checker for testing."""

    def __init__(
        self, obstacles: List[Tuple[np.ndarray, float]], robot_radius: float = 0.1
    ):
        self.obstacles = obstacles  # List of (center, radius)
        self.robot_radius = robot_radius

    def check_collision(self, config: Configuration) -> bool:
        pos = config.values[:2] if len(config.values) >= 2 else config.values
        for center, radius in self.obstacles:
            if len(center) >= 2:
                dist = np.linalg.norm(pos - center[:2])
                if dist < (radius + self.robot_radius):
                    return True
        return False

    def check_edge_collision(
        self, config1: Configuration, config2: Configuration, num_checks: int = 10
    ) -> bool:
        for i in range(num_checks + 1):
            alpha = i / num_checks
            config = Configuration(
                config1.values + alpha * (config2.values - config1.values)
            )
            if self.check_collision(config):
                return True
        return False


class Constraint(ABC):
    """Abstract constraint class."""

    @abstractmethod
    def is_satisfied(self, config: Configuration) -> bool:
        pass

    @abstractmethod
    def project(self, config: Configuration) -> Configuration:
        """Project configuration onto constraint manifold."""
        pass


class KinematicConstraints(Constraint):
    """Joint limit constraints."""

    def __init__(self, lower_limits: np.ndarray, upper_limits: np.ndarray):
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

    def is_satisfied(self, config: Configuration) -> bool:
        values = config.values
        return np.all(values >= self.lower_limits) and np.all(
            values <= self.upper_limits
        )

    def project(self, config: Configuration) -> Configuration:
        values = np.clip(config.values, self.lower_limits, self.upper_limits)
        return Configuration(values)


class DynamicConstraints(Constraint):
    """Velocity and acceleration constraints."""

    def __init__(self, max_velocities: np.ndarray, max_accelerations: np.ndarray):
        self.max_velocities = max_velocities
        self.max_accelerations = max_accelerations

    def is_satisfied(self, config: Configuration) -> bool:
        # For a single config, always satisfied
        return True

    def project(self, config: Configuration) -> Configuration:
        return config.copy()

    def check_trajectory(self, trajectory: Trajectory) -> bool:
        """Check if trajectory satisfies dynamic constraints."""
        if trajectory.velocities is not None:
            for vel in trajectory.velocities:
                if np.any(np.abs(vel) > self.max_velocities):
                    return False

        if trajectory.accelerations is not None:
            for acc in trajectory.accelerations:
                if np.any(np.abs(acc) > self.max_accelerations):
                    return False

        return True


class ConstraintSatisfaction:
    """Manager for multiple constraints."""

    def __init__(self):
        self.constraints: List[Constraint] = []

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)

    def check_all(self, config: Configuration) -> bool:
        return all(c.is_satisfied(config) for c in self.constraints)

    def project_all(self, config: Configuration, max_iter: int = 10) -> Configuration:
        """Project onto intersection of all constraints."""
        result = config.copy()
        for _ in range(max_iter):
            for constraint in self.constraints:
                result = constraint.project(result)
        return result


# ============================================================================
# Base Planner Interface
# ============================================================================


class MotionPlanner(ABC):
    """Abstract base class for motion planners."""

    def __init__(self, space: ConfigurationSpace):
        self.space = space
        self.stats = {"iterations": 0, "nodes_expanded": 0, "computation_time": 0.0}

    @abstractmethod
    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        """Plan a path from start to goal."""
        pass

    def reset_stats(self):
        self.stats = {"iterations": 0, "nodes_expanded": 0, "computation_time": 0.0}


# ============================================================================
# Sampling-Based Planners
# ============================================================================


class Node:
    """Node for tree/graph structures."""

    def __init__(self, config: Configuration, cost: float = 0.0):
        self.config = config
        self.cost = cost
        self.parent: Optional["Node"] = None
        self.children: List["Node"] = []

    def get_path(self) -> List[Configuration]:
        """Get path from root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current.config)
            current = current.parent
        return list(reversed(path))


class RRT(MotionPlanner):
    """Rapidly-exploring Random Tree planner."""

    def __init__(
        self,
        space: ConfigurationSpace,
        max_iter: int = 10000,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
    ):
        super().__init__(space)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.nodes: List[Node] = []

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        self.nodes = [Node(start)]

        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Sample random configuration
            if random.random() < self.goal_bias:
                random_config = goal
            else:
                random_config = self.space.sample_random()

            # Find nearest node
            nearest = self._nearest_node(random_config)

            # Steer towards random config
            new_config = self._steer(nearest.config, random_config)

            # Check if the new configuration is valid
            if self.space.is_valid(new_config):
                # Check edge collision
                local_path = self.space.local_planner(
                    nearest.config, new_config, self.step_size
                )
                if len(local_path) > 0:
                    new_node = Node(
                        new_config,
                        nearest.cost + nearest.config.distance_to(new_config),
                    )
                    new_node.parent = nearest
                    nearest.children.append(new_node)
                    self.nodes.append(new_node)

                    # Check if we can reach the goal
                    if new_config.distance_to(goal) < self.step_size:
                        if (
                            len(
                                self.space.local_planner(
                                    new_config, goal, self.step_size
                                )
                            )
                            > 0
                        ):
                            goal_node = Node(
                                goal, new_node.cost + new_config.distance_to(goal)
                            )
                            goal_node.parent = new_node
                            self.stats["computation_time"] = time.time() - start_time
                            return Path(goal_node.get_path(), goal_node.cost)

        self.stats["computation_time"] = time.time() - start_time
        return None

    def _nearest_node(self, config: Configuration) -> Node:
        """Find the nearest node to a configuration."""
        return min(self.nodes, key=lambda n: n.config.distance_to(config))

    def _steer(
        self, from_config: Configuration, to_config: Configuration
    ) -> Configuration:
        """Steer from one configuration towards another."""
        direction = to_config.values - from_config.values
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_config

        direction = direction / distance
        new_values = from_config.values + direction * self.step_size
        return Configuration(new_values)


class RRTStar(MotionPlanner):
    """RRT* - Optimal RRT with rewiring."""

    def __init__(
        self,
        space: ConfigurationSpace,
        max_iter: int = 10000,
        step_size: float = 0.1,
        goal_bias: float = 0.1,
        rewire_radius: Optional[float] = None,
    ):
        super().__init__(space)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
        self.nodes: List[Node] = []

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        self.nodes = [Node(start, 0.0)]
        best_path = None
        best_cost = float("inf")

        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Sample
            if random.random() < self.goal_bias:
                random_config = goal
            else:
                random_config = self.space.sample_random()

            # Find nearest
            nearest = self._nearest_node(random_config)
            new_config = self._steer(nearest.config, random_config)

            if self.space.is_valid(new_config):
                local_path = self.space.local_planner(
                    nearest.config, new_config, self.step_size
                )
                if len(local_path) > 0:
                    # Find neighbors within radius
                    neighbors = self._near_neighbors(new_config)

                    # Choose best parent
                    min_cost = nearest.cost + nearest.config.distance_to(new_config)
                    best_parent = nearest

                    for neighbor in neighbors:
                        cost = neighbor.cost + neighbor.config.distance_to(new_config)
                        if cost < min_cost:
                            local_path = self.space.local_planner(
                                neighbor.config, new_config, self.step_size
                            )
                            if len(local_path) > 0:
                                min_cost = cost
                                best_parent = neighbor

                    # Create new node
                    new_node = Node(new_config, min_cost)
                    new_node.parent = best_parent
                    best_parent.children.append(new_node)
                    self.nodes.append(new_node)

                    # Rewire
                    self._rewire(new_node, neighbors)

                    # Check goal
                    if new_config.distance_to(goal) < self.step_size:
                        local_path = self.space.local_planner(
                            new_config, goal, self.step_size
                        )
                        if len(local_path) > 0:
                            cost = new_node.cost + new_config.distance_to(goal)
                            if cost < best_cost:
                                best_cost = cost
                                goal_node = Node(goal, cost)
                                goal_node.parent = new_node
                                best_path = Path(goal_node.get_path(), cost)

        self.stats["computation_time"] = time.time() - start_time
        return best_path

    def _nearest_node(self, config: Configuration) -> Node:
        return min(self.nodes, key=lambda n: n.config.distance_to(config))

    def _near_neighbors(self, config: Configuration) -> List[Node]:
        if self.rewire_radius is None:
            # Adaptive radius based on number of nodes
            radius = min(
                5.0 * self.step_size,
                10.0
                * (
                    (math.log(len(self.nodes)) / len(self.nodes))
                    ** (1 / self.space.dim)
                ),
            )
        else:
            radius = self.rewire_radius

        return [n for n in self.nodes if n.config.distance_to(config) < radius]

    def _steer(
        self, from_config: Configuration, to_config: Configuration
    ) -> Configuration:
        direction = to_config.values - from_config.values
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_config

        direction = direction / distance
        new_values = from_config.values + direction * self.step_size
        return Configuration(new_values)

    def _rewire(self, new_node: Node, neighbors: List[Node]):
        """Rewire tree to maintain optimality."""
        for neighbor in neighbors:
            if neighbor == new_node.parent:
                continue

            new_cost = new_node.cost + new_node.config.distance_to(neighbor.config)
            if new_cost < neighbor.cost:
                local_path = self.space.local_planner(
                    new_node.config, neighbor.config, self.step_size
                )
                if len(local_path) > 0:
                    # Rewire
                    neighbor.parent.children.remove(neighbor)
                    neighbor.parent = new_node
                    new_node.children.append(neighbor)
                    neighbor.cost = new_cost
                    self._update_children_costs(neighbor)

    def _update_children_costs(self, node: Node):
        """Recursively update costs of children after rewiring."""
        for child in node.children:
            child.cost = node.cost + node.config.distance_to(child.config)
            self._update_children_costs(child)


class RRTConnect(MotionPlanner):
    """RRT-Connect with bidirectional search."""

    def __init__(
        self, space: ConfigurationSpace, max_iter: int = 10000, step_size: float = 0.1
    ):
        super().__init__(space)
        self.max_iter = max_iter
        self.step_size = step_size

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        # Two trees
        tree_start = [Node(start)]
        tree_goal = [Node(goal)]

        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Sample
            random_config = self.space.sample_random()

            # Extend from start tree
            extended_start = self._extend(tree_start, random_config)

            if extended_start is not None:
                # Try to connect from goal tree
                connected = self._connect(tree_goal, extended_start.config)

                if connected:
                    # Merge trees
                    start_node = self._nearest_in_tree(
                        tree_start, extended_start.config
                    )
                    goal_node = self._nearest_in_tree(tree_goal, extended_start.config)

                    path_start = start_node.get_path()
                    path_goal = goal_node.get_path()
                    path_goal.reverse()

                    full_path = path_start[:-1] + path_goal
                    cost = sum(
                        full_path[i].distance_to(full_path[i + 1])
                        for i in range(len(full_path) - 1)
                    )

                    self.stats["computation_time"] = time.time() - start_time
                    return Path(full_path, cost)

            # Swap trees
            tree_start, tree_goal = tree_goal, tree_start

        self.stats["computation_time"] = time.time() - start_time
        return None

    def _extend(self, tree: List[Node], config: Configuration) -> Optional[Node]:
        """Extend tree towards configuration."""
        nearest = min(tree, key=lambda n: n.config.distance_to(config))
        new_config = self._steer(nearest.config, config)

        if self.space.is_valid(new_config):
            local_path = self.space.local_planner(
                nearest.config, new_config, self.step_size
            )
            if len(local_path) > 0:
                new_node = Node(
                    new_config, nearest.cost + nearest.config.distance_to(new_config)
                )
                new_node.parent = nearest
                nearest.children.append(new_node)
                tree.append(new_node)

                if new_config.distance_to(config) < self.step_size:
                    return new_node
        return None

    def _connect(self, tree: List[Node], config: Configuration) -> bool:
        """Try to connect tree to configuration."""
        while True:
            nearest = min(tree, key=lambda n: n.config.distance_to(config))
            new_config = self._steer(nearest.config, config)

            if self.space.is_valid(new_config):
                local_path = self.space.local_planner(
                    nearest.config, new_config, self.step_size
                )
                if len(local_path) > 0:
                    new_node = Node(
                        new_config,
                        nearest.cost + nearest.config.distance_to(new_config),
                    )
                    new_node.parent = nearest
                    nearest.children.append(new_node)
                    tree.append(new_node)

                    if new_config.distance_to(config) < self.step_size:
                        return True
                else:
                    return False
            else:
                return False

    def _nearest_in_tree(self, tree: List[Node], config: Configuration) -> Node:
        return min(tree, key=lambda n: n.config.distance_to(config))

    def _steer(
        self, from_config: Configuration, to_config: Configuration
    ) -> Configuration:
        direction = to_config.values - from_config.values
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_config

        direction = direction / distance
        new_values = from_config.values + direction * self.step_size
        return Configuration(new_values)


class PRM(MotionPlanner):
    """Probabilistic Roadmap planner."""

    def __init__(
        self,
        space: ConfigurationSpace,
        num_samples: int = 1000,
        connection_radius: float = 1.0,
        max_neighbors: int = 10,
    ):
        super().__init__(space)
        self.num_samples = num_samples
        self.connection_radius = connection_radius
        self.max_neighbors = max_neighbors
        self.roadmap: Dict[int, Set[int]] = defaultdict(set)
        self.samples: List[Configuration] = []
        self.built = False

    def build_roadmap(self):
        """Build the probabilistic roadmap."""
        if self.built:
            return

        # Sample configurations
        self.samples = []
        while len(self.samples) < self.num_samples:
            config = self.space.sample_random()
            if self.space.is_valid(config):
                self.samples.append(config)

        # Build k-d tree for efficient nearest neighbor search
        sample_array = np.array([s.values for s in self.samples])
        kdtree = KDTree(sample_array)

        # Connect neighbors
        self.roadmap = defaultdict(set)
        for i, config in enumerate(self.samples):
            neighbors = kdtree.query_ball_point(config.values, self.connection_radius)
            neighbors = sorted(
                neighbors, key=lambda j: config.distance_to(self.samples[j])
            )

            for j in neighbors[: self.max_neighbors]:
                if i != j and j not in self.roadmap[i]:
                    local_path = self.space.local_planner(config, self.samples[j])
                    if len(local_path) > 0:
                        self.roadmap[i].add(j)
                        self.roadmap[j].add(i)

        self.built = True

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        # Build roadmap if not already built
        if not self.built:
            self.build_roadmap()

        # Add start and goal to roadmap temporarily
        start_idx = len(self.samples)
        goal_idx = len(self.samples) + 1
        all_samples = self.samples + [start, goal]

        # Connect start and goal
        sample_array = np.array([s.values for s in self.samples])
        if len(sample_array) > 0:
            kdtree = KDTree(sample_array)

            for idx, is_start in [(start_idx, True), (goal_idx, False)]:
                config = all_samples[idx]
                neighbors = kdtree.query_ball_point(
                    config.values, self.connection_radius
                )
                neighbors = sorted(
                    neighbors, key=lambda j: config.distance_to(self.samples[j])
                )

                for j in neighbors[: self.max_neighbors]:
                    local_path = self.space.local_planner(config, self.samples[j])
                    if len(local_path) > 0:
                        if is_start:
                            self.roadmap[start_idx].add(j)
                            self.roadmap[j].add(start_idx)
                        else:
                            self.roadmap[goal_idx].add(j)
                            self.roadmap[j].add(goal_idx)

        # Search for path using A*
        path_indices = self._astar_search(start_idx, goal_idx, all_samples)

        self.stats["computation_time"] = time.time() - start_time

        if path_indices is not None:
            configs = [all_samples[i] for i in path_indices]
            cost = sum(
                configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
            )
            return Path(configs, cost)

        return None

    def _astar_search(
        self, start_idx: int, goal_idx: int, samples: List[Configuration]
    ) -> Optional[List[int]]:
        """A* search on the roadmap."""
        open_set = [(0.0, start_idx)]
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start_idx] = 0.0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor in self.roadmap[current]:
                tentative_g = g_score[current] + samples[current].distance_to(
                    samples[neighbor]
                )

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + samples[neighbor].distance_to(
                        samples[goal_idx]
                    )
                    heapq.heappush(open_set, (f_score, neighbor))

        return None


class EST(MotionPlanner):
    """Expansive Space Trees planner."""

    def __init__(
        self, space: ConfigurationSpace, max_iter: int = 10000, step_size: float = 0.1
    ):
        super().__init__(space)
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes: List[Node] = []

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        self.nodes = [Node(start)]

        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Pick node with probability inversely proportional to density
            probabilities = self._compute_selection_probabilities()
            selected_idx = np.random.choice(len(self.nodes), p=probabilities)
            selected_node = self.nodes[selected_idx]

            # Sample in neighborhood
            random_config = self.space.sample_random_near(
                selected_node.config, self.step_size * 5
            )

            # Try to extend towards sample
            new_config = self._steer(selected_node.config, random_config)

            if self.space.is_valid(new_config):
                local_path = self.space.local_planner(
                    selected_node.config, new_config, self.step_size
                )
                if len(local_path) > 0:
                    new_node = Node(
                        new_config,
                        selected_node.cost
                        + selected_node.config.distance_to(new_config),
                    )
                    new_node.parent = selected_node
                    selected_node.children.append(new_node)
                    self.nodes.append(new_node)

                    # Check goal
                    if new_config.distance_to(goal) < self.step_size:
                        local_path = self.space.local_planner(
                            new_config, goal, self.step_size
                        )
                        if len(local_path) > 0:
                            goal_node = Node(
                                goal, new_node.cost + new_config.distance_to(goal)
                            )
                            goal_node.parent = new_node
                            self.stats["computation_time"] = time.time() - start_time
                            return Path(goal_node.get_path(), goal_node.cost)

        self.stats["computation_time"] = time.time() - start_time
        return None

    def _compute_selection_probabilities(self) -> np.ndarray:
        """Compute selection probabilities inversely proportional to local density."""
        densities = []
        for node in self.nodes:
            # Count neighbors within radius
            count = sum(
                1
                for other in self.nodes
                if node.config.distance_to(other.config) < self.step_size * 3
            )
            densities.append(1.0 / (count + 1))

        total = sum(densities)
        return np.array(densities) / total

    def _steer(
        self, from_config: Configuration, to_config: Configuration
    ) -> Configuration:
        direction = to_config.values - from_config.values
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_config

        direction = direction / distance
        new_values = from_config.values + direction * self.step_size
        return Configuration(new_values)


# ============================================================================
# Search-Based Planners
# ============================================================================


class AStar(MotionPlanner):
    """A* algorithm for grid-based or graph-based planning."""

    def __init__(
        self,
        space: ConfigurationSpace,
        grid_resolution: float = 0.1,
        diagonal_moves: bool = True,
    ):
        super().__init__(space)
        self.grid_resolution = grid_resolution
        self.diagonal_moves = diagonal_moves

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        # Convert to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # A* search
        open_set = [(0.0, start_grid)]
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start_grid] = 0.0

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            self.stats["nodes_expanded"] += 1

            if current == goal_grid:
                path = self._reconstruct_path(came_from, current, start)
                self.stats["computation_time"] = time.time() - start_time
                return path

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Check if neighbor is valid
                neighbor_config = self._grid_to_world(neighbor)
                if not self.space.is_valid(neighbor_config):
                    continue

                tentative_g = g_score[current] + self._grid_distance(current, neighbor)

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))

        self.stats["computation_time"] = time.time() - start_time
        return None

    def _world_to_grid(self, config: Configuration) -> Tuple[int, ...]:
        """Convert world coordinates to grid coordinates."""
        return tuple(int(c / self.grid_resolution) for c in config.values)

    def _grid_to_world(self, grid: Tuple[int, ...]) -> Configuration:
        """Convert grid coordinates to world coordinates."""
        values = np.array([c * self.grid_resolution for c in grid])
        return Configuration(values)

    def _get_neighbors(self, grid: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get neighboring grid cells."""
        neighbors = []
        dim = len(grid)

        # Generate all combinations of -1, 0, 1 for each dimension
        for offset in self._generate_offsets(dim):
            if all(o == 0 for o in offset):
                continue
            if not self.diagonal_moves and sum(abs(o) for o in offset) > 1:
                continue
            neighbor = tuple(grid[i] + offset[i] for i in range(dim))
            neighbors.append(neighbor)

        return neighbors

    def _generate_offsets(self, dim: int) -> List[Tuple[int, ...]]:
        """Generate all offset combinations."""
        if dim == 1:
            return [(-1,), (0,), (1,)]
        else:
            result = []
            for sub in self._generate_offsets(dim - 1):
                for i in [-1, 0, 1]:
                    result.append((i,) + sub)
            return result

    def _grid_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        """Distance between two grid cells."""
        diff = np.array([a[i] - b[i] for i in range(len(a))])
        euclidean = np.linalg.norm(diff) * self.grid_resolution
        return euclidean

    def _heuristic(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        """Heuristic estimate (Euclidean distance)."""
        return self._grid_distance(a, b)

    def _reconstruct_path(
        self, came_from: Dict, current: Tuple[int, ...], start_config: Configuration
    ) -> Path:
        """Reconstruct path from A* search."""
        grid_path = [current]
        while current in came_from:
            current = came_from[current]
            grid_path.append(current)

        grid_path.reverse()
        configs = [self._grid_to_world(g) for g in grid_path]
        configs[0] = start_config  # Use exact start

        cost = sum(
            configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
        )
        return Path(configs, cost)


class Dijkstra(AStar):
    """Dijkstra's algorithm (A* with zero heuristic)."""

    def _heuristic(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        return 0.0


class DStar(MotionPlanner):
    """D* Lite algorithm for dynamic replanning."""

    def __init__(self, space: ConfigurationSpace, grid_resolution: float = 0.1):
        super().__init__(space)
        self.grid_resolution = grid_resolution
        self.grid_changes: Set[Tuple[int, ...]] = set()

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # Initialize
        self.rhs = defaultdict(lambda: float("inf"))
        self.g = defaultdict(lambda: float("inf"))
        self.rhs[goal_grid] = 0

        self.open_set = []
        heapq.heappush(self.open_set, (self._key(goal_grid, start_grid), goal_grid))

        # Compute shortest path
        self._compute_shortest_path(start_grid)

        if self.g[start_grid] == float("inf"):
            self.stats["computation_time"] = time.time() - start_time
            return None

        path = self._extract_path(start_grid, goal_grid)
        self.stats["computation_time"] = time.time() - start_time
        return path

    def _key(self, s: Tuple[int, ...], start: Tuple[int, ...]) -> Tuple[float, float]:
        """Calculate priority key."""
        k1 = min(self.g[s], self.rhs[s]) + self._heuristic(s, start)
        k2 = min(self.g[s], self.rhs[s])
        return (k1, k2)

    def _heuristic(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        diff = np.array([a[i] - b[i] for i in range(len(a))])
        return np.linalg.norm(diff) * self.grid_resolution

    def _compute_shortest_path(self, start: Tuple[int, ...]):
        """Main D* Lite loop."""
        while self.open_set and (
            self.open_set[0][0] < self._key(start, start)
            or self.rhs[start] != self.g[start]
        ):
            k_old, u = heapq.heappop(self.open_set)

            if k_old < self._key(u, start):
                heapq.heappush(self.open_set, (self._key(u, start), u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self._get_predecessors(u):
                    self._update_vertex(s, start)
            else:
                self.g[u] = float("inf")
                self._update_vertex(u, start)
                for s in self._get_predecessors(u):
                    self._update_vertex(s, start)

    def _update_vertex(self, u: Tuple[int, ...], start: Tuple[int, ...]):
        """Update vertex in the search."""
        if u != self._goal_grid:
            self.rhs[u] = min(
                self._cost(s, u) + self.g[s] for s in self._get_successors(u)
            )

        # Remove from open set if present
        self.open_set = [(k, s) for k, s in self.open_set if s != u]
        heapq.heapify(self.open_set)

        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.open_set, (self._key(u, start), u))

    def _get_successors(self, u: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get successor nodes."""
        return self._get_neighbors(u)

    def _get_predecessors(self, u: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get predecessor nodes."""
        return self._get_neighbors(u)

    def _get_neighbors(self, u: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get neighboring grid cells."""
        neighbors = []
        dim = len(u)
        for offset in self._generate_offsets(dim):
            if all(o == 0 for o in offset):
                continue
            neighbor = tuple(u[i] + offset[i] for i in range(dim))
            # Check validity
            config = self._grid_to_world(neighbor)
            if self.space.is_valid(config):
                neighbors.append(neighbor)
        return neighbors

    def _generate_offsets(self, dim: int) -> List[Tuple[int, ...]]:
        if dim == 1:
            return [(-1,), (0,), (1,)]
        else:
            result = []
            for sub in self._generate_offsets(dim - 1):
                for i in [-1, 0, 1]:
                    result.append((i,) + sub)
            return result

    def _cost(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        """Cost between two cells."""
        return self._heuristic(a, b)

    def _world_to_grid(self, config: Configuration) -> Tuple[int, ...]:
        return tuple(int(c / self.grid_resolution) for c in config.values)

    def _grid_to_world(self, grid: Tuple[int, ...]) -> Configuration:
        values = np.array([c * self.grid_resolution for c in grid])
        return Configuration(values)

    def _extract_path(self, start: Tuple[int, ...], goal: Tuple[int, ...]) -> Path:
        """Extract path from computed costs."""
        path_grid = [start]
        current = start

        while current != goal:
            neighbors = self._get_successors(current)
            if not neighbors:
                break
            current = min(neighbors, key=lambda n: self.g[n] + self._cost(current, n))
            path_grid.append(current)

        configs = [self._grid_to_world(g) for g in path_grid]
        cost = sum(
            configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
        )
        return Path(configs, cost)


class ARAStar(MotionPlanner):
    """Anytime Repairing A* algorithm."""

    def __init__(
        self,
        space: ConfigurationSpace,
        grid_resolution: float = 0.1,
        initial_epsilon: float = 2.5,
        epsilon_decay: float = 0.8,
    ):
        super().__init__(space)
        self.grid_resolution = grid_resolution
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        epsilon = self.initial_epsilon
        best_path = None
        best_cost = float("inf")

        while epsilon >= 1.0:
            path = self._arastar_search(start_grid, goal_grid, epsilon)

            if path is not None:
                path_cost = path.cost
                if path_cost < best_cost:
                    best_cost = path_cost
                    best_path = path

            epsilon *= self.epsilon_decay
            if epsilon < 1.0:
                epsilon = 1.0

        self.stats["computation_time"] = time.time() - start_time
        return best_path

    def _arastar_search(
        self, start: Tuple[int, ...], goal: Tuple[int, ...], epsilon: float
    ) -> Optional[Path]:
        """Single ARA* search with given epsilon."""
        open_set = [(epsilon * self._heuristic(start, goal), start)]
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0.0

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)

            if current == goal:
                return self._reconstruct_path(came_from, current, start)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                neighbor_config = self._grid_to_world(neighbor)
                if not self.space.is_valid(neighbor_config):
                    continue

                tentative_g = g_score[current] + self._grid_distance(current, neighbor)

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + epsilon * self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def _world_to_grid(self, config: Configuration) -> Tuple[int, ...]:
        return tuple(int(c / self.grid_resolution) for c in config.values)

    def _grid_to_world(self, grid: Tuple[int, ...]) -> Configuration:
        values = np.array([c * self.grid_resolution for c in grid])
        return Configuration(values)

    def _heuristic(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        diff = np.array([a[i] - b[i] for i in range(len(a))])
        return np.linalg.norm(diff) * self.grid_resolution

    def _grid_distance(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        return self._heuristic(a, b)

    def _get_neighbors(self, grid: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        neighbors = []
        dim = len(grid)
        for offset in self._generate_offsets(dim):
            if all(o == 0 for o in offset):
                continue
            neighbor = tuple(grid[i] + offset[i] for i in range(dim))
            neighbors.append(neighbor)
        return neighbors

    def _generate_offsets(self, dim: int) -> List[Tuple[int, ...]]:
        if dim == 1:
            return [(-1,), (0,), (1,)]
        else:
            result = []
            for sub in self._generate_offsets(dim - 1):
                for i in [-1, 0, 1]:
                    result.append((i,) + sub)
            return result

    def _reconstruct_path(
        self, came_from: Dict, current: Tuple[int, ...], start_config: Configuration
    ) -> Path:
        grid_path = [current]
        while current in came_from:
            current = came_from[current]
            grid_path.append(current)

        grid_path.reverse()
        configs = [self._grid_to_world(g) for g in grid_path]

        cost = sum(
            configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
        )
        return Path(configs, cost)


class ThetaStar(AStar):
    """Theta* algorithm for any-angle path planning."""

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        open_set = [(self._heuristic(start_grid, goal_grid), start_grid)]
        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start_grid] = 0.0
        parent = {start_grid: start_grid}

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            self.stats["nodes_expanded"] += 1

            if current == goal_grid:
                path = self._reconstruct_path_theta(parent, current, start)
                self.stats["computation_time"] = time.time() - start_time
                return path

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                neighbor_config = self._grid_to_world(neighbor)
                if not self.space.is_valid(neighbor_config):
                    continue

                # Theta* - try to connect directly to parent's parent
                if self._line_of_sight(parent[current], neighbor):
                    tentative_g = g_score[parent[current]] + self._grid_distance(
                        parent[current], neighbor
                    )
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = parent[current]
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = parent[current]
                        f_score = tentative_g + self._heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score, neighbor))
                else:
                    tentative_g = g_score[current] + self._grid_distance(
                        current, neighbor
                    )
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        parent[neighbor] = current
                        f_score = tentative_g + self._heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score, neighbor))

        self.stats["computation_time"] = time.time() - start_time
        return None

    def _line_of_sight(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
        """Check if there's a clear line of sight between two grid cells."""
        # Bresenham's line algorithm with collision checking
        dim = len(a)
        if dim == 2:
            return self._line_of_sight_2d(a, b)
        else:
            # For higher dimensions, use simple interpolation
            num_checks = max(abs(a[i] - b[i]) for i in range(dim)) + 1
            for i in range(num_checks + 1):
                t = i / num_checks
                grid = tuple(int(a[j] + t * (b[j] - a[j])) for j in range(dim))
                config = self._grid_to_world(grid)
                if not self.space.is_valid(config):
                    return False
            return True

    def _line_of_sight_2d(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """Bresenham's algorithm for 2D."""
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            config = self._grid_to_world((x0, y0))
            if not self.space.is_valid(config):
                return False

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return True

    def _reconstruct_path_theta(
        self, parent: Dict, current: Tuple[int, ...], start_config: Configuration
    ) -> Path:
        """Reconstruct path from Theta* search."""
        grid_path = [current]
        while current != start_config:
            current = parent[current]
            grid_path.append(current)

        grid_path.reverse()
        configs = [self._grid_to_world(g) for g in grid_path]

        # Add shortcuts where line of sight exists
        simplified = self._simplify_path(configs)

        cost = sum(
            simplified[i].distance_to(simplified[i + 1])
            for i in range(len(simplified) - 1)
        )
        return Path(simplified, cost)

    def _simplify_path(self, configs: List[Configuration]) -> List[Configuration]:
        """Simplify path by removing unnecessary waypoints."""
        if len(configs) <= 2:
            return configs

        simplified = [configs[0]]
        i = 0

        while i < len(configs) - 1:
            # Look ahead as far as possible
            for j in range(len(configs) - 1, i, -1):
                if self._is_collision_free_edge(configs[i], configs[j]):
                    simplified.append(configs[j])
                    i = j
                    break
            else:
                i += 1
                simplified.append(configs[i])

        return simplified

    def _is_collision_free_edge(self, a: Configuration, b: Configuration) -> bool:
        """Check if edge is collision-free."""
        num_checks = int(a.distance_to(b) / self.grid_resolution) + 1
        for i in range(num_checks + 1):
            t = i / num_checks
            config = Configuration(a.values + t * (b.values - a.values))
            if not self.space.is_valid(config):
                return False
        return True


# ============================================================================
# Optimization-Based Planners
# ============================================================================


class CHOMP(MotionPlanner):
    """Covariant Hamiltonian Optimization for Motion Planning."""

    def __init__(
        self,
        space: ConfigurationSpace,
        max_iter: int = 100,
        step_size: float = 0.1,
        obst_cost_weight: float = 1.0,
        smoothness_weight: float = 0.1,
    ):
        super().__init__(space)
        self.max_iter = max_iter
        self.step_size = step_size
        self.obst_cost_weight = obst_cost_weight
        self.smoothness_weight = smoothness_weight

    def plan(
        self,
        start: Configuration,
        goal: Configuration,
        initial_path: Optional[Path] = None,
        **kwargs,
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        # Initialize trajectory
        if initial_path is None:
            # Straight line initialization
            num_waypoints = 20
            configs = []
            for i in range(num_waypoints):
                alpha = i / (num_waypoints - 1)
                config = self.space.interpolate(start, goal, alpha)
                configs.append(config)
            trajectory = np.array([c.values for c in configs])
        else:
            trajectory = np.array([c.values for c in initial_path.configurations])

        # CHOMP optimization
        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Compute gradients
            smooth_grad = self._compute_smoothness_gradient(trajectory)
            obst_grad = self._compute_obstacle_gradient(trajectory)

            # Update trajectory
            gradient = (
                self.smoothness_weight * smooth_grad + self.obst_cost_weight * obst_grad
            )
            trajectory = trajectory - self.step_size * gradient

            # Fix endpoints
            trajectory[0] = start.values
            trajectory[-1] = goal.values

            # Check convergence
            if np.linalg.norm(gradient) < 1e-6:
                break

        # Convert back to path
        configs = [Configuration(t) for t in trajectory]
        cost = self._compute_total_cost(trajectory)

        self.stats["computation_time"] = time.time() - start_time
        return Path(configs, cost)

    def _compute_smoothness_gradient(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute gradient of smoothness cost."""
        gradient = np.zeros_like(trajectory)
        for i in range(1, len(trajectory) - 1):
            gradient[i] = 2 * trajectory[i] - trajectory[i - 1] - trajectory[i + 1]
        return gradient

    def _compute_obstacle_gradient(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute gradient of obstacle cost."""
        gradient = np.zeros_like(trajectory)
        for i in range(1, len(trajectory) - 1):
            config = Configuration(trajectory[i])
            if not self.space.is_valid(config):
                # Push away from obstacle
                gradient[i] = self._compute_obstacle_potential_gradient(config)
        return gradient

    def _compute_obstacle_potential_gradient(self, config: Configuration) -> np.ndarray:
        """Compute gradient of obstacle potential."""
        # Simple numerical gradient
        eps = 1e-4
        grad = np.zeros(config.dimension)
        for i in range(config.dimension):
            config_plus = config.values.copy()
            config_plus[i] += eps
            config_minus = config.values.copy()
            config_minus[i] -= eps

            c_plus = Configuration(config_plus)
            c_minus = Configuration(config_minus)

            # Higher cost for invalid configs
            cost_plus = 0.0 if self.space.is_valid(c_plus) else 1.0
            cost_minus = 0.0 if self.space.is_valid(c_minus) else 1.0

            grad[i] = (cost_plus - cost_minus) / (2 * eps)

        return grad

    def _compute_total_cost(self, trajectory: np.ndarray) -> float:
        """Compute total trajectory cost."""
        # Smoothness cost
        smooth_cost = 0.0
        for i in range(1, len(trajectory)):
            smooth_cost += np.linalg.norm(trajectory[i] - trajectory[i - 1]) ** 2

        # Obstacle cost
        obst_cost = 0.0
        for i in range(len(trajectory)):
            config = Configuration(trajectory[i])
            if not self.space.is_valid(config):
                obst_cost += 1.0

        return self.smoothness_weight * smooth_cost + self.obst_cost_weight * obst_cost


class TrajOpt(MotionPlanner):
    """Trajectory optimization using sequential convex optimization."""

    def __init__(
        self,
        space: ConfigurationSpace,
        num_waypoints: int = 20,
        max_iter: int = 50,
        smoothness_weight: float = 0.01,
        collision_weight: float = 1.0,
        constraint_weight: float = 10.0,
    ):
        super().__init__(space)
        self.num_waypoints = num_waypoints
        self.max_iter = max_iter
        self.smoothness_weight = smoothness_weight
        self.collision_weight = collision_weight
        self.constraint_weight = constraint_weight

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        # Initialize trajectory as straight line
        dim = len(start.values)
        trajectory = np.zeros((self.num_waypoints, dim))
        for i in range(self.num_waypoints):
            alpha = i / (self.num_waypoints - 1)
            trajectory[i] = start.values + alpha * (goal.values - start.values)

        # Flatten for optimization
        x0 = trajectory[1:-1].flatten()  # Exclude endpoints

        # Optimize
        result = minimize(
            fun=lambda x: self._objective(x, start, goal),
            x0=x0,
            method="L-BFGS-B",
            jac=lambda x: self._gradient(x, start, goal),
            options={"maxiter": self.max_iter, "disp": False},
        )

        if result.success:
            # Reconstruct trajectory
            inner = result.x.reshape((self.num_waypoints - 2, dim))
            trajectory = np.vstack([start.values, inner, goal.values])
            configs = [Configuration(t) for t in trajectory]
            cost = result.fun

            self.stats["computation_time"] = time.time() - start_time
            return Path(configs, cost)

        self.stats["computation_time"] = time.time() - start_time
        return None

    def _objective(
        self, x: np.ndarray, start: Configuration, goal: Configuration
    ) -> float:
        """Compute objective function."""
        dim = len(start.values)
        inner = x.reshape((-1, dim))
        trajectory = np.vstack([start.values, inner, goal.values])

        cost = 0.0

        # Smoothness cost
        for i in range(1, len(trajectory)):
            cost += (
                self.smoothness_weight
                * np.linalg.norm(trajectory[i] - trajectory[i - 1]) ** 2
            )

        # Collision cost
        for i in range(len(trajectory)):
            config = Configuration(trajectory[i])
            if not self.space.is_valid(config):
                cost += self.collision_weight

        return cost

    def _gradient(
        self, x: np.ndarray, start: Configuration, goal: Configuration
    ) -> np.ndarray:
        """Compute gradient of objective."""
        eps = 1e-5
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            grad[i] = (
                self._objective(x_plus, start, goal)
                - self._objective(x_minus, start, goal)
            ) / (2 * eps)

        return grad


class STOMP(MotionPlanner):
    """Stochastic Trajectory Optimization for Motion Planning."""

    def __init__(
        self,
        space: ConfigurationSpace,
        num_waypoints: int = 20,
        num_rollouts: int = 10,
        max_iter: int = 50,
        noise_std: float = 0.1,
    ):
        super().__init__(space)
        self.num_waypoints = num_waypoints
        self.num_rollouts = num_rollouts
        self.max_iter = max_iter
        self.noise_std = noise_std

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        dim = len(start.values)

        # Initialize trajectory
        trajectory = np.zeros((self.num_waypoints, dim))
        for i in range(self.num_waypoints):
            alpha = i / (self.num_waypoints - 1)
            trajectory[i] = start.values + alpha * (goal.values - start.values)

        # Precompute smoothing matrix
        smoothing_matrix = self._compute_smoothing_matrix()

        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Generate noisy rollouts
            rollouts = []
            costs = []

            for _ in range(self.num_rollouts):
                noise = np.random.randn(self.num_waypoints - 2, dim) * self.noise_std
                noisy_traj = trajectory.copy()
                noisy_traj[1:-1] += noise

                cost = self._compute_trajectory_cost(noisy_traj)
                rollouts.append(noisy_traj)
                costs.append(cost)

            # Compute weights
            costs = np.array(costs)
            min_cost = np.min(costs)
            max_cost = np.max(costs)
            if max_cost > min_cost:
                weights = np.exp(-5 * (costs - min_cost) / (max_cost - min_cost))
            else:
                weights = np.ones_like(costs) / len(costs)
            weights = weights / np.sum(weights)

            # Update trajectory using weighted average
            update = np.zeros_like(trajectory)
            for i, rollout in enumerate(rollouts):
                update[1:-1] += weights[i] * (rollout[1:-1] - trajectory[1:-1])

            # Apply smoothing
            trajectory[1:-1] += smoothing_matrix @ update[1:-1]

            # Fix endpoints
            trajectory[0] = start.values
            trajectory[-1] = goal.values

        configs = [Configuration(t) for t in trajectory]
        cost = self._compute_trajectory_cost(trajectory)

        self.stats["computation_time"] = time.time() - start_time
        return Path(configs, cost)

    def _compute_smoothing_matrix(self) -> np.ndarray:
        """Compute smoothing convolution matrix."""
        n = self.num_waypoints - 2
        # Simple averaging filter
        kernel = np.array([0.25, 0.5, 0.25])
        matrix = np.eye(n)

        for i in range(n):
            for j, k in enumerate([-1, 0, 1]):
                if 0 <= i + k < n:
                    matrix[i, i + k] = kernel[j]

        return matrix

    def _compute_trajectory_cost(self, trajectory: np.ndarray) -> float:
        """Compute trajectory cost."""
        cost = 0.0

        # Smoothness cost
        for i in range(1, len(trajectory)):
            cost += np.linalg.norm(trajectory[i] - trajectory[i - 1]) ** 2

        # Collision cost
        for i in range(len(trajectory)):
            config = Configuration(trajectory[i])
            if not self.space.is_valid(config):
                cost += 10.0

        return cost


class ITOMP(MotionPlanner):
    """Incremental Trajectory Optimization."""

    def __init__(
        self,
        space: ConfigurationSpace,
        num_waypoints: int = 20,
        max_iter: int = 50,
        batch_size: int = 5,
    ):
        super().__init__(space)
        self.num_waypoints = num_waypoints
        self.max_iter = max_iter
        self.batch_size = batch_size

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        dim = len(start.values)

        # Start with coarse trajectory
        current_waypoints = max(5, self.batch_size)
        trajectory = self._initialize_trajectory(start, goal, current_waypoints)

        while current_waypoints <= self.num_waypoints:
            # Optimize current trajectory
            trajectory = self._optimize_trajectory(trajectory, start, goal)

            # Refine if needed
            if current_waypoints < self.num_waypoints:
                current_waypoints = min(
                    current_waypoints + self.batch_size, self.num_waypoints
                )
                trajectory = self._upsample_trajectory(trajectory, current_waypoints)

        configs = [Configuration(t) for t in trajectory]
        cost = self._compute_cost(trajectory)

        self.stats["computation_time"] = time.time() - start_time
        return Path(configs, cost)

    def _initialize_trajectory(
        self, start: Configuration, goal: Configuration, num_waypoints: int
    ) -> np.ndarray:
        """Initialize straight-line trajectory."""
        dim = len(start.values)
        trajectory = np.zeros((num_waypoints, dim))
        for i in range(num_waypoints):
            alpha = i / (num_waypoints - 1)
            trajectory[i] = start.values + alpha * (goal.values - start.values)
        return trajectory

    def _optimize_trajectory(
        self, trajectory: np.ndarray, start: Configuration, goal: Configuration
    ) -> np.ndarray:
        """Optimize trajectory using gradient descent."""
        optimized = trajectory.copy()
        learning_rate = 0.01

        for _ in range(self.max_iter):
            gradient = self._compute_gradient(optimized, start, goal)
            optimized[1:-1] -= learning_rate * gradient[1:-1]

            # Fix endpoints
            optimized[0] = start.values
            optimized[-1] = goal.values

        return optimized

    def _compute_gradient(
        self, trajectory: np.ndarray, start: Configuration, goal: Configuration
    ) -> np.ndarray:
        """Compute gradient of cost function."""
        gradient = np.zeros_like(trajectory)

        # Smoothness gradient
        for i in range(1, len(trajectory) - 1):
            gradient[i] = 2 * trajectory[i] - trajectory[i - 1] - trajectory[i + 1]

        # Collision gradient
        for i in range(len(trajectory)):
            config = Configuration(trajectory[i])
            if not self.space.is_valid(config):
                # Push towards valid space (simplified)
                gradient[i] += np.random.randn(len(start.values)) * 0.1

        return gradient

    def _upsample_trajectory(
        self, trajectory: np.ndarray, new_num_waypoints: int
    ) -> np.ndarray:
        """Upsample trajectory to more waypoints."""
        dim = trajectory.shape[1]
        old_num = len(trajectory)

        # Interpolate each dimension
        result = np.zeros((new_num_waypoints, dim))
        old_indices = np.linspace(0, 1, old_num)
        new_indices = np.linspace(0, 1, new_num_waypoints)

        for d in range(dim):
            result[:, d] = np.interp(new_indices, old_indices, trajectory[:, d])

        return result

    def _compute_cost(self, trajectory: np.ndarray) -> float:
        """Compute trajectory cost."""
        cost = 0.0

        # Smoothness
        for i in range(1, len(trajectory)):
            cost += np.linalg.norm(trajectory[i] - trajectory[i - 1]) ** 2

        # Collision
        for i in range(len(trajectory)):
            config = Configuration(trajectory[i])
            if not self.space.is_valid(config):
                cost += 10.0

        return cost


# ============================================================================
# Potential Field Methods
# ============================================================================


class PotentialFieldPlanner(MotionPlanner):
    """Base class for potential field planners."""

    def __init__(
        self,
        space: ConfigurationSpace,
        step_size: float = 0.01,
        max_iter: int = 10000,
        goal_tolerance: float = 0.1,
    ):
        super().__init__(space)
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()
        self.reset_stats()

        if not self.space.is_valid(start) or not self.space.is_valid(goal):
            return None

        path = [start]
        current = start

        for iteration in range(self.max_iter):
            self.stats["iterations"] = iteration

            # Compute potential gradient
            gradient = self._compute_gradient(current, goal)

            # Move in negative gradient direction
            new_values = current.values - self.step_size * gradient
            new_config = Configuration(new_values)

            # Check validity
            if not self.space.is_valid(new_config):
                # Try smaller step
                new_values = current.values - 0.5 * self.step_size * gradient
                new_config = Configuration(new_values)
                if not self.space.is_valid(new_config):
                    break

            current = new_config
            path.append(current)

            # Check if goal reached
            if current.distance_to(goal) < self.goal_tolerance:
                path.append(goal)
                cost = sum(
                    path[i].distance_to(path[i + 1]) for i in range(len(path) - 1)
                )
                self.stats["computation_time"] = time.time() - start_time
                return Path(path, cost)

        self.stats["computation_time"] = time.time() - start_time
        return None

    @abstractmethod
    def _compute_gradient(
        self, current: Configuration, goal: Configuration
    ) -> np.ndarray:
        """Compute potential gradient at current configuration."""
        pass


class APF(PotentialFieldPlanner):
    """Artificial Potential Field planner."""

    def __init__(
        self,
        space: ConfigurationSpace,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 1.0,
        influence_radius: float = 2.0,
        **kwargs,
    ):
        super().__init__(space, **kwargs)
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_radius = influence_radius

    def _compute_gradient(
        self, current: Configuration, goal: Configuration
    ) -> np.ndarray:
        # Attractive potential gradient
        attractive = self.attractive_gain * (current.values - goal.values)

        # Repulsive potential gradient (from obstacles)
        repulsive = np.zeros_like(current.values)

        # Sample nearby configurations to estimate obstacle gradient
        num_samples = 10
        for _ in range(num_samples):
            sample = self.space.sample_random_near(current, self.influence_radius)
            if not self.space.is_valid(sample):
                diff = current.values - sample.values
                dist = np.linalg.norm(diff)
                if dist < self.influence_radius and dist > 0.01:
                    repulsive += (
                        self.repulsive_gain
                        * (1.0 / dist - 1.0 / self.influence_radius)
                        / (dist**2)
                        * diff
                        / dist
                    )

        return attractive + repulsive


class HarmonicPotential(PotentialFieldPlanner):
    """Harmonic potential field planner."""

    def __init__(
        self, space: ConfigurationSpace, grid_resolution: float = 0.1, **kwargs
    ):
        super().__init__(space, **kwargs)
        self.grid_resolution = grid_resolution
        self.potential_grid = None

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()

        # Build harmonic potential field
        self._build_harmonic_potential(goal)

        # Call base planner
        result = super().plan(start, goal, **kwargs)

        if result is not None:
            result.cost = time.time() - start_time

        return result

    def _build_harmonic_potential(self, goal: Configuration):
        """Build harmonic potential field using relaxation."""
        # This is a simplified version - in practice, would use proper PDE solver
        bounds = self.space.bounds
        grid_shape = [
            int((b[1] - b[0]) / self.grid_resolution) + 1 for b in bounds[:2]
        ]  # 2D for simplicity

        self.potential_grid = np.ones(grid_shape)

        # Set goal potential to 0
        goal_grid = self._world_to_grid_2d(goal)
        if 0 <= goal_grid[0] < grid_shape[0] and 0 <= goal_grid[1] < grid_shape[1]:
            self.potential_grid[goal_grid] = 0.0

    def _world_to_grid_2d(self, config: Configuration) -> Tuple[int, int]:
        """Convert to 2D grid coordinates."""
        x = int((config.values[0] - self.space.bounds[0][0]) / self.grid_resolution)
        y = int((config.values[1] - self.space.bounds[1][0]) / self.grid_resolution)
        return (x, y)

    def _compute_gradient(
        self, current: Configuration, goal: Configuration
    ) -> np.ndarray:
        # Simplified gradient computation
        # In practice, would interpolate from harmonic potential grid
        return (
            self.attractive_gain * (current.values - goal.values)
            if hasattr(self, "attractive_gain")
            else (current.values - goal.values)
        )


class NavigationFunction(PotentialFieldPlanner):
    """Navigation function planner."""

    def __init__(
        self,
        space: ConfigurationSpace,
        grid_resolution: float = 0.1,
        kappa: float = 1.0,
        **kwargs,
    ):
        super().__init__(space, **kwargs)
        self.grid_resolution = grid_resolution
        self.kappa = kappa

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        import time

        start_time = time.time()

        # Build navigation function
        self._build_navigation_function(goal)

        result = super().plan(start, goal, **kwargs)

        if result is not None:
            result.cost = time.time() - start_time

        return result

    def _build_navigation_function(self, goal: Configuration):
        """Build navigation function using wavefront propagation."""
        # Simplified implementation
        pass

    def _compute_gradient(
        self, current: Configuration, goal: Configuration
    ) -> np.ndarray:
        # Navigation function gradient
        diff = goal.values - current.values
        dist = np.linalg.norm(diff)

        if dist > 0:
            # Gradient points towards goal
            return -diff / dist
        return np.zeros_like(current.values)


# ============================================================================
# Path Smoothing
# ============================================================================


class PathSmoother(ABC):
    """Abstract base class for path smoothers."""

    @abstractmethod
    def smooth(self, path: Path, space: ConfigurationSpace) -> Path:
        pass


class ShortcuttingSmoother(PathSmoother):
    """Path shortcutting smoother."""

    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations

    def smooth(self, path: Path, space: ConfigurationSpace) -> Path:
        if len(path) < 3:
            return path

        configs = path.configurations.copy()

        for _ in range(self.max_iterations):
            if len(configs) < 3:
                break

            # Random shortcut attempt
            i = random.randint(0, len(configs) - 3)
            j = random.randint(i + 2, len(configs) - 1)

            # Check if shortcut is valid
            local_path = space.local_planner(configs[i], configs[j])
            if len(local_path) > 0:
                # Apply shortcut
                configs = configs[: i + 1] + [configs[j]] + configs[j + 1 :]

        cost = sum(
            configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
        )
        return Path(configs, cost)


class BSplineSmoother(PathSmoother):
    """B-spline path smoother."""

    def __init__(self, num_points: int = 100, smoothing_factor: float = 0.0):
        self.num_points = num_points
        self.smoothing_factor = smoothing_factor

    def smooth(self, path: Path, space: ConfigurationSpace) -> Path:
        if len(path) < 4:
            return path

        # Extract waypoints
        points = np.array([c.values for c in path.configurations])
        dim = points.shape[1]

        try:
            # Fit B-spline
            tck, u = splprep(
                [points[:, d] for d in range(dim)], s=self.smoothing_factor
            )

            # Sample new points
            u_new = np.linspace(0, 1, self.num_points)
            smoothed = splev(u_new, tck)

            # Convert back to configurations
            configs = []
            for i in range(self.num_points):
                values = np.array([smoothed[d][i] for d in range(dim)])
                config = Configuration(values)
                if space.is_valid(config):
                    configs.append(config)

            if len(configs) < 2:
                return path

            cost = sum(
                configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
            )
            return Path(configs, cost)

        except Exception:
            # Fall back to original path
            return path


class GradientDescentSmoother(PathSmoother):
    """Gradient descent path smoother."""

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        smoothness_weight: float = 1.0,
        obstacle_weight: float = 1.0,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight

    def smooth(self, path: Path, space: ConfigurationSpace) -> Path:
        if len(path) < 3:
            return path

        points = np.array([c.values for c in path.configurations])

        for _ in range(self.max_iter):
            gradient = np.zeros_like(points)

            # Smoothness gradient
            for i in range(1, len(points) - 1):
                gradient[i] += self.smoothness_weight * (
                    2 * points[i] - points[i - 1] - points[i + 1]
                )

            # Obstacle gradient
            for i in range(len(points)):
                config = Configuration(points[i])
                if not space.is_valid(config):
                    # Push away from obstacle
                    gradient[i] += (
                        self.obstacle_weight * np.random.randn(len(points[i])) * 0.1
                    )

            # Update (keeping endpoints fixed)
            points[1:-1] -= self.learning_rate * gradient[1:-1]

        configs = [Configuration(p) for p in points]
        cost = sum(
            configs[i].distance_to(configs[i + 1]) for i in range(len(configs) - 1)
        )
        return Path(configs, cost)


# ============================================================================
# Utilities
# ============================================================================


class PathPlanner:
    """High-level path planning interface."""

    def __init__(self, space: ConfigurationSpace, algorithm: str = "rrt_star"):
        self.space = space
        self.algorithm = algorithm
        self.planner = self._create_planner(algorithm)

    def _create_planner(self, algorithm: str) -> MotionPlanner:
        """Create planner based on algorithm name."""
        planners = {
            "rrt": RRT,
            "rrt_star": RRTStar,
            "rrt_connect": RRTConnect,
            "prm": PRM,
            "est": EST,
            "astar": AStar,
            "dijkstra": Dijkstra,
            "dstar": DStar,
            "arastar": ARAStar,
            "thetastar": ThetaStar,
            "chomp": CHOMP,
            "trajopt": TrajOpt,
            "stomp": STOMP,
            "itomp": ITOMP,
            "apf": APF,
            "harmonic": HarmonicPotential,
            "navigation": NavigationFunction,
        }

        if algorithm.lower() not in planners:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return planners[algorithm.lower()](self.space)

    def plan(
        self, start: Configuration, goal: Configuration, **kwargs
    ) -> Optional[Path]:
        """Plan a path from start to goal."""
        return self.planner.plan(start, goal, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get planning statistics."""
        return self.planner.stats.copy()


class TrajectoryGenerator:
    """Generate time-parameterized trajectories from paths."""

    def __init__(self, max_velocity: float = 1.0, max_acceleration: float = 1.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate(self, path: Path, total_time: float = 10.0) -> Trajectory:
        """Generate a trajectory from a path."""
        num_points = len(path)
        timestamps = np.linspace(0, total_time, num_points)

        # Compute velocities using finite differences
        if num_points > 1:
            velocities = []
            for i in range(num_points):
                if i == 0:
                    vel = (path[1].values - path[0].values) / (
                        timestamps[1] - timestamps[0]
                    )
                elif i == num_points - 1:
                    vel = (path[-1].values - path[-2].values) / (
                        timestamps[-1] - timestamps[-2]
                    )
                else:
                    vel = (path[i + 1].values - path[i - 1].values) / (
                        timestamps[i + 1] - timestamps[i - 1]
                    )
                velocities.append(vel)
            velocities = np.array(velocities)
        else:
            velocities = np.zeros((num_points, path[0].dimension))

        # Compute accelerations
        if num_points > 2:
            accelerations = []
            for i in range(num_points):
                if i == 0:
                    acc = (velocities[1] - velocities[0]) / (
                        timestamps[1] - timestamps[0]
                    )
                elif i == num_points - 1:
                    acc = (velocities[-1] - velocities[-2]) / (
                        timestamps[-1] - timestamps[-2]
                    )
                else:
                    acc = (velocities[i + 1] - velocities[i - 1]) / (
                        timestamps[i + 1] - timestamps[i - 1]
                    )
                accelerations.append(acc)
            accelerations = np.array(accelerations)
        else:
            accelerations = np.zeros((num_points, path[0].dimension))

        return Trajectory(path, timestamps, velocities, accelerations)

    def time_optimize(self, path: Path) -> Trajectory:
        """Generate time-optimal trajectory."""
        # Simplified time-optimal trajectory
        # In practice, would use TOPP (Time-Optimal Path Parameterization)
        path_length = path.length()
        total_time = path_length / self.max_velocity
        return self.generate(path, total_time)


# ============================================================================
# Convenience Functions
# ============================================================================


def plan_path(
    start: np.ndarray,
    goal: np.ndarray,
    bounds: List[Tuple[float, float]],
    algorithm: str = "rrt_star",
    collision_checker: Optional[CollisionChecker] = None,
    **kwargs,
) -> Optional[Path]:
    """
    Convenience function for planning a path.

    Args:
        start: Start configuration
        goal: Goal configuration
        bounds: List of (min, max) bounds for each dimension
        algorithm: Planning algorithm to use
        collision_checker: Optional collision checker
        **kwargs: Additional arguments for the planner

    Returns:
        Path if successful, None otherwise
    """
    start_config = Configuration(start)
    goal_config = Configuration(goal)

    space = ConfigurationSpace(len(start), bounds, collision_checker)
    planner = PathPlanner(space, algorithm)

    return planner.plan(start_config, goal_config, **kwargs)


def smooth_path(
    path: Path, space: ConfigurationSpace, method: str = "shortcutting", **kwargs
) -> Path:
    """
    Smooth a path using the specified method.

    Args:
        path: Input path
        space: Configuration space
        method: Smoothing method ('shortcutting', 'bspline', 'gradient')
        **kwargs: Additional arguments for the smoother

    Returns:
        Smoothed path
    """
    smoothers = {
        "shortcutting": ShortcuttingSmoother,
        "bspline": BSplineSmoother,
        "gradient": GradientDescentSmoother,
    }

    if method not in smoothers:
        raise ValueError(f"Unknown smoothing method: {method}")

    smoother = smoothers[method](**kwargs)
    return smoother.smooth(path, space)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Plan a path in 2D with obstacles
    print("Fishstick Motion Planning Module")
    print("=" * 50)

    # Define configuration space
    bounds = [(0, 10), (0, 10)]

    # Create some obstacles
    obstacles = [
        (np.array([3, 3]), 1.0),
        (np.array([7, 7]), 1.5),
        (np.array([5, 2]), 0.8),
    ]
    collision_checker = SphereCollisionChecker(obstacles, robot_radius=0.2)
    space = ConfigurationSpace(2, bounds, collision_checker)

    # Define start and goal
    start = Configuration(np.array([1, 1]))
    goal = Configuration(np.array([9, 9]))

    # Test different planners
    planners_to_test = ["rrt", "rrt_star", "astar", "chomp"]

    for planner_name in planners_to_test:
        print(f"\nTesting {planner_name.upper()}...")
        try:
            planner = PathPlanner(space, planner_name)
            path = planner.plan(start, goal)

            if path is not None:
                stats = planner.get_stats()
                print(f"  Success! Path length: {path.length():.3f}")
                print(f"  Iterations: {stats.get('iterations', 'N/A')}")
                print(
                    f"  Computation time: {stats.get('computation_time', 0) * 1000:.2f} ms"
                )
            else:
                print("  Failed to find path")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 50)
    print("All tests completed!")
