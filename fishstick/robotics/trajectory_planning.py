"""
Trajectory Planning Algorithms.

Path and trajectory planning for robotics:
- RRT* (Rapidly-exploring Random Tree Star)
- CHOMP (Covariant Hamiltonian Optimization and Motion Planning)
- Time-optimal trajectory generation
- Jacobian-based IK solvers
"""

from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import numpy as np
from collections import defaultdict
import heapq

from .core import Trajectory, TrajectoryPoint, JointState


class TrajectoryPlanner(nn.Module):
    """
    Base class for trajectory planners.

    Provides common interface for path and trajectory planning.
    """

    def __init__(self, n_joints: int):
        super().__init__()
        self.n_joints = n_joints

    def plan(
        self,
        start: Tensor,
        goal: Tensor,
        obstacles: Optional[Tensor] = None,
    ) -> Trajectory:
        """Plan trajectory from start to goal."""
        raise NotImplementedError


class RRTStar(TrajectoryPlanner):
    """
    Rapidly-exploring Random Tree Star.

    Asymptotically optimal sampling-based motion planner.
    Builds a tree of collision-free configurations.

    Properties:
    - Probabilistic completeness
    - Asymptotic optimality
    - Efficient in high-dimensional spaces

    Args:
        n_joints: Number of joints
        joint_limits: Joint limits [n_joints, 2]
        step_size: Extension step size
        max_iterations: Maximum planning iterations
        goal_tolerance: Distance to goal to consider success
    """

    def __init__(
        self,
        n_joints: int,
        joint_limits: Tensor,
        step_size: float = 0.1,
        max_iterations: int = 1000,
        goal_tolerance: float = 0.05,
        rewire_radius: float = 0.5,
        collision_check_fn: Optional[Callable] = None,
    ):
        super().__init__(n_joints)

        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_tolerance = goal_tolerance
        self.rewire_radius = rewire_radius
        self.collision_check_fn = collision_check_fn

        self.register_buffer("joint_limits", joint_limits)

        self.tree: Dict[int, Tuple[Tensor, int]] = {}
        self.cost: Dict[int, float] = {}

    def plan(
        self,
        start: Tensor,
        goal: Tensor,
        obstacles: Optional[Tensor] = None,
    ) -> Trajectory:
        """
        Plan trajectory using RRT*.

        Args:
            start: Start configuration [n_joints]
            goal: Goal configuration [n_joints]
            obstacles: Obstacle positions [n_obstacles, 3]

        Returns:
            Planned trajectory
        """
        self.tree = {0: (start.clone(), -1)}
        self.cost = {0: 0.0}

        node_id = 1
        goal_node = -1

        for _ in range(self.max_iterations):
            if np.random.random() < 0.05:
                sample = goal.clone()
            else:
                sample = self._sample()

            nearest_id = self._nearest(sample)
            new_config = self._steer(self.tree[nearest_id][0], sample)

            if self._is_valid(new_config, obstacles):
                candidates = self._near_within_radius(new_config, self.rewire_radius)

                min_cost = self.cost[nearest_id] + self._distance(
                    self.tree[nearest_id][0], new_config
                )
                parent = nearest_id

                for cid in candidates:
                    if cid == nearest_id:
                        continue
                    cost = self.cost[cid] + self._distance(
                        self.tree[cid][0], new_config
                    )
                    if cost < min_cost:
                        if self._is_valid(
                            self._interpolate(self.tree[cid][0], new_config), obstacles
                        ):
                            min_cost = cost
                            parent = cid

                self.tree[node_id] = (new_config.clone(), parent)
                self.cost[node_id] = min_cost

                for cid in candidates:
                    if cid == parent:
                        continue
                    new_cost = min_cost + self._distance(new_config, self.tree[cid][0])
                    if new_cost < self.cost[cid]:
                        if self._is_valid(
                            self._interpolate(new_config, self.tree[cid][0]), obstacles
                        ):
                            self.tree[cid] = (self.tree[cid][0].clone(), node_id)
                            self.cost[cid] = new_cost

                if self._distance(new_config, goal) < self.goal_tolerance:
                    goal_node = node_id
                    break

                node_id += 1

        if goal_node == -1:
            _, nearest = min(
                ((self._distance(self.tree[i][0], goal), i) for i in self.tree),
                default=(float("inf"), -1),
            )
            goal_node = nearest

        path = self._extract_path(goal_node)

        return self._smooth_trajectory(path)

    def _sample(self) -> Tensor:
        """Sample random configuration."""
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return torch.rand_like(low) * (high - low) + low

    def _nearest(self, sample: Tensor) -> int:
        """Find nearest node in tree."""
        min_dist = float("inf")
        nearest = 0

        for i, (config, _) in self.tree.items():
            dist = self._distance(config, sample)
            if dist < min_dist:
                min_dist = dist
                nearest = i

        return nearest

    def _near_within_radius(self, config: Tensor, radius: float) -> List[int]:
        """Find all nodes within radius."""
        nearby = []
        for i, (c, _) in self.tree.items():
            if self._distance(c, config) < radius:
                nearby.append(i)
        return nearby

    def _steer(self, from_config: Tensor, to_config: Tensor) -> Tensor:
        """Extend from toward."""
        direction = to_config - from_config
        dist = torch.norm(direction)

        if dist < self.step_size:
            return to_config.clone()

        return from_config + (direction / dist) * self.step_size

    def _interpolate(self, from_config: Tensor, to_config: Tensor) -> Tensor:
        """Interpolate between configurations."""
        return (from_config + to_config) / 2

    def _distance(self, config1: Tensor, config2: Tensor) -> float:
        """Distance between configurations."""
        return torch.norm(config1 - config2).item()

    def _is_valid(
        self,
        config: Tensor,
        obstacles: Optional[Tensor] = None,
    ) -> bool:
        """Check if configuration is valid."""
        if torch.any(config < self.joint_limits[:, 0]) or torch.any(
            config > self.joint_limits[:, 1]
        ):
            return False

        if self.collision_check_fn is not None:
            return self.collision_check_fn(config, obstacles)

        return True

    def _extract_path(self, goal_node: int) -> List[Tensor]:
        """Extract path from tree."""
        path = []
        current = goal_node

        while current != -1:
            path.append(self.tree[current][0].clone())
            current = self.tree[current][1]

        return list(reversed(path))

    def _smooth_trajectory(self, path: List[Tensor]) -> Trajectory:
        """Create smooth trajectory from path."""
        points = []

        for i, config in enumerate(path):
            point = TrajectoryPoint(
                time=float(i) * 0.1,
                joint_position=config.clone(),
                joint_velocity=torch.zeros_like(config),
            )
            points.append(point)

        return Trajectory(points=points, dt=0.1, interpolation="cubic")


class CHOMP(TrajectoryPlanner):
    """
    Covariant Hamiltonian Optimization and Motion Planning.

    Gradient-based trajectory optimization that respects
    obstacles through a trajectory cost function.

    Properties:
    - Smooth, natural motions
    - Handles constraints
    - Gradient-based optimization

    Args:
        n_joints: Number of joints
        learning_rate: Optimization learning rate
        n_iterations: Number of optimization iterations
        obstacle_cost_weight: Weight for obstacle avoidance
    """

    def __init__(
        self,
        n_joints: int,
        learning_rate: float = 0.1,
        n_iterations: int = 100,
        obstacle_cost_weight: float = 10.0,
    ):
        super().__init__(n_joints)

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.obstacle_cost_weight = obstacle_cost_weight

    def plan(
        self,
        start: Tensor,
        goal: Tensor,
        obstacles: Optional[Tensor] = None,
    ) -> Trajectory:
        """
        Plan trajectory using CHOMP.

        Args:
            start: Start configuration [n_joints]
            goal: Goal configuration [n_joints]
            obstacles: Obstacle positions [n_obstacles, 3]

        Returns:
            Optimized trajectory
        """
        T = 50
        trajectory = torch.linspace(0, 1, T).unsqueeze(-1) * (goal - start) + start

        trajectory = trajectory.requires_grad_(True)

        for _ in range(self.n_iterations):
            cost = self._compute_cost(trajectory, start, goal, obstacles)

            grad = torch.autograd.grad(cost, trajectory, create_graph=True)[0]

            with torch.no_grad():
                trajectory = trajectory - self.learning_rate * grad

                trajectory[0] = start
                trajectory[-1] = goal

                trajectory = trajectory.requires_grad_(True)

        points = []
        for i, config in enumerate(trajectory.detach()):
            point = TrajectoryPoint(
                time=float(i) * 0.01,
                joint_position=config.clone(),
                joint_velocity=torch.zeros_like(config),
            )
            points.append(point)

        return Trajectory(points=points, dt=0.01, interpolation="cubic")

    def _compute_cost(
        self,
        trajectory: Tensor,
        start: Tensor,
        goal: Tensor,
        obstacles: Optional[Tensor],
    ) -> Tensor:
        """Compute CHOMP cost function."""
        smoothness_cost = self._smoothness_cost(trajectory)

        obstacle_cost = torch.tensor(0.0, device=trajectory.device)
        if obstacles is not None:
            obstacle_cost = self._obstacle_cost(trajectory, obstacles)

        goal_cost = torch.sum((trajectory[-1] - goal) ** 2)

        return (
            smoothness_cost
            + self.obstacle_cost_weight * obstacle_cost
            + 10.0 * goal_cost
        )

    def _smoothness_cost(self, trajectory: Tensor) -> Tensor:
        """Compute trajectory smoothness cost."""
        dt = 1.0 / (trajectory.shape[0] - 1)

        velocity = (trajectory[1:] - trajectory[:-1]) / dt
        acceleration = (velocity[1:] - velocity[:-1]) / dt

        return torch.sum(acceleration**2) * dt

    def _obstacle_cost(
        self,
        trajectory: Tensor,
        obstacles: Tensor,
    ) -> Tensor:
        """Compute obstacle avoidance cost."""
        cost = torch.tensor(0.0, device=trajectory.device)

        for config in trajectory:
            for obs in obstacles:
                dist = torch.norm(config[:3] - obs)
                if dist < 1.0:
                    cost += (1.0 - dist) ** 2

        return cost


class TimeOptimalProfile(nn.Module):
    """
    Time-optimal trajectory profile generation.

    Computes minimum-time trajectory respecting velocity/acceleration
    and torque limits.

    Args:
        n_joints: Number of joints
        velocity_limits: Max velocities [n_joints]
        acceleration_limits: Max accelerations [n_joints]
    """

    def __init__(
        self,
        n_joints: int,
        velocity_limits: Tensor,
        acceleration_limits: Tensor,
    ):
        super().__init__()
        self.n_joints = n_joints

        self.register_buffer("velocity_limits", velocity_limits)
        self.register_buffer("acceleration_limits", acceleration_limits)

    def compute_profile(
        self,
        path: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute time-optimal velocity profile.

        Args:
            path: Path waypoints [n_waypoints, n_joints]

        Returns:
            Tuple of (times, velocities) at each waypoint
        """
        n_waypoints = path.shape[0]
        distances = torch.norm(path[1:] - path[:-1], dim=1)

        max_velocities = self.velocity_limits.unsqueeze(0).expand(n_waypoints - 1, -1)

        v_max = torch.min(
            torch.sqrt(2 * distances.unsqueeze(-1) * self.acceleration_limits),
            max_velocities,
        )

        times = torch.zeros(n_waypoints)
        velocities = torch.zeros(n_waypoints - 1, self.n_joints)

        for i in range(1, n_waypoints):
            segment_time = distances[i - 1] / torch.clamp(
                velocities[i - 1].max(), min=1e-6
            )
            times[i] = times[i - 1] + segment_time

            if i < n_waypoints - 1:
                velocities[i] = torch.minimum(
                    velocities[i - 1] + self.acceleration_limits * segment_time,
                    v_max[i],
                )

        return times, velocities

    def forward(self, path: Tensor) -> Trajectory:
        """Generate time-optimal trajectory."""
        times, velocities = self.compute_profile(path)

        points = []
        for i in range(path.shape[0]):
            point = TrajectoryPoint(
                time=times[i].item(),
                joint_position=path[i].clone(),
                joint_velocity=velocities[i].clone()
                if i < velocities.shape[0]
                else torch.zeros(self.n_joints),
            )
            points.append(point)

        return Trajectory(points=points, dt=0.01, interpolation="cubic")


class JacobianInverse(nn.Module):
    """
    Jacobian-based inverse kinematics solvers.

    Implements various IK algorithms:
    - Pseudo-inverse Jacobian
    - Damped Least Squares (DLS)
    - Selectively damped least squares (SDLS)
    """

    def __init__(
        self,
        n_joints: int,
        task_dim: int = 3,
        damping: float = 0.1,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.task_dim = task_dim
        self.damping = damping

    def compute_jacobian(
        self,
        joint_positions: Tensor,
        forward_kinematics_fn: Callable,
    ) -> Tensor:
        """
        Compute task-space Jacobian.

        Args:
            joint_positions: Current joint angles [n_joints]
            forward_kinematics_fn: Function that computes FK

        Returns:
            Jacobian [task_dim, n_joints]
        """
        joint_positions = joint_positions.requires_grad_(True)

        ee_position = forward_kinematics_fn(joint_positions)

        jacobian = torch.zeros(
            self.task_dim, self.n_joints, device=joint_positions.device
        )

        for j in range(self.n_joints):
            grad = torch.autograd.grad(
                ee_position,
                joint_positions,
                grad_outputs=torch.eye(self.task_dim, device=joint_positions.device),
                create_graph=True,
            )[0]
            jacobian[:, j] = grad[:, j]

        return jacobian

    def solve_pseudo_inverse(
        self,
        delta_task: Tensor,
        jacobian: Tensor,
    ) -> Tensor:
        """
        Solve using pseudo-inverse.

        q_delta = J^+ * delta_x
        """
        return torch.linalg.lstsq(jacobian, delta_task).solution

    def solve_damped_ls(
        self,
        delta_task: Tensor,
        jacobian: Tensor,
    ) -> Tensor:
        """
        Solve using Damped Least Squares.

        q_delta = J^T * (J * J^T + lambda^2 * I)^-1 * delta_x
        """
        jjt = jacobian @ jacobian.T
        regularization = self.damping**2 * torch.eye(
            self.task_dim, device=jacobian.device
        )

        try:
            inv_term = torch.inverse(jjt + regularization)
        except:
            inv_term = torch.linalg.pinv(jjt + regularization)

        return jacobian.T @ inv_term @ delta_task

    def solve_sdls(
        self,
        delta_task: Tensor,
        jacobian: Tensor,
        lambda_min: float = 0.1,
    ) -> Tensor:
        """
        Selectively Damped Least Squares.

        Adaptively adjusts damping based on singular values.
        """
        u, s, vh = torch.linalg.svd(jacobian, full_matrices=False)

        task_norm = torch.norm(delta_task)

        threshold = lambda_min * task_norm

        s_inv = torch.where(s > threshold, 1.0 / s, s / (lambda_min**2 + s**2))

        delta_joint = vh.T @ (s_inv.unsqueeze(-1) * (u.T @ delta_task))

        return delta_joint

    def step(
        self,
        joint_positions: Tensor,
        target_position: Tensor,
        method: str = "dls",
    ) -> Tensor:
        """
        Single IK iteration step.

        Args:
            joint_positions: Current joint positions [n_joints]
            target_position: Target task position [task_dim]
            method: "pseudo_inverse", "dls", or "sdls"

        Returns:
            Joint position update
        """
        jacobian = self._compute_numerical_jacobian(joint_positions, target_position)

        current_ee = self._compute_fk(joint_positions)
        delta_task = target_position - current_ee

        if method == "pseudo_inverse":
            delta_joint = self.solve_pseudo_inverse(delta_task, jacobian)
        elif method == "sdls":
            delta_joint = self.solve_sdls(delta_task, jacobian)
        else:
            delta_joint = self.solve_damped_ls(delta_task, jacobian)

        return joint_positions + delta_joint

    def _compute_numerical_jacobian(
        self,
        joint_positions: Tensor,
        target_position: Tensor,
        epsilon: float = 1e-6,
    ) -> Tensor:
        """Compute Jacobian numerically."""
        fk_current = self._compute_fk(joint_positions)

        jacobian = torch.zeros(
            self.task_dim, self.n_joints, device=joint_positions.device
        )

        for i in range(self.n_joints):
            delta = torch.zeros_like(joint_positions)
            delta[i] = epsilon

            fk_delta = self._compute_fk(joint_positions + delta)

            jacobian[:, i] = (fk_delta - fk_current) / epsilon

        return jacobian

    def _compute_fk(self, joint_positions: Tensor) -> Tensor:
        """Simple FK (override with actual FK)."""
        return joint_positions[: self.task_dim]

    def solve(
        self,
        start: Tensor,
        target: Tensor,
        n_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> Tensor:
        """
        Solve full IK problem.

        Args:
            start: Starting joint positions
            target: Target task position
            n_iterations: Max iterations
            tolerance: Convergence tolerance

        Returns:
            Solution joint positions
        """
        q = start.clone()

        for _ in range(n_iterations):
            q = self.step(q, target)

            current_ee = self._compute_fk(q)
            error = torch.norm(target - current_ee)

            if error < tolerance:
                break

        return q


class TrajectoryOptimizer(nn.Module):
    """
    General trajectory optimization.

    Refines trajectories using gradient-based optimization
    with various cost functions.
    """

    def __init__(
        self,
        n_joints: int,
        n_points: int = 50,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.n_points = n_points

    def optimize(
        self,
        initial_trajectory: Trajectory,
        cost_fn: Callable[[Tensor], Tensor],
        n_iterations: int = 100,
    ) -> Trajectory:
        """Optimize trajectory to minimize cost."""
        traj_tensor = torch.zeros(self.n_points, self.n_joints)

        for i, point in enumerate(initial_trajectory.points):
            if point.joint_position is not None:
                traj_tensor[i] = point.joint_position

        traj_tensor = traj_tensor.requires_grad_(True)
        optimizer = torch.optim.Adam([traj_tensor], lr=0.01)

        for _ in range(n_iterations):
            cost = cost_fn(traj_tensor)

            optimizer.zero_grad()
            cost.backward()
            optimizer.grad()

            with torch.no_grad():
                traj_tensor[0] = initial_trajectory.points[0].joint_position
                traj_tensor[-1] = initial_trajectory.points[-1].joint_position

        points = []
        for i in range(self.n_points):
            point = TrajectoryPoint(
                time=float(i) / self.n_points,
                joint_position=traj_tensor[i].detach().clone(),
            )
            points.append(point)

        return Trajectory(points=points, dt=0.01)
