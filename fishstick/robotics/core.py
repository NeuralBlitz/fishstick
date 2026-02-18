"""
Core types and utilities for robotics.

Defines fundamental data structures for robot state, control,
dynamics, and trajectory representation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable, Dict, Any
from enum import Enum
import torch
from torch import Tensor
import numpy as np


class ControlMode(Enum):
    """Control mode for robot operation."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"
    HYBRID = "hybrid"


class PlanningAlgorithm(Enum):
    """Motion planning algorithm types."""

    RRT_STAR = "rrt_star"
    PRM = "prm"
    CHOMP = "chomp"
    STOMP = "stomp"
    MPNet = "mpnet"


@dataclass
class JointState:
    """
    Robot joint state representation.

    Represents the state of robot joints in configuration space.
    """

    position: Tensor  # Joint positions [n_joints]
    velocity: Tensor  # Joint velocities [n_joints]
    acceleration: Optional[Tensor] = None  # Joint accelerations [n_joints]
    torque: Optional[Tensor] = None  # Joint torques [n_joints]
    timestamp: Optional[float] = None

    @property
    def n_joints(self) -> int:
        return self.position.shape[-1]

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return self.position.cpu().numpy(), self.velocity.cpu().numpy()

    @classmethod
    def from_numpy(
        cls,
        position: np.ndarray,
        velocity: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> "JointState":
        """Create from numpy arrays."""
        return cls(
            position=torch.FloatTensor(position),
            velocity=torch.FloatTensor(velocity),
            timestamp=timestamp,
        )

    def zero_velocity_like(self) -> "JointState":
        """Create a copy with zero velocity."""
        return JointState(
            position=self.position.clone(),
            velocity=torch.zeros_like(self.velocity),
            timestamp=self.timestamp,
        )


@dataclass
class TaskState:
    """
    Robot task-space state.

    Represents end-effector pose and velocity in task/operational space.
    """

    position: Tensor  # End-effector position [3]
    orientation: Tensor  # End-effector orientation (quaternion) [4]
    linear_velocity: Tensor  # Linear velocity [3]
    angular_velocity: Tensor  # Angular velocity [3]
    timestamp: Optional[float] = None

    @property
    def pose(self) -> Tensor:
        """Full SE(3) pose as [position, quaternion]."""
        return torch.cat([self.position, self.orientation])

    @property
    def twist(self) -> Tensor:
        """Full task-space velocity (6D)."""
        return torch.cat([self.linear_velocity, self.angular_velocity])

    def to_homogeneous(self) -> Tensor:
        """Convert to 4x4 homogeneous transformation matrix."""
        pos = self.position
        q = self.orientation

        w, x, y, z = q[0], q[1], q[2], q[3]
        R = torch.tensor(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ],
            device=q.device,
            dtype=q.dtype,
        )

        T = torch.eye(4, device=q.device, dtype=q.dtype)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T


@dataclass
class ControlCommand:
    """
    Robot control command.

    Encapsulates a control input to the robot.
    """

    mode: ControlMode
    target: Tensor  # Target based on mode (position/velocity/torque/impedance)
    stiffness: Optional[Tensor] = None  # Impedance stiffness [6]
    damping: Optional[Tensor] = None  # Impedance damping [6]
    duration: float = 0.01  # Command duration in seconds

    @property
    def n_joints(self) -> int:
        return self.target.shape[-1]


@dataclass
class TrajectoryPoint:
    """
    A single point along a trajectory.

    Contains both the desired state and timing information.
    """

    time: float
    joint_position: Optional[Tensor] = None
    joint_velocity: Optional[Tensor] = None
    task_position: Optional[Tensor] = None
    task_orientation: Optional[Tensor] = None
    acceleration: Optional[Tensor] = None


@dataclass
class Trajectory:
    """
    Robot trajectory representation.

    A sequence of trajectory points defining a motion path.
    """

    points: List[TrajectoryPoint]
    dt: float = 0.01
    interpolation: str = "cubic"  # "linear", "cubic", "quintic"

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def duration(self) -> float:
        if not self.points:
            return 0.0
        return self.points[-1].time - self.points[0].time

    def sample(self, t: float) -> Tuple[Tensor, Tensor]:
        """
        Sample trajectory at time t.

        Returns:
            Tuple of (position, velocity) at time t
        """
        if not self.points:
            raise ValueError("Empty trajectory")

        times = torch.tensor([p.time for p in self.points])

        if t <= times[0]:
            p = self.points[0].joint_position
            v = (
                self.points[0].joint_velocity
                if self.points[0].joint_velocity is not None
                else torch.zeros_like(p)
            )
            return p, v

        if t >= times[-1]:
            p = self.points[-1].joint_position
            v = (
                self.points[-1].joint_velocity
                if self.points[-1].joint_velocity is not None
                else torch.zeros_like(p)
            )
            return p, v

        idx = torch.searchsorted(times, t)

        if self.interpolation == "linear":
            t0, t1 = times[idx - 1], times[idx]
            alpha = (t - t0) / (t1 - t0)
            p0 = self.points[idx - 1].joint_position
            p1 = self.points[idx].joint_position
            position = (1 - alpha) * p0 + alpha * p1
            velocity = (p1 - p0) / (t1 - t0)
        else:
            p0 = self.points[idx - 1].joint_position
            p1 = self.points[idx].joint_position
            v0 = (
                self.points[idx - 1].joint_velocity
                if self.points[idx - 1].joint_velocity is not None
                else torch.zeros_like(p0)
            )
            v1 = (
                self.points[idx].joint_velocity
                if self.points[idx].joint_velocity is not None
                else torch.zeros_like(p1)
            )
            t0, t1 = times[idx - 1], times[idx]
            alpha = (t - t0) / (t1 - t0)

            position = (
                (2 * alpha**3 - 3 * alpha**2 + 1) * p0
                + (-2 * alpha**3 + 3 * alpha**2) * p1
                + (alpha**3 - 2 * alpha**2 + alpha) * (t1 - t0) * v0
                + (alpha**3 - alpha**2) * (t1 - t0) * v1
            )

            velocity = (
                (6 * alpha**2 - 6 * alpha) * (p1 - p0) / (t1 - t0)
                + (3 * alpha**2 - 4 * alpha + 1) * v0
                + (3 * alpha**2 - 2 * alpha) * v1
            )

        return position, velocity


@dataclass
class RobotModel:
    """
    Robot model parameters.

    Contains kinematic and dynamic parameters of a robot.
    """

    n_joints: int
    n_links: int
    link_lengths: Tensor  # [n_links]
    link_masses: Tensor  # [n_links]
    link_inertia: Tensor  # [n_links, 3, 3] COM inertia tensors
    joint_limits: Optional[Tensor] = None  # [n_joints, 2] (min, max)
    velocity_limits: Optional[Tensor] = None  # [n_joints]
    acceleration_limits: Optional[Tensor] = None  # [n_joints]
    torque_limits: Optional[Tensor] = None  # [n_joints]

    @property
    def total_mass(self) -> Tensor:
        return self.link_masses.sum()

    def is_within_limits(self, state: JointState) -> Tuple[bool, Dict]:
        """Check if state is within physical limits."""
        violations = {}
        valid = True

        if self.joint_limits is not None:
            below_min = (state.position < self.joint_limits[:, 0]).any()
            above_max = (state.position > self.joint_limits[:, 1]).any()
            if below_min or above_max:
                violations["position"] = True
                valid = False

        if self.velocity_limits is not None:
            exceeds = torch.abs(state.velocity) > self.velocity_limits
            if exceeds.any():
                violations["velocity"] = True
                valid = False

        return valid, violations


@dataclass
class ControlLimits:
    """Control input limits."""

    min_output: Optional[Tensor] = None
    max_output: Optional[Tensor] = None
    max_rate: Optional[Tensor] = None  # Max change per step

    def apply(self, u: Tensor) -> Tensor:
        """Apply saturation limits to control input."""
        if self.min_output is not None:
            u = torch.maximum(u, self.min_output)
        if self.max_output is not None:
            u = torch.minimum(u, self.max_output)
        return u


@dataclass
class DynamicsParameters:
    """Parameters for robot dynamics model."""

    gravity: Tensor = field(default_factory=lambda: torch.tensor([0, 0, -9.81]))
    friction_model: str = "viscous"  # "viscous", "coulomb", "combined"
    coulomb_friction: Optional[Tensor] = None
    viscous_friction: Optional[Tensor] = None
    damping: Optional[Tensor] = None


@dataclass
class Observation:
    """
    Robot observation for RL or control.

    Full state observation including proprioceptive and exteroceptive data.
    """

    joint_state: JointState
    task_state: Optional[TaskState] = None
    external_wrench: Optional[Tensor] = None  # [6] force/torque
    contact: Optional[Tensor] = None  # Contact flags
    timestamp: float = 0.0

    @property
    def proprioceptive(self) -> Tensor:
        """Proprioceptive sensing (joint positions, velocities)."""
        return torch.cat([self.joint_state.position, self.joint_state.velocity])

    def to_vector(self) -> Tensor:
        """Flatten all observations into a single vector."""
        vectors = [self.proprioceptive]
        if self.task_state is not None:
            vectors.extend(
                [
                    self.task_state.position,
                    self.task_state.orientation,
                    self.task_state.linear_velocity,
                    self.task_state.angular_velocity,
                ]
            )
        if self.external_wrench is not None:
            vectors.append(self.external_wrench)
        return torch.cat(vectors)


@dataclass
class Action:
    """
    Robot action for RL or control.

    Action can be in joint space or task space.
    """

    joint_torques: Optional[Tensor] = None
    target_position: Optional[Tensor] = None
    target_velocity: Optional[Tensor] = None
    impedance_targets: Optional[Tensor] = None  # [position, orientation]

    @property
    def is_joint_space(self) -> bool:
        return self.joint_torques is not None

    @property
    def is_task_space(self) -> bool:
        return self.target_position is not None
