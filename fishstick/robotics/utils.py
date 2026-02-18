"""
Utility functions for robotics.

Helpers for:
- Trajectory processing
- Robot model utilities
- Visualization helpers
- Common transformations
"""

from typing import Optional, Tuple, List, Callable
import torch
from torch import Tensor
import numpy as np

from .core import JointState, TaskState, Trajectory, TrajectoryPoint, ControlCommand


def compute_trajectory_velocity(
    trajectory: Trajectory,
    filter_window: int = 5,
) -> Trajectory:
    """
    Compute velocities from trajectory positions.

    Args:
        trajectory: Position trajectory
        filter_window: Smoothing window size

    Returns:
        Trajectory with velocities
    """
    positions = torch.stack([p.joint_position for p in trajectory.points])
    times = torch.tensor([p.time for p in trajectory.points])

    velocities = torch.zeros_like(positions)

    for i in range(1, len(positions) - 1):
        dt = times[i + 1] - times[i - 1]
        velocities[i] = (positions[i + 1] - positions[i - 1]) / dt

    velocities[0] = velocities[1]
    velocities[-1] = velocities[-2]

    if filter_window > 1:
        kernel = torch.ones(filter_window) / filter_window
        for j in range(positions.shape[1]):
            velocities[:, j] = torch.conv1d(
                velocities[:, j].unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=filter_window // 2,
            ).squeeze()

    new_points = []
    for i, point in enumerate(trajectory.points):
        new_point = TrajectoryPoint(
            time=point.time,
            joint_position=point.joint_position,
            joint_velocity=velocities[i],
        )
        new_points.append(new_point)

    return Trajectory(
        points=new_points,
        dt=trajectory.dt,
        interpolation=trajectory.interpolation,
    )


def smooth_trajectory(
    trajectory: Trajectory,
    alpha: float = 0.1,
) -> Trajectory:
    """
    Apply exponential smoothing to trajectory.

    Args:
        trajectory: Input trajectory
        alpha: Smoothing factor (0-1)

    Returns:
        Smoothed trajectory
    """
    positions = torch.stack([p.joint_position for p in trajectory.points])

    smoothed = positions.clone()

    for i in range(1, len(positions)):
        smoothed[i] = alpha * positions[i] + (1 - alpha) * smoothed[i - 1]

    new_points = []
    for i, point in enumerate(trajectory.points):
        new_point = TrajectoryPoint(
            time=point.time,
            joint_position=smoothed[i],
            joint_velocity=point.joint_velocity,
        )
        new_points.append(new_point)

    return Trajectory(points=new_points, dt=trajectory.dt)


def resample_trajectory(
    trajectory: Trajectory,
    target_n_points: int,
) -> Trajectory:
    """
    Resample trajectory to different number of points.

    Args:
        trajectory: Input trajectory
        target_n_points: Target number of points

    Returns:
        Resampled trajectory
    """
    positions = torch.stack([p.joint_position for p in trajectory.points])
    times = torch.tensor([p.time for p in trajectory.points])

    original_times = times
    original_positions = positions

    new_times = torch.linspace(times[0], times[-1], target_n_points)

    new_positions = torch.zeros(target_n_points, positions.shape[1])

    for j in range(positions.shape[1]):
        new_positions[:, j] = torch.interp(
            new_times, original_times, original_positions[:, j]
        )

    new_points = []
    for i in range(target_n_points):
        point = TrajectoryPoint(
            time=new_times[i].item(),
            joint_position=new_positions[i],
        )
        new_points.append(point)

    return Trajectory(points=new_points, dt=0.01)


def compute_trajectory_length(trajectory: Trajectory) -> float:
    """
    Compute total arc length of trajectory.

    Args:
        trajectory: Input trajectory

    Returns:
        Total length
    """
    positions = torch.stack([p.joint_position for p in trajectory.points])

    diffs = positions[1:] - positions[:-1]
    segment_lengths = torch.norm(diffs, dim=1)

    return segment_lengths.sum().item()


def compute_jerk(trajectory: Trajectory) -> Tensor:
    """
    Compute jerk (derivative of acceleration) along trajectory.

    Args:
        trajectory: Input trajectory

    Returns:
        Jerk values
    """
    positions = torch.stack([p.joint_position for p in trajectory.points])
    times = torch.tensor([p.time for p in trajectory.points])

    dt = times[1] - times[0]

    velocity = torch.gradient(positions, dim=0, spacing=dt)[0]
    acceleration = torch.gradient(velocity, dim=0, spacing=dt)[0]
    jerk = torch.gradient(acceleration, dim=0, spacing=dt)[0]

    return jerk


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """
    Multiply two quaternions.

    Args:
        q1: First quaternion [4] (w, x, y, z)
        q2: Second quaternion [4]

    Returns:
        Product quaternion [4]
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.tensor([w, x, y, z], device=q1.device, dtype=q1.dtype)


def quat_conjugate(q: Tensor) -> Tensor:
    """
    Compute quaternion conjugate.

    Args:
        q: Quaternion [4]

    Returns:
        Conjugate [4]
    """
    return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device, dtype=q.dtype)


def quat_inverse(q: Tensor) -> Tensor:
    """
    Compute quaternion inverse.

    Args:
        q: Quaternion [4]

    Returns:
        Inverse [4]
    """
    conj = quat_conjugate(q)
    norm_sq = torch.sum(q**2)
    return conj / norm_sq


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """
    Rotate vector by quaternion.

    Args:
        q: Rotation quaternion [4]
        v: Vector to rotate [3]

    Returns:
        Rotated vector [3]
    """
    q_conj = quat_conjugate(q)

    v_quat = torch.tensor([0, v[0], v[1], v[2]], device=q.device, dtype=q.dtype)

    result = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return result[1:]


def quat_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """
    Create quaternion from axis-angle representation.

    Args:
        axis: Rotation axis [3]
        angle: Rotation angle (radians)

    Returns:
        Quaternion [4]
    """
    half_angle = angle / 2
    s = torch.sin(half_angle)

    return torch.tensor(
        [
            torch.cos(half_angle),
            axis[0] * s,
            axis[1] * s,
            axis[2] * s,
        ],
        device=axis.device,
        dtype=axis.dtype,
    )


def quat_to_euler(q: Tensor) -> Tensor:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q: Quaternion [4]

    Returns:
        Euler angles [3]
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = torch.asin(2 * (w * y - z * x))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw], device=q.device, dtype=q.dtype)


def compute_distance_joint_space(q1: Tensor, q2: Tensor) -> float:
    """
    Compute distance in joint space.

    Args:
        q1: First configuration
        q2: Second configuration

    Returns:
        Distance
    """
    return torch.norm(q1 - q2).item()


def compute_distance_task_space(
    pose1: Tuple[Tensor, Tensor],
    pose2: Tuple[Tensor, Tensor],
    orientation_weight: float = 0.1,
) -> float:
    """
    Compute distance in task space.

    Args:
        pose1: Tuple of (position, orientation)
        pose2: Tuple of (position, orientation)
        orientation_weight: Weight for orientation error

    Returns:
        Task space distance
    """
    pos_dist = torch.norm(pose1[0] - pose2[0])

    q1 = pose1[1] / torch.norm(pose1[1])
    q2 = pose2[1] / torch.norm(pose2[1])

    dot = torch.clamp(torch.dot(q1, q2), -1, 1)
    ori_dist = 2 * torch.acos(torch.abs(dot))

    return (pos_dist + orientation_weight * ori_dist).item()


def interpolate_joint_space(
    q_start: Tensor,
    q_end: Tensor,
    alpha: float,
) -> Tensor:
    """
    Interpolate linearly in joint space.

    Args:
        q_start: Start configuration
        q_end: End configuration
        alpha: Interpolation factor [0, 1]

    Returns:
        Interpolated configuration
    """
    return (1 - alpha) * q_start + alpha * q_end


def interpolate_task_space(
    pose1_start: Tuple[Tensor, Tensor],
    pose1_end: Tuple[Tensor, Tensor],
    pose2_start: Tuple[Tensor, Tensor],
    pose2_end: Tuple[Tensor, Tensor],
    alpha: float,
) -> Tuple[Tensor, Tensor]:
    """
    Interpolate in task space with SLERP for orientation.

    Args:
        pose1_start: Start pose at alpha=0
        pose1_end: End pose at alpha=0
        pose2_start: Start pose at alpha=1
        pose2_end: End pose at alpha=1
        alpha: Interpolation factor

    Returns:
        Interpolated pose (position, orientation)
    """
    pos_start = (1 - alpha) * pose1_start[0] + alpha * pose1_end[0]
    pos_end = (1 - alpha) * pose2_start[0] + alpha * pose2_end[0]

    position = (1 - alpha) * pos_start + alpha * pos_end

    q1 = pose1_start[1]
    q2 = pose2_end[1]

    dot = torch.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    dot = torch.clamp(dot, -1, 1)
    theta = torch.acos(dot)

    if torch.abs(theta) < 1e-6:
        orientation = q1
    else:
        s = torch.sin(theta)
        a = torch.sin((1 - alpha) * theta) / s
        b = torch.sin(alpha * theta) / s
        orientation = a * q1 + b * q2

    return position, orientation / torch.norm(orientation)


def wrap_to_limits(value: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """
    Wrap value to be within limits.

    Args:
        value: Input value
        low: Lower limit
        high: Upper limit

    Returns:
        Wrapped value
    """
    range_size = high - low
    return low + ((value - low) % range_size + range_size) % range_size


def apply_velocity_smoothing(
    velocity: Tensor,
    max_acceleration: float,
    dt: float,
) -> Tensor:
    """
    Apply velocity smoothing with acceleration limits.

    Args:
        velocity: Raw velocity command
        max_acceleration: Maximum allowed acceleration
        dt: Time step

    Returns:
        Smoothed velocity
    """
    smoothed = velocity.clone()

    for i in range(1, len(velocity)):
        delta = velocity[i] - smoothed[i - 1]
        delta_clamped = torch.clamp(
            delta, -max_acceleration * dt, max_acceleration * dt
        )
        smoothed[i] = smoothed[i - 1] + delta_clamped

    return smoothed


def compute_manipulability(jacobian: Tensor) -> float:
    """
    Compute manipulability measure from Jacobian.

    Args:
        jacobian: Jacobian matrix [6, n]

    Returns:
        Manipulability measure (0 = singular)
    """
    jjt = jacobian @ jacobian.T

    eigenvalues = torch.linalg.eigvalsh(jjt)

    return torch.sqrt(eigenvalues.prod()).item()


def check_collision(
    config: Tensor,
    obstacles: Tensor,
    link_positions: Optional[Tensor] = None,
    radius: float = 0.05,
) -> bool:
    """
    Simple collision check.

    Args:
        config: Joint configuration
        obstacles: Obstacle positions
        link_positions: Precomputed link positions
        radius: Collision radius

    Returns:
        True if collision detected
    """
    if link_positions is None:
        return False

    for link_pos in link_positions:
        for obs in obstacles:
            if torch.norm(link_pos - obs) < radius:
                return True

    return False


def joint_limits_penalty(
    config: Tensor,
    joint_limits: Tensor,
    penalty_scale: float = 1.0,
) -> Tensor:
    """
    Compute penalty for approaching joint limits.

    Args:
        config: Current configuration
        joint_limits: [n_joints, 2] (min, max)
        penalty_scale: Penalty scaling factor

    Returns:
        Penalty value
    """
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    range_size = upper - lower

    normalized = (config - lower) / range_size

    margin = 0.1
    violation = torch.zeros_like(config)

    violation = torch.where(
        normalized < margin, ((margin - normalized) / margin) ** 2, violation
    )

    violation = torch.where(
        normalized > (1 - margin),
        ((normalized - (1 - margin)) / margin) ** 2,
        violation,
    )

    return penalty_scale * violation.sum()


def normalize_joint_positions(
    config: Tensor,
    joint_limits: Tensor,
) -> Tensor:
    """
    Normalize joint positions to [0, 1].

    Args:
        config: Joint configuration
        joint_limits: [n_joints, 2]

    Returns:
        Normalized configuration
    """
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    return (config - lower) / (upper - lower)


def denormalize_joint_positions(
    normalized: Tensor,
    joint_limits: Tensor,
) -> Tensor:
    """
    Denormalize joint positions from [0, 1] to actual limits.

    Args:
        normalized: Normalized configuration
        joint_limits: [n_joints, 2]

    Returns:
        Denormalized configuration
    """
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    return normalized * (upper - lower) + lower
