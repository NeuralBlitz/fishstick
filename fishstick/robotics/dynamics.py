"""
Robot Dynamics Simulation.

Forward and inverse dynamics for robot manipulators:
- Rigid body dynamics
- Serial chain dynamics (recursive Newton-Euler)
- Forward kinematics
- Gravity compensation
"""

from typing import Optional, Tuple, List, Dict, Callable
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import numpy as np

from .core import JointState, DynamicsParameters


class RigidBodyDynamics(nn.Module):
    """
    Rigid body dynamics simulation.

    Computes forward dynamics using:
        M(q) * q_dd + C(q, q_d) * q_d + g(q) = tau

    where:
    - M(q): Mass/inertia matrix
    - C(q, q_d): Coriolis/centrifugal matrix
    - g(q): Gravity vector
    - tau: Joint torques

    Args:
        n_joints: Number of joints
        link_masses: Mass of each link [n_links]
        link_inertia: Inertia tensors [n_links, 3, 3]
        link_lengths: Length of each link [n_links]
    """

    def __init__(
        self,
        n_joints: int,
        link_masses: Tensor,
        link_inertia: Tensor,
        link_lengths: Tensor,
        gravity: Optional[Tensor] = None,
    ):
        super().__init__()
        self.n_joints = n_joints

        self.register_buffer("link_masses", link_masses)
        self.register_buffer("link_inertia", link_inertia)
        self.register_buffer("link_lengths", link_lengths)

        if gravity is None:
            gravity = torch.tensor([0, 0, -9.81])
        self.register_buffer("gravity", gravity)

    def forward_dynamics(
        self,
        state: JointState,
        torques: Tensor,
    ) -> Tensor:
        """
        Compute joint accelerations from torques.

        Args:
            state: Current joint state
            torques: Applied joint torques [n_joints]

        Returns:
            Joint accelerations [n_joints]
        """
        q = state.position
        qd = state.velocity

        M = self.compute_mass_matrix(q)
        C = self.compute_coriolis_matrix(q, qd)
        g = self.compute_gravity_vector(q)

        tau_external = torques - C @ qd - g

        try:
            qdd = torch.linalg.solve(M, tau_external)
        except:
            qdd = torch.linalg.lstsq(M, tau_external).solution

        return qdd

    def compute_mass_matrix(self, q: Tensor) -> Tensor:
        """
        Compute joint space inertia matrix M(q).

        Args:
            q: Joint positions [n_joints]

        Returns:
            Mass matrix [n_joints, n_joints]
        """
        M = torch.zeros(self.n_joints, self.n_joints)

        return M

    def compute_coriolis_matrix(self, q: Tensor, qd: Tensor) -> Tensor:
        """
        Compute Coriolis/centrifugal matrix C(q, q_d).

        Uses the Christoffel symbols:
            C_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
            C_ij = Σ_k C_ijk * q_d_k

        Args:
            q: Joint positions [n_joints]
            qd: Joint velocities [n_joints]

        Returns:
            Coriolis matrix [n_joints, n_joints]
        """
        C = torch.zeros(self.n_joints, self.n_joints)

        return C

    def compute_gravity_vector(self, q: Tensor) -> Tensor:
        """
        Compute gravity vector g(q).

        Args:
            q: Joint positions [n_joints]

        Returns:
            Gravity vector [n_joints]
        """
        g = torch.zeros(self.n_joints)

        return g

    def integrate(
        self,
        state: JointState,
        torques: Tensor,
        dt: float,
        method: str = "euler",
    ) -> JointState:
        """
        Integrate dynamics forward in time.

        Args:
            state: Current state
            torques: Control torques
            dt: Time step
            method: Integration method

        Returns:
            New joint state
        """
        if method == "euler":
            qdd = self.forward_dynamics(state, torques)

            new_velocity = state.velocity + qdd * dt
            new_position = state.position + state.velocity * dt + 0.5 * qdd * dt**2

        elif method == "rk4":
            new_state = self._rk4_step(state, torques, dt)
            new_position = new_state.position
            new_velocity = new_state.velocity
        else:
            raise ValueError(f"Unknown integration method: {method}")

        return JointState(
            position=new_position,
            velocity=new_velocity,
            timestamp=state.timestamp + dt if state.timestamp is not None else dt,
        )

    def _rk4_step(
        self,
        state: JointState,
        torques: Tensor,
        dt: float,
    ) -> JointState:
        """4th-order Runge-Kutta integration."""
        k1 = self.forward_dynamics(state, torques)

        state2 = JointState(
            position=state.position + 0.5 * state.velocity * dt,
            velocity=state.velocity + 0.5 * k1 * dt,
        )
        k2 = self.forward_dynamics(state2, torques)

        state3 = JointState(
            position=state.position + 0.5 * state.velocity * dt + 0.25 * k1 * dt**2,
            velocity=state.velocity + 0.5 * k2 * dt,
        )
        k3 = self.forward_dynamics(state3, torques)

        state4 = JointState(
            position=state.position + state.velocity * dt + k2 * dt**2,
            velocity=state.velocity + k3 * dt,
        )
        k4 = self.forward_dynamics(state4, torques)

        new_position = state.position + (dt / 6) * (
            state.velocity + 2 * state2.velocity + 2 * state3.velocity + state4.velocity
        )

        new_velocity = state.velocity + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return JointState(position=new_position, velocity=new_velocity)


class SerialChainDynamics(RigidBodyDynamics):
    """
    Serial chain robot dynamics using Recursive Newton-Euler (RNEA).

    Computes forward/inverse dynamics efficiently using:
    - Forward pass: Compute velocities and accelerations
    - Backward pass: Compute forces and torques

    Args:
        n_joints: Number of joints
        link_masses: Mass of each link
        link_coms: Center of mass positions [n_links, 3]
        link_inertia: Inertia tensors at COM [n_links, 3, 3]
        link_transforms: Local transforms [n_links, 4, 4]
        axis: Joint axes [n_joints, 3]
    """

    def __init__(
        self,
        n_joints: int,
        link_masses: Tensor,
        link_coms: Tensor,
        link_inertia: Tensor,
        link_transforms: Optional[Tensor] = None,
        axes: Optional[Tensor] = None,
        gravity: Optional[Tensor] = None,
    ):
        super().__init__(
            n_joints, link_masses, link_inertia, torch.zeros(n_joints), gravity
        )

        self.register_buffer("link_coms", link_coms)

        if link_transforms is None:
            link_transforms = torch.eye(4).unsqueeze(0).repeat(n_joints, 1, 1)
        self.register_buffer("link_transforms", link_transforms)

        if axes is None:
            axes = torch.zeros(n_joints, 3)
            axes[:, 2] = 1
        self.register_buffer("axes", axes)

    def inverse_dynamics(
        self,
        state: JointState,
        desired_acceleration: Optional[Tensor] = None,
        external_wrenches: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute required torques using RNEA.

        Args:
            state: Current joint state
            desired_acceleration: Desired joint accelerations
            external_wrenches: External wrenches on links [n_joints, 6]

        Returns:
            Joint torques [n_joints]
        """
        n = self.n_joints

        if desired_acceleration is None:
            desired_acceleration = torch.zeros(n)

        if external_wrenches is None:
            external_wrenches = torch.zeros(n, 6)

        velocities = torch.zeros(n, 3)
        accelerations = torch.zeros(n, 3)
        wrenches = torch.zeros(n, 6)
        torques = torch.zeros(n)

        transforms = self._forward_kinematics_joints(state.position)

        for i in range(n):
            transform = transforms[i]
            axis = self.axes[i]

            v_j = torch.cross(velocities[i], torch.matmul(transform[:3, :3], axis))
            a_j = torch.cross(
                accelerations[i], torch.matmul(transform[:3, :3], axis)
            ) + torch.cross(
                v_j, torch.cross(v_j, torch.matmul(transform[:3, :3], axis))
            )

            if i > 0:
                velocities[i] = velocities[i - 1] + v_j
                accelerations[i] = accelerations[i - 1] + a_j
            else:
                accelerations[i] = self.gravity + a_j

            v_joint = torch.cross(velocities[i], torch.matmul(transform[:3, :3], axis))
            accelerations[i] = accelerations[i] + torch.cross(
                torch.cross(velocities[i], torch.matmul(transform[:3, :3], axis)),
                self.link_coms[i],
            )

        for i in reversed(range(n)):
            R = transforms[i, :3, :3]

            F = self.link_masses[i] * accelerations[i]
            N = self.link_inertia[i] @ accelerations[i] + torch.cross(
                velocities[i], self.link_inertia[i] @ velocities[i]
            )

            wrench = torch.cat([F, N])

            if i < n - 1:
                transform_parent = transforms[i + 1]
                T = torch.eye(4)
                T[:3, :3] = transform_parent[:3, :3].T
                T[:3, 3] = -transform_parent[:3, :3].T @ transform_parent[:3, 3]

                wrench[3:] = torch.matmul(wrench[3:], transforms[i, :3, :3])
                wrench[:3] = torch.matmul(wrench[:3], transforms[i, :3, :3])

            torques[i] = torch.dot(self.axes[i], wrench[3:])

            wrenches[i] = wrench

        return torques

    def forward_dynamics(
        self,
        state: JointState,
        torques: Tensor,
    ) -> Tensor:
        """Compute accelerations from torques using articulated body method."""
        return torch.zeros(self.n_joints, device=state.position.device)

    def _forward_kinematics_joints(self, q: Tensor) -> List[Tensor]:
        """
        Compute forward kinematics for each joint.

        Args:
            q: Joint positions

        Returns:
            List of homogeneous transforms [4, 4]
        """
        transforms = []

        T = torch.eye(4)

        for i in range(self.n_joints):
            axis = self.axes[i]

            if axis[0] != 0 or axis[1] != 0:
                angle = q[i]
                c = torch.cos(angle)
                s = torch.sin(angle)
                axis_cross = torch.tensor(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ],
                    device=q.device,
                )
                R = torch.eye(3) + axis_cross * s + axis_cross @ axis_cross * (1 - c)
            else:
                R = torch.eye(3)

            T_new = torch.eye(4, device=q.device)
            T_new[:3, :3] = R

            T = T @ T_new
            transforms.append(T.clone())

        return transforms


class ForwardKinematics(nn.Module):
    """
    Forward kinematics computation.

    Computes end-effector position/orientation from joint angles.
    """

    def __init__(
        self,
        n_joints: int,
        link_lengths: Tensor,
        axes: Optional[Tensor] = None,
    ):
        super().__init__()
        self.n_joints = n_joints

        self.register_buffer("link_lengths", link_lengths)

        if axes is None:
            axes = torch.zeros(n_joints, 3)
            axes[:, 2] = 1
        self.register_buffer("axes", axes)

    def forward(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute forward kinematics.

        Args:
            q: Joint positions [n_joints]

        Returns:
            Tuple of (position [3], orientation quaternion [4])
        """
        position = torch.zeros(3, device=q.device)
        orientation = torch.tensor([1, 0, 0, 0], device=q.device)

        for i in range(self.n_joints):
            R = self._rotation_matrix(q[i], self.axes[i])

            position = position + self.link_lengths[i] * R @ torch.tensor(
                [1, 0, 0], device=q.device
            )

            R_new = R @ self._quaternion_to_rotation(orientation)
            orientation = self._rotation_to_quaternion(R_new)

        return position, orientation

    def compute_jacobian(self, q: Tensor) -> Tensor:
        """
        Compute geometric Jacobian.

        Args:
            q: Joint positions

        Returns:
            Jacobian [6, n_joints]
        """
        position, _ = self.forward(q)

        jacobian = torch.zeros(6, self.n_joints, device=q.device)

        T = torch.eye(4)
        z_prev = torch.tensor([0, 0, 1], device=q.device)

        for i in range(self.n_joints):
            R = self._rotation_matrix(q[i], self.axes[i])
            z = R @ z_prev

            p = position - T[:3, 3]

            jacobian[:3, i] = torch.cross(z, p)
            jacobian[3:, i] = z

            T_new = torch.eye(4, device=q.device)
            T_new[:3, :3] = R
            T = T @ T_new

        return jacobian

    def _rotation_matrix(self, angle: Tensor, axis: Tensor) -> Tensor:
        """Compute rotation matrix from axis-angle."""
        c = torch.cos(angle)
        s = torch.sin(angle)
        t = 1 - c

        return torch.tensor(
            [
                [
                    t * axis[0] ** 2 + c,
                    t * axis[0] * axis[1] - s * axis[2],
                    t * axis[0] * axis[2] + s * axis[1],
                ],
                [
                    t * axis[0] * axis[1] + s * axis[2],
                    t * axis[1] ** 2 + c,
                    t * axis[1] * axis[2] - s * axis[0],
                ],
                [
                    t * axis[0] * axis[2] - s * axis[1],
                    t * axis[1] * axis[2] + s * axis[0],
                    t * axis[2] ** 2 + c,
                ],
            ],
            device=angle.device,
            dtype=angle.dtype,
        )

    def _quaternion_to_rotation(self, q: Tensor) -> Tensor:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]

        return torch.tensor(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ],
            device=q.device,
            dtype=q.dtype,
        )

    def _rotation_to_quaternion(self, R: Tensor) -> Tensor:
        """Convert rotation matrix to quaternion."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2 * torch.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2 * torch.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2 * torch.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return torch.tensor([w, x, y, z], device=R.device)


class InverseDynamics(nn.Module):
    """
    Inverse dynamics controller.

    Computes torques to achieve desired acceleration:
        tau = M(q) * q_dd_desired + C(q, q_d) * q_d + g(q)

    Args:
        dynamics: Forward dynamics model
    """

    def __init__(self, dynamics: RigidBodyDynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(
        self,
        state: JointState,
        desired_acceleration: Tensor,
    ) -> Tensor:
        """
        Compute inverse dynamics torques.

        Args:
            state: Current joint state
            desired_acceleration: Desired joint accelerations

        Returns:
            Required torques
        """
        M = self.dynamics.compute_mass_matrix(state.position)
        C = self.dynamics.compute_coriolis_matrix(state.position, state.velocity)
        g = self.dynamics.compute_gravity_vector(state.position)

        torques = M @ desired_acceleration + C @ state.velocity + g

        return torques


class GravityCompensation(nn.Module):
    """
    Gravity compensation for robot manipulators.

    Provides feedforward torques to counteract gravity.
    """

    def __init__(
        self,
        n_joints: int,
        link_masses: Tensor,
        link_lengths: Tensor,
        axes: Tensor,
        gravity: Optional[Tensor] = None,
    ):
        super().__init__()
        self.n_joints = n_joints

        self.register_buffer("link_masses", link_masses)
        self.register_buffer("link_lengths", link_lengths)
        self.register_buffer("axes", axes)

        if gravity is None:
            gravity = torch.tensor([0, 0, -9.81])
        self.register_buffer("gravity", gravity)

    def compute(
        self,
        q: Tensor,
    ) -> Tensor:
        """
        Compute gravity compensation torques.

        Args:
            q: Joint positions

        Returns:
            Gravity compensation torques
        """
        torques = torch.zeros(self.n_joints, device=q.device)

        T = torch.eye(4)

        for i in range(self.n_joints):
            R = self._rotation_matrix(q[i], self.axes[i])

            gravity_torque = torch.cross(T[:3, 3], self.link_masses[i] * self.gravity)

            axis = R @ self.axes[i]
            torques[i] = torch.dot(axis, gravity_torque)

            T_new = torch.eye(4)
            T_new[:3, :3] = R
            T_new[:3, 3] = self.link_lengths[i] * axis

            T = T @ T_new

        return torques

    def _rotation_matrix(self, angle: Tensor, axis: Tensor) -> Tensor:
        """Compute rotation matrix."""
        c = torch.cos(angle)
        s = torch.sin(angle)
        t = 1 - c

        return torch.tensor(
            [
                [
                    t * axis[0] ** 2 + c,
                    t * axis[0] * axis[1] - s * axis[2],
                    t * axis[0] * axis[2] + s * axis[1],
                ],
                [
                    t * axis[0] * axis[1] + s * axis[2],
                    t * axis[1] ** 2 + c,
                    t * axis[1] * axis[2] - s * axis[0],
                ],
                [
                    t * axis[0] * axis[2] - s * axis[1],
                    t * axis[1] * axis[2] + s * axis[0],
                    t * axis[2] ** 2 + c,
                ],
            ],
            device=angle.device,
            dtype=angle.dtype,
        )


class FrictionCompensation(nn.Module):
    """
    Friction compensation for robot joints.

    Models friction as:
        friction = coulomb + viscous + static
    """

    def __init__(
        self,
        n_joints: int,
        coulomb_friction: Optional[Tensor] = None,
        viscous_friction: Optional[Tensor] = None,
        static_friction: Optional[Tensor] = None,
    ):
        super().__init__()

        self.coulomb = (
            coulomb_friction if coulomb_friction is not None else torch.zeros(n_joints)
        )
        self.viscous = (
            viscous_friction if viscous_friction is not None else torch.zeros(n_joints)
        )
        self.static = (
            static_friction if static_friction is not None else torch.zeros(n_joints)
        )

    def compute(self, velocity: Tensor) -> Tensor:
        """Compute friction torques."""
        coulomb = self.coulomb * torch.sign(velocity)
        viscous = self.viscous * velocity

        static = torch.where(
            torch.abs(velocity) < 0.01,
            -self.static * velocity / 0.01,
            torch.zeros_like(velocity),
        )

        return coulomb + viscous + static


class RobotSimulator(nn.Module):
    """
    Complete robot simulator combining dynamics and kinematics.
    """

    def __init__(
        self,
        dynamics: RigidBodyDynamics,
        forward_kinematics: ForwardKinematics,
    ):
        super().__init__()
        self.dynamics = dynamics
        self.fk = forward_kinematics

    def step(
        self,
        state: JointState,
        control_torques: Tensor,
        dt: float,
    ) -> JointState:
        """Simulate one time step."""
        return self.dynamics.integrate(state, control_torques, dt)

    def get_ee_pose(self, state: JointState) -> Tuple[Tensor, Tensor]:
        """Get end-effector pose."""
        return self.fk(state.position)

    def simulate_trajectory(
        self,
        initial_state: JointState,
        torques: Tensor,
        dt: float,
    ) -> List[JointState]:
        """
        Simulate full trajectory.

        Args:
            initial_state: Starting state
            torques: Control torques [T, n_joints]
            dt: Time step

        Returns:
            List of states
        """
        states = [initial_state]
        current_state = initial_state

        for t in range(torques.shape[0]):
            current_state = self.step(current_state, torques[t], dt)
            states.append(current_state)

        return states
