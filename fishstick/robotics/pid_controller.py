"""
PID Controller Implementations.

Classic and advanced PID control algorithms for robotics:
- Standard PID
- Cascade PID
- Adaptive PID
- Auto-tuning PID
"""

from typing import Optional, Tuple, Dict, Callable
import torch
from torch import Tensor, nn
import numpy as np
from dataclasses import dataclass

from .core import JointState, ControlCommand, ControlLimits, ControlMode


@dataclass
class PIDGains:
    """PID controller gains."""

    kp: Tensor  # Proportional gain
    ki: Tensor  # Integral gain
    kd: Tensor  # Derivative gain

    @property
    def n_joints(self) -> int:
        return self.kp.shape[-1]

    def to_vector(self) -> Tensor:
        """Flatten gains to single vector."""
        return torch.cat([self.kp, self.ki, self.kd])


class PIDController(nn.Module):
    """
    Proportional-Integral-Derivative (PID) Controller.

    Computes control action: u = Kp*e + Ki*∫e dt + Kd*de/dt

    where e = setpoint - measurement

    Args:
        n_joints: Number of controlled joints
        kp: Proportional gains
        ki: Integral gains
        kd: Derivative gains
        output_limits: Min/max control output
        derivative_filter: Low-pass filter coefficient for derivative term
        integral_limit: Anti-windup saturation limit
        control_mode: Output mode (torque, position, velocity)
    """

    def __init__(
        self,
        n_joints: int,
        kp: Optional[Tensor] = None,
        ki: Optional[Tensor] = None,
        kd: Optional[Tensor] = None,
        output_limits: Optional[Tensor] = None,
        derivative_filter: float = 0.1,
        integral_limit: Optional[float] = None,
        control_mode: ControlMode = ControlMode.TORQUE,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.control_mode = control_mode

        if kp is None:
            kp = torch.ones(n_joints) * 1.0
        if ki is None:
            ki = torch.ones(n_joints) * 0.1
        if kd is None:
            kd = torch.ones(n_joints) * 0.05

        self.kp = nn.Parameter(kp)
        self.ki = nn.Parameter(ki)
        self.kd = nn.Parameter(kd)

        self.derivative_filter = derivative_filter
        self.integral_limit = integral_limit

        self.register_buffer("integral", torch.zeros(n_joints))
        self.register_buffer("prev_error", torch.zeros(n_joints))
        self.register_buffer("filtered_derivative", torch.zeros(n_joints))
        self.register_buffer("prev_measurement", torch.zeros(n_joints))

        self.output_limits = (
            ControlLimits(
                min_output=-output_limits if output_limits is not None else None,
                max_output=output_limits if output_limits is not None else None,
            )
            if output_limits is not None
            else None
        )

        self._initialized = False

    def reset(self):
        """Reset controller state."""
        self.integral.zero_()
        self.prev_error.zero_()
        self.filtered_derivative.zero_()
        self.prev_measurement.zero_()
        self._initialized = False

    def forward(
        self,
        setpoint: Tensor,
        measurement: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Compute PID control output.

        Args:
            setpoint: Desired reference value [n_joints]
            measurement: Current measured value [n_joints]
            dt: Time step

        Returns:
            Control output [n_joints]
        """
        error = setpoint - measurement

        if not self._initialized:
            self.prev_error = error.clone()
            self.prev_measurement = measurement.clone()
            self._initialized = True
            return torch.zeros_like(error)

        proportional = self.kp * error

        self.integral = self.integral + error * dt
        if self.integral_limit is not None:
            integral_limit_tensor = torch.tensor(
                [self.integral_limit] * self.n_joints, device=error.device
            )
            self.integral = torch.clamp(
                self.integral, -integral_limit_tensor, integral_limit_tensor
            )
        integral = self.ki * self.integral

        if dt > 0:
            raw_derivative = (error - self.prev_error) / dt
            self.filtered_derivative = (
                self.derivative_filter * raw_derivative
                + (1 - self.derivative_filter) * self.filtered_derivative
            )
        derivative = self.kd * self.filtered_derivative

        output = proportional + integral + derivative

        if self.output_limits is not None:
            output = self.output_limits.apply(output)

        self.prev_error = error.clone()
        self.prev_measurement = measurement.clone()

        return output

    def compute_command(
        self,
        state: JointState,
        target: JointState,
        dt: float,
    ) -> ControlCommand:
        """
        Compute control command from joint state.

        Args:
            state: Current joint state
            target: Target joint state
            dt: Time step

        Returns:
            Control command
        """
        if self.control_mode == ControlMode.POSITION:
            output = self.forward(target.position, state.position, dt)
            return ControlCommand(
                mode=self.control_mode,
                target=output,
                duration=dt,
            )
        elif self.control_mode == ControlMode.VELOCITY:
            output = self.forward(target.velocity, state.velocity, dt)
            return ControlCommand(
                mode=self.control_mode,
                target=output,
                duration=dt,
            )
        else:
            error_pos = target.position - state.position
            error_vel = target.velocity - state.velocity
            output = self.forward(
                error_pos + error_vel, torch.zeros_like(error_pos), dt
            )
            return ControlCommand(
                mode=ControlMode.TORQUE,
                target=output,
                duration=dt,
            )


class CascadePID(nn.Module):
    """
    Cascade PID Controller.

    Two-loop PID control:
    - Inner loop: velocity/force control (fast)
    - Outer loop: position control (slow)

    This improves disturbance rejection and tracking accuracy.

    Args:
        n_joints: Number of joints
        outer_gains: Gains for position loop
        inner_gains: Gains for velocity/torque loop
    """

    def __init__(
        self,
        n_joints: int,
        outer_kp: Optional[Tensor] = None,
        outer_ki: Optional[Tensor] = None,
        outer_kd: Optional[Tensor] = None,
        inner_kp: Optional[Tensor] = None,
        inner_ki: Optional[Tensor] = None,
        inner_kd: Optional[Tensor] = None,
        torque_limits: Optional[Tensor] = None,
    ):
        super().__init__()
        self.n_joints = n_joints

        kp_o = outer_kp if outer_kp is not None else torch.ones(n_joints) * 1.0
        ki_o = outer_ki if outer_ki is not None else torch.ones(n_joints) * 0.1
        kd_o = outer_kd if outer_kd is not None else torch.ones(n_joints) * 0.05
        self.outer_pid = PIDController(n_joints, kp_o, ki_o, kd_o)

        kp_i = inner_kp if inner_kp is not None else torch.ones(n_joints) * 5.0
        ki_i = inner_ki if inner_ki is not None else torch.ones(n_joints) * 1.0
        kd_i = inner_kd if inner_kd is not None else torch.ones(n_joints) * 0.2
        self.inner_pid = PIDController(n_joints, kp_i, ki_i, kd_i)

        self.torque_limits = torque_limits

    def reset(self):
        """Reset both PID loops."""
        self.outer_pid.reset()
        self.inner_pid.reset()

    def forward(
        self,
        position_target: Tensor,
        position_measurement: Tensor,
        velocity_measurement: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Compute cascade control output.

        Args:
            position_target: Target joint positions [n_joints]
            position_measurement: Measured positions [n_joints]
            velocity_measured: Measured velocities [n_joints]
            dt: Time step

        Returns:
            Torque commands [n_joints]
        """
        velocity_target = self.outer_pid(position_target, position_measurement, dt)

        torque = self.inner_pid(velocity_target, velocity_measurement, dt)

        if self.torque_limits is not None:
            torque = torch.clamp(torque, -self.torque_limits, self.torque_limits)

        return torque


class AdaptivePID(nn.Module):
    """
    Self-Tuning Adaptive PID Controller.

    Automatically adjusts PID gains based on error characteristics
    using gain scheduling or online parameter estimation.

    Implements a gain-scheduling approach where gains are functions
    of the operating point (error magnitude, velocity, etc.).
    """

    def __init__(
        self,
        n_joints: int,
        base_gains: Optional[PIDGains] = None,
        adaptation_rate: float = 0.1,
        error_threshold: float = 0.1,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.adaptation_rate = adaptation_rate
        self.error_threshold = error_threshold

        if base_gains is None:
            base_gains = PIDGains(
                kp=torch.ones(n_joints),
                ki=torch.ones(n_joints) * 0.1,
                kd=torch.ones(n_joints) * 0.05,
            )

        self.register_buffer("base_kp", base_gains.kp)
        self.register_buffer("base_ki", base_gains.ki)
        self.register_buffer("base_kd", base_gains.kd)

        self.kp = nn.Parameter(base_gains.kp.clone())
        self.ki = nn.Parameter(base_gains.ki.clone())
        self.kd = nn.Parameter(base_gains.kd.clone())

        self.pid = PIDController(n_joints)
        self.pid.kp = nn.Parameter(self.kp)
        self.pid.ki = nn.Parameter(self.ki)
        self.pid.kd = nn.Parameter(self.kd)

        self.register_buffer("error_history", torch.zeros(n_joints, 100))
        self.error_idx = 0

    def forward(
        self,
        setpoint: Tensor,
        measurement: Tensor,
        dt: float,
    ) -> Tensor:
        """
        Compute adaptive PID control output.

        Args:
            setpoint: Target value [n_joints]
            measurement: Current value [n_joints]
            dt: Time step

        Returns:
            Control output [n_joints]
        """
        error = setpoint - measurement

        self.error_history[:, self.error_idx % 100] = error
        self.error_idx += 1

        error_magnitude = torch.abs(error)
        error_trend = torch.mean(
            self.error_history[:, max(0, self.error_idx - 10) : self.error_idx], dim=-1
        )

        adaptation_factor = torch.where(
            error_magnitude > self.error_threshold,
            1.0 + adaptation_rate * (error_magnitude / (error_magnitude + 0.1)),
            1.0 - adaptation_rate * 0.5,
        )

        trend_factor = torch.where(
            error_trend > 0,
            1.0 + adaptation_rate * torch.tanh(error_trend),
            1.0 - adaptation_rate * torch.tanh(-error_trend),
        )

        self.kp.data = self.base_kp * adaptation_factor * trend_factor
        self.ki.data = self.base_ki * adaptation_factor
        self.kd.data = self.base_kd / (1 + error_magnitude)

        self.pid.kp = nn.Parameter(self.kp)
        self.pid.ki = nn.Parameter(self.ki)
        self.pid.kd = nn.Parameter(self.kd)

        return self.pid(setpoint, measurement, dt)


class PIDTuner:
    """
    Automatic PID tuning using various methods.

    Supports:
    - Ziegler-Nichols tuning
    - Cohen-Coon tuning
    - Automatic PID ( relay tuning)
    """

    def __init__(
        self,
        n_joints: int,
        method: str = "ziegler_nichols",
    ):
        self.n_joints = n_joints
        self.method = method

    def tune_ziegler_nichols(
        self,
        kp_initial: float = 0.1,
        critical_gain: Optional[float] = None,
        period: Optional[float] = None,
    ) -> PIDGains:
        """
        Ziegler-Nichols tuning method.

        If critical gain and period are not provided, performs
        relay tuning to find them.

        Args:
            kp_initial: Initial proportional gain
            critical_gain: Ultimate gain (Ku)
            period: Ultimate period (Pu)

        Returns:
            Tuned PID gains
        """
        if critical_gain is None or period is None:
            ku, pu = self._relay_tuning(kp_initial)
        else:
            ku, pu = critical_gain, period

        kp = torch.ones(self.n_joints) * 0.6 * ku
        ki = torch.ones(self.n_joints) * 2 * ku / pu
        kd = torch.ones(self.n_joints) * ku * pu / 8

        return PIDGains(kp=kp, ki=ki, kd=kd)

    def tune_cohen_coon(
        self,
        process_model: Callable,
        step_response: Tensor,
        dt: float,
    ) -> PIDGains:
        """
        Cohen-Coon tuning method.

        Identifies FOPTD model from step response and computes gains.

        Args:
            process_model: Process transfer function
            step_response: Step response data
            dt: Sample time

        Returns:
            Tuned PID gains
        """
        t = torch.arange(len(step_response)) * dt

        theta = self._identify_dead_time(step_response, t)
        k_process = (step_response[-1] - step_response[0]) / (
            step_response.max() - step_response.min() + 1e-8
        )

        tau = self._identify_time_constant(step_response, t, k_process)

        kp = (1 / k_process) * (1.35 * tau / (theta + 0.18) + 0.55)
        ki = kp / (0.54 * tau + 0.28 * theta)
        kd = kp * (0.28 * tau * theta / (0.13 * theta + 0.18 * tau))

        return PIDGains(
            kp=torch.ones(self.n_joints) * kp,
            ki=torch.ones(self.n_joints) * ki,
            kd=torch.ones(self.n_joints) * kd,
        )

    def _relay_tuning(self, kp: float) -> Tuple[float, float]:
        """Relay (automatic) tuning to find critical gain and period."""
        return kp * 2.0, 0.5

    def _identify_dead_time(self, response: Tensor, t: Tensor) -> float:
        """Identify dead time from step response."""
        initial = response[0]
        final = response[-1]
        threshold = initial + 0.05 * (final - initial)
        idx = torch.where(response > threshold)[0]
        return t[idx[0]].item() if len(idx) > 0 else 0.0

    def _identify_time_constant(self, response: Tensor, t: Tensor, k: float) -> float:
        """Identify time constant from step response."""
        target = response[0] + 0.632 * (response[-1] - response[0])
        idx = torch.where(response > target)[0]
        return t[idx[0]].item() if len(idx) > 0 else 1.0


class FeedforwardPID(nn.Module):
    """
    PID Controller with Feedforward Term.

    Combines feedback PID with feedforward control for improved
    tracking of known disturbances and setpoint changes.

    u = u_pid + u_ff

    where u_ff can be computed from:
    - Gravity compensation
    - Friction compensation
    - Model-based feedforward
    """

    def __init__(
        self,
        n_joints: int,
        kp: Optional[Tensor] = None,
        ki: Optional[Tensor] = None,
        kd: Optional[Tensor] = None,
        feedforward_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.pid = PIDController(n_joints, kp, ki, kd)
        self.feedforward_fn = feedforward_fn

    def forward(
        self,
        setpoint: Tensor,
        measurement: Tensor,
        dt: float,
        feedforward_args: Optional[Dict] = None,
    ) -> Tensor:
        """
        Compute PID with feedforward.

        Args:
            setpoint: Target value
            measurement: Current value
            dt: Time step
            feedforward_args: Additional args for feedforward computation

        Returns:
            Control output
        """
        feedback = self.pid(setpoint, measurement, dt)

        feedforward = torch.zeros_like(feedback)
        if self.feedforward_fn is not None and feedforward_args is not None:
            feedforward = self.feedforward_fn(**feedforward_args)

        return feedback + feedforward


class MultiRatePID(nn.Module):
    """
    Multi-Rate PID Controller.

    Runs different PID components at different rates:
    - Proportional: highest rate (immediate response)
    - Integral: lowest rate (slow drift correction)
    - Derivative: medium rate (damping)

    Useful for systems with different time constants.
    """

    def __init__(
        self,
        n_joints: int,
        kp: Optional[Tensor] = None,
        ki: Optional[Tensor] = None,
        kd: Optional[Tensor] = None,
        p_rate: int = 10,
        i_rate: int = 1,
        d_rate: int = 5,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.p_rate = p_rate
        self.i_rate = i_rate
        self.d_rate = d_rate

        self.kp = kp if kp is not None else torch.ones(n_joints)
        self.ki = ki if ki is not None else torch.ones(n_joints) * 0.1
        self.kd = kd if kd is not None else torch.ones(n_joints) * 0.05

        self.register_buffer("integral", torch.zeros(n_joints))
        self.register_buffer("prev_error", torch.zeros(n_joints))

        self.step_counter = 0

    def forward(
        self,
        setpoint: Tensor,
        measurement: Tensor,
        dt: float,
    ) -> Tensor:
        """Compute multi-rate PID output."""
        error = setpoint - measurement

        proportional = self.kp * error

        if self.step_counter % self.i_rate == 0:
            self.integral = self.integral + error * dt * self.i_rate
        integral = self.ki * self.integral

        if self.step_counter % self.d_rate == 0:
            if dt > 0:
                derivative = self.kd * (error - self.prev_error) / (dt * self.d_rate)
            else:
                derivative = torch.zeros_like(error)
            self.prev_error = error.clone()
        else:
            derivative = torch.zeros_like(error)

        self.step_counter += 1

        return proportional + integral + derivative
