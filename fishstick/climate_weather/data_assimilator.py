"""
Data Assimilation for Climate and Weather

Kalman filter and related methods for data assimilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AssimilationState:
    """State estimate from data assimilation.

    Args:
        mean: State mean
        covariance: State covariance matrix
        timestamp: Current timestamp
    """

    mean: Tensor
    covariance: Tensor
    timestamp: float


class KalmanFilter:
    """Kalman Filter for linear-Gaussian data assimilation.

    Args:
        state_dim: Dimension of state vector
        obs_dim: Dimension of observation vector
        process_noise: Process noise covariance
        observation_noise: Observation noise covariance
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        process_noise: Optional[Tensor] = None,
        observation_noise: Optional[Tensor] = None,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        if process_noise is None:
            self.Q = torch.eye(state_dim) * 0.1
        else:
            self.Q = process_noise

        if observation_noise is None:
            self.R = torch.eye(obs_dim) * 1.0
        else:
            self.R = observation_noise

        self.P = torch.eye(state_dim)
        self.x = torch.zeros(state_dim)
        self.F = torch.eye(state_dim)
        self.H = torch.zeros(obs_dim, state_dim)

    def predict(
        self,
        control_input: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict step (forecast).

        Args:
            control_input: Optional control input

        Returns:
            Predicted state
        """
        if control_input is not None:
            self.x = self.F @ self.x + control_input
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(
        self,
        observation: Tensor,
    ) -> Tensor:
        """Update step (analysis).

        Args:
            observation: Observation vector

        Returns:
            Updated state
        """
        y = observation - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ torch.inverse(S)

        self.x = self.x + K @ y

        I = torch.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    def step(
        self,
        observation: Tensor,
        control_input: Optional[Tensor] = None,
    ) -> Tensor:
        """Combined predict and update step.

        Args:
            observation: Observation vector
            control_input: Optional control input

        Returns:
            Updated state
        """
        self.predict(control_input)
        return self.update(observation)


class ExtendedKalmanFilter:
    """Extended Kalman Filter for non-linear data assimilation.

    Args:
        state_dim: Dimension of state vector
        obs_dim: Dimension of observation vector
        process_noise: Process noise covariance
        observation_noise: Observation noise covariance
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        process_noise: Optional[Tensor] = None,
        observation_noise: Optional[Tensor] = None,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        if process_noise is None:
            self.Q = torch.eye(state_dim) * 0.1
        else:
            self.Q = process_noise

        if observation_noise is None:
            self.R = torch.eye(obs_dim) * 1.0
        else:
            self.R = observation_noise

        self.P = torch.eye(state_dim)
        self.x = torch.zeros(state_dim)

        self.state_transition_fn = None
        self.observation_fn = None

    def set_state_transition(
        self,
        fn: callable,
    ):
        """Set the state transition function.

        Args:
            fn: Non-linear state transition function
        """
        self.state_transition_fn = fn

    def set_observation(
        self,
        fn: callable,
    ):
        """Set the observation function.

        Args:
            fn: Non-linear observation function
        """
        self.observation_fn = fn

    def predict(
        self,
        control_input: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict step using linearization.

        Args:
            control_input: Optional control input

        Returns:
            Predicted state
        """
        if self.state_transition_fn is not None:
            with torch.no_grad():
                self.x = self.state_transition_fn(self.x, control_input)

            J = torch.autograd.functional.jacobian(
                lambda s: self.state_transition_fn(s, control_input),
                self.x,
            )
            F = torch.eye(self.state_dim) + J
        else:
            F = torch.eye(self.state_dim)

        self.P = F @ self.P @ F.T + self.Q

        return self.x.copy()

    def update(
        self,
        observation: Tensor,
    ) -> Tensor:
        """Update step using linearization.

        Args:
            observation: Observation vector

        Returns:
            Updated state
        """
        if self.observation_fn is not None:
            with torch.no_grad():
                z_pred = self.observation_fn(self.x)

            J = torch.autograd.functional.jacobian(self.observation_fn, self.x)
            H = J
        else:
            H = torch.zeros(self.obs_dim, self.state_dim)
            z_pred = torch.zeros(self.obs_dim)

        y = observation - z_pred

        S = H @ self.P @ H.T + self.R

        K = self.P @ H.T @ torch.inverse(S)

        self.x = self.x + K @ y

        I = torch.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        return self.x.copy()


class EnsembleKalmanFilter:
    """Ensemble Kalman Filter for data assimilation.

    Args:
        state_dim: Dimension of state vector
        obs_dim: Dimension of observation vector
        ensemble_size: Number of ensemble members
        process_noise: Process noise covariance
        observation_noise: Observation noise covariance
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        ensemble_size: int = 50,
        process_noise: Optional[Tensor] = None,
        observation_noise: Optional[Tensor] = None,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.ensemble_size = ensemble_size

        if process_noise is None:
            self.Q = torch.eye(state_dim) * 0.1
        else:
            self.Q = process_noise

        if observation_noise is None:
            self.R = torch.eye(obs_dim) * 1.0
        else:
            self.R = observation_noise

        self.ensemble = torch.randn(ensemble_size, state_dim)
        self.mean = self.ensemble.mean(dim=0)
        self.cov = torch.eye(state_dim)

        self.state_transition_fn = None
        self.observation_fn = None

    def set_state_transition(self, fn: callable):
        self.state_transition_fn = fn

    def set_observation(self, fn: callable):
        self.observation_fn = fn

    def predict(self) -> Tensor:
        """Predict step.

        Returns:
            Predicted state mean
        """
        for i in range(self.ensemble_size):
            noise = torch.randn(self.state_dim) @ torch.cholesky(self.Q)
            if self.state_transition_fn is not None:
                self.ensemble[i] = self.state_transition_fn(self.ensemble[i])
            self.ensemble[i] = self.ensemble[i] + noise

        self.mean = self.ensemble.mean(dim=0)

        centered = self.ensemble - self.mean.unsqueeze(0)
        self.cov = (centered.t() @ centered) / (self.ensemble_size - 1)

        return self.mean.copy()

    def update(self, observation: Tensor) -> Tensor:
        """Update step.

        Args:
            observation: Observation vector

        Returns:
            Updated state mean
        """
        obs_ensemble = torch.zeros(self.ensemble_size, self.obs_dim)

        for i in range(self.ensemble_size):
            if self.observation_fn is not None:
                obs_ensemble[i] = self.observation_fn(self.ensemble[i])
            else:
                obs_ensemble[i] = self.ensemble[i, : self.obs_dim]

        obs_mean = obs_ensemble.mean(dim=0)

        centered_state = self.ensemble - self.mean.unsqueeze(0)
        centered_obs = obs_ensemble - obs_mean.unsqueeze(0)

        S = (centered_obs.t() @ centered_obs) / (self.ensemble_size - 1) + self.R

        cross_cov = (centered_state.t() @ centered_obs) / (self.ensemble_size - 1)

        K = cross_cov @ torch.inverse(S)

        for i in range(self.ensemble_size):
            noise = torch.randn(self.obs_dim) @ torch.cholesky(self.R)
            innovation = observation - obs_ensemble[i] - noise
            self.ensemble[i] = self.ensemble[i] + K @ innovation

        self.mean = self.ensemble.mean(dim=0)

        return self.mean.copy()


class ObservationOperator(nn.Module):
    """Observation operator for data assimilation.

    Args:
        state_dim: Dimension of state vector
        obs_dim: Dimension of observation vector
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.H = nn.Linear(state_dim, obs_dim, bias=False)

    def forward(self, state: Tensor) -> Tensor:
        return self.H(state)

    def set_matrix(self, H: Tensor):
        with torch.no_grad():
            self.H.weight.copy_(H)


class FourDimensionalVar:
    """4D-Var data assimilation.

    Args:
        state_dim: Dimension of state vector
        obs_dim: Dimension of observation vector
        background_cov: Background error covariance
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        background_cov: Optional[Tensor] = None,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        if background_cov is None:
            self.B = torch.eye(state_dim)
        else:
            self.B = background_cov

        self.R = torch.eye(obs_dim)
        self.observation_operator = None
        self.forward_model = None

    def set_observation_operator(self, H: Tensor):
        self.H = H

    def set_forward_model(self, fn: callable):
        self.forward_model = fn

    def minimize(
        self,
        x_b: Tensor,
        observations: List[Tuple[float, Tensor]],
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> Tensor:
        """Minimize cost function to find optimal analysis.

        Args:
            x_b: Background state
            observations: List of (time, observation) tuples
            max_iter: Maximum iterations
            lr: Learning rate

        Returns:
            Analyzed state
        """
        x = x_b.clone()
        x.requires_grad = True

        optimizer = torch.optim.LBFGS([x], max_iter=10, line_search_fn="strong_wolfe")

        for iteration in range(max_iter):
            optimizer.zero_grad()

            J_background = 0.5 * ((x - x_b).t() @ torch.inverse(self.B) @ (x - x_b))

            J_observation = 0.0
            for _, obs in observations:
                if self.observation_operator is not None:
                    hx = self.observation_operator(x)
                else:
                    hx = x[: self.obs_dim]

                innovation = obs - hx
                J_observation = J_observation + 0.5 * (
                    innovation.t() @ torch.inverse(self.R) @ innovation
                )

            J = J_background + J_observation

            J.backward()
            optimizer.step()

            if J.item() < 1e-6:
                break

        return x.detach()


class ParticleFilter:
    """Particle Filter for non-linear data assimilation.

    Args:
        state_dim: Dimension of state vector
        obs_dim: Dimension of observation vector
        num_particles: Number of particles
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        num_particles: int = 100,
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles

        self.particles = torch.randn(num_particles, state_dim)
        self.weights = torch.ones(num_particles) / num_particles

        self.resample_threshold = num_particles // 2

    def predict(
        self,
        transition_fn: callable,
        process_noise: float = 0.1,
    ):
        """Prediction step.

        Args:
            transition_fn: State transition function
            process_noise: Process noise standard deviation
        """
        for i in range(self.num_particles):
            self.particles[i] = transition_fn(self.particles[i])
            self.particles[i] = (
                self.particles[i] + torch.randn(self.state_dim) * process_noise
            )

    def update(
        self,
        observation: Tensor,
        likelihood_fn: callable,
    ):
        """Update step.

        Args:
            observation: Observation vector
            likelihood_fn: Function to compute observation likelihood
        """
        for i in range(self.num_particles):
            self.weights[i] = likelihood_fn(self.particles[i], observation)

        self.weights = self.weights / self.weights.sum()

        if (self.weights < 1 / self.resample_threshold).sum():
            self.resample()

    def resample(self):
        """Resample particles based on weights."""
        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices]
        self.weights = torch.ones(self.num_particles) / self.num_particles

    def get_state(self) -> Tensor:
        """Get estimated state."""
        return (self.particles * self.weights.unsqueeze(1)).sum(dim=0)


def compute_innovation(
    observation: Tensor,
    forecast: Tensor,
    observation_operator: Tensor,
) -> Tensor:
    """Compute innovation vector.

    Args:
        observation: Observation vector
        forecast: Forecast state
        observation_operator: Observation operator matrix

    Returns:
        Innovation vector
    """
    hx = observation_operator @ forecast
    return observation - hx


def compute_analysis_increment(
    innovation: Tensor,
    kalman_gain: Tensor,
) -> Tensor:
    """Compute analysis increment.

    Args:
        innovation: Innovation vector
        kalman_gain: Kalman gain matrix

    Returns:
        Analysis increment
    """
    return kalman_gain @ innovation
