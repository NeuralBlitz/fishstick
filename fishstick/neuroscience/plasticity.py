"""Synaptic plasticity mechanisms: STDP, Oja's rule, and variants."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class STDP(nn.Module):
    """Spike-Timing-Dependent Plasticity (STDP) learning rule.

    STDP modifies synaptic weights based on the relative timing of pre-
    and post-synaptic spikes:
        Δw = A⁺ * exp(-Δt/τ⁺) if Δt > 0
        Δw = A⁻ * exp(Δt/τ⁻) if Δt < 0

    where Δt = t_post - t_pre

    Args:
        n_synapses: Number of synapses
        a_plus: Potentiation amplitude
        a_minus: Depression amplitude
        tau_plus: Potentiation time constant (ms)
        tau_minus: Depression time constant (ms)
        w_min: Minimum weight
        w_max: Maximum weight
    """

    def __init__(
        self,
        n_synapses: int,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        super().__init__()
        self.n_synapses = n_synapses
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

        self.weight = nn.Parameter(torch.rand(n_synapses) * 0.5 + 0.5)

        self.register_buffer("trace_pre", torch.zeros(n_synapses))
        self.register_buffer("trace_post", torch.zeros(n_synapses))

    def forward(
        self,
        spike_pre: Tensor,
        spike_post: Tensor,
    ) -> Tensor:
        """Compute weight updates based on spike timing.

        Args:
            spike_pre: Pre-synaptic spikes (batch, n_synapses)
            spike_post: Post-synaptic spikes (batch, n_neurons)

        Returns:
            Updated weights
        """
        if not self.training:
            return self.weight

        pre_mean = spike_pre.float().mean(dim=0)
        post_mean = spike_post.float().mean(dim=0)

        self.trace_pre = (
            self.trace_pre * torch.exp(torch.tensor(-1.0 / self.tau_plus)) + pre_mean
        )
        self.trace_post = (
            self.trace_post * torch.exp(torch.tensor(-1.0 / self.tau_minus)) + post_mean
        )

        delta_w = torch.zeros_like(self.weight)

        potentiation = self.a_plus * self.trace_pre * (1 - self.weight / self.w_max)
        depression = -self.a_minus * self.trace_post * (self.weight / self.w_min)

        delta_w = potentiation + depression

        self.weight.data = torch.clamp(self.weight + delta_w, self.w_min, self.w_max)

        return self.weight


class OjaRule(nn.Module):
    """Oja's rule for stable Hebbian learning.

    Oja's rule normalizes Hebbian learning by including a forgetting term:
        Δw = η * y * (x - w * y)

    This ensures weights don't grow unbounded while maintaining correlation
    between pre-synaptic input and post-synaptic output.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        learning_rate: Learning rate η
        beta: Forgetting coefficient
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.01,
        beta: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.beta = beta

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with Oja's learning rule.

        Args:
            x: Input (batch, in_features)

        Returns:
            Tuple of (output, updated_weights)
        """
        output = torch.matmul(x, self.weight.t())

        if self.training:
            y = output.mean(dim=0, keepdim=True)
            x_centered = x - torch.matmul(y, self.weight)

            delta_w = (
                self.learning_rate * torch.matmul(y.t(), x_centered)
                - self.beta * self.weight
            )

            self.weight.data = self.weight + delta_w

        return output, self.weight


class BCMPlasticity(nn.Module):
    """Bienenstock-Cooper-Munro (BCM) plasticity rule.

    The BCM rule implements activity-dependent plasticity with a sliding
    threshold (plasticity threshold):
        Δw = η * y * (y - θ) * x

    where θ is a running average of post-synaptic activity.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        learning_rate: Learning rate
        tau_threshold: Time constant for threshold adaptation
        theta_init: Initial threshold
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.001,
        tau_threshold: float = 1000.0,
        theta_init: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.tau_threshold = tau_threshold

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.threshold = nn.Parameter(torch.full((out_features,), theta_init))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass with BCM learning."""
        output = torch.matmul(x, self.weight.t())

        if self.training:
            activity = output.detach().mean(dim=0)

            threshold_update = (activity**2 - self.threshold) / self.tau_threshold
            self.threshold.data = self.threshold + threshold_update

            pre = x.float()
            post = (output - self.threshold).clamp(min=0).detach()

            delta_w = self.learning_rate * torch.matmul(
                post.unsqueeze(-1), pre.unsqueeze(1)
            )
            self.weight.data = self.weight + delta_w.mean(dim=0)

        return output, self.threshold


class HomeostaticPlasticity(nn.Module):
    """Homeostatic plasticity module that regulates neural activity.

    Implements synaptic scaling and intrinsic plasticity to maintain
    stable firing rates:
        w → w * (target_rate / actual_rate)^α

    Args:
        n_neurons: Number of neurons
        target_rate: Target firing rate
        tau_homeo: Homeostatic time constant
        alpha: Scaling exponent
    """

    def __init__(
        self,
        n_neurons: int,
        target_rate: float = 0.1,
        tau_homeo: float = 100.0,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.target_rate = target_rate
        self.tau_homeo = tau_homeo
        self.alpha = alpha

        self.gain = nn.Parameter(torch.ones(n_neurons))
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(
        self,
        spikes: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Apply homeostatic regulation.

        Args:
            spikes: Spike tensor (batch, n_neurons)

        Returns:
            Tuple of (scaled_output, (gain, bias))
        """
        if not self.training:
            return spikes * self.gain + self.bias, (self.gain, self.bias)

        actual_rate = spikes.float().mean(dim=0)

        rate_ratio = (self.target_rate / (actual_rate + 1e-8)).clamp(0.1, 10.0)
        target_gain = self.gain * (rate_ratio**self.alpha)

        self.gain.data = self.gain + (target_gain - self.gain) / self.tau_homeo

        output = spikes * self.gain + self.bias

        return output, (self.gain, self.bias)


class TripletSTDP(nn.Module):
    """Triplet STDP - extended STDP with three-factor learning rule.

    Triplet STDP uses both pair-based and triplet-based interactions:
        Δw = a₁ * ō₊ * ē₊ + a₂ * ō₊ * ē₋ + a₃ * ō₋ * ē₊ + a₄ * ō₋ * ē₋

    Args:
        n_synapses: Number of synapses
        a_plus: Pair potentiation
        a_minus: Pair depression
        a_triplet_plus: Triplet potentiation
        a_triplet_minus: Triplet depression
    """

    def __init__(
        self,
        n_synapses: int,
        a_plus: float = 0.005,
        a_minus: float = 0.00525,
        a_triplet_plus: float = 0.0001,
        a_triplet_minus: float = 0.0001,
    ):
        super().__init__()
        self.n_synapses = n_synapses
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.a_triplet_plus = a_triplet_plus
        self.a_triplet_minus = a_triplet_minus

        self.weight = nn.Parameter(torch.rand(n_synapses) * 0.5 + 0.5)

        self.register_buffer("trace1_pre", torch.zeros(n_synapses))
        self.register_buffer("trace1_post", torch.zeros(n_synapses))
        self.register_buffer("trace2_pre", torch.zeros(n_synapses))
        self.register_buffer("trace2_post", torch.zeros(n_synapses))

    def forward(self, spike_pre: Tensor, spike_post: Tensor) -> Tensor:
        """Apply triplet STDP update."""
        pre_mean = spike_pre.float().mean(dim=0)
        post_mean = spike_post.float().mean(dim=0)

        self.trace1_pre = self.trace1_pre * 0.9 + pre_mean
        self.trace1_post = self.trace1_post * 0.9 + post_mean
        self.trace2_pre = self.trace2_pre * 0.99 + pre_mean
        self.trace2_post = self.trace2_post * 0.99 + post_mean

        delta_w = (
            self.a_plus * self.trace1_pre * self.trace1_post
            - self.a_minus * self.trace1_post * (1 - self.trace1_pre)
            + self.a_triplet_plus * self.trace2_pre * self.trace1_post
            - self.a_triplet_minus * self.trace2_post * (1 - self.trace1_pre)
        )

        self.weight.data = torch.clamp(self.weight + delta_w, 0.0, 1.0)

        return self.weight


class VoltageBasedSTDP(nn.Module):
    """Voltage-based STDP with eligibility trace.

    Combines pre-synaptic spikes, post-synaptic voltage, and a calcium
    signal to gate plasticity:
        Δw = ∫ η(t) * pre(t) * post_v(t) * Ca(t) dt

    Args:
        n_synapses: Number of synapses
        tau_voltage: Voltage trace time constant
        tau_calcium: Calcium trace time constant
    """

    def __init__(
        self,
        n_synapses: int,
        tau_voltage: float = 20.0,
        tau_calcium: float = 200.0,
    ):
        super().__init__()
        self.n_synapses = n_synapses
        self.tau_voltage = tau_voltage
        self.tau_calcium = tau_calcium

        self.weight = nn.Parameter(torch.rand(n_synapses) * 0.5 + 0.5)

        self.register_buffer("voltage_trace", torch.zeros(n_synapses))
        self.register_buffer("calcium_trace", torch.zeros(n_synapses))

    def forward(
        self,
        spike_pre: Tensor,
        voltage_post: Tensor,
        spike_post: Tensor,
    ) -> Tensor:
        """Apply voltage-based STDP."""
        pre_mean = spike_pre.float().mean(dim=0)
        post_mean = spike_post.float().mean(dim=0)
        voltage_mean = voltage_post.float().mean(dim=0)

        self.voltage_trace = (
            self.voltage_trace * torch.exp(torch.tensor(-1.0 / self.tau_voltage))
            + voltage_mean
        )
        self.calcium_trace = (
            self.calcium_trace * torch.exp(torch.tensor(-1.0 / self.tau_calcium))
            + post_mean
        )

        potentiation = 0.01 * pre_mean * self.voltage_trace * self.calcium_trace
        depression = -0.01 * self.voltage_trace * (1 - pre_mean) * self.calcium_trace

        delta_w = potentiation + depression
        self.weight.data = torch.clamp(self.weight + delta_w, 0.0, 1.0)

        return self.weight
