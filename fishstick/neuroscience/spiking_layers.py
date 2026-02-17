"""Spiking neural network layers and utilities."""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from torch import Tensor


class SpikingDense(nn.Module):
    """Dense spiking neural network layer with LIF neurons.

    Implements a fully connected layer of spiking neurons using the
    Leaky Integrate-and-Fire model with rate-coded inputs.

    Args:
        in_features: Number of input features
        out_features: Number of output neurons
        tau_mem: Membrane time constant (ms)
        tau_syn_exc: Excitatory synaptic time constant (ms)
        tau_syn_inh: Inhibitory synaptic time constant (ms)
        v_thresh: Spike threshold (mV)
        v_reset: Reset voltage (mV)
        v_rest: Resting voltage (mV)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_mem: float = 20.0,
        tau_syn_exc: float = 5.0,
        tau_syn_inh: float = 10.0,
        v_thresh: float = -50.0,
        v_reset: float = -70.0,
        v_rest: float = -70.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_mem = tau_mem
        self.tau_syn_exc = tau_syn_exc
        self.tau_syn_inh = tau_syn_inh
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))
        self.alpha_exc = torch.exp(torch.tensor(-1.0 / tau_syn_exc))
        self.alpha_inh = torch.exp(torch.tensor(-1.0 / tau_syn_inh))

    def forward(
        self,
        rates: Tensor,
        state: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass with rate-coded input.

        Args:
            rates: Input firing rates (batch, in_features)
            state: Optional (v_mem, i_exc, i_inh, spike_history)

        Returns:
            Tuple of (spikes, new_state)
        """
        batch_size = rates.shape[0]
        device = rates.device

        if state is None:
            v_mem = torch.full(
                (batch_size, self.out_features), self.v_rest, device=device
            )
            i_exc = torch.zeros(batch_size, self.out_features, device=device)
            i_inh = torch.zeros(batch_size, self.out_features, device=device)
            spike_history = torch.zeros(batch_size, self.out_features, device=device)
        else:
            v_mem, i_exc, i_inh, spike_history = state

        i_exc = self.alpha_exc * i_exc + (1 - self.alpha_exc) * torch.matmul(
            rates, self.weight.t()
        )
        i_inh = self.alpha_inh * i_inh

        current = i_exc - i_inh + self.bias
        v_mem = self.alpha_mem * v_mem + (1 - self.alpha_mem) * (
            self.v_rest - v_mem + current
        )

        spikes = (v_mem >= self.v_thresh).float()
        v_mem = torch.where(spikes.bool(), self.v_reset, v_mem)

        spike_history = spikes

        return spikes, (v_mem, i_exc, i_inh, spike_history)


class SpikingConv2d(nn.Module):
    """2D Convolutional spiking neural network layer.

    Implements convolutional spiking neurons for processing spatial data.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Convolution stride
        padding: Padding size
        tau_mem: Membrane time constant (ms)
        v_thresh: Spike threshold (mV)
        v_reset: Reset voltage (mV)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        tau_mem: float = 20.0,
        v_thresh: float = -50.0,
        v_reset: float = -70.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.tau_mem = tau_mem
        self.v_thresh = v_thresh
        self.v_reset = v_reset

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))

    def forward(
        self,
        spikes: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            spikes: Input spikes (batch, channels, height, width)
            state: Optional (v_mem, spike_history)

        Returns:
            Tuple of (output_spikes, new_state)
        """
        batch_size = spikes.shape[0]
        device = spikes.device

        if state is None:
            v_mem = torch.zeros_like(spikes[:, : self.out_channels, :, :])
            spike_history = torch.zeros_like(v_mem)
        else:
            v_mem, spike_history = state

        current = self.conv(spikes.float())
        v_mem = self.alpha_mem * v_mem + (1 - self.alpha_mem) * current

        out_spikes = (v_mem >= self.v_thresh).float()
        v_mem = torch.where(out_spikes.bool(), self.v_reset, v_mem)

        return out_spikes, (v_mem, out_spikes)


class LiquidStateMachine(nn.Module):
    """Liquid State Machine (LSM) - recurrent spiking network with liquid connectivity.

    The LSM consists of a randomly connected recurrent network of spiking
    neurons that projects to a read-out layer. It serves as a reservoir
    computing framework for temporal pattern recognition.

    Args:
        input_dim: Input dimension
        reservoir_size: Number of neurons in reservoir
        connectivity: Connection probability
        tau_mem: Membrane time constant (ms)
        tau_syn: Synaptic time constant (ms)
        spectral_radius: Spectral radius of reservoir weight matrix
    """

    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        connectivity: float = 0.1,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        spectral_radius: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size

        self.input_weight = nn.Parameter(torch.randn(reservoir_size, input_dim) * 0.1)

        reservoir_weight = torch.randn(reservoir_size, reservoir_size)
        mask = torch.rand(reservoir_size, reservoir_size) < connectivity
        reservoir_weight = reservoir_weight * mask.float()

        eigenvalues = torch.linalg.eigvalsh(reservoir_weight)
        max_eigenvalue = eigenvalues[-1]
        reservoir_weight = reservoir_weight * (spectral_radius / max_eigenvalue)

        self.register_buffer("reservoir_weight", reservoir_weight)

        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))
        self.alpha_syn = torch.exp(torch.tensor(-1.0 / tau_syn))

    def forward(
        self,
        rates: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
        n_steps: int = 10,
    ) -> Tuple[Tensor, Tensor]:
        """Simulate reservoir for multiple time steps.

        Args:
            rates: Input rates (batch, input_dim)
            state: Optional (v_mem, current)
            n_steps: Number of simulation steps

        Returns:
            Tuple of (readout, final_state)
        """
        batch_size = rates.shape[0]
        device = rates.device

        if state is None:
            v_mem = torch.zeros(batch_size, self.reservoir_size, device=device)
            current = torch.zeros(batch_size, self.reservoir_size, device=device)
        else:
            v_mem, current = state

        outputs = []

        for _ in range(n_steps):
            input_current = torch.matmul(rates, self.input_weight.t())
            recurrent_current = torch.matmul(
                (current > 0).float(), self.reservoir_weight
            )

            current = self.alpha_syn * current + (1 - self.alpha_syn) * (
                input_current + recurrent_current
            )
            v_mem = self.alpha_mem * v_mem + (1 - self.alpha_mem) * current

            spikes = (v_mem > 0).float()
            outputs.append(spikes)

        readout = torch.stack(outputs, dim=1).mean(dim=1)

        return readout, (v_mem, current)


class SpikingAttention(nn.Module):
    """Spiking neural network with attention mechanism.

    Implements attention-based communication between spiking neuron groups.

    Args:
        n_groups: Number of neuron groups
        group_size: Neurons per group
        tau_mem: Membrane time constant
        attention_dim: Dimension for attention computation
    """

    def __init__(
        self,
        n_groups: int,
        group_size: int,
        tau_mem: float = 20.0,
        attention_dim: int = 64,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.group_size = group_size
        self.attention_dim = attention_dim

        self.query = nn.Linear(group_size, attention_dim)
        self.key = nn.Linear(group_size, attention_dim)
        self.value = nn.Linear(group_size, attention_dim)

        self.v_mem = nn.Parameter(torch.randn(n_groups, group_size) * 0.1)

        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))

    def forward(
        self,
        group_spikes: Tensor,
    ) -> Tensor:
        """Apply spiking attention.

        Args:
            group_spikes: Spikes from each group (batch, n_groups, group_size)

        Returns:
            Attended output (batch, n_groups, group_size)
        """
        batch_size = group_spikes.shape[0]

        q = self.query(group_spikes)
        k = self.key(group_spikes)
        v = self.value(group_spikes)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attended = torch.matmul(attn_weights, v)

        return attended


class ThresholdDependentPlasticity(nn.Module):
    """Layer with threshold-dependent plasticity (TDP).

    Implements a learning rule where synaptic plasticity depends on
    the post-synaptic neuron's membrane potential threshold.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        tau_mem: Membrane time constant
        learning_rate: STDP learning rate
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_mem: float = 20.0,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_mem = tau_mem
        self.learning_rate = learning_rate

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.v_mem = nn.Parameter(torch.zeros(out_features))

        self.alpha_mem = torch.exp(torch.tensor(-1.0 / tau_mem))

    def forward(
        self,
        spikes_in: Tensor,
    ) -> Tensor:
        """Forward pass with online plasticity.

        Args:
            spikes_in: Input spikes (batch, in_features)

        Returns:
            Output spikes (batch, out_features)
        """
        current = torch.matmul(spikes_in.float(), self.weight.t())

        v_mem = self.alpha_mem * self.v_mem + (1 - self.alpha_mem) * current

        self.v_mem.data = v_mem.detach()

        spikes_out = (v_mem > 0).float()

        if self.training:
            pre = spikes_in.float().mean(dim=0)
            post = spikes_out.detach().float().mean(dim=0)

            delta_w = self.learning_rate * torch.outer(post, pre)
            self.weight.data = self.weight.data + delta_w

        return spikes_out
