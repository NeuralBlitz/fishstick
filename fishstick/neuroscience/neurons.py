"""Neuron models: Leaky Integrate-and-Fire and Hodgkin-Huxley."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class LeakyIntegrateAndFire(nn.Module):
    """Leaky Integrate-and-Fire (LIF) neuron model.

    The LIF neuron integrates synaptic inputs with leak conductance and fires
    a spike when the membrane potential reaches threshold, then resets.

    Dynamics:
        τ_m * dv/dt = -(v - v_rest) + R * I(t)
        v(t+) = V_reset after spike

    Args:
        n_neurons: Number of neurons in the layer
        tau_mem: Membrane time constant (ms)
        tau_ref: Refractory period (ms)
        v_thresh: Spike threshold voltage (mV)
        v_reset: Reset voltage after spike (mV)
        v_rest: Resting potential (mV)
        r_mem: Membrane resistance (MΩ)
        dt: Simulation time step (ms)
    """

    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,
        tau_ref: float = 2.0,
        v_thresh: float = -50.0,
        v_reset: float = -70.0,
        v_rest: float = -70.0,
        r_mem: float = 10.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_mem = tau_mem
        self.tau_ref = tau_ref
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.r_mem = r_mem
        self.dt = dt

        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        self.beta = r_mem * (1 - torch.exp(torch.tensor(-dt / tau_mem)))

    def forward(
        self,
        current: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of LIF neuron.

        Args:
            current: Input current tensor (batch, n_neurons)
            state: Optional (membrane_potential, refractory_time) tuple

        Returns:
            Tuple of (spikes, new_state)
            - spikes: Binary spike tensor (batch, n_neurons)
            - new_state: (membrane_potential, refractory_time) tuple
        """
        batch_size = current.shape[0]

        if state is None:
            v_m = torch.full(
                (batch_size, self.n_neurons), self.v_rest, device=current.device
            )
            refractory = torch.zeros(batch_size, self.n_neurons, device=current.device)
        else:
            v_m, refractory = state

        v_m = self.alpha * v_m + self.beta * current

        spikes = (v_m >= self.v_thresh).float()
        v_m = torch.where(spikes.bool(), self.v_reset, v_m)

        refractory = torch.clamp(refractory - self.dt, min=0.0)
        refractory = torch.where(spikes.bool(), self.tau_ref, refractory)

        v_m = torch.where(refractory > 0, self.v_rest, v_m)

        return spikes, (v_m, refractory)


class HodgkinHuxley(nn.Module):
    """Hodgkin-Huxley neuron model.

    The HH model describes membrane potential dynamics using ionic currents:
        C_m * dV/dt = I_ext - I_Na - I_K - I_L

    With channel dynamics:
        I_Na = g_Na * m^3 * h * (V - E_Na)
        I_K = g_K * n^4 * (V - E_K)
        I_L = g_L * (V - E_L)

    Args:
        n_neurons: Number of neurons
        dt: Simulation time step (ms)
        c_m: Membrane capacitance (μF/cm²)
        g_na: Maximum Na conductance (mS/cm²)
        g_k: Maximum K conductance (mS/cm²)
        g_l: Leak conductance (mS/cm²)
        e_na: Na reversal potential (mV)
        e_k: K reversal potential (mV)
        e_l: Leak reversal potential (mV)
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 0.05,
        c_m: float = 1.0,
        g_na: float = 120.0,
        g_k: float = 36.0,
        g_l: float = 0.3,
        e_na: float = 50.0,
        e_k: float = -77.0,
        e_l: float = -54.387,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.c_m = c_m
        self.g_na = g_na
        self.g_k = g_k
        self.g_l = g_l
        self.e_na = e_na
        self.e_k = e_k
        self.e_l = e_l

    def _alpha_n(self, v: Tensor) -> Tensor:
        return 0.01 * (v + 55) / (1 - torch.exp(-(v + 55) / 10))

    def _beta_n(self, v: Tensor) -> Tensor:
        return 0.125 * torch.exp(-(v + 65) / 80)

    def _alpha_m(self, v: Tensor) -> Tensor:
        return 0.1 * (v + 40) / (1 - torch.exp(-(v + 40) / 10))

    def _beta_m(self, v: Tensor) -> Tensor:
        return 4.0 * torch.exp(-(v + 65) / 18)

    def _alpha_h(self, v: Tensor) -> Tensor:
        return 0.07 * torch.exp(-(v + 65) / 20)

    def _beta_h(self, v: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-(v + 35) / 10))

    def forward(
        self,
        current: Tensor,
        state: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass of Hodgkin-Huxley neuron.

        Args:
            current: Input current tensor (batch, n_neurons)
            state: Optional (v, m, n, h) state tuple

        Returns:
            Tuple of (spikes, new_state)
            - spikes: Binary spike tensor
            - new_state: (v, m, n, h) tuple
        """
        batch_size = current.shape[0]
        device = current.device

        if state is None:
            v = torch.full((batch_size, self.n_neurons), -65.0, device=device)
            m = torch.full((batch_size, self.n_neurons), 0.05, device=device)
            n = torch.full((batch_size, self.n_neurons), 0.32, device=device)
            h = torch.full((batch_size, self.n_neurons), 0.6, device=device)
        else:
            v, m, n, h = state

        alpha_n = self._alpha_n(v)
        beta_n = self._beta_n(v)
        n = n + self.dt * (alpha_n * (1 - n) - beta_n * n)

        alpha_m = self._alpha_m(v)
        beta_m = self._beta_m(v)
        m = m + self.dt * (alpha_m * (1 - m) - beta_m * m)

        alpha_h = self._alpha_h(v)
        beta_h = self._beta_h(v)
        h = h + self.dt * (alpha_h * (1 - h) - beta_h * h)

        i_na = self.g_na * m**3 * h * (v - self.e_na)
        i_k = self.g_k * n**4 * (v - self.e_k)
        i_l = self.g_l * (v - self.e_l)

        dv = (current - i_na - i_k - i_l) / self.c_m
        v = v + self.dt * dv

        spikes = (v >= 30.0).float()

        return spikes, (v, m, n, h)


class Izhikevich(nn.Module):
    """Izhikevich neuron model - simple yet biologically realistic.

    Combines simplicity of LIF with rich dynamics of HH model:
        v' = 0.04v² + 5v + 140 - u + I
        u' = a(bv - u)

    After spike: v = c, u = u + d

    Args:
        n_neurons: Number of neurons
        dt: Simulation time step (ms)
        a: Time scale of recovery variable
        b: Sensitivity of recovery to subthreshold oscillations
        c: After-spike reset of membrane potential
        d: After-spike reset of recovery variable
    """

    def __init__(
        self,
        n_neurons: int,
        dt: float = 1.0,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(
        self,
        current: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass of Izhikevich neuron.

        Args:
            current: Input current tensor (batch, n_neurons)
            state: Optional (v, u) state tuple

        Returns:
            Tuple of (spikes, new_state)
        """
        batch_size = current.shape[0]
        device = current.device

        if state is None:
            v = torch.full((batch_size, self.n_neurons), -65.0, device=device)
            u = torch.full((batch_size, self.n_neurons), self.b * -65.0, device=device)
        else:
            v, u = state

        v_prev = v.clone()

        v = v + self.dt * (0.04 * v**2 + 5 * v + 140 - u + current)
        u = u + self.dt * self.a * (self.b * v_prev - u)

        spikes = (v >= 30.0).float()

        v = torch.where(spikes.bool(), self.c, v)
        u = torch.where(spikes.bool(), u + self.d, u)

        return spikes, (v, u)


class AdaptiveLIF(nn.Module):
    """Adaptive Leaky Integrate-and-Fire neuron with spike-frequency adaptation.

    Adds adaptation current that increases with each spike:
        τ_a * da/dt = -a
        a += b * spike

    Args:
        n_neurons: Number of neurons
        tau_mem: Membrane time constant (ms)
        tau_adapt: Adaptation time constant (ms)
        a_adapt: Subthreshold adaptation (mV)
        b_adapt: Spike-triggered adaptation (mV)
    """

    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,
        tau_adapt: float = 200.0,
        a_adapt: float = 0.0,
        b_adapt: float = 10.0,
        v_thresh: float = -50.0,
        v_reset: float = -70.0,
        v_rest: float = -70.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.a_adapt = a_adapt
        self.b_adapt = b_adapt
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt

        self.alpha_mem = torch.exp(torch.tensor(-dt / tau_mem))
        self.alpha_adapt = torch.exp(torch.tensor(-dt / tau_adapt))

    def forward(
        self,
        current: Tensor,
        state: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """Forward pass of adaptive LIF neuron."""
        batch_size = current.shape[0]

        if state is None:
            v_m = torch.full(
                (batch_size, self.n_neurons), self.v_rest, device=current.device
            )
            adapt = torch.zeros(batch_size, self.n_neurons, device=current.device)
            refractory = torch.zeros(batch_size, self.n_neurons, device=current.device)
        else:
            v_m, adapt, refractory = state

        v_m = self.alpha_mem * v_m + (1 - self.alpha_mem) * (
            self.v_rest - adapt + current
        )

        spikes = (v_m >= self.v_thresh).float()

        v_m = torch.where(spikes.bool(), self.v_reset, v_m)
        adapt = (
            self.alpha_adapt * adapt + (1 - self.alpha_adapt) * self.b_adapt * spikes
        )

        refractory = torch.clamp(refractory - self.dt, min=0.0)
        v_m = torch.where(refractory > 0, self.v_rest, v_m)

        return spikes, (v_m, adapt, refractory)
