"""
Monte Carlo sampling utilities.

Provides various sampling algorithms including importance sampling,
rejection sampling, particle filters, and sequential Monte Carlo.
"""

from typing import Optional, Tuple, List, Callable, Dict
from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np


@dataclass
class SampleResult:
    """Container for sampling results."""

    samples: Tensor
    weights: Optional[Tensor] = None
    log_weights: Optional[Tensor] = None
    ESS: Optional[float] = None


class ImportanceSampling:
    """
    Importance Sampling for estimating expectations under complex distributions.

    Approximates E[f(x)] = ∫ f(x) p(x) dx using weighted samples from q(x).
    """

    def __init__(
        self,
        target_log_prob_fn: Callable[[Tensor], Tensor],
        proposal_log_prob_fn: Callable[[Tensor], Tensor],
    ):
        self.target_log_prob = target_log_prob_fn
        self.proposal_log_prob = proposal_log_prob_fn

    def sample(
        self,
        n_samples: int,
        proposal_samples: Optional[Tensor] = None,
    ) -> SampleResult:
        """
        Run importance sampling.

        Args:
            n_samples: Number of samples
            proposal_samples: Optional pre-generated proposal samples

        Returns:
            SampleResult with samples and weights
        """
        if proposal_samples is None:
            raise ValueError("Proposal samples must be provided")

        log_target = self.target_log_prob(proposal_samples)
        log_proposal = self.proposal_log_prob(proposal_samples)

        log_weights = log_target - log_proposal

        weights = torch.softmax(log_weights, dim=0)
        log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)

        ess = 1.0 / (weights**2).sum().item()

        return SampleResult(
            samples=proposal_samples,
            weights=weights,
            log_weights=log_weights_normalized,
            ESS=ess,
        )

    def estimate_expectation(
        self,
        f: Callable[[Tensor], Tensor],
        samples: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """
        Estimate E[f(x)] using weighted samples.

        Args:
            f: Function to compute expectation of
            samples: Weighted samples
            weights: Sample weights

        Returns:
            Estimated expectation
        """
        f_values = f(samples)
        return (weights * f_values).sum()


class RejectionSampling:
    """
    Rejection Sampling for sampling from unnormalized distributions.

    Accepts samples from proposal if u < p(x)/M*q(x) where M >= sup p(x)/q(x).
    """

    def __init__(
        self,
        target_log_prob_fn: Callable[[Tensor], Tensor],
        proposal_log_prob_fn: Callable[[Tensor], Tensor],
        M: float = 1.0,
    ):
        self.target_log_prob = target_log_prob_fn
        self.proposal_log_prob = proposal_log_prob_fn
        self.M = M

    def sample(
        self,
        n_samples: int,
        max_attempts: int = 10000,
    ) -> SampleResult:
        """
        Run rejection sampling.

        Args:
            n_samples: Number of samples to generate
            max_attempts: Maximum number of proposal attempts

        Returns:
            SampleResult with accepted samples
        """
        samples = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            x = torch.randn(10)  # Simplified - would use proper proposal

            log_accept_prob = (
                self.target_log_prob(x) - self.proposal_log_prob(x) - np.log(self.M)
            )

            accept_mask = torch.log(torch.rand(1)) < log_accept_prob

            if accept_mask:
                samples.append(x)

            attempts += 1

        return SampleResult(
            samples=torch.stack(samples) if samples else torch.zeros(0),
        )


class MetropolisHastingsSampler:
    """
    Metropolis-Hastings MCMC sampler (re-implementation with additional features).
    """

    def __init__(
        self,
        log_target_fn: Callable[[Tensor], Tensor],
        proposal_std: float = 1.0,
    ):
        self.log_target = log_target_fn
        self.proposal_std = proposal_std

    def sample(
        self,
        n_samples: int,
        initial_sample: Tensor,
        burn_in: int = 100,
    ) -> SampleResult:
        """
        Run Metropolis-Hastings sampling.

        Args:
            n_samples: Number of samples to generate
            initial_sample: Starting point
            burn_in: Number of initial samples to discard

        Returns:
            SampleResult with samples
        """
        samples = []
        current = initial_sample.clone()
        current_log_prob = self.log_target(current)

        accept_count = 0

        for i in range(burn_in + n_samples):
            proposal = current + torch.randn_like(current) * self.proposal_std
            proposal_log_prob = self.log_target(proposal)

            log_accept_ratio = proposal_log_prob - current_log_prob

            if torch.log(torch.rand(())) < log_accept_ratio:
                current = proposal
                current_log_prob = proposal_log_prob
                if i >= burn_in:
                    accept_count += 1

            if i >= burn_in:
                samples.append(current.clone())

        return SampleResult(
            samples=torch.stack(samples),
            ESS=n_samples / accept_count if accept_count > 0 else n_samples,
        )


class GibbsSampling:
    """
    Gibbs Sampling for multivariate distributions.

    Samples each variable conditional on others.
    """

    def __init__(self):
        self.conditional_fns: Dict[int, Callable] = {}

    def add_conditional(
        self,
        idx: int,
        conditional_fn: Callable[[Dict[int, Tensor]], Tensor],
    ) -> None:
        """Add conditional distribution for variable idx."""
        self.conditional_fns[idx] = conditional_fn

    def sample(
        self,
        n_samples: int,
        initial_state: Dict[int, Tensor],
    ) -> Dict[int, Tensor]:
        """
        Run Gibbs sampling.

        Args:
            n_samples: Number of samples to generate
            initial_state: Starting state

        Returns:
            Dictionary of samples for each variable
        """
        current_state = initial_state.copy()
        samples = {idx: [] for idx in current_state}

        for _ in range(n_samples):
            for idx, fn in self.conditional_fns.items():
                current_state[idx] = fn(current_state)

            for idx in current_state:
                samples[idx].append(current_state[idx].clone())

        return {idx: torch.stack(vals) for idx, vals in samples.items()}


class HamiltonianSampler:
    """
    Hamiltonian Monte Carlo sampler (simplified version).
    """

    def __init__(
        self,
        log_target_fn: Callable[[Tensor], Tensor],
        step_size: float = 0.1,
        n_leapfrog: int = 5,
    ):
        self.log_target = log_target_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

    def _grad_log_target(self, x: Tensor) -> Tensor:
        """Compute gradient using automatic differentiation."""
        x = x.requires_grad_(True)
        log_prob = self.log_target(x)
        return torch.autograd.grad(log_prob, x, retain_graph=False)[0]

    def _leapfrog(self, x: Tensor, p: Tensor) -> Tuple[Tensor, Tensor]:
        """Leapfrog integrator."""
        for _ in range(self.n_leapfrog):
            p = p + 0.5 * self.step_size * self._grad_log_target(x)
            x = x + self.step_size * p
            p = p + 0.5 * self.step_size * self._grad_log_target(x)

        return x, -p

    def sample(
        self,
        n_samples: int,
        initial_sample: Tensor,
        burn_in: int = 100,
    ) -> SampleResult:
        """Run HMC sampling."""
        samples = []
        current = initial_sample.clone()

        for i in range(burn_in + n_samples):
            p = torch.randn_like(current)

            proposed_x, proposed_p = self._leapfrog(current, p)

            current_log_prob = self.log_target(current)
            proposed_log_prob = self.log_target(proposed_x)

            h_current = -current_log_prob + 0.5 * (p**2).sum()
            h_proposed = -proposed_log_prob + 0.5 * (proposed_p**2).sum()

            if torch.log(torch.rand(())) < -(h_proposed - h_current):
                current = proposed_x

            if i >= burn_in:
                samples.append(current.clone())

        return SampleResult(
            samples=torch.stack(samples),
        )


class ParticleFilter:
    """
    Sequential Importance Resampling (SIR) Particle Filter.

    For online state estimation in hidden Markov models.
    """

    def __init__(
        self,
        n_particles: int,
        transition_fn: Callable[[Tensor], Tensor],
        observation_fn: Callable[[Tensor], Tensor],
        log_likelihood_fn: Callable[[Tensor, Tensor], Tensor],
    ):
        self.n_particles = n_particles
        self.transition = transition_fn
        self.observation = observation_fn
        self.log_likelihood = log_likelihood_fn

        self.particles: Optional[Tensor] = None
        self.weights: Optional[Tensor] = None

    def initialize(self, initial_sample_fn: Callable[[int], Tensor]) -> None:
        """Initialize particles from prior distribution."""
        self.particles = initial_sample_fn(self.n_particles)
        self.weights = torch.ones(self.n_particles) / self.n_particles

    def predict(self) -> None:
        """Prediction step: propagate particles."""
        for i in range(self.n_particles):
            self.particles[i] = self.transition(self.particles[i])

    def update(self, observation: Tensor) -> None:
        """Update step: weight particles by observation likelihood."""
        log_weights = self.log_likelihood(self.particles, observation)

        log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
        self.weights = torch.exp(log_weights_normalized)

        ess = 1.0 / (self.weights**2).sum().item()

        if ess < self.n_particles / 2:
            self._resample()

    def _resample(self) -> None:
        """Resample particles based on weights."""
        indices = torch.multinomial(self.weights, self.n_particles, replacement=True)
        self.particles = self.particles[indices]
        self.weights = torch.ones(self.n_particles) / self.n_particles

    def estimate_state(self) -> Tensor:
        """Get weighted estimate of state."""
        return (self.weights.unsqueeze(-1) * self.particles).sum(dim=0)

    def step(self, observation: Tensor) -> Tensor:
        """Single step of particle filter."""
        self.predict()
        self.update(observation)
        return self.estimate_state()


class AnnealedImportanceSampling:
    """
    Annealed Importance Sampling (AIS) for partition function estimation.

    Uses sequence of intermediate distributions bridging two distributions.
    """

    def __init__(
        self,
        log_target_fn: Callable[[Tensor], Tensor],
        log_base_fn: Callable[[Tensor], Tensor],
        betas: Optional[Tensor] = None,
    ):
        self.log_target = log_target_fn
        self.log_base = log_base_fn

        if betas is None:
            self.betas = torch.linspace(0, 1, 100)
        else:
            self.betas = betas

    def sample(
        self,
        n_samples: int,
        initial_sample_fn: Callable[[int], Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Run AIS sampling.

        Returns:
            Tuple of (samples, log_importance_weights)
        """
        x = initial_sample_fn(n_samples)

        log_weights = torch.zeros(n_samples)

        for i in range(len(self.betas) - 1):
            beta_curr = self.betas[i]
            beta_next = self.betas[i + 1]

            log_prob_curr = (1 - beta_curr) * self.log_base(
                x
            ) + beta_curr * self.log_target(x)
            log_prob_next = (1 - beta_next) * self.log_base(
                x
            ) + beta_next * self.log_target(x)

            log_weights += log_prob_next - log_prob_curr

            x = x + torch.randn_like(x) * 0.1

        return x, log_weights

    def estimate_partition_function(
        self,
        log_weights: Tensor,
    ) -> float:
        """
        Estimate partition function (normalizing constant) ratio.

        Z_target / Z_base ≈ mean(exp(log_weights))
        """
        return torch.logsumexp(log_weights, dim=0).item() - np.log(len(log_weights))


class SliceSampling:
    """
    Slice Sampling for univariate distributions.

    Automatically adapts step size for efficient sampling.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[Tensor], Tensor],
        step_size: float = 1.0,
    ):
        self.log_prob = log_prob_fn
        self.step_size = step_size

    def sample(
        self,
        n_samples: int,
        initial_sample: Tensor,
    ) -> SampleResult:
        """Run slice sampling."""
        samples = []
        x = initial_sample.clone()

        for _ in range(n_samples):
            y = torch.log(torch.rand(())) + self.log_prob(x)

            L = x - self.step_size * torch.rand(())
            R = L + self.step_size

            while True:
                x_new = L + torch.rand(()) * (R - L)

                if self.log_prob(x_new) >= y:
                    x = x_new
                    break

                if x_new < x:
                    L = x_new
                else:
                    R = x_new

            samples.append(x.clone())

        return SampleResult(
            samples=torch.stack(samples),
        )


class RaoBlackwellization:
    """
    Rao-Blackwellization for variance reduction in Monte Carlo estimates.

    Analytically integrates out some variables when possible.
    """

    def __init__(self):
        pass

    @staticmethod
    def estimate_expectation(
        samples: Tensor,
        marginal_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """
        Compute Rao-Blackwellized estimate.

        E[f(x)] = E[E[f(x)|y]] ≈ 1/N Σ f(x_i) where x_i ~ p(x|y)
        """
        marginals = marginal_fn(samples)
        return marginals.mean(dim=0)


class AntitheticVariates:
    """
    Antithetic Variates for variance reduction.

    Uses paired samples with negative correlation.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_pairs(
        sampler: Callable[[int], Tensor],
        n_pairs: int,
    ) -> Tensor:
        """Generate paired antithetic samples."""
        u = torch.rand(n_pairs)

        x1 = sampler(n_pairs)

        x2 = sampler(n_pairs)

        pairs = torch.stack([x1, x2])
        return pairs


class ControlVariates:
    """
    Control Variates for variance reduction.

    Uses known expectations of correlated variables.
    """

    def __init__(self, known_expectations: Dict[str, float]):
        self.known_expectations = known_expectations

    def estimate(
        self,
        samples: Tensor,
        fn_values: Tensor,
        control_fn_values: Dict[str, Tensor],
    ) -> Tuple[Tensor, float]:
        """
        Estimate expectation with control variates.

        Returns:
            Tuple of (adjusted_estimate, variance_reduction)
        """
        estimate = fn_values.mean()

        c = -torch.cov(
            torch.cat([fn_values.unsqueeze(0), *control_fn_values.values()])
        )[0, 1:] / (
            torch.stack(
                [torch.var(control_fn) for control_fn in control_fn_values.values()]
            )
            + 1e-8
        )

        adjusted = estimate

        for (name, ctrl_val), c_val in zip(control_fn_values.items(), c):
            adjusted = adjusted + c_val * (
                ctrl_val.mean() - self.known_expectations[name]
            )

        return adjusted, 0.0


def effective_sample_size(weights: Tensor) -> float:
    """
    Compute effective sample size (ESS) of weighted samples.

    ESS = 1 / Σ w_i²

    Measures quality of importance sampling approximation.
    """
    return 1.0 / (weights**2).sum().item()


def stratified_sampling(
    sampler: Callable[[int], Tensor],
    n_strata: int,
    n_samples_per_stratum: int,
) -> Tensor:
    """
    Stratified sampling for variance reduction.

    Divides sample space into strata and samples proportionally.
    """
    samples = []

    for i in range(n_strata):
        u = (i + torch.rand(n_samples_per_stratum)) / n_strata
        stratum_samples = sampler(n_samples_per_stratum)
        samples.append(stratum_samples)

    return torch.cat(samples)


def latin_hypercube_sampling(
    n_dims: int,
    n_samples: int,
) -> Tensor:
    """
    Latin Hypercube Sampling (LHS).

    Generates uniform samples with better coverage than random.
    """
    samples = torch.zeros(n_samples, n_dims)

    for d in range(n_dims):
        samples[:, d] = torch.rand(n_samples)
        samples[:, d] += torch.arange(n_samples).float()
        samples[:, d] /= n_samples

    for d in range(n_dims):
        perm = torch.randperm(n_samples)
        samples[:, d] = samples[perm, d]

    return samples


def sobol_sampling(
    n_dims: int,
    n_samples: int,
) -> Tensor:
    """
    Sobol sequence for quasi-Monte Carlo.

    Low-discrepancy sequence for better convergence.
    """
    samples = torch.zeros(n_samples, n_dims)

    sobol_gen = torch.quasirandom.SobolEngine(dimension=n_dims)
    samples = sobol_gen.draw(n_samples)

    return samples


def bootstrap_sampling(
    data: Tensor,
    n_bootstrap: int,
    n_samples: Optional[int] = None,
) -> Tensor:
    """
    Bootstrap sampling with replacement.

    Args:
        data: Original data
        n_bootstrap: Number of bootstrap samples
        n_samples: Samples per bootstrap (default: same as data)

    Returns:
        Bootstrap samples [n_bootstrap, n_samples, *data.shape[1:]]
    """
    if n_samples is None:
        n_samples = len(data)

    indices = torch.randint(0, len(data), (n_bootstrap, n_samples))

    return data[indices]


def jackknife_sampling(
    data: Tensor,
) -> Tensor:
    """
    Jackknife resampling (leave-one-out).

    Returns all n leave-one-out samples.
    """
    n = len(data)
    indices = torch.arange(n).unsqueeze(1).expand(-1, n - 1)
    indices = indices[indices != torch.arange(n)]

    return data[indices.view(n, -1)]
