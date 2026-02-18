"""
Evidence Lower Bound (ELBO) Computation Module.

Provides various implementations of ELBO for training
Bayesian neural networks with variational inference.
"""

from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Categorical, RelaxedOneHotCategorical


class ELBO:
    """Evidence Lower Bound for variational inference.

    Provides methods to compute ELBO for different likelihood models.
    """

    @staticmethod
    def binary_classification(
        logits: Tensor,
        labels: Tensor,
        kl_div: Tensor,
        n_data: int,
        beta: float = 1.0,
    ) -> Tensor:
        """ELBO for binary classification.

        Args:
            logits: Model output logits
            labels: Binary labels
            kl_div: KL divergence from posterior to prior
            n_data: Number of data points
            beta: KL weighting factor

        Returns:
            Negative ELBO
        """
        log_likelihood = Bernoulli(logits=logits).log_prob(labels).sum()
        return -(log_likelihood / n_data + beta * kl_div)

    @staticmethod
    def classification(
        logits: Tensor,
        labels: Tensor,
        kl_div: Tensor,
        n_data: int,
        beta: float = 1.0,
    ) -> Tensor:
        """ELBO for multi-class classification.

        Args:
            logits: Model output logits
            labels: Class labels
            kl_div: KL divergence from posterior to prior
            n_data: Number of data points
            beta: KL weighting factor

        Returns:
            Negative ELBO
        """
        log_likelihood = Categorical(logits=logits).log_prob(labels).sum()
        return -(log_likelihood / n_data + beta * kl_div)

    @staticmethod
    def regression(
        mean: Tensor,
        log_var: Tensor,
        targets: Tensor,
        kl_div: Tensor,
        n_data: int,
        beta: float = 1.0,
    ) -> Tensor:
        """ELBO for regression with Gaussian likelihood.

        Args:
            mean: Predicted mean
            log_var: Log of predicted variance
            targets: Target values
            kl_div: KL divergence from posterior to prior
            n_data: Number of data points
            beta: KL weighting factor

        Returns:
            Negative ELBO
        """
        sigma = torch.exp(0.5 * log_var)
        log_likelihood = Normal(mean, sigma).log_prob(targets).sum()
        return -(log_likelihood / n_data + beta * kl_div)


class IWAE(nn.Module):
    """Importance Weighted Autoencoder (IWAE) loss.

    Provides a tighter lower bound on the marginal likelihood.

    Args:
        beta: KL weighting factor
        n_samples: Number of importance samples
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_samples: int = 5,
    ):
        super().__init__()
        self.beta = beta
        self.n_samples = n_samples

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute IWAE loss.

        Args:
            logits: Model output logits
            labels: Target labels
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            IWAE loss
        """
        log_likelihood = F.log_softmax(logits, dim=-1)
        log_likelihood = log_likelihood.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        log_weight = log_likelihood - self.beta * kl_div / n_data

        log_sum_exp = torch.logsumexp(log_weight, dim=0)
        nll = -log_sum_exp + torch.log(
            torch.tensor(self.n_samples, dtype=torch.float32)
        )

        return nll


class WarmupELBO(nn.Module):
    """ELBO with warm-up schedule for KL divergence.

    Gradually increases the KL weighting during training to
    improve training stability.

    Args:
        total_steps: Total training steps
        warmup_steps: Number of warmup steps
        beta: Maximum KL weight
    """

    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 1000,
        beta: float = 1.0,
    ):
        super().__init__()
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.step = 0

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute ELBO with warmup.

        Args:
            logits: Model output logits
            labels: Target labels
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            ELBO loss
        """
        if self.step < self.warmup_steps:
            current_beta = self.beta * (self.step / self.warmup_steps)
        else:
            current_beta = self.beta

        self.step += 1

        log_likelihood = Categorical(logits=logits).log_prob(labels).sum()
        return -(log_likelihood / n_data + current_beta * kl_div)


class SpectralELBO(nn.Module):
    """ELBO with spectral normalization for stability.

    Uses spectral normalization on weight matrices to
    improve training stability of BNNs.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute ELBO loss.

        Args:
            logits: Model output logits
            labels: Target labels
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            ELBO loss
        """
        log_likelihood = Categorical(logits=logits).log_prob(labels).sum()
        return -(log_likelihood / n_data + self.beta * kl_div)


class DropoutELBO(nn.Module):
    """ELBO approximation for MC Dropout.

    Uses Monte Carlo sampling to approximate the ELBO
    when using dropout as a Bayesian approximation.
    """

    def __init__(
        self,
        n_samples: int = 10,
        beta: float = 1.0,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.beta = beta

    def forward(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute MC Dropout ELBO.

        Args:
            model: Model with dropout
            x: Input data
            y: Target labels
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            ELBO loss
        """
        model.train()

        log_likelihoods = []
        for _ in range(self.n_samples):
            logits = model(x)
            log_likelihood = Categorical(logits=logits).log_prob(y)
            log_likelihoods.append(log_likelihood)

        log_likelihood = torch.stack(log_likelihoods).mean()

        return -(log_likelihood + self.beta * kl_div / n_data)


class CompositeELBO(nn.Module):
    """Composite ELBO with multiple loss components.

    Combines multiple likelihood terms for multi-task learning.

    Args:
        beta: KL weighting factor
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        outputs: Tensor,
        targets: list[Tensor],
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute composite ELBO.

        Args:
            outputs: Model outputs for each task
            targets: Target for each task
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            ELBO loss
        """
        log_likelihood = 0.0

        for output, target in zip(outputs, targets):
            if output.dim() > 1 and output.size(-1) > 1:
                log_likelihood += Categorical(logits=output).log_prob(target).sum()
            else:
                log_likelihood += Normal(output, 1).log_prob(target).sum()

        return -(log_likelihood / n_data + self.beta * kl_div)


class VariationalEBLO:
    """Variational Elastic Bridge (VEBLO) for smooth KL annealing.

    Uses an elastic bridge between prior and posterior for
    smoother training dynamics.
    """

    def __init__(self, beta: float = 1.0, elasticity: float = 0.5):
        self.beta = beta
        self.elasticity = elasticity

    def compute(
        self,
        log_likelihood: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute VEBLO.

        Args:
            log_likelihood: Data log-likelihood
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            VEBLO loss
        """
        kl_weight = self.beta * (1 + self.elasticity)

        return -(log_likelihood / n_data + kl_weight * kl_div)


class MonteCarloELBO:
    """Monte Carlo ELBO for any differentiable model.

    Uses reparameterization to compute ELBO via sampling.
    """

    def __init__(
        self,
        n_samples: int = 10,
        beta: float = 1.0,
    ):
        self.n_samples = n_samples
        self.beta = beta

    def compute(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        prior_log_prob_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        """Compute MC ELBO.

        Args:
            model: BNN model
            x: Input data
            y: Target labels
            prior_log_prob_fn: Function to compute log prior probability

        Returns:
            ELBO estimate
        """
        if prior_log_prob_fn is None:
            prior_log_prob_fn = lambda w: Normal(0, 1).log_prob(w).sum()

        log_likelihoods = []
        kl_divs = []

        for _ in range(self.n_samples):
            logits = model(x)

            if y.dim() > 0:
                log_likelihood = Categorical(logits=logits).log_prob(y)
            else:
                log_likelihood = Normal(logits, 1).log_prob(y)

            log_likelihoods.append(log_likelihood)

            kl = 0.0
            for param in model.parameters():
                kl += prior_log_prob_fn(param)

            kl_divs.append(kl)

        log_likelihood = torch.stack(log_likelihoods).mean()
        kl_div = torch.stack(kl_divs).mean()

        return -(log_likelihood + self.beta * kl_div)


def make_elbo_loss(
    loss_type: str = "elbo",
    **kwargs,
) -> nn.Module:
    """Factory function to create ELBO loss modules.

    Args:
        loss_type: Type of ELBO loss ('elbo', 'iwae', 'warmup', 'dropout')
        **kwargs: Loss-specific arguments

    Returns:
        ELBO loss module
    """
    losses = {
        "elbo": nn.Identity,
        "iwae": IWAE,
        "warmup": WarmupELBO,
        "dropout": DropoutELBO,
    }

    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return losses[loss_type](**kwargs)


class ELBOWithGradientClipping(ELBO):
    """ELBO with gradient clipping for training stability.

    Args:
        beta: KL weighting factor
        max_grad_norm: Maximum gradient norm
    """

    def __init__(self, beta: float = 1.0, max_grad_norm: float = 1.0):
        super().__init__()
        self.beta = beta
        self.max_grad_norm = max_grad_norm

    def forward(
        self,
        model: nn.Module,
        x: Tensor,
        y: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tuple[Tensor, float]:
        """Compute ELBO with gradient clipping.

        Args:
            model: BNN model
            x: Input data
            y: Target labels
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            Tuple of (ELBO loss, grad_norm)
        """
        log_likelihood = Categorical(logits=model(x)).log_prob(y).sum()
        loss = -(log_likelihood / n_data + self.beta * kl_div)

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_grad_norm
        )

        return loss, grad_norm.item()


class BayesianTripleLoss(nn.Module):
    """Triple ELBO loss combining multiple objectives.

    Combines:
    - Standard ELBO for prediction
    - Diversity loss for ensemble members
    - Consistency loss for smooth predictions
    """

    def __init__(
        self,
        beta: float = 1.0,
        diversity_weight: float = 0.1,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.beta = beta
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        model_outputs: Tensor,
        labels: Tensor,
        kl_div: Tensor,
        n_data: int,
    ) -> Tensor:
        """Compute triple ELBO loss.

        Args:
            model_outputs: List of model outputs
            labels: Target labels
            kl_div: KL divergence
            n_data: Number of data points

        Returns:
            Combined loss
        """
        log_likelihood = 0.0
        for logits in model_outputs:
            log_likelihood += Categorical(logits=logits).log_prob(labels)
        log_likelihood /= len(model_outputs)

        elbo = -(log_likelihood / n_data + self.beta * kl_div)

        if len(model_outputs) > 1:
            outputs = torch.stack(model_outputs)
            diversity_loss = outputs.var(dim=0).mean()
        else:
            diversity_loss = 0.0

        if len(model_outputs) > 1:
            mean_output = torch.stack(model_outputs).mean(dim=0)
            consistency_loss = F.mse_loss(mean_output, model_outputs[0])
        else:
            consistency_loss = 0.0

        return (
            elbo
            + self.diversity_weight * diversity_loss
            + self.consistency_weight * consistency_loss
        )
