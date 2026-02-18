"""
Adversarial Loss Functions

Advanced adversarial loss implementations including WGAN-GP,
hinge-based losses, and distribution matching losses.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WGAN_GPLoss(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty Loss.

    Implements WGAN-GP for stable GAN training with improved gradient behavior.

    Args:
        lambda_gp: Gradient penalty coefficient.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        lambda_gp: float = 10.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_gp = lambda_gp
        self.reduction = reduction

    def forward(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
        fake_samples: Tensor,
        real_samples: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        real_loss = -real_scores.mean()
        fake_loss = fake_scores.mean()

        gp = self._gradient_penalty(real_samples, fake_samples)

        d_loss = real_loss + fake_loss + self.lambda_gp * gp
        g_loss = -fake_scores.mean()

        if self.reduction == "sum":
            return d_loss.sum(), g_loss.sum()
        return d_loss, g_loss

    def _gradient_penalty(
        self,
        real_samples: Tensor,
        fake_samples: Tensor,
    ) -> Tensor:
        batch_size = real_samples.size(0)
        device = real_samples.device

        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        interpolated_scores = self._get_discriminator_scores(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def _get_discriminator_scores(self, x: Tensor) -> Tensor:
        return x.mean()


class HingeLossGAN(nn.Module):
    """
    Hinge Loss for GAN Training.

    Stable GAN training using hinge loss for discriminator.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        real_loss = F.relu(1.0 - real_scores).mean()
        fake_loss = F.relu(1.0 + fake_scores).mean()

        d_loss = real_loss + fake_loss
        g_loss = -fake_scores.mean()

        if self.reduction == "sum":
            return d_loss.sum(), g_loss.sum()
        return d_loss, g_loss


class LeastSquaresGANLoss(nn.Module):
    """
    Least Squares GAN Loss.

    Uses least squares objective for more stable training.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        real_loss = ((real_scores - 1) ** 2).mean()
        fake_loss = (fake_scores**2).mean()

        d_loss = 0.5 * (real_loss + fake_loss)
        g_loss = ((fake_scores - 1) ** 2).mean()

        if self.reduction == "sum":
            return d_loss.sum(), g_loss.sum()
        return d_loss, g_loss


class DistributionMatchingLoss(nn.Module):
    """
    Distribution Matching Loss.

    Minimizes the discrepancy between generated and real distributions
    using multiple statistical tests.

    Args:
        matching_type: Type of distribution matching ('mmd', 'kl', 'js').
        kernel_sigma: Sigma for RBF kernel in MMD.
    """

    def __init__(
        self,
        matching_type: str = "mmd",
        kernel_sigma: float = 1.0,
    ):
        super().__init__()
        self.matching_type = matching_type
        self.kernel_sigma = kernel_sigma

    def forward(self, real_samples: Tensor, fake_samples: Tensor) -> Tensor:
        if self.matching_type == "mmd":
            return self._mmd_loss(real_samples, fake_samples)
        elif self.matching_type == "kl":
            return self._kl_divergence_loss(real_samples, fake_samples)
        elif self.matching_type == "js":
            return self._js_divergence_loss(real_samples, fake_samples)
        else:
            raise ValueError(f"Unknown matching type: {self.matching_type}")

    def _rbf_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(1).expand(batch_size, batch_size, -1)
        y = y.unsqueeze(0).expand(batch_size, batch_size, -1)
        diff = x - y
        diff = diff.pow(2).sum(dim=2)
        return torch.exp(-diff / (2 * self.kernel_sigma**2))

    def _mmd_loss(self, real_samples: Tensor, fake_samples: Tensor) -> Tensor:
        batch_size = real_samples.size(0)

        real_kernel = self._rbf_kernel(real_samples, real_samples)
        fake_kernel = self._rbf_kernel(fake_samples, fake_samples)
        cross_kernel = self._rbf_kernel(real_samples, fake_samples)

        mmd = real_kernel.mean() + fake_kernel.mean() - 2 * cross_kernel.mean()
        return mmd

    def _kl_divergence_loss(self, real_samples: Tensor, fake_samples: Tensor) -> Tensor:
        real_mean = real_samples.mean(dim=0)
        fake_mean = fake_samples.mean(dim=0)
        real_var = real_samples.var(dim=0) + 1e-6
        fake_var = fake_samples.var(dim=0) + 1e-6

        kl = 0.5 * (
            (real_var / fake_var)
            + (fake_mean - real_mean) ** 2 / fake_var
            - 1
            - torch.log(real_var / fake_var)
        )
        return kl.mean()

    def _js_divergence_loss(self, real_samples: Tensor, fake_samples: Tensor) -> Tensor:
        real_mean = real_samples.mean(dim=0)
        fake_mean = fake_samples.mean(dim=0)
        real_var = real_samples.var(dim=0) + 1e-6
        fake_var = fake_samples.var(dim=0) + 1e-6

        m_mean = 0.5 * (real_mean + fake_mean)
        m_var = 0.5 * (real_var + fake_var)

        kl_real = 0.5 * (
            (real_var / m_var)
            + (m_mean - real_mean) ** 2 / m_var
            - 1
            - torch.log(real_var / m_var)
        )
        kl_fake = 0.5 * (
            (fake_var / m_var)
            + (m_mean - fake_mean) ** 2 / m_var
            - 1
            - torch.log(fake_var / m_var)
        )

        return 0.5 * (kl_real.mean() + kl_fake.mean())


class GradientPenaltyLoss(nn.Module):
    """
    Gradient Penalty Loss for various GAN variants.

    Args:
        penalty_weight: Weight for the gradient penalty term.
    """

    def __init__(self, penalty_weight: float = 10.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        discriminator: nn.Module,
        real_samples: Tensor,
        fake_samples: Tensor,
    ) -> Tensor:
        batch_size = real_samples.size(0)
        device = real_samples.device

        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        interpolated_scores = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return self.penalty_weight * gradient_penalty


class SpectralNormPenalty(nn.Module):
    """
    Spectral Norm Penalty for discriminator regularization.

    Args:
        penalty_weight: Weight for the spectral norm penalty.
    """

    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, discriminator: nn.Module, real_samples: Tensor) -> Tensor:
        output = discriminator(real_samples)
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=real_samples,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm = grad.pow(2).sum(dim=1).mean()
        return self.penalty_weight * grad_norm


class RelativisticAverageLoss(nn.Module):
    """
    Relativistic Average GAN Loss.

    Uses relativistic discriminator formulation for improved stability.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        real_mean = real_scores.mean()
        fake_mean = fake_scores.mean()

        real_loss = -torch.log(torch.sigmoid(real_scores - fake_mean)).mean()
        fake_loss = -torch.log(torch.sigmoid(fake_scores - real_mean)).mean()

        d_loss = real_loss + fake_loss
        g_loss = -torch.log(torch.sigmoid(real_scores - fake_scores)).mean()

        if self.reduction == "sum":
            return d_loss.sum(), g_loss.sum()
        return d_loss, g_loss


class ConsistencyLoss(nn.Module):
    """
    Consistency Loss for improved GAN training.

    Encourages consistent predictions under small perturbations.

    Args:
        penalty_weight: Weight for consistency penalty.
    """

    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        discriminator: nn.Module,
        real_samples: Tensor,
        noise_scale: float = 0.1,
    ) -> Tensor:
        device = real_samples.device

        noise = torch.randn_like(real_samples) * noise_scale
        perturbed_samples = real_samples + noise
        perturbed_samples.requires_grad_(True)

        original_scores = discriminator(real_samples)
        perturbed_scores = discriminator(perturbed_samples)

        consistency_penalty = (original_scores - perturbed_scores).pow(2).mean()

        return self.penalty_weight * consistency_penalty


class MultiScaleLoss(nn.Module):
    """
    Multi-Scale Adversarial Loss.

    Applies adversarial loss at multiple scales for better detail preservation.

    Args:
        scales: List of scales to apply loss.
        reduction: Specifies the reduction to apply.
    """

    def __init__(
        self,
        scales: Optional[List[int]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.scales = scales if scales is not None else [1, 2, 4]
        self.reduction = reduction

    def forward(
        self,
        real_outputs: List[Tensor],
        fake_outputs: List[Tensor],
    ) -> Tensor:
        total_loss = 0.0
        for scale in self.scales:
            if scale < len(real_outputs):
                real_out = real_outputs[scale]
                fake_out = fake_outputs[scale]

                loss = F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()
                total_loss += loss

        if self.reduction == "mean":
            return total_loss / len(self.scales)
        return total_loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss.

    Matches intermediate feature representations between real and generated samples.

    Args:
        reduction: Specifies the reduction to apply.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        real_features: List[Tensor],
        fake_features: List[Tensor],
    ) -> Tensor:
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            real_mean = real_feat.mean(dim=0)
            fake_mean = fake_feat.mean(dim=0)
            loss += ((real_mean - fake_mean) ** 2).mean()

        if self.reduction == "mean":
            return loss / len(real_features)
        return loss


class PerceptualAdversarialLoss(nn.Module):
    """
    Perceptual Adversarial Loss using pretrained networks.

    Combines adversarial loss with perceptual loss for better visual quality.

    Args:
        perceptual_network: Pretrained network for feature extraction.
        adversarial_weight: Weight for adversarial term.
        perceptual_weight: Weight for perceptual term.
    """

    def __init__(
        self,
        perceptual_network: Optional[nn.Module] = None,
        adversarial_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ):
        super().__init__()
        self.perceptual_network = perceptual_network
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight

    def forward(
        self,
        real_samples: Tensor,
        fake_samples: Tensor,
        discriminator: Optional[nn.Module] = None,
    ) -> Tensor:
        loss = 0.0

        if discriminator is not None:
            real_scores = discriminator(real_samples)
            fake_scores = discriminator(fake_samples)
            adv_loss = -fake_scores.mean()
            loss += self.adversarial_weight * adv_loss

        if self.perceptual_network is not None:
            with torch.no_grad():
                real_features = self.perceptual_network(real_samples)
            fake_features = self.perceptual_network(fake_samples)

            for real_feat, fake_feat in zip(real_features, fake_features):
                loss += self.perceptual_weight * ((real_feat - fake_feat) ** 2).mean()

        return loss


__all__ = [
    "WGAN_GPLoss",
    "HingeLossGAN",
    "LeastSquaresGANLoss",
    "DistributionMatchingLoss",
    "GradientPenaltyLoss",
    "SpectralNormPenalty",
    "RelativisticAverageLoss",
    "ConsistencyLoss",
    "MultiScaleLoss",
    "FeatureMatchingLoss",
    "PerceptualAdversarialLoss",
]
