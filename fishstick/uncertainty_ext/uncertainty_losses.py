"""
Uncertainty-Aware Training Losses

Training losses that incorporate uncertainty estimation.
"""

from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FocalLossWithUncertainty(nn.Module):
    """Focal loss with uncertainty weighting.

    Args:
        gamma: Focal loss gamma parameter
        alpha: Class weighting factor
        uncertainty_weight: Weight for uncertainty term
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        uncertainty_weight: float = 0.1,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        uncertainty: Optional[Tensor] = None,
    ) -> Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        loss = focal_loss.mean()

        if uncertainty is not None and self.uncertainty_weight > 0:
            uncertainty_loss = uncertainty.mean()
            loss = loss + self.uncertainty_weight * uncertainty_loss

        return loss


class LabelSmoothingWithUncertainty(nn.Module):
    """Label smoothing with uncertainty estimation.

    Args:
        smoothing: Label smoothing factor
        uncertainty_weight: Weight for uncertainty term
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        uncertainty_weight: float = 0.1,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        uncertainty: Optional[Tensor] = None,
    ) -> Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        targets_onehot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )

        targets_smooth = (
            targets_onehot * (1 - self.smoothing) + self.smoothing / n_classes
        )

        loss = -(targets_smooth * log_probs).sum(dim=-1).mean()

        if uncertainty is not None and self.uncertainty_weight > 0:
            uncertainty_loss = uncertainty.mean()
            loss = loss + self.uncertainty_weight * uncertainty_loss

        return loss


class EvidentialRegressionLoss(nn.Module):
    """Evidential loss for deep evidential regression.

    Args:
        evidence_func: Function to convert logits to evidence
    """

    def __init__(self, evidence_func: str = "softplus"):
        super().__init__()
        self.evidence_func = evidence_func

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
        return_uncertainty: bool = False,
    ) -> Tensor:
        gamma, nu, alpha, beta = (
            predictions[..., 0],
            predictions[..., 1],
            predictions[..., 2],
            predictions[..., 3],
        )

        if self.evidence_func == "softplus":
            alpha = F.softplus(alpha) + 1
            beta = F.softplus(beta) + 1e-6
        elif self.evidence_func == "exp":
            alpha = torch.exp(alpha) + 1
            beta = torch.exp(beta) + 1e-6

        error = (targets - gamma) ** 2

        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(2 * beta * (1 + nu * error / (2 * beta - 1)))
            + (alpha + 0.5) * torch.log(nu * error + 2 * beta - 1)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        loss = nll.mean()

        if return_uncertainty:
            uncertainty = beta / (alpha - 1)
            return loss, uncertainty

        return loss


class EvidentialClassificationLoss(nn.Module):
    """Evidential loss for deep evidential classification.

    Args:
        num_classes: Number of classes
        reg_weight: Regularization weight for evidence
    """

    def __init__(self, num_classes: int, reg_weight: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.reg_weight = reg_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        return_uncertainty: bool = False,
    ) -> Tensor:
        evidence = F.softplus(logits)
        alpha = evidence + 1
        beta = torch.ones_like(evidence)

        target_onehot = F.one_hot(targets, self.num_classes).float()

        dirichlet = torch.distributions.Dirichlet(alpha)
        log_likelihood = dirichlet.log_prob(target_onehot)
        loss = -log_likelihood.mean()

        if self.reg_weight > 0:
            evidence_reg = evidence.sum(dim=-1).mean()
            loss = loss + self.reg_weight * evidence_reg

        if return_uncertainty:
            strength = alpha.sum(dim=-1, keepdim=True)
            uncertainty = self.num_classes / strength
            return loss, uncertainty.squeeze(-1)

        return loss


class UncertaintyAwareContrastiveLoss(nn.Module):
    """Contrastive loss with uncertainty weighting.

    Args:
        temperature: Temperature parameter
        uncertainty_weight: Weight for uncertainty
    """

    def __init__(
        self,
        temperature: float = 0.07,
        uncertainty_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor,
        uncertainties: Optional[Tensor] = None,
    ) -> Tensor:
        embeddings = F.normalize(embeddings, dim=1)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        logits = similarity - similarity.max(dim=1, keepdim=True)[0].detach()

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        mask_positive = mask.float()
        positive_loss = -(mask_positive * log_prob).sum() / mask_positive.sum()

        mask_negative = (~mask).float()
        negative_loss = -((1 - mask_positive) * log_prob).sum() / mask_negative.sum()

        loss = positive_loss + negative_loss

        if uncertainties is not None and self.uncertainty_weight > 0:
            uncertainty_loss = uncertainties.mean()
            loss = loss + self.uncertainty_weight * uncertainty_loss

        return loss


class MixupUncertaintyLoss(nn.Module):
    """Mixup loss with uncertainty weighting.

    Args:
        alpha: Mixup alpha parameter
        uncertainty_weight: Weight for uncertainty term
    """

    def __init__(
        self,
        alpha: float = 0.2,
        uncertainty_weight: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        uncertainties: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = logits.size(0)

        indices = torch.randperm(batch_size, device=logits.device)

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample()

        mixed_logits = lam * logits + (1 - lam) * logits[indices]

        mixed_targets = torch.where(
            torch.rand(batch_size, device=logits.device) < lam,
            targets,
            targets[indices],
        )

        loss = F.cross_entropy(mixed_logits, mixed_targets)

        if uncertainties is not None and self.uncertainty_weight > 0:
            mixed_uncertainties = (
                lam * uncertainties + (1 - lam) * uncertainties[indices]
            )
            uncertainty_loss = mixed_uncertainties.mean()
            loss = loss + self.uncertainty_weight * uncertainty_loss

        return loss


class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic loss for regression with uncertainty.

    Args:
        reduction: Reduction method
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        mean, log_var = predictions

        precision = torch.exp(-log_var)

        loss = precision * (targets - mean) ** 2 + log_var

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss, log_var.mean()


class BootstrapUncertaintyLoss(nn.Module):
    """Bootstrap loss with uncertainty from multiple forward passes.

    Args:
        n_bootstrap: Number of bootstrap samples
        uncertainty_weight: Weight for uncertainty term
    """

    def __init__(
        self,
        n_bootstrap: int = 5,
        uncertainty_weight: float = 0.1,
    ):
        super().__init__()
        self.n_bootstrap = n_bootstrap
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        model.eval()

        with torch.no_grad():
            bootstrap_preds = []

            for _ in range(self.n_bootstrap):
                mask = torch.rand_like(inputs) > 0.5

                if mask.sum() == 0:
                    mask = torch.ones_like(inputs, dtype=torch.bool)

                masked_input = inputs * mask.float()

                pred = model(masked_input)
                bootstrap_preds.append(pred)

            predictions = torch.stack(bootstrap_preds, dim=0)

            mean_pred = predictions.mean(dim=0)
            variance = predictions.var(dim=0)

        loss = F.mse_loss(mean_pred, targets)

        if self.uncertainty_weight > 0:
            uncertainty_loss = variance.mean()
            loss = loss + self.uncertainty_weight * uncertainty_loss

        return loss


class ConfidenceWeightedLoss(nn.Module):
    """Confidence-weighted loss for selective classification.

    Args:
        threshold: Confidence threshold for weighting
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)

        weights = torch.where(
            max_probs >= self.threshold,
            torch.ones_like(max_probs),
            torch.zeros_like(max_probs),
        )

        if weights.sum() == 0:
            return F.cross_entropy(logits, targets)

        loss = F.cross_entropy(logits, targets, reduction="none")

        weighted_loss = (loss * weights).sum() / (weights.sum() + 1e-10)

        return weighted_loss


class DistributionUncertaintyLoss(nn.Module):
    """Loss that accounts for distributional uncertainty.

    Args:
        weight: Weight for distributional uncertainty term
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        ce_loss = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=-1)
        dist_uncertainty = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        loss = ce_loss + self.weight * dist_uncertainty

        return loss


class SemanticUncertaintyLoss(nn.Module):
    """Semantic uncertainty loss for structured predictions.

    Args:
        n_classes: Number of classes
        tau: Temperature for sharpening
    """

    def __init__(
        self,
        n_classes: int,
        tau: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.tau = tau

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        probs = F.softmax(logits / self.tau, dim=-1)

        target_onehot = F.one_hot(targets, self.n_classes).float()

        sharpening = probs ** (1 / self.tau)
        sharpening = sharpening / sharpening.sum(dim=-1, keepdim=True)

        loss = -(target_onehot * torch.log(sharpening + 1e-10)).sum(dim=-1).mean()

        return loss


class SoftTargetLoss(nn.Module):
    """Soft target loss with uncertainty from teacher ensemble.

    Args:
        tau: Temperature for soft targets
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        student_logits: Tensor,
        teacher_probs: Tensor,
    ) -> Tensor:
        student_log_probs = F.log_softmax(student_logits / self.tau, dim=-1)

        soft_targets = teacher_probs ** (1 / self.tau)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)

        loss = -(soft_targets * student_log_probs).sum(dim=-1).mean()

        return loss * (self.tau**2)


class VariancePenaltyLoss(nn.Module):
    """Loss with variance penalty for stable predictions.

    Args:
        penalty_weight: Weight for variance penalty
    """

    def __init__(self, penalty_weight: float = 0.1):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
        n_passes: int = 5,
    ) -> Tensor:
        model.eval()

        predictions = []

        for _ in range(n_passes):
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean_pred = predictions.mean(dim=0)

        variance_penalty = predictions.var(dim=0).mean()

        if targets.dim() > 1 and targets.size(-1) > 1:
            main_loss = F.cross_entropy(mean_pred, targets.argmax(dim=-1))
        else:
            main_loss = F.cross_entropy(mean_pred, targets)

        loss = main_loss + self.penalty_weight * variance_penalty

        return loss
