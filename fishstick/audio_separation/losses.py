"""
Loss Functions for Audio Source Separation

Provides various loss functions for training audio source separation models,
including signal-level losses, permutation-invariant training, and deep
clustering losses.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparationLoss(nn.Module):
    """Base class for separation losses."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError


class SISDRLoss(SeparationLoss):
    """Scale-Invariant Source-to-Distortion Ratio Loss.

    SI-SDR is a widely used metric for audio source separation that measures
    the ratio of signal power to distortion power. It is more correlated with
    human perception than SDR.

    Reference:
        "SDR - half-baked or well done?" (Wisdom et al., 2019)
    """

    def __init__(
        self,
        reduction: str = "mean",
        epsilon: float = 1e-8,
        zero_nan: bool = True,
    ):
        super().__init__(reduction)
        self.epsilon = epsilon
        self.zero_nan = zero_nan

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SI-SDR loss.

        Args:
            predictions: Predicted source signals (n_sources, batch, channels, time)
            targets: Target source signals (n_sources, batch, channels, time)

        Returns:
            SI-SDR loss value
        """
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        dot_product = torch.sum(predictions * targets, dim=-1, keepdim=True)
        target_energy = torch.sum(targets**2, dim=-1, keepdim=True) + self.epsilon

        scale = dot_product / target_energy
        scaled_target = scale * targets
        noise = predictions - scaled_target

        signal_power = torch.sum(scaled_target**2, dim=-1)
        noise_power = torch.sum(noise**2, dim=-1)

        sisdr = 10 * torch.log10(
            (signal_power + self.epsilon) / (noise_power + self.epsilon)
        )

        if self.zero_nan:
            sisdr = torch.where(torch.isfinite(sisdr), sisdr, torch.zeros_like(sisdr))

        if self.reduction == "mean":
            return -sisdr.mean()
        elif self.reduction == "sum":
            return -sisdr.sum()
        return -sisdr


class SDRLoss(SeparationLoss):
    """Source-to-Distortion Ratio Loss.

    SDR measures the ratio of target signal power to total error power,
    including interference from other sources and artifacts.
    """

    def __init__(
        self,
        reduction: str = "mean",
        epsilon: float = 1e-8,
    ):
        super().__init__(reduction)
        self.epsilon = epsilon

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SDR loss.

        Args:
            predictions: Predicted sources
            targets: Target sources

        Returns:
            SDR loss value
        """
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        target_power = torch.sum(targets**2, dim=-1)
        error = predictions - targets
        error_power = torch.sum(error**2, dim=-1)

        sdr = 10 * torch.log10(
            (target_power + self.epsilon) / (error_power + self.epsilon)
        )

        if self.reduction == "mean":
            return -sdr.mean()
        elif self.reduction == "sum":
            return -sdr.sum()
        return -sdr


class PITLoss(SeparationLoss):
    """Permutation Invariant Training Loss.

    PIT solves the permutation ambiguity problem in source separation by
    finding the best permutation between predictions and targets during
    training.

    Reference:
        "Permutation Invariant Training of Deep Neural Networks for
        Source Separation" (Kolbaek et al., 2017)
    """

    def __init__(
        self,
        loss_fn: Optional[nn.Module] = None,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        self.loss_fn = loss_fn or SISDRLoss(reduction="none")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PIT loss with optimal permutation.

        Args:
            predictions: Predicted sources (n_sources, batch, channels, time)
            targets: Target sources (n_sources, batch, channels, time)

        Returns:
            Tuple of (PIT loss, permutation indices)
        """
        n_sources = predictions.shape[0]
        batch_size = predictions.shape[1]

        predictions = predictions.transpose(0, 1)
        targets = targets.transpose(0, 1)

        loss_matrix = torch.zeros(
            batch_size, n_sources, n_sources, device=predictions.device
        )

        for i in range(n_sources):
            for j in range(n_sources):
                loss_matrix[:, i, j] = -self.loss_fn(predictions[:, i], targets[:, j])

        permutations = self._get_best_permutations(loss_matrix)

        loss = self._compute_permuted_loss(loss_matrix, permutations, batch_size)

        if self.reduction == "mean":
            return loss.mean(), permutations
        return loss, permutations

    def _get_best_permutations(
        self,
        loss_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Find optimal permutation using Hungarian algorithm."""
        batch_size, n_sources, _ = loss_matrix.shape

        permutations = torch.zeros(batch_size, n_sources, dtype=torch.long)
        permutations = permutations.to(loss_matrix.device)

        for b in range(batch_size):
            cost_matrix = loss_matrix[b]
            indices = torch.argmin(cost_matrix, dim=1)
            permutations[b] = indices

        return permutations

    def _compute_permuted_loss(
        self,
        loss_matrix: torch.Tensor,
        permutations: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        losses = torch.zeros(batch_size, device=loss_matrix.device)
        n_sources = loss_matrix.shape[1]

        for b in range(batch_size):
            for i in range(n_sources):
                losses[b] += loss_matrix[b, i, permutations[b, i]]
            losses[b] /= n_sources

        return losses


class DeepClusteringLoss(nn.Module):
    """Deep Clustering Loss for source separation.

    Deep clustering learns embeddings for each time-frequency bin and uses
    clustering to assign bins to different sources.

    Reference:
        "Deep Clustering: Discriminative Embeddings for Segmentation
        and Separation of Speech" (Hershey et al., 2016)
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep clustering loss.

        Args:
            embeddings: Embeddings of shape (batch, freq, time, embed_dim)
            assignments: Binary assignments of shape (batch, freq, time, n_sources)

        Returns:
            Deep clustering loss value
        """
        batch_size, freq, time, embed_dim = embeddings.shape
        n_sources = assignments.shape[-1]

        embeddings = embeddings.reshape(batch_size, freq * time, embed_dim)
        assignments = assignments.reshape(batch_size, freq * time, n_sources)

        V = assignments.float()
        VVT = torch.bmm(V, V.transpose(1, 2))

        E = embeddings
        EET = torch.bmm(E, E.transpose(1, 2))

        loss = torch.norm(VVT - EET, p="fro") ** 2

        return loss / (batch_size * freq * time)


class CompositeLoss(nn.Module):
    """Composite loss combining multiple separation losses.

    Combines different loss functions with learnable or fixed weights
    for multi-task learning.
    """

    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None,
        learnable: bool = False,
    ):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.n_losses = len(losses)

        if weights is None:
            weights = [1.0] * self.n_losses

        if learnable:
            self.log_vars = nn.Parameter(torch.zeros(self.n_losses))
        else:
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

        self.learnable = learnable

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """Compute weighted composite loss.

        Args:
            predictions: Predicted sources
            targets: Target sources
            **kwargs: Additional arguments for individual losses

        Returns:
            Tuple of (total loss, loss breakdown dict)
        """
        loss_values = {}

        for i, loss_fn in enumerate(self.losses):
            loss_val = loss_fn(predictions, targets, **kwargs)
            loss_values[f"loss_{i}"] = loss_val

        if self.learnable:
            weights = F.softmax(self.log_vars, dim=0)
            total = sum(w * loss_values[f"loss_{i}"] for i, w in enumerate(weights))
        else:
            total = sum(
                w * loss_values[f"loss_{i}"] for i, w in enumerate(self.weights)
            )

        loss_values["total"] = total
        return total, loss_values


class ConsistencyLoss(nn.Module):
    """Consistency loss for ensuring mask consistency across time/frequency.

    Encourages smooth masks that don't have abrupt changes.
    """

    def __init__(
        self,
        time_weight: float = 0.5,
        freq_weight: float = 0.5,
    ):
        super().__init__()
        self.time_weight = time_weight
        self.freq_weight = freq_weight

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss.

        Args:
            masks: Predicted masks (batch, n_sources, freq, time)

        Returns:
            Consistency loss value
        """
        time_diff = torch.diff(masks, dim=-1)
        freq_diff = torch.diff(masks, dim=2)

        time_loss = torch.mean(time_diff**2)
        freq_loss = torch.mean(freq_diff**2)

        return self.time_weight * time_loss + self.freq_weight * freq_loss


class FrequencyReconstructionLoss(nn.Module):
    """Loss for frequency-domain reconstruction.

    Computes loss in the STFT domain to focus on spectral quality.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        reduction: str = "mean",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute frequency reconstruction loss.

        Args:
            predictions: Predicted waveforms
            targets: Target waveforms

        Returns:
            Frequency domain loss
        """
        pred_stft = torch.stft(
            predictions.reshape(-1, predictions.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )
        target_stft = torch.stft(
            targets.reshape(-1, targets.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
        )

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        loss = F.mse_loss(pred_mag, target_mag, reduction=self.reduction)

        return loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for training separator with discriminator.

    Uses a discriminator to improve the realism of separated sources.
    """

    def __init__(
        self,
        discriminator: Optional[nn.Module] = None,
        loss_type: str = "hinge",
    ):
        super().__init__()
        self.discriminator = discriminator
        self.loss_type = loss_type

    def generator_loss(
        self,
        fake_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute generator (separator) adversarial loss.

        Args:
            fake_outputs: Discriminator outputs on generated sources

        Returns:
            Generator loss
        """
        if self.loss_type == "hinge":
            loss = -torch.stack([torch.mean(out) for out in fake_outputs]).mean()
        elif self.loss_type == "bce":
            real_labels = torch.ones_like(fake_outputs[0])
            loss = F.binary_cross_entropy_with_logits(
                torch.cat(fake_outputs, dim=0),
                torch.cat([real_labels] * len(fake_outputs), dim=0),
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def discriminator_loss(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute discriminator loss.

        Args:
            real_outputs: Discriminator outputs on real sources
            fake_outputs: Discriminator outputs on generated sources

        Returns:
            Discriminator loss
        """
        if self.loss_type == "hinge":
            real_loss = -torch.stack(
                [torch.mean(F.relu(1.0 - out)) for out in real_outputs]
            ).mean()
            fake_loss = -torch.stack(
                [torch.mean(F.relu(1.0 + out)) for out in fake_outputs]
            ).mean()
            loss = real_loss + fake_loss
        elif self.loss_type == "bce":
            real_labels = torch.ones_like(real_outputs[0])
            fake_labels = torch.zeros_like(fake_outputs[0])
            real_loss = F.binary_cross_entropy_with_logits(
                torch.cat(real_outputs, dim=0),
                torch.cat([real_labels] * len(real_outputs), dim=0),
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                torch.cat(fake_outputs, dim=0),
                torch.cat([fake_labels] * len(fake_outputs), dim=0),
            )
            loss = real_loss + fake_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class FrequencyAwareLoss(nn.Module):
    """Frequency-aware loss with band-wise weighting.

    Applies different weights to different frequency bands to focus
    on perceptually important regions.
    """

    def __init__(
        self,
        n_fft: int = 512,
        sample_rate: int = 16000,
        weight_bands: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate

        if weight_bands is None:
            freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
            weight_bands = self._compute_perceptual_weights(freqs)

        self.register_buffer("weight_bands", weight_bands)

    def _compute_perceptual_weights(
        self,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual weighting based on frequency."""
        weights = torch.ones_like(freqs)

        low_freq_mask = freqs < 200
        weights[low_freq_mask] = 2.0

        high_freq_mask = freqs > 4000
        weights[high_freq_mask] = 1.5

        weights = weights / weights.mean()

        return weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute frequency-weighted MSE loss.

        Args:
            predictions: Predicted waveforms
            targets: Target waveforms

        Returns:
            Frequency-weighted loss
        """
        pred_stft = torch.stft(
            predictions.reshape(-1, predictions.shape[-1]),
            n_fft=self.n_fft,
            return_complex=True,
        )
        target_stft = torch.stft(
            targets.reshape(-1, targets.shape[-1]),
            n_fft=self.n_fft,
            return_complex=True,
        )

        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)

        weighted_diff = (pred_mag - target_mag) ** 2
        weighted_diff = weighted_diff * self.weight_bands.view(1, -1, 1)

        return weighted_diff.mean()
