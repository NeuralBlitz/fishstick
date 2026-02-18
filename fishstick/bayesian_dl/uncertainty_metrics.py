"""
Uncertainty Metrics for Bayesian Deep Learning.

Provides comprehensive metrics for evaluating uncertainty
estimates from Bayesian neural networks and ensemble methods.
"""

from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


class UncertaintyMetrics:
    """Collection of uncertainty quantification metrics."""

    @staticmethod
    def entropy(probs: Tensor) -> Tensor:
        """Compute entropy of probability distribution.

        Args:
            probs: Probability tensor [batch, classes]

        Returns:
            Entropy values
        """
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    @staticmethod
    def mutual_information(probs: Tensor) -> Tensor:
        """Compute mutual information as uncertainty measure.

        Args:
            probs: Sampled probabilities [n_samples, batch, classes]

        Returns:
            Mutual information
        """
        mean_probs = probs.mean(dim=0)

        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        mean_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)

        return entropy_mean - mean_entropy

    @staticmethod
    def expected_calibration_error(
        probs: Tensor,
        labels: Tensor,
        n_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            probs: Predicted probabilities [batch, classes]
            labels: True labels [batch]
            n_bins: Number of bins

        Returns:
            ECE score
        """
        confidences, predictions = probs.max(dim=1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * (accuracy_in_bin - avg_confidence_in_bin).abs()

        return ece.item()

    @staticmethod
    def nll(logits: Tensor, labels: Tensor) -> Tensor:
        """Compute negative log-likelihood.

        Args:
            logits: Model logits [batch, classes]
            labels: True labels [batch]

        Returns:
            NLL value
        """
        return F.cross_entropy(logits, labels)

    @staticmethod
    def brier_score(probs: Tensor, labels: Tensor) -> Tensor:
        """Compute Brier score.

        Args:
            probs: Predicted probabilities [batch, classes]
            labels: True labels [batch]

        Returns:
            Brier score
        """
        one_hot = F.one_hot(labels, probs.size(-1)).float()
        return ((probs - one_hot) ** 2).sum(dim=-1).mean()

    @staticmethod
    def accuracy(probs: Tensor, labels: Tensor) -> float:
        """Compute accuracy.

        Args:
            probs: Predicted probabilities [batch, classes]
            labels: True labels [batch]

        Returns:
            Accuracy
        """
        predictions = probs.argmax(dim=1)
        return predictions.eq(labels).float().mean().item()

    @staticmethod
    def auroc(probs: Tensor, labels: Tensor) -> float:
        """Compute Area Under ROC curve for OOD detection.

        Args:
            probs: Maximum probabilities [batch]
            labels: Binary labels (ID=1, OOD=0) [batch]

        Returns:
            AUROC score
        """
        sorted_indices = torch.argsort(probs)
        labels_sorted = labels[sorted_indices]

        tp = (labels_sorted == 1).cumsum(dim=0).float()
        fp = (labels_sorted == 0).cumsum(dim=0).float()

        tp = tp / (labels == 1).sum()
        fp = fp / (labels == 0).sum()

        auroc = (tp[1:] - tp[:-1]) * (fp[1:] + fp[:-1]) / 2
        return auroc.sum().item()

    @staticmethod
    def precision_recall(
        probs: Tensor,
        labels: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1.

        Args:
            probs: Predicted probabilities [batch]
            labels: True binary labels [batch]
            threshold: Classification threshold

        Returns:
            Precision, Recall, F1
        """
        predictions = (probs > threshold).float()

        tp = (predictions * labels).sum().item()
        fp = (predictions * (1 - labels)).sum().item()
        fn = ((1 - predictions) * labels).sum().item()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return precision, recall, f1


class RegressionUncertaintyMetrics:
    """Uncertainty metrics for regression tasks."""

    @staticmethod
    def rmse(predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute Root Mean Squared Error.

        Args:
            predictions: Predicted values [batch]
            targets: True values [batch]

        Returns:
            RMSE
        """
        return torch.sqrt(F.mse_loss(predictions, targets))

    @staticmethod
    def mae(predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute Mean Absolute Error.

        Args:
            predictions: Predicted values [batch]
            targets: True values [batch]

        Returns:
            MAE
        """
        return F.l1_loss(predictions, targets)

    @staticmethod
    def calibration_curve(
        mean_pred: Tensor,
        var_pred: Tensor,
        targets: Tensor,
        n_bins: int = 10,
    ) -> Tuple[list[float], list[float]]:
        """Compute reliability diagram for regression.

        Args:
            mean_pred: Predicted means [batch]
            var_pred: Predicted variances [batch]
            targets: True values [batch]
            n_bins: Number of bins

        Returns:
            Mean predicted values and fraction of true values in each bin
        """
        std_pred = torch.sqrt(var_pred)

        z_scores = (targets - mean_pred).abs() / (std_pred + 1e-10)

        bin_boundaries = torch.linspace(0, 3, n_bins + 1)

        mean_preds = []
        fractions = []

        for i in range(n_bins):
            in_bin = (z_scores > bin_boundaries[i]) & (
                z_scores <= bin_boundaries[i + 1]
            )
            if in_bin.sum() > 0:
                mean_preds.append(mean_pred[in_bin].mean().item())
                fractions.append(in_bin.float().mean().item())

        return mean_preds, fractions

    @staticmethod
    def negative_log_likelihood(
        mean_pred: Tensor,
        var_pred: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Compute negative log-likelihood for Gaussian.

        Args:
            mean_pred: Predicted means [batch]
            var_pred: Predicted variances [batch]
            targets: True values [batch]

        Returns:
            NLL
        """
        sigma = torch.sqrt(var_pred + 1e-6)
        nll = 0.5 * (
            torch.log(sigma**2)
            + ((targets - mean_pred) ** 2) / (sigma**2)
            + torch.log(torch.tensor(2 * 3.14159))
        )
        return nll.mean()

    @staticmethod
    def coverage(
        mean_pred: Tensor,
        var_pred: Tensor,
        targets: Tensor,
        confidence: float = 0.95,
    ) -> float:
        """Compute prediction interval coverage.

        Args:
            mean_pred: Predicted means [batch]
            var_pred: Predicted variances [batch]
            targets: True values [batch]
            confidence: Confidence level

        Returns:
            Coverage fraction
        """
        std_pred = torch.sqrt(var_pred)

        z = torch.tensor(1.96) if confidence == 0.95 else torch.tensor(1.645)

        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred

        covered = ((targets >= lower) & (targets <= upper)).float()

        return covered.mean().item()

    @staticmethod
    def interval_width(
        var_pred: Tensor,
        confidence: float = 0.95,
    ) -> Tensor:
        """Compute average prediction interval width.

        Args:
            var_pred: Predicted variances [batch]
            confidence: Confidence level

        Returns:
            Average interval width
        """
        std_pred = torch.sqrt(var_pred)

        z = torch.tensor(1.96) if confidence == 0.95 else torch.tensor(1.645)

        return (2 * z * std_pred).mean()


class UncertaintyDecomposition:
    """Decompose total uncertainty into epistemic and aleatoric."""

    @staticmethod
    def decompose(
        samples: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Decompose uncertainty from ensemble/BNN samples.

        Args:
            samples: Model predictions [n_samples, batch, ...]

        Returns:
            epistemic: Epistemic uncertainty (model uncertainty)
            aleatoric: Aleatoric uncertainty (data uncertainty)
        """
        mean = samples.mean(dim=0)
        epistemic = samples.var(dim=0)
        aleatoric = ((samples - mean.unsqueeze(0)) ** 2).mean(dim=0)

        return epistemic, aleatoric

    @staticmethod
    def mutual_information_decomposition(
        samples: Tensor,
    ) -> Tensor:
        """Compute mutual information as epistemic uncertainty.

        Args:
            samples: Model predictions [n_samples, batch, classes]

        Returns:
            Mutual information (epistemic uncertainty)
        """
        probs = F.softmax(samples, dim=-1)
        mean_probs = probs.mean(dim=0)

        entropy_mean = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)
        mean_entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=0)

        return entropy_mean - mean_entropy


class OODDetectionMetrics:
    """Metrics for Out-of-Distribution detection."""

    @staticmethod
    def fpr_at_95_tpr(
        id_scores: Tensor,
        ood_scores: Tensor,
    ) -> float:
        """Compute False Positive Rate at 95% True Positive Rate.

        Args:
            id_scores: ID sample scores (higher = more ID)
            ood_scores: OOD sample scores

        Returns:
            FPR at 95% TPR
        """
        all_scores = torch.cat([id_scores, ood_scores])
        labels = torch.cat([torch.ones(len(id_scores)), torch.zeros(len(ood_scores))])

        sorted_indices = torch.argsort(all_scores, descending=True)
        labels_sorted = labels[sorted_indices]

        tp = (labels_sorted == 1).cumsum(dim=0).float()
        fp = (labels_sorted == 0).cumsum(dim=0).float()

        tpr = tp / (labels == 1).sum()
        fpr = fp / (labels == 0).sum()

        idx = (tpr >= 0.95).nonzero(as_tuple=True)[0]

        if len(idx) > 0:
            return fpr[idx[0]].item()
        return 1.0

    @staticmethod
    def detection_error(
        id_scores: Tensor,
        ood_scores: Tensor,
    ) -> float:
        """Compute detection error (probability of error).

        Args:
            id_scores: ID sample scores
            ood_scores: OOD sample scores

        Returns:
            Detection error
        """
        all_scores = torch.cat([id_scores, ood_scores])
        labels = torch.cat([torch.ones(len(id_scores)), torch.zeros(len(ood_scores))])

        threshold_idx = torch.argmin(
            torch.abs(all_scores.unsqueeze(1) - all_scores.unsqueeze(0)).mean(
                dim=(1, 2)
            )
        )

        predictions = (all_scores > all_scores[threshold_idx]).float()

        return 1 - predictions.eq(labels).float().mean().item()

    @staticmethod
    def auroc_ood(
        id_scores: Tensor,
        ood_scores: Tensor,
    ) -> float:
        """Compute AUROC for OOD detection.

        Args:
            id_scores: ID sample scores (higher = more confident ID)
            ood_scores: OOD sample scores

        Returns:
            AUROC
        """
        all_scores = torch.cat([id_scores, ood_scores])
        labels = torch.cat([torch.ones(len(id_scores)), torch.zeros(len(ood_scores))])

        sorted_indices = torch.argsort(all_scores)
        labels_sorted = labels[sorted_indices]

        tp = (labels_sorted == 1).cumsum(dim=0).float()
        fp = (labels_sorted == 0).cumsum(dim=0).float()

        tp = tp / (labels == 1).sum()
        fp = fp / (labels == 0).sum()

        auroc = (tp[1:] - tp[:-1]) * (fp[1:] + fp[:-1]) / 2
        return auroc.sum().item()


class ConfidenceCalibration:
    """Methods for calibrating confidence estimates."""

    @staticmethod
    def temperature_scale(
        logits: Tensor,
        labels: Tensor,
        val_logits: Optional[Tensor] = None,
        val_labels: Optional[Tensor] = None,
    ) -> Tuple[float, float]:
        """Fit temperature scaling for calibration.

        Args:
            logits: Training logits
            labels: Training labels
            val_logits: Validation logits (optional)
            val_labels: Validation labels (optional)

        Returns:
            Temperature value and validation NLL
        """
        temperature = nn.Parameter(torch.tensor(1.0))

        if val_logits is None:
            val_logits = logits
            val_labels = labels

        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            scaled = logits / temperature
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        val_scaled = val_logits / temperature
        val_nll = F.cross_entropy(val_scaled, val_labels).item()

        return temperature.item(), val_nll

    @staticmethod
    def platt_scaling(
        logits: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fit Platt scaling for calibration.

        Args:
            logits: Model logits
            labels: True labels

        Returns:
            Weight and bias parameters
        """
        weight = nn.Parameter(torch.ones(1, logits.size(1)))
        bias = nn.Parameter(torch.zeros(1))

        optimizer = torch.optim.Adam([weight, bias], lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()

            scaled = logits * weight + bias
            loss = F.cross_entropy(scaled, labels)

            loss.backward()
            optimizer.step()

        return weight.data, bias.data

    @staticmethod
    def isotonic_regression(
        confidences: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fit isotonic regression for calibration.

        Args:
            confidences: Model confidences
            labels: Binary correctness labels

        Returns:
            Mapping functions for confidence
        """
        confidences_np = confidences.cpu().numpy()
        labels_np = labels.cpu().numpy()

        from scipy.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(confidences_np, labels_np)

        calibrated = iso.transform(confidences_np)

        return (
            torch.from_numpy(calibrated).to(confidences.device),
            torch.from_numpy(iso.transform(np.linspace(0, 1, 100))).to(
                confidences.device
            ),
        )
