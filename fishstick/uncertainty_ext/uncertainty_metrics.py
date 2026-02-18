"""
Uncertainty Metrics

Comprehensive metrics for evaluating uncertainty estimation and calibration.
"""

from typing import Optional, Tuple, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def expected_calibration_error(
    logits: Tensor,
    labels: Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        logits: Model logits
        labels: True labels
        n_bins: Number of bins for calibration

    Returns:
        ECE score
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, predictions = probs.max(dim=-1)

    correct = (predictions == labels).float()

    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):
        in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])

        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = max_probs[in_bin].mean()

            ece += (in_bin.float().sum() / len(labels)) * abs(bin_acc - bin_conf)

    return ece.item()


def maximum_calibration_error(
    logits: Tensor,
    labels: Tensor,
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error (MCE).

    Args:
        logits: Model logits
        labels: True labels
        n_bins: Number of bins

    Returns:
        MCE score
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, predictions = probs.max(dim=-1)

    correct = (predictions == labels).float()

    bin_edges = np.linspace(0, 1, n_bins + 1)

    max_error = 0.0

    for i in range(n_bins):
        in_bin = (max_probs >= bin_edges[i]) & (max_probs < bin_edges[i + 1])

        if in_bin.sum() > 0:
            bin_acc = correct[in_bin].mean()
            bin_conf = max_probs[in_bin].mean()

            error = abs(bin_acc - bin_conf)
            max_error = max(max_error, error)

    return max_error


def negative_log_likelihood(
    logits: Tensor,
    labels: Tensor,
) -> float:
    """Compute Negative Log-Likelihood (NLL).

    Args:
        logits: Model logits
        labels: True labels

    Returns:
        NLL score
    """
    nll = F.cross_entropy(logits, labels)
    return nll.item()


def brier_score(
    logits: Tensor,
    labels: Tensor,
) -> float:
    """Compute Brier Score.

    Args:
        logits: Model logits
        labels: True labels

    Returns:
        Brier score (lower is better)
    """
    probs = F.softmax(logits, dim=-1)

    labels_onehot = F.one_hot(labels, probs.size(-1)).float()

    brier = ((probs - labels_onehot) ** 2).sum(dim=-1).mean()

    return brier.item()


def brier_decomposition(
    logits: Tensor,
    labels: Tensor,
) -> Dict[str, float]:
    """Decompose Brier Score into components.

    Args:
        logits: Model logits
        labels: True labels

    Returns:
        Dictionary with calibration, refinement, and uncertainty
    """
    probs = F.softmax(logits, dim=-1)

    labels_onehot = F.one_hot(labels, probs.size(-1)).float()

    labels_idx = labels

    confidence = probs.max(dim=-1)[0]
    accuracy = (probs.argmax(dim=-1) == labels).float()

    calibration = ((confidence - accuracy) ** 2).mean()

    class_probs = probs.gather(1, labels_idx.unsqueeze(1)).squeeze()
    refinement = (probs**2 - class_probs**2).mean()

    uncertainty = (1 - (probs**2).sum(dim=-1)).mean()

    return {
        "calibration": calibration.item(),
        "refinement": refinement.item(),
        "uncertainty": uncertainty.item(),
        "brier": (calibration + refinement).item(),
    }


def auroc_ood_detection(
    id_logits: Tensor,
    ood_logits: Tensor,
    method: str = "energy",
) -> Tuple[float, float]:
    """Compute AUROC for OOD detection.

    Args:
        id_logits: In-distribution logits
        ood_logits: Out-of-distribution logits
        method: Method to use ('energy', 'max_prob', 'entropy')

    Returns:
        Tuple of (AUROC, AUPR)
    """
    if method == "energy":
        id_scores = torch.logsumexp(id_logits, dim=-1)
        ood_scores = torch.logsumexp(ood_logits, dim=-1)
    elif method == "max_prob":
        id_scores = F.softmax(id_logits, dim=-1).max(dim=-1)[0]
        ood_scores = F.softmax(ood_logits, dim=-1).max(dim=-1)[0]
    elif method == "entropy":
        id_scores = -(
            F.softmax(id_logits, dim=-1) * F.log_softmax(id_logits, dim=-1)
        ).sum(dim=-1)
        ood_scores = -(
            F.softmax(ood_logits, dim=-1) * F.log_softmax(ood_logits, dim=-1)
        ).sum(dim=-1)

    id_labels = torch.ones(id_scores.size(0))
    ood_labels = torch.zeros(ood_scores.size(0))

    scores = torch.cat([id_scores, ood_scores])
    labels = torch.cat([id_labels, ood_labels])

    auroc = roc_auc_score(labels.numpy(), scores.numpy())
    aupr = average_precision_score(labels.numpy(), scores.numpy())

    return auroc, aupr


def confidence_accuracy_correlation(
    logits: Tensor,
    labels: Tensor,
) -> float:
    """Compute correlation between confidence and accuracy.

    Args:
        logits: Model logits
        labels: True labels

    Returns:
        Pearson correlation coefficient
    """
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1)[0]
    accuracy = (probs.argmax(dim=-1) == labels).float()

    correlation = torch.corrcoef(torch.stack([confidence, accuracy]))[0, 1]

    return correlation.item()


def selective_accuracy_at_recall(
    logits: Tensor,
    labels: Tensor,
    recall_levels: Optional[Tensor] = None,
) -> Dict[float, float]:
    """Compute selective accuracy at various recall levels.

    Args:
        logits: Model logits
        labels: True labels
        recall_levels: Recall levels to evaluate

    Returns:
        Dictionary of recall -> selective accuracy
    """
    if recall_levels is None:
        recall_levels = torch.tensor([0.8, 0.85, 0.9, 0.95, 1.0])

    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    correct = (predictions == labels).float()

    sorted_indices = torch.argsort(confidence, descending=True)
    correct_sorted = correct[sorted_indices]

    n_samples = len(labels)
    results = {}

    for recall in recall_levels:
        n_select = int(recall * n_samples)
        n_select = max(1, min(n_select, n_samples))

        selected_correct = correct_sorted[:n_select]
        selective_acc = selected_correct.mean().item()

        results[recall.item()] = selective_acc

    return results


def nll_decomposition(
    logits: Tensor,
    labels: Tensor,
) -> Dict[str, float]:
    """Decompose NLL into uncertainty components.

    Args:
        logits: Model logits
        labels: True labels

    Returns:
        Dictionary with NLL components
    """
    probs = F.softmax(logits, dim=-1)

    labels_idx = labels

    true_prob = probs.gather(1, labels_idx.unsqueeze(1)).squeeze()

    nll = -torch.log(true_prob + 1e-10).mean()

    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

    cross_entropy = -torch.log(probs.gather(1, labels_idx.unsqueeze(1)) + 1e-10).mean()

    return {
        "nll": nll.item(),
        "entropy": entropy.item(),
        "cross_entropy": cross_entropy.item(),
    }


def uncertainty_ranking_correlation(
    logits: Tensor,
    labels: Tensor,
    ood_logits: Optional[Tensor] = None,
) -> float:
    """Compute ranking correlation between uncertainty and error.

    Args:
        logits: Model logits
        labels: True labels
        ood_logits: Optional OOD logits

    Returns:
        Spearman correlation
    """
    probs = F.softmax(logits, dim=-1)

    confidence = probs.max(dim=-1)[0]
    predictions = probs.argmax(dim=-1)
    errors = (predictions != labels).float()

    from scipy.stats import spearmanr

    if ood_logits is not None:
        ood_confidence = F.softmax(ood_logits, dim=-1).max(dim=-1)[0]

        all_confidence = torch.cat([confidence, ood_confidence])
        all_labels = torch.cat(
            [torch.zeros(confidence.size(0)), torch.ones(ood_confidence.size(0))]
        )

        correlation, _ = spearmanr(all_confidence.numpy(), all_labels.numpy())
    else:
        correlation, _ = spearmanr(confidence.numpy(), errors.numpy())

    return correlation


def calibration_curve(
    logits: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Generate calibration curve data.

    Args:
        logits: Model logits
        labels: True labels
        n_bins: Number of bins

    Returns:
        Tuple of (mean_confidence, accuracy) per bin
    """
    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    correct = (predictions == labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    mean_confidences = []
    accuracies = []

    for i in range(n_bins):
        in_bin = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])

        if in_bin.sum() > 0:
            mean_confidences.append(confidence[in_bin].mean())
            accuracies.append(correct[in_bin].mean())

    return torch.stack(mean_confidences), torch.stack(accuracies)


def test_calibration(
    logits: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Perform statistical test for calibration.

    Args:
        logits: Model logits
        labels: True labels
        n_bins: Number of bins

    Returns:
        Dictionary with calibration metrics and test results
    """
    probs = F.softmax(logits, dim=-1)
    confidence, predictions = probs.max(dim=-1)

    correct = (predictions == labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])

        if in_bin.sum() > 0:
            bin_confs.append(confidence[in_bin].mean().item())
            bin_accs.append(correct[in_bin].mean().item())
            bin_counts.append(in_bin.sum().item())

    bin_confs = torch.tensor(bin_confs)
    bin_accs = torch.tensor(bin_accs)
    bin_counts = torch.tensor(bin_counts)

    ece = ((bin_counts / bin_counts.sum()) * (bin_confs - bin_accs).abs()).sum()

    from scipy.stats import chi2

    chi2_stat = ((bin_counts * (bin_confs - bin_accs) ** 2) / (bin_confs + 1e-10)).sum()

    p_value = 1 - chi2.cdf(chi2_stat.item(), n_bins - 1)

    return {
        "ece": ece.item(),
        "chi2_statistic": chi2_stat.item(),
        "p_value": p_value,
        "is_calibrated": p_value > 0.05,
    }


def compute_uncertainty_metrics(
    logits: Tensor,
    labels: Tensor,
    ood_logits: Optional[Tensor] = None,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute all uncertainty metrics.

    Args:
        logits: Model logits
        labels: True labels
        ood_logits: Optional OOD logits
        n_bins: Number of bins for ECE

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "ece": expected_calibration_error(logits, labels, n_bins),
        "mce": maximum_calibration_error(logits, labels, n_bins),
        "nll": negative_log_likelihood(logits, labels),
        "brier": brier_score(logits, labels),
        "correlation": confidence_accuracy_correlation(logits, labels),
    }

    brier_decomp = brier_decomposition(logits, labels)
    metrics.update(brier_decomp)

    if ood_logits is not None:
        auroc, aupr = auroc_ood_detection(logits, ood_logits, method="energy")
        metrics["auroc_ood"] = auroc
        metrics["aupr_ood"] = aupr

    return metrics


class UncertaintyMetricTracker:
    """Track uncertainty metrics over training.

    Args:
        n_bins: Number of bins for ECE
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.history: Dict[str, List[float]] = {
            "ece": [],
            "nll": [],
            "brier": [],
            "auroc": [],
        }

    def update(
        self,
        logits: Tensor,
        labels: Tensor,
        ood_logits: Optional[Tensor] = None,
    ):
        """Update metrics with new batch.

        Args:
            logits: Model logits
            labels: True labels
            ood_logits: Optional OOD logits
        """
        ece = expected_calibration_error(logits, labels, self.n_bins)
        nll = negative_log_likelihood(logits, labels)
        brier = brier_score(logits, labels)

        self.history["ece"].append(ece)
        self.history["nll"].append(nll)
        self.history["brier"].append(brier)

        if ood_logits is not None:
            auroc, _ = auroc_ood_detection(logits, ood_logits)
            self.history["auroc"].append(auroc)

    def get_averages(self) -> Dict[str, float]:
        """Get average metrics over history.

        Returns:
            Dictionary of average metrics
        """
        return {k: np.mean(v) if v else 0.0 for k, v in self.history.items()}

    def reset(self):
        """Reset history."""
        for key in self.history:
            self.history[key] = []
