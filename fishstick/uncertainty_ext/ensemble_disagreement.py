"""
Ensemble Disagreement Measures

Various methods for measuring disagreement and uncertainty in model ensembles.
"""

from typing import Optional, List, Tuple, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


class EnsembleDisagreement:
    """Base class for ensemble disagreement measures.

    Args:
        models: List of models in the ensemble
    """

    def __init__(self, models: nn.ModuleList):
        self.models = models

    def get_predictions(self, x: Tensor) -> List[Tensor]:
        """Get predictions from all models."""
        self._set_eval()
        predictions = []

        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                predictions.append(logits)

        return predictions

    def get_probabilities(self, x: Tensor) -> Tensor:
        """Get probability distributions from all models.

        Returns:
            Tensor of shape (n_models, batch_size, n_classes)
        """
        predictions = self.get_predictions(x)
        probs = [F.softmax(pred, dim=-1) for pred in predictions]
        return torch.stack(probs, dim=0)

    def _set_eval(self):
        """Set all models to eval mode."""
        for model in self.models:
            model.eval()


class VarianceDisagreement(EnsembleDisagreement):
    """Variance-based disagreement measure.

    Computes variance of predictions across ensemble members.
    """

    def compute(self, x: Tensor) -> Tensor:
        """Compute variance disagreement.

        Returns:
            Tensor of shape (batch_size,) with variance per sample
        """
        probs = self.get_probabilities(x)

        variance = probs.var(dim=0)

        return variance.mean(dim=-1)


class EntropyDisagreement(EnsembleDisagreement):
    """Entropy-based disagreement measure.

    Uses predictive entropy of mean prediction.
    """

    def compute(self, x: Tensor) -> Tensor:
        """Compute entropy disagreement.

        Returns:
            Tensor of shape (batch_size,) with entropy per sample
        """
        probs = self.get_probabilities(x)

        mean_probs = probs.mean(dim=0)

        entropy_vals = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        return entropy_vals


class MutualInformationDisagreement(EnsembleDisagreement):
    """Mutual information based disagreement.

    Computes MI between input and prediction: I(x, y) = H[y] - H[y|x]
    """

    def compute(self, x: Tensor) -> Tensor:
        """Compute mutual information disagreement.

        Returns:
            Tensor of shape (batch_size,) with MI per sample
        """
        probs = self.get_probabilities(x)

        mean_probs = probs.mean(dim=0)
        mean_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=-1)

        entropy_per_model = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        expected_entropy = entropy_per_model.mean(dim=0)

        mi = mean_entropy - expected_entropy

        return mi


class CosineDisagreement(EnsembleDisagreement):
    """Cosine similarity based disagreement.

    Measures disagreement as 1 - cosine similarity between predictions.
    """

    def compute(self, x: Tensor) -> Tensor:
        """Compute cosine disagreement.

        Returns:
            Tensor of shape (batch_size,) with cosine disagreement per sample
        """
        predictions = self.get_predictions(x)
        predictions_stack = torch.stack(predictions, dim=0)

        predictions_norm = F.normalize(predictions_stack, p=2, dim=-1)

        cosine_matrix = torch.einsum("ijk,ilk->ijl", predictions_norm, predictions_norm)

        n_models = len(self.models)
        cosine_matrix = cosine_matrix.mean(dim=0)

        disagreement = 1 - cosine_matrix.diagonal(dim1=-2, dim2=-1)

        return disagreement.mean(dim=-1)


class PairwiseKLDisagreement(EnsembleDisagreement):
    """Pairwise KL divergence disagreement.

    Computes average KL divergence between all pairs of ensemble members.
    """

    def compute(self, x: Tensor) -> Tensor:
        """Compute pairwise KL disagreement.

        Returns:
            Tensor of shape (batch_size,) with KL disagreement per sample
        """
        probs = self.get_probabilities(x)

        n_models = probs.size(0)
        batch_size = probs.size(1)
        n_classes = probs.size(2)

        kl_sum = torch.zeros(batch_size, device=probs.device)

        for i in range(n_models):
            for j in range(i + 1, n_models):
                kl_div = F.kl_div(
                    probs[i].log(),
                    probs[j],
                    reduction="none",
                ).sum(dim=-1)
                kl_sum += kl_div

        n_pairs = n_models * (n_models - 1) / 2
        return kl_sum / n_pairs


class DisagreementEnsemble:
    """Combined ensemble disagreement with multiple measures.

    Args:
        models: List of models in the ensemble
        weights: Optional weights for combining measures
    """

    def __init__(
        self,
        models: nn.ModuleList,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.models = models

        if weights is None:
            weights = {
                "variance": 0.25,
                "entropy": 0.25,
                "mutual_info": 0.25,
                "cosine": 0.125,
                "kl": 0.125,
            }

        self.weights = weights

        self.variance = VarianceDisagreement(models)
        self.entropy = EntropyDisagreement(models)
        self.mutual_info = MutualInformationDisagreement(models)
        self.cosine = CosineDisagreement(models)
        self.kl = PairwiseKLDisagreement(models)

    def compute_all(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute all disagreement measures.

        Returns:
            Dictionary of disagreement measures
        """
        measures = {
            "variance": self.variance.compute(x),
            "entropy": self.entropy.compute(x),
            "mutual_info": self.mutual_info.compute(x),
            "cosine": self.cosine.compute(x),
            "kl": self.kl.compute(x),
        }

        return measures

    def compute_combined(self, x: Tensor) -> Tensor:
        """Compute weighted combination of all measures.

        Returns:
            Combined disagreement score
        """
        measures = self.compute_all(x)

        combined = sum(self.weights[k] * measures[k] for k in self.weights)

        return combined


class DoubleSoftmaxDisagreement(EnsembleDisagreement):
    """Double softmax disagreement measure.

    Uses double softmax for more confident predictions.
    """

    def compute(self, x: Tensor) -> Tensor:
        """Compute double softmax disagreement.

        Returns:
            Tensor of shape (batch_size,)
        """
        predictions = self.get_predictions(x)
        probs_list = [F.softmax(pred, dim=-1) for pred in predictions]
        probs = torch.stack(probs_list, dim=0)

        double_softmax = F.softmax(probs.mean(dim=0), dim=-1)

        confidence = double_softmax.max(dim=-1)[0]

        return 1 - confidence


class SelectiveDisagreement(EnsembleDisagreement):
    """Selective disagreement for active learning.

    Computes disagreement only on high-uncertainty samples.
    """

    def __init__(self, models: nn.ModuleList, threshold: float = 0.5):
        super().__init__(models)
        self.threshold = threshold

    def compute(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute disagreement with selection mask.

        Returns:
            Tuple of (disagreement, select_mask)
        """
        probs = self.get_probabilities(x)

        mean_probs = probs.mean(dim=0)
        confidence = mean_probs.max(dim=-1)[0]

        select_mask = confidence < self.threshold

        if select_mask.sum() == 0:
            return torch.zeros(x.size(0), device=x.device), select_mask

        variance = probs.var(dim=0).mean(dim=-1)

        return variance, select_mask


class DiversityEnsemble:
    """Measures diversity in ensemble predictions.

    Args:
        models: List of models
    """

    def __init__(self, models: nn.ModuleList):
        self.models = models

    def compute_diversity(self, x: Tensor) -> Dict[str, Tensor]:
        """Compute various diversity measures.

        Returns:
            Dictionary of diversity scores
        """
        predictions = self.get_predictions(x)
        probs_list = [F.softmax(pred, dim=-1) for pred in predictions]
        probs = torch.stack(probs_list, dim=0)

        mean_probs = probs.mean(dim=0)

        agreement = (probs.argmax(dim=-1) == mean_probs.argmax(dim=-1)).float()
        diversity = 1 - agreement.mean(dim=0)

        return {
            "diversity": diversity,
            "mean_agreement": agreement.mean(dim=0),
            "variance": probs.var(dim=0).mean(dim=-1),
        }


class QueryByDisagreement:
    """Active learning query by disagreement strategy.

    Args:
        models: List of models for ensemble
    """

    def __init__(self, models: nn.ModuleList):
        self.models = models

    def query(self, unlabeled_x: Tensor, n_query: int) -> Tuple[Tensor, Tensor]:
        """Query most disagreement samples.

        Args:
            unlabeled_x: Unlabeled input data
            n_query: Number of samples to query

        Returns:
            Tuple of (queried_indices, disagreement_scores)
        """
        probs = self.get_probabilities(unlabeled_x)

        variance = probs.var(dim=0).mean(dim=-1)

        _, top_indices = variance.topk(min(n_query, variance.size(0)))

        return top_indices, variance[top_indices]


class AdversarialDisagreement:
    """Adversarial disagreement using adversarial perturbations.

    Args:
        models: List of models
        epsilon: Perturbation magnitude
    """

    def __init__(self, models: nn.ModuleList, epsilon: float = 0.1):
        self.models = models
        self.epsilon = epsilon

    def compute_adversarial_disagreement(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute disagreement on original and adversarial examples.

        Returns:
            Tuple of (original_disagreement, adversarial_disagreement)
        """
        self._set_eval()

        original_probs = self.get_probabilities(x)

        grad_signs = []
        for model in self.models:
            model.zero_grad()
            x_temp = x.detach().requires_grad_(True)
            logits = model(x_temp)

            loss = logits.sum()
            loss.backward()

            grad_signs.append(x_temp.grad.sign())

        avg_grad_sign = torch.stack(grad_signs).mean(dim=0)

        x_adv = x + self.epsilon * avg_grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)

        adv_probs = self.get_probabilities(x_adv)

        original_variance = original_probs.var(dim=0).mean(dim=-1)
        adv_variance = adv_probs.var(dim=0).mean(dim=-1)

        return original_variance, adv_variance

    def _set_eval(self):
        for model in self.models:
            model.eval()


class DisagreementVisualizer:
    """Visualization utilities for ensemble disagreement.

    Args:
        models: List of models
    """

    def __init__(self, models: nn.ModuleList):
        self.models = models

    def compute_confusion_matrix_ensemble(self, x: Tensor, labels: Tensor) -> Tensor:
        """Compute ensemble prediction confusion matrix.

        Returns:
            Confusion matrix tensor
        """
        probs = self.get_probabilities(x)

        mean_preds = probs.mean(dim=0).argmax(dim=-1)

        n_classes = probs.size(2)
        confusion = torch.zeros(n_classes, n_classes)

        for true, pred in zip(labels, mean_preds):
            confusion[true, pred] += 1

        return confusion

    def get_predictions(self, x: Tensor) -> List[Tensor]:
        self._set_eval()
        predictions = []
        with torch.no_grad():
            for model in self.models:
                predictions.append(model(x))
        return predictions

    def get_probabilities(self, x: Tensor) -> Tensor:
        predictions = self.get_predictions(x)
        probs = [F.softmax(pred, dim=-1) for pred in predictions]
        return torch.stack(probs, dim=0)

    def _set_eval(self):
        for model in self.models:
            model.eval()
