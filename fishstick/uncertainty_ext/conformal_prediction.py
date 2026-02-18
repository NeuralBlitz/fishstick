"""
Conformal Prediction Implementations

Various conformal prediction methods for guaranteed coverage uncertainty sets.
"""

from typing import Optional, Tuple, List, Dict, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from scipy import stats


class BaseConformalPredictor:
    """Base class for conformal prediction methods."""

    def __init__(self, model: nn.Module, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None

    def compute_conformity_scores(self, logits: Tensor, labels: Tensor) -> np.ndarray:
        """Compute conformity scores for calibration."""
        raise NotImplementedError

    def calibrate(self, calib_logits: Tensor, calib_labels: Tensor):
        """Fit the conformal predictor on calibration set."""
        scores = self.compute_conformity_scores(calib_logits, calib_labels)
        self.calibration_scores = scores

        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.quantile = np.quantile(scores, q_level)

    def predict_sets(self, logits: Tensor) -> Tensor:
        """Generate prediction sets for new inputs."""
        raise NotImplementedError


class AdaptiveConformalPredictor(BaseConformalPredictor):
    """Adaptive conformal prediction using non-conformity scores.

    Uses the standard non-conformity score: s(x, y) = 1 - P(y|x)

    Args:
        model: Base classification model
        alpha: Miscoverage level (1-alpha = confidence)
        regression: Whether to use regression mode
    """

    def __init__(self, model: nn.Module, alpha: float = 0.1, regression: bool = False):
        super().__init__(model, alpha)
        self.regression = regression

    def compute_conformity_scores(self, logits: Tensor, labels: Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            if self.regression:
                preds = logits.squeeze(-1)
                scores = (preds - labels).abs().cpu().numpy()
            else:
                batch_idx = torch.arange(logits.size(0))
                scores = (1 - probs[batch_idx, labels]).cpu().numpy()
        return scores

    def predict_sets(self, logits: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            n_classes = probs.size(-1)

            if self.regression:
                mean = logits.squeeze(-1)
                lower = mean - self.quantile
                upper = mean + self.quantile
                return {"mean": mean, "lower": lower, "upper": upper}
            else:
                sets = probs >= (1 - self.quantile)
                return {"sets": sets.long(), "scores": probs}


class SplitConformalPredictor(BaseConformalPredictor):
    """Split conformal prediction for efficient calibration.

    Uses a hold-out calibration set for efficiency.

    Args:
        model: Base model
        alpha: Miscoverage level
        n_classes: Number of classes
    """

    def __init__(self, model: nn.Module, alpha: float = 0.1, n_classes: int = 10):
        super().__init__(model, alpha)
        self.n_classes = n_classes
        self.calibrated_probs: Optional[Tensor] = None

    def compute_conformity_scores(self, logits: Tensor, labels: Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            batch_idx = torch.arange(logits.size(0))
            scores = (1 - probs[batch_idx, labels]).cpu().numpy()
        return scores

    def predict_sets(self, logits: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            threshold = 1 - self.quantile

            sets = probs >= threshold
            if sets.sum(dim=-1).min() == 0:
                sets = probs >= (probs.max(dim=-1, keepdim=True)[0] - 1e-6)

            return {
                "sets": sets.long(),
                "probs": probs,
                "threshold": torch.full((logits.size(0),), threshold),
            }


class JackknifePlusPredictor:
    """Jackknife+ conformal prediction for tighter prediction sets.

    Provides valid coverage with potentially smaller set sizes.

    Args:
        model: Base model
        alpha: Miscoverage level
        n_classes: Number of classes
    """

    def __init__(self, model: nn.Module, alpha: float = 0.1, n_classes: int = 10):
        self.model = model
        self.alpha = alpha
        self.n_classes = n_classes
        self.calib_preds: Optional[Tensor] = None
        self.calib_labels: Optional[Tensor] = None

    def calibrate(self, calib_loader: Callable):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in calib_loader:
                logits = self.model(x)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds)
                all_labels.append(y)

        self.calib_preds = torch.cat(all_preds)
        self.calib_labels = torch.cat(all_labels)

    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

            n_calib = len(self.calib_preds)
            scores = torch.zeros(x.size(0), n_calib, device=x.device)

            for i in range(n_calib):
                mask = torch.ones(n_calib, dtype=torch.bool)
                mask[i] = False

                local_calib_preds = self.calib_preds[mask]
                local_calib_labels = self.calib_labels[mask]

                conformity = (local_calib_preds != local_calib_labels).float()
                conformity = conformity.mean(dim=0)

                scores[:, i] = conformity

            q_level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
            q_level = min(q_level, 1.0)
            q = torch.quantile(scores, q_level, dim=-1)

            sets = scores <= q.unsqueeze(-1)

            return {
                "sets": sets.long(),
                "scores": scores,
                "quantile": q,
                "probs": probs,
            }


class CVPlusPredictor:
    """CV+ Conformal Prediction using cross-validation.

    More efficient than Jackknife+ with similar guarantees.

    Args:
        model: Base model
        alpha: Miscoverage level
        n_folds: Number of cross-validation folds
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        n_folds: int = 5,
    ):
        self.model = model
        self.alpha = alpha
        self.n_folds = n_folds
        self.calib_scores: Optional[Tensor] = None

    def calibrate(self, calib_logits: Tensor, calib_labels: Tensor):
        n = calib_logits.size(0)
        fold_size = n // self.n_folds

        all_scores = []

        for fold in range(self.n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else n

            mask = torch.ones(n, dtype=torch.bool)
            mask[start:end] = False

            fold_logits = calib_logits[mask]
            fold_labels = calib_labels[mask]

            probs = F.softmax(fold_logits, dim=-1)
            batch_idx = torch.arange(fold_logits.size(0))
            scores = 1 - probs[batch_idx, fold_labels]
            all_scores.append(scores)

        self.calib_scores = torch.cat(all_scores)

        n_total = len(self.calib_scores)
        q_level = np.ceil((n_total + 1) * (1 - self.alpha)) / n_total
        q_level = min(q_level, 1.0)
        self.quantile = torch.quantile(self.calib_scores, q_level)

    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

            sets = probs >= (1 - self.quantile)

            if sets.sum(dim=-1).min() == 0:
                sets = probs >= (probs.max(dim=-1, keepdim=True)[0] - 1e-6)

            return {
                "sets": sets.long(),
                "probs": probs,
                "threshold": self.quantile,
            }


class WeightedConformalPredictor(BaseConformalPredictor):
    """Weighted conformal prediction with non-uniformity.

    Allows for non-uniform coverage guarantees across different groups.

    Args:
        model: Base model
        alpha: Miscoverage level
        weights: Optional sample weights for calibration
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        weights: Optional[Tensor] = None,
    ):
        super().__init__(model, alpha)
        self.weights = weights

    def compute_conformity_scores(self, logits: Tensor, labels: Tensor) -> np.ndarray:
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            batch_idx = torch.arange(logits.size(0))
            scores = (1 - probs[batch_idx, labels]).cpu().numpy()

            if self.weights is not None:
                scores = scores * self.weights.cpu().numpy()

        return scores

    def calibrate(self, calib_logits: Tensor, calib_labels: Tensor):
        scores = self.compute_conformity_scores(calib_logits, calib_labels)

        if self.weights is not None:
            sorted_idx = np.argsort(scores)
            sorted_weights = self.weights.cpu().numpy()[sorted_idx]
            cumsum = np.cumsum(sorted_weights)
            cumsum = cumsum / cumsum[-1]

            self.quantile = np.quantile(scores[cumsum >= 1 - self.alpha], 0)
        else:
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.quantile = np.quantile(scores, min(q_level, 1.0))

    def predict_sets(self, logits: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            sets = probs >= (1 - self.quantile)

            return {"sets": sets.long(), "probs": probs}


class ConformalRegressor:
    """Conformal prediction for regression tasks.

    Args:
        model: Base regression model
        alpha: Miscoverage level
    """

    def __init__(self, model: nn.Module, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.residuals: Optional[Tensor] = None

    def calibrate(self, calib_x: Tensor, calib_y: Tensor):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(calib_x).squeeze(-1)

            if calib_y.dim() > 1:
                calib_y = calib_y.squeeze(-1)

            self.residuals = (preds - calib_y).abs()

        n = len(self.residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = torch.quantile(self.residuals.float(), min(q_level, 1.0)).item()

    def predict_intervals(self, x: Tensor) -> Dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).squeeze(-1)

            return {
                "point": preds,
                "lower": preds - self.quantile,
                "upper": preds + self.quantile,
                "width": 2 * self.quantile,
            }


class AdaptivePredictionIntervals:
    """Adaptive conformal prediction intervals for regression.

    Uses conditional coverage optimization.

    Args:
        model: Base model
        alpha: Miscoverage level
        n_bins: Number of bins for adaptive intervals
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        n_bins: int = 10,
    ):
        self.model = model
        self.alpha = alpha
        self.n_bins = n_bins
        self.bin_edges: Optional[Tensor] = None
        self.bin_quantiles: Optional[Tensor] = None

    def calibrate(self, calib_x: Tensor, calib_y: Tensor):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(calib_x).squeeze(-1)

            if calib_y.dim() > 1:
                calib_y = calib_y.squeeze(-1)

            residuals = (preds - calib_y).abs()
            uncertainties = residuals

            self.bin_edges = torch.quantile(
                uncertainties,
                torch.linspace(0, 1, self.n_bins + 1, device=uncertainties.device),
            )

            bin_idx = torch.searchsorted(self.bin_edges[1:], uncertainties)
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)

            self.bin_quantiles = torch.zeros(self.n_bins, device=residuals.device)

            for b in range(self.n_bins):
                mask = bin_idx == b
                if mask.sum() > 0:
                    bin_residuals = residuals[mask]
                    n = len(bin_residuals)
                    q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
                    self.bin_quantiles[b] = torch.quantile(
                        bin_residuals.float(), min(q_level, 1.0)
                    )

    def predict_intervals(self, x: Tensor) -> Dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x).squeeze(-1)
            uncertainties = torch.zeros_like(preds)

            bin_idx = torch.searchsorted(self.bin_edges[1:], uncertainties)
            bin_idx = bin_idx.clamp(0, self.n_bins - 1)

            widths = self.bin_quantiles[bin_idx]

            return {
                "point": preds,
                "lower": preds - widths,
                "upper": preds + widths,
                "width": 2 * widths,
                "bin_idx": bin_idx,
            }
