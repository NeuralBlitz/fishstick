"""
Temperature Scaling for Model Calibration

Various temperature scaling and calibration methods for neural networks.
"""

from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from scipy import optimize
from scipy.special import expit


class TemperatureScaling(nn.Module):
    """Single temperature scaling for calibrated predictions.

    Scales logits by a learned temperature parameter to improve calibration.

    Args:
        model: Base classifier
        init_temp: Initial temperature value
    """

    def __init__(self, model: nn.Module, init_temp: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Find optimal temperature using NLL minimization.

        Args:
            logits: Model logits on calibration set
            labels: True labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Optimal temperature value
        """
        self.temperature.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            nll = F.cross_entropy(scaled, labels)
            nll.backward()
            return nll

        optimizer.step(eval)
        self.temperature.requires_grad_(False)

        return self.temperature.item()


class VectorTemperatureScaling(nn.Module):
    """Vector temperature scaling with per-class temperatures.

    Uses different temperature for each class for finer calibration.

    Args:
        model: Base classifier
        num_classes: Number of classes
    """

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.temperatures = nn.Parameter(torch.ones(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        temps = self.temperatures.unsqueeze(0).expand(logits.size(0), -1)
        return logits / temps

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> Tensor:
        """Find optimal per-class temperatures."""
        self.temperatures.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.temperatures], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            temps = self.temperatures.unsqueeze(0).expand(logits.size(0), -1)
            scaled = logits / temps
            nll = F.cross_entropy(scaled, labels)
            nll.backward()
            return nll

        optimizer.step(eval)
        self.temperatures.requires_grad_(False)

        return self.temperatures.data.clone()


class ClassWiseTemperatureScaling(nn.Module):
    """Class-wise temperature scaling with calibration per class.

    Uses separate calibration for each class with Platt scaling.

    Args:
        model: Base classifier
        num_classes: Number of classes
    """

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.ones(num_classes))
        self.beta = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        alpha = F.softplus(self.alpha)
        beta = self.beta.unsqueeze(0).expand(logits.size(0), -1)
        return (logits * alpha) + beta

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """Calibrate alpha and beta parameters."""
        self.alpha.requires_grad_(True)
        self.beta.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.alpha, self.beta], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            preds = self.forward(logits)
            nll = F.cross_entropy(preds, labels)
            nll.backward()
            return nll

        optimizer.step(eval)
        self.alpha.requires_grad_(False)
        self.beta.requires_grad_(False)

        return F.softplus(self.alpha.data), self.beta.data


class PlattScaling(nn.Module):
    """Platt scaling for probability calibration.

    Applies logistic regression on logits for calibration.

    Args:
        model: Base classifier
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        scaled = logits * self.weight + self.bias
        return scaled

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> Tuple[float, float]:
        """Calibrate weight and bias."""
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias], lr=lr, max_iter=max_iter
        )

        def eval():
            optimizer.zero_grad()
            preds = self.forward(logits)
            nll = F.cross_entropy(preds, labels)
            nll.backward()
            return nll

        optimizer.step(eval)
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

        return self.weight.item(), self.bias.item()


class BetaCalibration(nn.Module):
    """Beta calibration for multi-class probabilities.

    Uses beta distribution parameters for calibration.

    Args:
        model: Base classifier
        num_classes: Number of classes
    """

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.a = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.ones(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)

        a = F.softplus(self.a)
        b = F.softplus(self.b)

        calibrated = (probs * a) / (probs * a + (1 - probs) * b + 1e-10)

        return calibrated

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """Calibrate a and b parameters."""
        self.a.requires_grad_(True)
        self.b.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            probs = self.forward(logits)
            nll = -torch.log(probs.gather(1, labels.unsqueeze(1)) + 1e-10).mean()
            nll.backward()
            return nll

        optimizer.step(eval)
        self.a.requires_grad_(False)
        self.b.requires_grad_(False)

        return F.softplus(self.a.data), F.softplus(self.b.data)


class IsotonicCalibrator:
    """Isotonic regression calibration.

    Non-parametric monotonic calibration.

    Args:
        n_classes: Number of classes
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.fits: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, logits: Tensor, labels: Tensor):
        """Fit isotonic calibration per class."""
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        labels_onehot = F.one_hot(labels, self.n_classes).cpu().numpy()

        for c in range(self.n_classes):
            mask = labels_onehot[:, c] == 1
            if mask.sum() < 2:
                self.fits[c] = (np.array([0, 1]), np.array([0, 1]))
                continue

            from sklearn.isotonic import IsotonicRegression

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs[:, c], labels_onehot[:, c])
            self.fits[c] = (ir.X_thresholds_, ir.y_thresholds_)

    def transform(self, logits: Tensor) -> Tensor:
        """Apply isotonic calibration."""
        probs = F.softmax(logits, dim=-1)
        calibrated = torch.zeros_like(probs)

        for c in range(self.n_classes):
            if c in self.fits:
                x_thresh, y_thresh = self.fits[c]
                calibrated[:, c] = torch.from_numpy(
                    np.interp(probs[:, c].cpu().numpy(), x_thresh, y_thresh)
                ).to(probs.device)

        return calibrated

    def fit_transform(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Fit and transform in one step."""
        self.fit(logits, labels)
        return self.transform(logits)


class HistogramCalibrator:
    """Histogram-based calibration using binning.

    Args:
        n_bins: Number of histogram bins
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_edges: Optional[np.ndarray] = None
        self.binaccuracies: Optional[np.ndarray] = None

    def fit(self, logits: Tensor, labels: Tensor):
        """Compute calibration histogram."""
        probs = F.softmax(logits, dim=-1)
        max_probs, pred_labels = probs.max(dim=-1)

        correct = (pred_labels == labels).float()

        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self.binaccuracies = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (max_probs >= self.bin_edges[i]) & (
                max_probs < self.bin_edges[i + 1]
            )
            if mask.sum() > 0:
                self.binaccuracies[i] = correct[mask].mean().item()

        self.bin_edges[-1] = 1.0 + 1e-6

    def transform(self, logits: Tensor) -> Tensor:
        """Apply histogram calibration."""
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)

        bin_idx = np.digitize(max_probs.cpu().numpy(), self.bin_edges[1:]) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        calibrated = torch.zeros_like(probs)
        for i in range(probs.size(0)):
            calibrated[i] = probs[i] * (
                self.binaccuracies[bin_idx[i]] / (max_probs[i].item() + 1e-10)
            )

        calibrated = calibrated / calibrated.sum(dim=-1, keepdim=True)

        return calibrated


class FocalTemperatureScaling(nn.Module):
    """Temperature scaling with focal loss for hard example mining.

    Args:
        model: Base classifier
        gamma: Focal loss gamma parameter
    """

    def __init__(self, model: nn.Module, gamma: float = 2.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Calibrate using focal loss objective."""
        self.temperature.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            probs = F.softmax(scaled, dim=-1)

            pt = probs.gather(1, labels.unsqueeze(1)).squeeze()
            focal_weight = (1 - pt) ** self.gamma

            focal_loss = F.cross_entropy(scaled, labels, reduction="none")
            loss = (focal_weight * focal_loss).mean()
            loss.backward()
            return loss

        optimizer.step(eval)
        self.temperature.requires_grad_(False)

        return self.temperature.item()


class MixupCalibrator(nn.Module):
    """Calibration with mixup data augmentation.

    Uses mixup to create virtual training examples.

    Args:
        model: Base classifier
        alpha: Mixup alpha parameter
    """

    def __init__(self, model: nn.Module, alpha: float = 0.2):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_mixup_logits(self, logits1: Tensor, logits2: Tensor, lam: float) -> Tensor:
        """Compute mixed logits."""
        return lam * logits1 + (1 - lam) * logits2

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Calibrate using mixup augmented calibration set."""
        self.model.train()

        n = logits.size(0)
        indices = torch.randperm(n)

        lam = np.random.beta(self.alpha, self.alpha)
        mixed_logits = self.get_mixup_logits(logits, logits[indices], lam)

        mixed_labels = torch.where(torch.rand(n) < lam, labels, labels[indices])

        self.model.eval()
        with torch.no_grad():
            scaled = mixed_logits
            probs = F.softmax(scaled, dim=-1)

        self.temperature = nn.Parameter(torch.ones(1))
        self.temperature.requires_grad_(True)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            scaled = mixed_logits / self.temperature
            nll = F.cross_entropy(scaled, mixed_labels)
            nll.backward()
            return nll

        optimizer.step(eval)
        temp = self.temperature.item()
        self.temperature.requires_grad_(False)

        return temp


class OnlineCalibrator:
    """Online adaptive calibration with sliding window.

    Continuously adapts calibration during inference.

    Args:
        window_size: Size of sliding window for calibration
        alpha: Smoothing factor for exponential moving average
    """

    def __init__(self, window_size: int = 1000, alpha: float = 0.01):
        self.window_size = window_size
        self.alpha = alpha
        self.temperature = 1.0

        self.recent_logits: List[Tensor] = []
        self.recent_labels: List[Tensor] = []

    def update(self, logits: Tensor, labels: Tensor):
        """Update calibration with new samples."""
        self.recent_logits.append(logits)
        self.recent_labels.append(labels)

        total_len = sum(x.size(0) for x in self.recent_logits)

        if total_len > self.window_size:
            excess = total_len - self.window_size
            while excess > 0 and self.recent_logits:
                if self.recent_logits[0].size(0) <= excess:
                    excess -= self.recent_logits[0].size(0)
                    self.recent_logits.pop(0)
                    self.recent_labels.pop(0)
                else:
                    self.recent_logits[0] = self.recent_logits[0][excess:]
                    self.recent_labels[0] = self.recent_labels[0][excess:]
                    excess = 0

    def calibrate_step(self, lr: float = 0.01):
        """Perform one calibration step."""
        if not self.recent_logits:
            return

        all_logits = torch.cat(self.recent_logits)
        all_labels = torch.cat(self.recent_labels)

        def nll(t: float) -> float:
            scaled = all_logits / t
            return F.cross_entropy(scaled, all_labels).item()

        result = optimize.minimize_scalar(nll, bounds=(0.01, 10), method="bounded")

        new_temp = result.x
        self.temperature = (1 - self.alpha) * self.temperature + self.alpha * new_temp

    def get_temperature(self) -> float:
        """Get current temperature."""
        return self.temperature


from typing import Dict, List
