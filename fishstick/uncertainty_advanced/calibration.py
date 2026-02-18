import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod
import numpy as np
from sklearn.isotonic import IsotonicRegression as SkIsotonicRegression


class TemperatureScaling(nn.Module):
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
        verbose: bool = False,
    ) -> "TemperatureScaling":
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)

        if verbose:
            print(f"Optimal temperature: {self.temperature.item():.4f}")

        return self

    def predict(self, logits: Tensor) -> Dict[str, Tensor]:
        self.eval()
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probs = F.softmax(scaled_logits, dim=-1)
            return {
                "logits": scaled_logits,
                "probs": probs,
                "predictions": probs.argmax(dim=-1),
            }


class PlattScaling(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.weight = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, logits: Tensor) -> Tensor:
        scaled_logits = logits * self.weight + self.bias
        return scaled_logits

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "PlattScaling":
        self.train()
        optimizer = torch.optim.Adam([self.weight, self.bias], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, logits: Tensor) -> Dict[str, Tensor]:
        self.eval()
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            probs = F.softmax(scaled_logits, dim=-1)
            return {
                "logits": scaled_logits,
                "probs": probs,
                "predictions": probs.argmax(dim=-1),
            }


class IsotonicRegressionCalibrator(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.calibrators: List[SkIsotonicRegression] = []

    def fit(self, logits: Tensor, labels: Tensor) -> "IsotonicRegressionCalibrator":
        probs = F.softmax(logits, dim=-1)

        self.calibrators = []
        for c in range(self.n_classes):
            binary_labels = (labels == c).float().numpy()
            calibrator = SkIsotonicRegression(out_of_bounds="clip")
            calibrator.fit(probs[:, c].cpu().numpy(), binary_labels)
            self.calibrators.append(calibrator)

        return self

    def predict(self, logits: Tensor) -> Dict[str, Tensor]:
        self.eval()
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            calibrated_probs = torch.zeros_like(probs)

            for c in range(self.n_classes):
                calibrated_probs[:, c] = torch.tensor(
                    self.calibrators[c].predict(probs[:, c].cpu().numpy())
                )

            calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=-1, keepdim=True)

            return {
                "probs": calibrated_probs,
                "predictions": calibrated_probs.argmax(dim=-1),
            }


class ReliabilityDiagram:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)

    def compute_reliability_data(
        self,
        probs: Tensor,
        labels: Tensor,
    ) -> Dict[str, Any]:
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels)

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(self.n_bins):
            bin_lower = self.bin_boundaries[i]
            bin_upper = self.bin_boundaries[i + 1]

            mask = (confidences > bin_lower) & (confidences <= bin_upper)
            if mask.sum() > 0:
                bin_accuracies.append(accuracies[mask].float().mean().item())
                bin_confidences.append(confidences[mask].mean().item())
                bin_counts.append(mask.sum().item())

        return {
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "n_filled_bins": len(bin_accuracies),
        }

    def plot(
        self,
        probs: Tensor,
        labels: Tensor,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        reliability_data = self.compute_reliability_data(probs, labels)

        if save_path:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))

            bin_conf = reliability_data["bin_confidences"]
            bin_acc = reliability_data["bin_accuracies"]

            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.bar(
                bin_conf,
                bin_acc,
                width=0.1,
                alpha=0.7,
                label="Reliability",
            )
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title("Reliability Diagram")
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            plt.savefig(save_path)
            plt.close()

        return reliability_data


class ExpectedCalibrationError:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.diagram = ReliabilityDiagram(n_bins=n_bins)

    def compute(self, probs: Tensor, labels: Tensor) -> float:
        data = self.diagram.compute_reliability_data(probs, labels)

        ece = 0.0
        total_count = sum(data["bin_counts"])
        for i in range(len(data["bin_counts"])):
            weight = data["bin_counts"][i] / total_count
            ece += weight * abs(data["bin_confidences"][i] - data["bin_accuracies"][i])

        return ece

    def __call__(self, probs: Tensor, labels: Tensor) -> float:
        return self.compute(probs, labels)


class MaximumCalibrationError:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.diagram = ReliabilityDiagram(n_bins=n_bins)

    def compute(self, probs: Tensor, labels: Tensor) -> float:
        data = self.diagram.compute_reliability_data(probs, labels)

        max_error = 0.0
        for i in range(len(data["bin_counts"])):
            error = abs(data["bin_confidences"][i] - data["bin_accuracies"][i])
            max_error = max(max_error, error)

        return max_error

    def __call__(self, probs: Tensor, labels: Tensor) -> float:
        return self.compute(probs, labels)


class CalibratedClassifier:
    def __init__(
        self,
        model: nn.Module,
        method: str = "temperature",
        n_classes: int = 10,
    ):
        self.model = model
        self.method = method
        self.n_classes = n_classes
        self.calibrator = self._create_calibrator()

    def _create_calibrator(self) -> nn.Module:
        if self.method == "temperature":
            return TemperatureScaling()
        elif self.method == "platt":
            return PlattScaling(self.n_classes)
        elif self.method == "isotonic":
            return IsotonicRegressionCalibrator(self.n_classes)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "CalibratedClassifier":
        self.model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for data, labels in train_loader:
                logits = self.model(data)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        if self.method == "isotonic":
            self.calibrator.fit(logits, labels)
        else:
            self.calibrator.fit(logits, labels, lr=lr, max_iter=max_iter)

        return self

    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)

        if isinstance(self.calibrator, IsotonicRegressionCalibrator):
            return self.calibrator.predict(logits)
        else:
            return self.calibrator.predict(logits)

    def predict_with_uncertainty(
        self, x: Tensor
    ) -> Dict[str, Tensor]:
        result = self.predict(x)

        probs = result["probs"]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        result["entropy"] = entropy

        return result


class Adaptive Calibration:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.temperature_per_class = nn.Parameter(torch.ones(n_classes))

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature_per_class.unsqueeze(0)

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "Adaptive Calibration":
        self.train()
        optimizer = torch.optim.Adam([self.temperature_per_class], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            optimizer.step()

        return self


class DirichletCalibration(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = nn.Parameter(torch.ones(n_classes))

    def forward(self, logits: Tensor) -> Tensor:
        alpha = F.softplus(self.alpha).unsqueeze(0)
        probs = F.softmax(logits, dim=-1) * alpha
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "DirichletCalibration":
        self.train()
        optimizer = torch.optim.Adam([self.alpha], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            probs = self.forward(logits)
            loss = F.nll_loss(torch.log(probs + 1e-10), labels)
            loss.backward()
            optimizer.step()

        return self


class HistogramBinning:
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_acc = None

    def fit(self, probs: Tensor, labels: Tensor) -> "HistogramBinning":
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(labels).float()

        self.bin_acc = torch.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (
                (confidences > self.bin_boundaries[i])
                & (confidences <= self.bin_boundaries[i + 1])
            )
            if mask.sum() > 0:
                self.bin_acc[i] = accuracies[mask].mean()

        return self

    def predict(self, probs: Tensor) -> Tensor:
        confidences, _ = probs.max(dim=-1)
        calibrated_probs = torch.zeros_like(probs)

        for i in range(self.n_bins):
            mask = (
                (confidences > self.bin_boundaries[i])
                & (confidences <= self.bin_boundaries[i + 1])
            )
            calibrated_probs[mask] = self.bin_acc[i]

        return calibrated_probs


def compute_calibration_metrics(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    ece = ExpectedCalibrationError(n_bins=n_bins)
    mce = MaximumCalibrationError(n_bins=n_bins)

    return {
        "ECE": ece.compute(probs, labels),
        "MCE": mce.compute(probs, labels),
    }


def temperature_scale_loss(
    logits: Tensor,
    labels: Tensor,
    temperature: Tensor,
) -> Tensor:
    scaled_logits = logits / temperature
    return F.cross_entropy(scaled_logits, labels)


def find_optimal_temperature(
    logits: Tensor,
    labels: Tensor,
    lr: float = 0.01,
    max_iter: int = 50,
) -> float:
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval_fn():
        loss = temperature_scale_loss(logits, labels, temperature)
        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(eval_fn)

    return temperature.item()
