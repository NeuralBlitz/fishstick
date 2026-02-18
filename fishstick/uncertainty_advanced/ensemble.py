import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod
import copy
import numpy as np


class DeepEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        n_models: Optional[int] = None,
    ):
        super().__init__()
        if n_models is not None:
            self.models = nn.ModuleList(
                [copy.deepcopy(models[0]) for _ in range(n_models)]
            )
            self._init_models(models[0])
        else:
            self.models = nn.ModuleList(models)

    def _init_models(self, base_model: nn.Module) -> None:
        for model in self.models:
            for p1, p2 in zip(model.parameters(), base_model.parameters()):
                if p1.shape == p2.shape:
                    p1.data = p2.data.clone()

    def forward(self, x: Tensor) -> Tensor:
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    def predict_with_uncertainty(
        self, x: Tensor, return_individual: bool = False
    ) -> Dict[str, Any]:
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)
        std = torch.sqrt(variance)

        probs = F.softmax(mean, dim=-1)
        predictions = mean.argmax(dim=-1)

        result = {
            "mean": mean,
            "variance": variance,
            "std": std,
            "predictions": predictions,
            "probs": probs,
        }

        if return_individual:
            result["individual_outputs"] = outputs

        return result

    def predict_with_diversity(self, x: Tensor) -> Dict[str, Any]:
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)
        std = torch.sqrt(variance)

        probs_list = [F.softmax(out, dim=-1) for out in outputs]
        probs_stack = torch.stack(probs_list)

        diversity = compute_diversity_measures(probs_stack)

        return {
            "mean": mean,
            "variance": variance,
            "std": std,
            "predictions": mean.argmax(dim=-1),
            "probs": F.softmax(mean, dim=-1),
            **diversity,
        }

    @staticmethod
    def create_ensemble(
        model_fn: Callable[[], nn.Module],
        n_models: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "DeepEnsemble":
        models = [model_fn() for _ in range(n_models)]
        for i, model in enumerate(models):
            if i > 0:
                for p1, p2 in zip(model.parameters(), models[0].parameters()):
                    if p1.shape == p2.shape:
                        p1.data = p2.data.clone() + torch.randn_like(p1) * 0.01
            model.to(device)
        return DeepEnsemble(models)


class BatchEnsemble(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        n_ensembles: int = 4,
    ):
        super().__init__()
        self.base_model = base_model
        self.n_ensembles = n_ensembles

        self.alpha = nn.ParameterList(
            [
                nn.Parameter(torch.randn(base_model.fc.out_features))
                for _ in range(n_ensembles)
            ]
        )
        self.beta = nn.ParameterList(
            [
                nn.Parameter(torch.randn(base_model.fc.out_features))
                for _ in range(n_ensembles)
            ]
        )

    def forward(self, x: Tensor, ensemble_idx: int = 0) -> Tensor:
        output = self.base_model(x)
        alpha = torch.sigmoid(self.alpha[ensemble_idx])
        beta = self.beta[ensemble_idx]
        return alpha * output + beta

    def predict_with_uncertainty(self, x: Tensor) -> Dict[str, Tensor]:
        outputs = [self.forward(x, i) for i in range(self.n_ensembles)]
        outputs = torch.stack(outputs)

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return {
            "mean": mean,
            "variance": variance,
            "std": torch.sqrt(variance),
            "predictions": mean.argmax(dim=-1),
            "probs": F.softmax(mean, dim=-1),
        }


class SWAEnsemble(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        n_swa_points: int = 5,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_swa_points = n_swa_points
        self.swa_weights: Optional[Tensor] = None

    def compute_swa_weights(self, x: Tensor) -> None:
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        self.swa_weights = 1.0 / (variance + 1e-8)

    def forward(self, x: Tensor) -> Tensor:
        if self.swa_weights is not None:
            outputs = [model(x) for model in self.models]
            outputs = torch.stack(outputs)
            weighted = outputs * self.swa_weights.unsqueeze(0)
            return weighted.sum(dim=0) / self.swa_weights.sum()

        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    def predict_with_uncertainty(self, x: Tensor) -> Dict[str, Tensor]:
        self.compute_swa_weights(x)
        mean = self.forward(x)

        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)
        variance = outputs.var(dim=0)

        return {
            "mean": mean,
            "variance": variance,
            "std": torch.sqrt(variance),
            "predictions": mean.argmax(dim=-1),
            "probs": F.softmax(mean, dim=-1),
        }


class WeightAveragedEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], method: str = "swa"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method

        if method == "swa":
            self._init_swa()
        elif method == "ewa":
            self._init_ewa()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _init_swa(self) -> None:
        self.swa_weights: List[Tensor] = []
        for model in self.models:
            state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            self.swa_weights.append(state_dict)

    def _init_ewa(self) -> None:
        self.ema_weights: Dict[str, Tensor] = {}
        self.decay = 0.99

    def forward(self, x: Tensor) -> Tensor:
        if self.method == "swa":
            return self._forward_swa(x)
        else:
            return self._forward_ewa(x)

    def _forward_swa(self, x: Tensor) -> Tensor:
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        return torch.stack(outputs).mean(dim=0)

    def _forward_ewa(self, x: Tensor) -> Tensor:
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        return torch.stack(outputs).mean(dim=0)

    def predict_with_uncertainty(self, x: Tensor) -> Dict[str, Tensor]:
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)

        return {
            "mean": mean,
            "variance": variance,
            "std": torch.sqrt(variance),
            "predictions": mean.argmax(dim=-1),
            "probs": F.softmax(mean, dim=-1),
        }


def compute_diversity_measures(
    ensemble_probs: Tensor,
    epsilon: float = 1e-10,
) -> Dict[str, Tensor]:
    n_ensembles = ensemble_probs.shape[0]

    entropy_individual = -(ensemble_probs * torch.log(ensemble_probs + epsilon)).sum(
        dim=-1
    )
    avg_entropy = entropy_individual.mean(dim=0)

    mean_probs = ensemble_probs.mean(dim=0)
    entropy_avg = -(mean_probs * torch.log(mean_probs + epsilon)).sum(dim=-1)

    mutual_info = avg_entropy - entropy_avg

    disagreement = (
        (ensemble_probs.argmax(dim=-1) != mean_probs.argmax(dim=-1)).float().mean()
    )

    cos_sim_matrix = torch.zeros(n_ensembles, n_ensembles)
    for i in range(n_ensembles):
        for j in range(n_ensembles):
            if i != j:
                cos_sim_matrix[i, j] = F.cosine_similarity(
                    ensemble_probs[i], ensemble_probs[j], dim=-1
                ).mean()

    avg_cos_sim = (cos_sim_matrix.sum() - n_ensembles) / (
        n_ensembles * (n_ensembles - 1)
    )

    return {
        "entropy": avg_entropy,
        "mutual_information": mutual_info,
        "disagreement": disagreement,
        "avg_cosine_similarity": avg_cos_sim,
    }


def compute_ensemble_variance(
    ensemble_outputs: Tensor,
    reduction: str = "mean",
) -> Tensor:
    variance = ensemble_outputs.var(dim=0)

    if reduction == "mean":
        return variance.mean()
    elif reduction == "sum":
        return variance.sum()
    elif reduction == "none":
        return variance
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_diversity_loss(
    ensemble_probs: Tensor,
    target: Tensor,
    alpha: float = 0.1,
) -> Tensor:
    n_ensembles = ensemble_probs.shape[0]

    mean_probs = ensemble_probs.mean(dim=0)
    ce_loss = F.cross_entropy(torch.log(mean_probs + 1e-10), target)

    diversity = compute_diversity_measures(ensemble_probs)
    diversity_loss = -diversity["mutual_information"]

    return ce_loss + alpha * diversity_loss


class DiversityRegularizer(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, ensemble_outputs: Tensor, target: Tensor) -> Tensor:
        probs = F.softmax(ensemble_outputs, dim=-1)
        return compute_diversity_loss(probs, target, self.alpha)


class EnsembleCalibrator:
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    def compute_reliability(
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

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                bin_accuracies.append(accuracies[in_bin].float().mean())
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())

        if len(bin_accuracies) == 0:
            return {"reliability": None, "n_bins": 0}

        bin_accuracies = torch.stack(bin_accuracies)
        bin_confidences = torch.stack(bin_confidences)
        bin_counts = torch.stack(bin_counts)

        return {
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "reliability": (bin_confidences - bin_accuracies).abs().mean(),
        }


class FastDepthEnsemble(DeepEnsemble):
    def __init__(self, models: List[nn.Module]):
        super().__init__(models)

    def predict_depth_uncertainty(self, x: Tensor) -> Dict[str, Tensor]:
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs)

        mean = outputs.mean(dim=0)
        variance = outputs.var(dim=0)
        std = torch.sqrt(variance)

        aleatoric = variance.mean(dim=0)
        epistemic = (outputs.var(dim=0).mean(dim=0) - aleatoric).clamp(min=0)

        return {
            "depth_map": mean,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": variance.mean(dim=0),
            "std": std.mean(dim=0),
        }
