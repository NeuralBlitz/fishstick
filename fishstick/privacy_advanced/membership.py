import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from collections import defaultdict


class MembershipInferenceAttack:
    def __init__(
        self,
        target_model: nn.Module,
        attack_model: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.target_model = target_model
        self.attack_model = attack_model
        self.device = device
        self.target_model.to(device)

        self.attack_train_data: List[Tuple[torch.Tensor, int, float]] = []
        self.attack_test_data: List[Tuple[torch.Tensor, int, float]] = []

    def extract_losses(
        self, data_loader: torch.utils.data.DataLoader, is_member: bool = True
    ) -> List[Tuple[torch.Tensor, int]]:
        self.target_model.eval()
        losses = []
        labels = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.target_model(inputs)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, targets)

                for i in range(len(loss)):
                    losses.append(loss[i].item())
                    labels.append(1 if is_member else 0)

        return list(zip(losses, labels))

    def prepare_attack_dataset(
        self,
        member_loader: torch.utils.data.DataLoader,
        non_member_loader: torch.utils.data.DataLoader,
    ) -> Tuple[List[float], List[int]]:
        member_losses = self.extract_losses(member_loader, is_member=True)
        non_member_losses = self.extract_losses(non_member_loader, is_member=False)

        all_losses = [x[0] for x in member_losses] + [x[0] for x in non_member_losses]
        all_labels = [x[1] for x in member_losses] + [x[1] for x in non_member_losses]

        return all_losses, all_labels

    def attack_threshold(
        self, losses: List[float], labels: List[int], threshold: Optional[float] = None
    ) -> Dict[str, float]:
        losses = np.array(losses)
        labels = np.array(labels)

        if threshold is None:
            threshold = np.percentile(losses, 50)

        predictions = (losses < threshold).astype(int)

        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))

        accuracy = (true_positives + true_negatives) / len(labels)
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
            "attack_advantage": abs(precision - recall),
        }

    def compute_privacy_loss(
        self,
        member_loader: torch.utils.data.DataLoader,
        non_member_loader: torch.utils.data.DataLoader,
    ) -> float:
        member_losses = self.extract_losses(member_loader, is_member=True)
        non_member_losses = self.extract_losses(non_member_loader, is_member=False)

        member_losses = [x[0] for x in member_losses]
        non_member_losses = [x[0] for x in non_member_losses]

        member_mean = np.mean(member_losses)
        member_std = np.std(member_losses)
        non_member_mean = np.mean(non_member_losses)
        non_member_std = np.std(non_member_losses)

        privacy_loss = abs(member_mean - non_member_mean) / (
            member_std + non_member_std + 1e-10
        )

        return privacy_loss


class LabelOnlyAttack(MembershipInferenceAttack):
    def __init__(
        self,
        target_model: nn.Module,
        distance_metric: str = "euclidean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(target_model, None, device)
        self.distance_metric = distance_metric

    def compute_output_distance(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> float:
        self.target_model.eval()

        with torch.no_grad():
            outputs = self.target_model(inputs)

            if self.distance_metric == "euclidean":
                distances = torch.norm(outputs, dim=1)
            elif self.distance_metric == "cosine":
                normalized = torch.nn.functional.normalize(outputs, dim=1)
                distances = 1 - torch.sum(normalized * normalized, dim=1)
            else:
                distances = torch.max(outputs, dim=1)[0] - torch.min(outputs, dim=1)[0]

        return distances.mean().item()

    def attack(
        self,
        member_loader: torch.utils.data.DataLoader,
        non_member_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        member_distances = []
        for inputs, labels in member_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            dist = self.compute_output_distance(inputs, labels)
            member_distances.append(dist)

        non_member_distances = []
        for inputs, labels in non_member_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            dist = self.compute_output_distance(inputs, labels)
            non_member_distances.append(dist)

        member_mean = np.mean(member_distances)
        non_member_mean = np.mean(non_member_distances)

        all_distances = member_distances + non_member_distances
        all_labels = [1] * len(member_distances) + [0] * len(non_member_distances)

        return self.attack_threshold(all_distances, all_labels)


class ShadowModelsAttack(MembershipInferenceAttack):
    def __init__(
        self,
        target_model: nn.Module,
        shadow_model_fn: Callable[[], nn.Module],
        num_shadow_models: int = 10,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(target_model, None, device)
        self.shadow_model_fn = shadow_model_fn
        self.num_shadow_models = num_shadow_models
        self.shadow_models: List[nn.Module] = []

    def train_shadow_models(
        self,
        member_data: torch.utils.data.Dataset,
        non_member_data: torch.utils.data.Dataset,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> None:
        member_loader = torch.utils.data.DataLoader(
            member_data, batch_size=batch_size, shuffle=True
        )
        non_member_loader = torch.utils.data.DataLoader(
            non_member_data, batch_size=batch_size, shuffle=True
        )

        for i in range(self.num_shadow_models):
            shadow_model = self.shadow_model_fn().to(self.device)
            optimizer = torch.optim.SGD(shadow_model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                for inputs, targets in member_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = shadow_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            self.shadow_models.append(shadow_model)

    def attack_with_shadows(
        self, attack_loader: torch.utils.data.DataLoader
    ) -> Tuple[List[float], List[int]]:
        all_confidences = []
        all_labels = []

        for shadow_model in self.shadow_models:
            shadow_model.eval()

            with torch.no_grad():
                for inputs, targets in attack_loader:
                    inputs = inputs.to(self.device)
                    outputs = shadow_model(inputs)
                    probs = torch.softmax(outputs, dim=1)

                    max_probs, _ = torch.max(probs, dim=1)

                    for prob in max_probs:
                        all_confidences.append(prob.item())
                        all_labels.append(1)

        self.attack_test_data = list(zip(all_confidences, all_labels))

        return all_confidences, all_labels

    def compute_mia_advantage(
        self, member_confidences: List[float], non_member_confidences: List[float]
    ) -> float:
        member_confidences = np.array(member_confidences)
        non_member_confidences = np.array(non_member_confidences)

        member_mean = np.mean(member_confidences)
        non_member_mean = np.mean(non_member_confidences)

        advantage = abs(member_mean - non_member_mean)

        return advantage


class PrivacyAuditor:
    def __init__(
        self,
        model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.reference_model = reference_model
        self.device = device
        self.model.to(device)

        self.audit_results: Dict[str, Any] = {}

    def audit_membership(
        self,
        train_data: torch.utils.data.Dataset,
        test_data: torch.utils.data.Dataset,
        num_samples: int = 1000,
    ) -> Dict[str, float]:
        train_subset = torch.utils.data.Subset(
            train_data, range(min(num_samples, len(train_data)))
        )
        test_subset = torch.utils.data.Subset(
            test_data, range(min(num_samples, len(test_data)))
        )

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32)

        attack = MembershipInferenceAttack(self.model, device=self.device)

        member_losses = self._extract_losses(train_loader)
        non_member_losses = self._extract_losses(test_loader)

        combined_losses = member_losses + non_member_losses
        combined_labels = [1] * len(member_losses) + [0] * len(non_member_losses)

        results = attack.attack_threshold(combined_losses, combined_labels)

        self.audit_results["membership_inference"] = results

        return results

    def _extract_losses(self, data_loader: torch.utils.data.DataLoader) -> List[float]:
        self.model.eval()
        losses = []

        criterion = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                losses.extend(loss.cpu().numpy().tolist())

        return losses

    def audit_model_inversion(
        self, target_class: int, num_iterations: int = 100, lr: float = 0.1
    ) -> torch.Tensor:
        self.model.eval()

        dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True, device=self.device)
        optimizer = torch.optim.SGD([dummy_input], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            output = self.model(dummy_input)
            loss = -output[0, target_class]

            loss.backward()
            optimizer.step()

        self.audit_results["model_inversion"] = {
            "target_class": target_class,
            "iterations": num_iterations,
        }

        return dummy_input.detach()

    def audit_attribute_inference(
        self,
        sensitive_attribute_fn: Callable[[torch.Tensor], int],
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 100,
    ) -> Dict[str, float]:
        self.model.eval()

        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for i, (inputs, _) in enumerate(data_loader):
                if i >= num_samples:
                    break

                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                for j, pred in enumerate(predictions):
                    if j < len(inputs):
                        actual_attr = sensitive_attribute_fn(inputs[j])
                        if pred.item() == actual_attr:
                            correct_predictions += 1
                        total_samples += 1

        inference_accuracy = correct_predictions / max(total_samples, 1)

        self.audit_results["attribute_inference"] = {
            "accuracy": inference_accuracy,
            "num_samples": num_samples,
        }

        return {"accuracy": inference_accuracy}

    def generate_audit_report(self) -> Dict[str, Any]:
        return {
            "model_architecture": str(self.model.__class__.__name__),
            "audit_results": self.audit_results,
            "privacy_risk_level": self._compute_risk_level(),
        }

    def _compute_risk_level(self) -> str:
        if not self.audit_results:
            return "unknown"

        membership_results = self.audit_results.get("membership_inference", {})
        accuracy = membership_results.get("accuracy", 0.5)

        if accuracy > 0.9:
            return "high"
        elif accuracy > 0.7:
            return "medium"
        else:
            return "low"


def apply_defense(model: nn.Module, defense_type: str, **kwargs) -> nn.Module:
    if defense_type == "dropout":
        dropout_rate = kwargs.get("dropout_rate", 0.5)
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    elif defense_type == "label_smoothing":
        smoothing = kwargs.get("smoothing", 0.1)

        def new_forward(self, x):
            log_probs = torch.nn.functional.log_softmax(x, dim=-1)
            n_classes = x.size(-1)
            return (1 - smoothing) * log_probs + smoothing / n_classes

        for module in model.modules():
            if hasattr(module, "forward"):
                original_forward = module.forward
                module.forward = lambda x: new_forward(module, x)

    elif defense_type == "temperature_scaling":
        temperature = kwargs.get("temperature", 2.0)

        def scaled_forward(self, x):
            return x / temperature

        for module in model.modules():
            if hasattr(module, "forward"):
                original_forward = module.forward
                module.forward = lambda x: scaled_forward(module, x)

    elif defense_type == "adversarial_regularization":
        regularization_weight = kwargs.get("weight", 0.1)

        def adv_forward(self, x):
            return self.original_forward(x)

        for name, param in model.named_parameters():
            if "weight" in name:
                reg_loss = regularization_weight * torch.sum(param**2)
                param.register_hook(
                    lambda grad: grad + regularization_weight * 2 * param.data
                )

    return model


def compute_privacy_budget(epsilon: float, delta: float) -> Dict[str, float]:
    if epsilon < 0 or delta < 0:
        return {"status": "invalid", "epsilon": epsilon, "delta": delta}

    if epsilon <= 1:
        privacy_level = "strong"
    elif epsilon <= 10:
        privacy_level = "moderate"
    else:
        privacy_level = "weak"

    return {
        "status": "valid",
        "epsilon": epsilon,
        "delta": delta,
        "privacy_level": privacy_level,
        "recommendation": f"Privacy budget provides {privacy_level} privacy guarantees",
    }
