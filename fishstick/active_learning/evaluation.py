import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
from typing import Optional, List, Callable, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ActiveLearningState:
    labeled_indices: List[int]
    unlabeled_indices: List[int]
    queried_count: int
    round_number: int
    performance_history: List[float]


class ActiveLearningEvaluator:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        initial_labeled_size: int = 10,
        query_size: int = 10,
        n_rounds: int = 10,
        batch_size: int = 32,
        n_epochs: int = 10,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.initial_labeled_size = initial_labeled_size
        self.query_size = query_size
        self.n_rounds = n_rounds
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        self.device = device

        self.model.to(self.device)

        self.state: Optional[ActiveLearningState] = None

    def initialize(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

        n_samples = len(self.train_dataset)
        all_indices = np.random.permutation(n_samples).tolist()

        labeled_indices = all_indices[: self.initial_labeled_size]
        unlabeled_indices = all_indices[self.initial_labeled_size :]

        self.state = ActiveLearningState(
            labeled_indices=labeled_indices,
            unlabeled_indices=unlabeled_indices,
            queried_count=self.initial_labeled_size,
            round_number=0,
            performance_history=[],
        )

    def train_model(self, labeled_indices: List[int]) -> float:
        labeled_subset = Subset(self.train_dataset, labeled_indices)
        train_loader = DataLoader(
            labeled_subset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        total_loss = 0.0

        for epoch in range(self.n_epochs):
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / (len(train_loader) * self.n_epochs)

    def evaluate(self) -> float:
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def run(
        self, query_strategy: Callable, features_extractor: Optional[nn.Module] = None
    ) -> Dict:
        if self.state is None:
            self.initialize()

        learning_curve = []

        initial_accuracy = self.evaluate()
        learning_curve.append(initial_accuracy)

        for round_num in range(self.n_rounds):
            if len(self.state.unlabeled_indices) == 0:
                break

            labeled_features = self._get_labeled_features(features_extractor)
            unlabeled_features = self._get_unlabeled_features(features_extractor)

            query_indices = query_strategy(
                unlabeled_features=unlabeled_features,
                n_query=self.query_size,
                labeled_features=labeled_features,
            )

            new_labeled_indices = [
                self.state.unlabeled_indices[i] for i in query_indices.tolist()
            ]

            self.state.labeled_indices.extend(new_labeled_indices)
            self.state.unlabeled_indices = [
                idx
                for i, idx in enumerate(self.state.unlabeled_indices)
                if i not in query_indices.tolist()
            ]

            self.state.round_number = round_num + 1
            self.state.queried_count += len(new_labeled_indices)

            self.train_model(self.state.labeled_indices)

            accuracy = self.evaluate()
            learning_curve.append(accuracy)
            self.state.performance_history.append(accuracy)

        return {
            "learning_curve": learning_curve,
            "labeled_count": len(self.state.labeled_indices),
            "total_rounds": self.state.round_number,
            "final_accuracy": learning_curve[-1] if learning_curve else 0.0,
        }

    def _get_labeled_features(
        self, feature_extractor: Optional[nn.Module]
    ) -> Optional[torch.Tensor]:
        if feature_extractor is None:
            return None

        labeled_subset = Subset(self.train_dataset, self.state.labeled_indices)
        loader = DataLoader(labeled_subset, batch_size=self.batch_size)

        features = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                feats = feature_extractor(inputs.to(self.device))
                features.append(feats.cpu())

        return torch.cat(features, dim=0)

    def _get_unlabeled_features(
        self, feature_extractor: Optional[nn.Module]
    ) -> torch.Tensor:
        if feature_extractor is None:
            unlabeled_subset = Subset(self.train_dataset, self.state.unlabeled_indices)
            loader = DataLoader(unlabeled_subset, batch_size=self.batch_size)

            inputs = []
            for batch in loader:
                batch_inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                inputs.append(batch_inputs)

            return torch.cat(inputs, dim=0)

        unlabeled_subset = Subset(self.train_dataset, self.state.unlabeled_indices)
        loader = DataLoader(unlabeled_subset, batch_size=self.batch_size)

        features = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                feats = feature_extractor(inputs.to(self.device))
                features.append(feats.cpu())

        return torch.cat(features, dim=0)


def simulate_labeling(
    model: nn.Module,
    query_strategy: Callable,
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    initial_size: int = 10,
    query_size: int = 10,
    n_rounds: int = 10,
    batch_size: int = 32,
    n_epochs: int = 10,
    device: str = "cpu",
) -> Tuple[List[float], List[int]]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_samples = X_pool.size(0)
    labeled_indices = np.random.choice(n_samples, initial_size, replace=False)
    unlabeled_indices = np.array(
        [i for i in range(n_samples) if i not in labeled_indices]
    )

    learning_curve = []
    label_counts = []

    for round_num in range(n_rounds):
        train_dataset = TensorDataset(X_pool[labeled_indices], y_pool[labeled_indices])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(n_epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        learning_curve.append(accuracy)
        label_counts.append(len(labeled_indices))

        if len(unlabeled_indices) == 0:
            break

        query_indices = (
            query_strategy(X_pool[unlabeled_indices].to(device), query_size)
            .cpu()
            .numpy()
        )

        new_indices = unlabeled_indices[query_indices]
        labeled_indices = np.concatenate([labeled_indices, new_indices])
        unlabeled_indices = np.array(
            [i for i in range(n_samples) if i not in labeled_indices]
        )

    return learning_curve, label_counts


def compute_learning_curve(
    results: List[Dict], metric: str = "accuracy"
) -> Dict[str, List[float]]:
    curves = {}
    for name, result in results.items():
        if metric in result:
            curves[name] = result[metric]
        elif "learning_curve" in result:
            curves[name] = result["learning_curve"]

    return curves


def compute_label_efficiency(
    learning_curves: Dict[str, List[float]],
    baseline_performance: float = 0.9,
    budget_percentile: float = 0.1,
) -> Dict[str, float]:
    efficiency_scores = {}

    for name, curve in learning_curves.items():
        curve = np.array(curve)

        target_perf = (
            baseline_performance * max(curve)
            if baseline_performance < 1.0
            else baseline_performance
        )

        above_target = np.where(curve >= target_perf)[0]

        if len(above_target) > 0:
            efficiency_scores[name] = float(above_target[0] + 1) / len(curve)
        else:
            efficiency_scores[name] = 1.0

    return efficiency_scores


def compute_area_under_curve(learning_curve: List[float]) -> float:
    curve = np.array(learning_curve)
    x = np.arange(len(curve))
    auc = np.trapz(curve, x)
    return float(auc)


def compare_strategies(
    strategies: Dict[str, Callable],
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    initial_size: int = 10,
    query_size: int = 10,
    n_rounds: int = 10,
    n_trials: int = 3,
    **train_kwargs,
) -> Dict[str, Dict]:
    results = {}

    for name, strategy in strategies.items():
        trial_curves = []

        for trial in range(n_trials):
            model = train_kwargs.get("model_class")(
                **train_kwargs.get("model_kwargs", {})
            )

            def create_strategy(features, n):
                return strategy(model, features, n)

            curve, counts = simulate_labeling(
                model=model,
                query_strategy=create_strategy,
                X_pool=X_pool,
                y_pool=y_pool,
                X_test=X_test,
                y_test=y_test,
                initial_size=initial_size,
                query_size=query_size,
                n_rounds=n_rounds,
                batch_size=train_kwargs.get("batch_size", 32),
                n_epochs=train_kwargs.get("n_epochs", 10),
                device=train_kwargs.get("device", "cpu"),
            )

            trial_curves.append(curve)

        avg_curve = np.mean(trial_curves, axis=0).tolist()

        results[name] = {
            "learning_curve": avg_curve,
            "final_accuracy": avg_curve[-1] if avg_curve else 0.0,
            "auc": compute_area_under_curve(avg_curve),
            "std": np.std(trial_curves, axis=0).tolist(),
        }

    return results
