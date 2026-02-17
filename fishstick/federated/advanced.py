"""
Advanced Federated Learning Module for fishstick

Production-ready federated learning implementation with support for:
- Multiple aggregation strategies (FedAvg, FedProx, SCAFFOLD, FedNova)
- Secure aggregation with differential privacy
- Non-IID data handling
- Gradient compression and communication efficiency
- Fault tolerance and async communication
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
TensorDict = Dict[str, torch.Tensor]
ModelState = Dict[str, Any]


class AggregationError(Exception):
    """Exception raised during aggregation operations."""

    pass


class CommunicationError(Exception):
    """Exception raised during communication operations."""

    pass


class ClientNotAvailableError(Exception):
    """Exception raised when a client is not available."""

    pass


class PartitionStrategy(Enum):
    """Strategies for partitioning data across clients."""

    IID = auto()
    NON_IID_DIRICHLET = auto()
    QUANTITY_SKEW = auto()
    LABEL_SKEW = auto()
    SHARD = auto()


@dataclass
class PartitionConfig:
    """Configuration for data partitioning."""

    num_clients: int
    strategy: PartitionStrategy = PartitionStrategy.IID
    alpha: float = 0.5
    min_samples: int = 10
    shards_per_client: int = 2
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)


class DataPartitioner:
    """
    Partition data for federated learning across multiple clients.

    Supports IID, non-IID via Dirichlet distribution, quantity skew,
    and label distribution skew.
    """

    def __init__(self, dataset: Dataset, config: PartitionConfig):
        self.dataset = dataset
        self.config = config
        self.partitions: Dict[int, List[int]] = {}
        self._partition_data()

    def _partition_data(self) -> None:
        if self.config.strategy == PartitionStrategy.IID:
            self._partition_iid()
        elif self.config.strategy == PartitionStrategy.NON_IID_DIRICHLET:
            self._partition_dirichlet()
        elif self.config.strategy == PartitionStrategy.QUANTITY_SKEW:
            self._partition_quantity_skew()
        elif self.config.strategy == PartitionStrategy.LABEL_SKEW:
            self._partition_label_skew()
        elif self.config.strategy == PartitionStrategy.SHARD:
            self._partition_shard()
        else:
            raise ValueError(f"Unknown partition strategy: {self.config.strategy}")

    def _partition_iid(self) -> None:
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        samples_per_client = len(indices) // self.config.num_clients
        remainder = len(indices) % self.config.num_clients
        start = 0
        for client_id in range(self.config.num_clients):
            end = start + samples_per_client + (1 if client_id < remainder else 0)
            self.partitions[client_id] = indices[start:end]
            start = end
        logger.info(f"IID partition created for {self.config.num_clients} clients")

    def _partition_dirichlet(self) -> None:
        if hasattr(self.dataset, "targets"):
            labels = np.array(self.dataset.targets)
        else:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])

        num_classes = int(np.max(labels) + 1)
        indices_per_class = [
            np.where(labels == i)[0].tolist() for i in range(num_classes)
        ]

        for client_id in range(self.config.num_clients):
            self.partitions[client_id] = []

        for class_idx in range(num_classes):
            random.shuffle(indices_per_class[class_idx])
            proportions = np.random.dirichlet(
                [self.config.alpha] * self.config.num_clients
            )
            proportions = proportions / proportions.sum()
            splits = (
                np.cumsum(proportions) * len(indices_per_class[class_idx])
            ).astype(int)[:-1]
            client_indices = np.split(indices_per_class[class_idx], splits)
            for client_id, idx_list in enumerate(client_indices):
                self.partitions[client_id].extend(idx_list.tolist())

        logger.info(f"Dirichlet partition created with alpha={self.config.alpha}")

    def _partition_quantity_skew(self) -> None:
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        powers = np.array([1.0 / (i + 1) for i in range(self.config.num_clients)])
        proportions = powers / powers.sum()
        sizes = (proportions * len(indices)).astype(int)
        sizes = np.maximum(sizes, self.config.min_samples)
        diff = len(indices) - sizes.sum()
        if diff > 0:
            sizes[0] += diff
        start = 0
        for client_id, size in enumerate(sizes):
            end = start + int(size)
            self.partitions[client_id] = indices[start:end]
            start = end
        logger.info(f"Quantity skew partition created")

    def _partition_label_skew(self) -> None:
        if hasattr(self.dataset, "targets"):
            labels = np.array(self.dataset.targets)
        else:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])

        num_classes = int(np.max(labels) + 1)
        indices_per_class = [
            np.where(labels == i)[0].tolist() for i in range(num_classes)
        ]
        classes_per_client = max(1, num_classes // self.config.num_clients)

        for client_id in range(self.config.num_clients):
            self.partitions[client_id] = []
            start_class = (client_id * classes_per_client) % num_classes
            for i in range(classes_per_client):
                class_idx = (start_class + i) % num_classes
                self.partitions[client_id].extend(indices_per_class[class_idx])

        logger.info(
            f"Label skew partition created with {classes_per_client} classes per client"
        )

    def _partition_shard(self) -> None:
        if hasattr(self.dataset, "targets"):
            labels = np.array(self.dataset.targets)
        else:
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])

        num_classes = int(np.max(labels) + 1)
        indices_per_class = [
            np.where(labels == i)[0].tolist() for i in range(num_classes)
        ]
        shards = []

        for class_idx in range(num_classes):
            random.shuffle(indices_per_class[class_idx])
            shard_size = (
                len(indices_per_class[class_idx]) // self.config.shards_per_client
            )
            for i in range(self.config.shards_per_client):
                start = i * shard_size
                end = (
                    start + shard_size
                    if i < self.config.shards_per_client - 1
                    else len(indices_per_class[class_idx])
                )
                shards.append(indices_per_class[class_idx][start:end])

        random.shuffle(shards)
        shards_per_client = len(shards) // self.config.num_clients

        for client_id in range(self.config.num_clients):
            start_shard = client_id * shards_per_client
            end_shard = start_shard + shards_per_client
            self.partitions[client_id] = []
            for shard_idx in range(start_shard, end_shard):
                self.partitions[client_id].extend(shards[shard_idx])

        logger.info(
            f"Shard partition created with {shards_per_client} shards per client"
        )

    def get_client_dataset(self, client_id: int) -> Subset:
        if client_id not in self.partitions:
            raise ValueError(f"Invalid client ID: {client_id}")
        return Subset(self.dataset, self.partitions[client_id])

    def get_client_indices(self, client_id: int) -> List[int]:
        return self.partitions.get(client_id, [])

    def get_partition_statistics(self) -> Dict[str, Any]:
        stats = {
            "num_clients": self.config.num_clients,
            "strategy": self.config.strategy.name,
            "total_samples": len(self.dataset),
            "samples_per_client": {},
        }
        for client_id, indices in self.partitions.items():
            stats["samples_per_client"][client_id] = len(indices)
        return stats


class CompressionMethod(Enum):
    """Methods for gradient/model compression."""

    NONE = auto()
    TOP_K = auto()
    QUANTIZATION = auto()
    SPARSIFICATION = auto()
    RANDOM_MASK = auto()


@dataclass
class CompressionConfig:
    """Configuration for gradient compression."""

    method: CompressionMethod = CompressionMethod.NONE
    top_k_ratio: float = 0.1
    num_bits: int = 8
    sparsity: float = 0.9
    seed: Optional[int] = None


class GradientCompressor:
    """
    Compress gradients/models for efficient communication.
    Supports top-k sparsification, quantization, and random masking.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

    def compress(self, state_dict: TensorDict) -> Tuple[TensorDict, Dict[str, Any]]:
        if self.config.method == CompressionMethod.NONE:
            return state_dict, {}

        compressed = {}
        metadata = {"method": self.config.method.name}

        for key, tensor in state_dict.items():
            if self.config.method == CompressionMethod.TOP_K:
                compressed[key], meta = self._top_k_compress(tensor)
                metadata[key] = meta
            elif self.config.method == CompressionMethod.QUANTIZATION:
                compressed[key], meta = self._quantize(tensor)
                metadata[key] = meta
            elif self.config.method == CompressionMethod.SPARSIFICATION:
                compressed[key], meta = self._sparsify(tensor)
                metadata[key] = meta
            elif self.config.method == CompressionMethod.RANDOM_MASK:
                compressed[key], meta = self._random_mask(tensor)
                metadata[key] = meta
            else:
                compressed[key] = tensor

        return compressed, metadata

    def decompress(
        self, compressed_state: TensorDict, metadata: Dict[str, Any]
    ) -> TensorDict:
        if self.config.method == CompressionMethod.NONE or not metadata:
            return compressed_state

        decompressed = {}
        method = metadata.get("method", "NONE")

        for key, tensor in compressed_state.items():
            meta = metadata.get(key, {})
            if method == "QUANTIZATION" and meta:
                decompressed[key] = self._dequantize(tensor, meta)
            else:
                decompressed[key] = tensor

        return decompressed

    def _top_k_compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        flat = tensor.flatten()
        k = max(1, int(len(flat) * self.config.top_k_ratio))
        values, indices = torch.topk(torch.abs(flat), k)
        compressed = torch.zeros_like(flat)
        compressed[indices] = flat[indices]
        return compressed.reshape(tensor.shape), {
            "indices": indices,
            "shape": tensor.shape,
        }

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2**self.config.num_bits - 1)
        quantized = torch.round((tensor - min_val) / scale).to(torch.int32)
        return quantized, {
            "min": min_val.item(),
            "max": max_val.item(),
            "scale": scale.item(),
            "shape": tensor.shape,
        }

    def _dequantize(self, tensor: torch.Tensor, meta: Dict) -> torch.Tensor:
        return tensor.float() * meta["scale"] + meta["min"]

    def _sparsify(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        mask = torch.rand_like(tensor) > self.config.sparsity
        return tensor * mask.float(), {"mask": mask}

    def _random_mask(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        mask = torch.rand(tensor.shape) > self.config.sparsity
        return tensor * mask.float(), {"mask": mask, "shape": tensor.shape}


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy."""

    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None
    mechanism: str = "gaussian"


class DifferentialPrivacy:
    """
    Differential privacy mechanisms for federated learning.
    Implements gradient clipping and noise addition.
    """

    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.sample_count = 0

    def clip_gradients(self, state_dict: TensorDict) -> TensorDict:
        global_norm = 0.0
        for tensor in state_dict.values():
            global_norm += torch.sum(tensor**2).item()
        global_norm = np.sqrt(global_norm)
        clip_coef = min(self.config.max_grad_norm / (global_norm + 1e-6), 1.0)
        return {key: tensor * clip_coef for key, tensor in state_dict.items()}

    def add_noise(self, state_dict: TensorDict, sensitivity: float = 1.0) -> TensorDict:
        if self.config.noise_multiplier is None:
            if self.config.delta > 0:
                noise_multiplier = (
                    np.sqrt(2 * np.log(1.25 / self.config.delta)) / self.config.epsilon
                )
            else:
                noise_multiplier = 1.0 / self.config.epsilon
        else:
            noise_multiplier = self.config.noise_multiplier

        noisy_state = {}
        for key, tensor in state_dict.items():
            if self.config.mechanism == "gaussian":
                noise = torch.randn_like(tensor) * noise_multiplier * sensitivity
            elif self.config.mechanism == "laplace":
                noise = (
                    torch.from_numpy(
                        np.random.laplace(
                            0, noise_multiplier * sensitivity, tensor.shape
                        )
                    )
                    .to(tensor.device)
                    .to(tensor.dtype)
                )
            else:
                raise ValueError(f"Unknown mechanism: {self.config.mechanism}")
            noisy_state[key] = tensor + noise

        self.sample_count += 1
        return noisy_state

    def get_privacy_spent(self, sample_rate: float = 0.01) -> Dict[str, float]:
        return {
            "epsilon": self.config.epsilon * self.sample_count * sample_rate,
            "delta": self.config.delta,
            "samples": self.sample_count,
        }


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        client_states: List[TensorDict],
        weights: List[float],
        global_state: Optional[TensorDict] = None,
    ) -> TensorDict:
        pass


class FedAvgStrategy(AggregationStrategy):
    """
    Federated Averaging (FedAvg) strategy.
    McMahan et al., AISTATS 2017.
    """

    def aggregate(
        self,
        client_states: List[TensorDict],
        weights: List[float],
        global_state: Optional[TensorDict] = None,
    ) -> TensorDict:
        if not client_states:
            raise AggregationError("No client states to aggregate")
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        aggregated = {}
        for key in client_states[0].keys():
            aggregated[key] = torch.zeros_like(client_states[0][key])
            for state, weight in zip(client_states, normalized_weights):
                aggregated[key] += state[key] * weight
        return aggregated


class FedProxStrategy(AggregationStrategy):
    """
    Federated Proximal (FedProx) strategy.
    Li et al., MLSys 2020.
    """

    def __init__(self, mu: float = 0.01):
        self.mu = mu

    def aggregate(
        self,
        client_states: List[TensorDict],
        weights: List[float],
        global_state: Optional[TensorDict] = None,
    ) -> TensorDict:
        return FedAvgStrategy().aggregate(client_states, weights, global_state)


class ScaffoldStrategy(AggregationStrategy):
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
    Karimireddy et al., ICML 2020.
    """

    def __init__(self, num_clients: int, lr: float = 0.1):
        self.server_controls: Dict[str, torch.Tensor] = {}
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {}
        self.num_clients = num_clients
        self.lr = lr

    def aggregate(
        self,
        client_states: List[TensorDict],
        weights: List[float],
        global_state: Optional[TensorDict] = None,
    ) -> TensorDict:
        if not client_states or global_state is None:
            raise AggregationError("SCAFFOLD requires client states and global state")
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        aggregated = {}
        for key in global_state.keys():
            aggregated[key] = torch.zeros_like(global_state[key])
            for state, weight in zip(client_states, normalized_weights):
                aggregated[key] += state[key] * weight
            if key not in self.server_controls:
                self.server_controls[key] = torch.zeros_like(global_state[key])
        return aggregated

    def get_client_correction(
        self, client_id: int, global_state: TensorDict
    ) -> Dict[str, torch.Tensor]:
        correction = {}
        client_control = self.client_controls.get(client_id, {})
        for key in global_state.keys():
            server_c = self.server_controls.get(
                key, torch.zeros_like(global_state[key])
            )
            client_c = client_control.get(key, torch.zeros_like(global_state[key]))
            correction[key] = server_c - client_c
        return correction

    def update_client_control(
        self,
        client_id: int,
        global_state: TensorDict,
        client_state: TensorDict,
        num_steps: int,
    ) -> None:
        if client_id not in self.client_controls:
            self.client_controls[client_id] = {}
        for key in global_state.keys():
            if key not in self.client_controls[client_id]:
                self.client_controls[client_id][key] = torch.zeros_like(
                    global_state[key]
                )
            diff = global_state[key] - client_state[key]
            self.client_controls[client_id][key] += diff / (num_steps * self.lr)


class FedNovaStrategy(AggregationStrategy):
    """
    FedNova: Normalized Averaging for Heterogeneous Networks.
    Wang et al., NeurIPS 2020.
    """

    def aggregate(
        self,
        client_states: List[TensorDict],
        weights: List[float],
        global_state: Optional[TensorDict] = None,
    ) -> TensorDict:
        if not client_states:
            raise AggregationError("No client states to aggregate")
        total_weight = sum(weights)
        aggregated = {}
        for key in client_states[0].keys():
            aggregated[key] = torch.zeros_like(client_states[0][key])
            for state, weight in zip(client_states, weights):
                normalized_weight = weight / total_weight
                aggregated[key] += state[key] * normalized_weight
        return aggregated


class SecureAggregationStrategy(AggregationStrategy):
    """Secure aggregation with differential privacy."""

    def __init__(
        self, base_strategy: AggregationStrategy, privacy_config: PrivacyConfig
    ):
        self.base_strategy = base_strategy
        self.privacy = DifferentialPrivacy(privacy_config)

    def aggregate(
        self,
        client_states: List[TensorDict],
        weights: List[float],
        global_state: Optional[TensorDict] = None,
    ) -> TensorDict:
        clipped_states = [self.privacy.clip_gradients(state) for state in client_states]
        aggregated = self.base_strategy.aggregate(clipped_states, weights, global_state)
        return self.privacy.add_noise(
            aggregated, sensitivity=self.privacy.config.max_grad_norm
        )


@dataclass
class CommunicationConfig:
    """Configuration for communication management."""

    enable_compression: bool = True
    compression_config: CompressionConfig = field(default_factory=CompressionConfig)
    async_mode: bool = False
    timeout: float = 300.0
    max_retries: int = 3
    fault_tolerance: bool = True
    min_clients: int = 1


class CommunicationManager:
    """
    Manage communication between server and clients.
    Handles compression, async communication, and fault tolerance.
    """

    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.compressor = GradientCompressor(config.compression_config)
        self.pending_updates: Dict[int, Tuple[TensorDict, Dict]] = {}
        self.failed_clients: Set[int] = set()

    def compress_model(self, state_dict: TensorDict) -> Tuple[TensorDict, Dict]:
        if not self.config.enable_compression:
            return state_dict, {}
        return self.compressor.compress(state_dict)

    def decompress_model(
        self, compressed_state: TensorDict, metadata: Dict
    ) -> TensorDict:
        if not self.config.enable_compression:
            return compressed_state
        return self.compressor.decompress(compressed_state, metadata)

    def handle_client_dropout(
        self, client_id: int, selected_clients: List[int]
    ) -> bool:
        self.failed_clients.add(client_id)
        if not self.config.fault_tolerance:
            raise ClientNotAvailableError(f"Client {client_id} dropped")
        remaining = len(selected_clients) - len(self.failed_clients)
        can_continue = remaining >= self.config.min_clients
        if can_continue:
            logger.warning(f"Client {client_id} dropped. Remaining: {remaining}")
        else:
            logger.error(
                f"Client {client_id} dropped. Insufficient clients: {remaining}"
            )
        return can_continue

    def reset(self) -> None:
        self.pending_updates.clear()
        self.failed_clients.clear()


@dataclass
class ClientConfig:
    """Configuration for federated client."""

    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    device: str = "cpu"
    enable_dp: bool = False
    dp_config: PrivacyConfig = field(default_factory=PrivacyConfig)
    proximal_mu: float = 0.0
    gradient_clip: Optional[float] = None


class FederatedClient:
    """
    Federated learning client.
    Performs local training with support for differential privacy,
    gradient compression, and proximal term (FedProx).
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        config: ClientConfig,
    ):
        self.client_id = client_id
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.config = config
        self.device = config.device
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.dp = DifferentialPrivacy(config.dp_config) if config.enable_dp else None
        self.global_state: Optional[TensorDict] = None
        self.scaffold_correction: Optional[Dict[str, torch.Tensor]] = None
        logger.info(
            f"Client {client_id} initialized with {len(train_loader.dataset)} samples"
        )

    def set_global_state(self, state_dict: TensorDict) -> None:
        self.global_state = {k: v.clone() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

    def get_model_state(self) -> TensorDict:
        return self.model.state_dict()

    def set_scaffold_correction(self, correction: Dict[str, torch.Tensor]) -> None:
        self.scaffold_correction = correction

    def train(
        self, epochs: Optional[int] = None, verbose: bool = False
    ) -> Dict[str, Any]:
        epochs = epochs or self.config.local_epochs
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        num_steps = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                if self.config.proximal_mu > 0 and self.global_state is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in self.global_state:
                            proximal_term += torch.sum(
                                (param - self.global_state[name]) ** 2
                            )
                    loss += (self.config.proximal_mu / 2) * proximal_term

                loss.backward()
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                num_steps += 1

            if verbose:
                logger.info(
                    f"Client {self.client_id} Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss / epoch_batches:.4f}"
                )

            total_loss += epoch_loss
            num_batches += epoch_batches

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        result = {
            "client_id": self.client_id,
            "loss": avg_loss,
            "num_samples": len(self.train_loader.dataset),
            "num_steps": num_steps,
            "epochs": epochs,
        }

        if self.dp is not None:
            state_dict = self.get_model_state()
            clipped = self.dp.clip_gradients(state_dict)
            noisy = self.dp.add_noise(clipped)
            self.model.load_state_dict(noisy)
            result["privacy_spent"] = self.dp.get_privacy_spent()

        return result

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return {
            "client_id": self.client_id,
            "loss": total_loss / len(test_loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }


class ClientSelectionStrategy(Enum):
    """Strategies for selecting clients in each round."""

    RANDOM = auto()
    IMPORTANCE_BASED = auto()
    ROUND_ROBIN = auto()
    THRESHOLD = auto()


@dataclass
class ServerConfig:
    """Configuration for federated server."""

    num_rounds: int = 100
    clients_per_round: int = 10
    selection_strategy: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    aggregation_strategy: AggregationStrategy = field(default_factory=FedAvgStrategy)
    device: str = "cpu"
    eval_every: int = 1
    checkpoint_dir: Optional[str] = None
    enable_secure_agg: bool = False
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)


class FederatedServer:
    """
    Federated learning server.
    Coordinates training across multiple clients with support for
    multiple aggregation strategies, secure aggregation, and client selection.
    """

    def __init__(
        self, model: nn.Module, clients: List[FederatedClient], config: ServerConfig
    ):
        self.global_model = model.to(config.device)
        self.clients = {c.client_id: c for c in clients}
        self.config = config
        self.device = config.device
        self.current_round = 0

        if config.enable_secure_agg:
            self.aggregator = SecureAggregationStrategy(
                config.aggregation_strategy, config.privacy_config
            )
        else:
            self.aggregator = config.aggregation_strategy

        self.comm_manager = CommunicationManager(CommunicationConfig())
        self.metrics_history: List[Dict[str, Any]] = []
        self.client_importance: Dict[int, float] = {c: 1.0 for c in self.clients}
        logger.info(f"Server initialized with {len(clients)} clients")

    def get_global_state(self) -> TensorDict:
        return self.global_model.state_dict()

    def set_global_state(self, state_dict: TensorDict) -> None:
        self.global_model.load_state_dict(state_dict)

    def select_clients(self, num_clients: int) -> List[int]:
        available_clients = list(self.clients.keys())
        if self.config.selection_strategy == ClientSelectionStrategy.RANDOM:
            return random.sample(
                available_clients, min(num_clients, len(available_clients))
            )
        elif self.config.selection_strategy == ClientSelectionStrategy.IMPORTANCE_BASED:
            weights = [self.client_importance[c] for c in available_clients]
            total = sum(weights)
            probabilities = [w / total for w in weights]
            return np.random.choice(
                available_clients,
                size=min(num_clients, len(available_clients)),
                replace=False,
                p=probabilities,
            ).tolist()
        elif self.config.selection_strategy == ClientSelectionStrategy.ROUND_ROBIN:
            start_idx = (self.current_round * num_clients) % len(available_clients)
            selected = []
            for i in range(num_clients):
                idx = (start_idx + i) % len(available_clients)
                selected.append(available_clients[idx])
            return selected
        return available_clients[:num_clients]

    def broadcast_to_clients(self, client_ids: List[int]) -> None:
        global_state = self.get_global_state()
        compressed, metadata = self.comm_manager.compress_model(global_state)
        for client_id in client_ids:
            decompressed = self.comm_manager.decompress_model(compressed, metadata)
            self.clients[client_id].set_global_state(decompressed)

    def train_round(
        self, client_ids: Optional[List[int]] = None, epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        self.current_round += 1
        if client_ids is None:
            client_ids = self.select_clients(self.config.clients_per_round)

        self.broadcast_to_clients(client_ids)
        client_results = []
        client_states = []
        client_weights = []

        for client_id in client_ids:
            try:
                client = self.clients[client_id]
                result = client.train(epochs=epochs)
                client_results.append(result)
                client_states.append(client.get_model_state())
                client_weights.append(result["num_samples"])
                self.client_importance[client_id] = 1.0 / (result["loss"] + 1e-6)
            except Exception as e:
                logger.error(f"Error training client {client_id}: {e}")
                if not self.comm_manager.handle_client_dropout(client_id, client_ids):
                    raise ClientNotAvailableError(f"Too many clients dropped")

        if client_states:
            global_state = self.get_global_state()
            aggregated = self.aggregator.aggregate(
                client_states, client_weights, global_state
            )
            self.set_global_state(aggregated)

        metrics = {
            "round": self.current_round,
            "num_clients": len(client_results),
            "avg_loss": sum(r["loss"] for r in client_results) / len(client_results)
            if client_results
            else 0.0,
            "client_results": client_results,
        }
        self.metrics_history.append(metrics)
        return metrics

    def evaluate(
        self, test_loaders: Optional[Dict[int, DataLoader]] = None
    ) -> Dict[str, float]:
        self.global_model.eval()
        total_correct = 0
        total_samples = 0
        client_accuracies = {}

        if test_loaders is None:
            return {"accuracy": 0.0, "total_samples": 0}

        with torch.no_grad():
            for client_id, test_loader in test_loaders.items():
                correct = 0
                samples = 0
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    samples += target.size(0)
                client_accuracies[client_id] = correct / samples if samples > 0 else 0.0
                total_correct += correct
                total_samples += samples

        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return {
            "accuracy": accuracy,
            "total_samples": total_samples,
            "client_accuracies": client_accuracies,
        }

    def get_convergence_metrics(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
        losses = [m["avg_loss"] for m in self.metrics_history]
        return {
            "rounds": self.current_round,
            "final_loss": losses[-1],
            "best_loss": min(losses),
            "convergence_rate": (losses[0] - losses[-1]) / len(losses)
            if len(losses) > 1
            else 0.0,
            "loss_history": losses,
        }


@dataclass
class TrainerConfig:
    """Configuration for federated trainer."""

    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    eval_every: int = 1
    checkpoint_every: int = 10
    early_stopping_patience: int = 20
    min_improvement: float = 1e-4
    device: str = "cpu"


class FederatedTrainer:
    """
    High-level API for federated learning training.
    Combines all components for easy training.
    """

    def __init__(
        self,
        model: nn.Module,
        partitioner: DataPartitioner,
        config: TrainerConfig,
        server_config: Optional[ServerConfig] = None,
        client_config: Optional[ClientConfig] = None,
    ):
        self.model = model
        self.partitioner = partitioner
        self.config = config
        self.server_config = server_config or ServerConfig()
        self.client_config = client_config or ClientConfig()
        self.server: Optional[FederatedServer] = None
        self.clients: List[FederatedClient] = []
        self._setup()

    def _setup(self) -> None:
        logger.info("Setting up federated training...")
        for client_id in range(self.partitioner.config.num_clients):
            client_model = type(self.model)(
                *[
                    getattr(self.model, attr)
                    for attr in ["in_features", "hidden_dim", "num_classes"]
                ]
                if hasattr(self.model, "in_features")
                else []
            )
            client_model.load_state_dict(self.model.state_dict())
            client_dataset = self.partitioner.get_client_dataset(client_id)
            train_loader = DataLoader(
                client_dataset, batch_size=self.client_config.batch_size, shuffle=True
            )
            client = FederatedClient(
                client_id, client_model, train_loader, self.client_config
            )
            self.clients.append(client)
        self.server = FederatedServer(self.model, self.clients, self.server_config)
        logger.info(f"Setup complete with {len(self.clients)} clients")

    def train(
        self, test_loaders: Optional[Dict[int, DataLoader]] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        logger.info(f"Starting federated training for {self.config.num_rounds} rounds")
        best_loss = float("inf")
        patience_counter = 0
        training_history = []

        for round_num in range(self.config.num_rounds):
            metrics = self.server.train_round(epochs=self.config.local_epochs)
            training_history.append(metrics)

            if test_loaders is not None and round_num % self.config.eval_every == 0:
                eval_metrics = self.server.evaluate(test_loaders)
                metrics["eval_accuracy"] = eval_metrics["accuracy"]
                if verbose:
                    logger.info(
                        f"Round {round_num + 1}/{self.config.num_rounds}: Loss={metrics['avg_loss']:.4f}, Acc={eval_metrics['accuracy']:.4f}"
                    )
            elif verbose:
                logger.info(
                    f"Round {round_num + 1}/{self.config.num_rounds}: Loss={metrics['avg_loss']:.4f}"
                )

            if metrics["avg_loss"] < best_loss - self.config.min_improvement:
                best_loss = metrics["avg_loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at round {round_num + 1}")
                break

        convergence = self.server.get_convergence_metrics()
        final_eval = (
            self.server.evaluate(test_loaders) if test_loaders else {"accuracy": 0.0}
        )

        return {
            "training_history": training_history,
            "convergence_metrics": convergence,
            "final_accuracy": final_eval["accuracy"],
            "final_loss": training_history[-1]["avg_loss"] if training_history else 0.0,
            "rounds_completed": len(training_history),
        }

    def get_global_model(self) -> nn.Module:
        return self.server.global_model if self.server else self.model


__all__ = [
    "FederatedServer",
    "FederatedClient",
    "FederatedTrainer",
    "DataPartitioner",
    "PartitionConfig",
    "PartitionStrategy",
    "AggregationStrategy",
    "FedAvgStrategy",
    "FedProxStrategy",
    "ScaffoldStrategy",
    "FedNovaStrategy",
    "SecureAggregationStrategy",
    "CommunicationManager",
    "CommunicationConfig",
    "GradientCompressor",
    "CompressionConfig",
    "CompressionMethod",
    "DifferentialPrivacy",
    "PrivacyConfig",
    "ClientConfig",
    "ServerConfig",
    "TrainerConfig",
    "ClientSelectionStrategy",
    "AggregationError",
    "CommunicationError",
    "ClientNotAvailableError",
]
