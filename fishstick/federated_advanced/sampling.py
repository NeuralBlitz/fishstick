import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ClientStats:
    client_id: int
    num_samples: int
    compute_power: float
    network_bandwidth: float
    latency: float


class HeterogeneousSampler:
    def __init__(
        self,
        num_clients: int,
        client_stats: List[ClientStats],
        strategy: str = "uniform",
    ):
        self.num_clients = num_clients
        self.client_stats = {cs.client_id: cs for cs in client_stats}
        self.strategy = strategy

    def sample_clients(
        self,
        num_to_sample: int,
        round_number: int,
    ) -> List[int]:
        if self.strategy == "uniform":
            return self._uniform_sample(num_to_sample)
        elif self.strategy == "power_of_choice":
            return self._power_of_choice_sample(num_to_sample)
        elif self.strategy == "bandwidth_aware":
            return self._bandwidth_aware_sample(num_to_sample)
        else:
            return self._uniform_sample(num_to_sample)

    def _uniform_sample(self, num_to_sample: int) -> List[int]:
        all_clients = list(range(self.num_clients))
        indices = np.random.choice(all_clients, size=num_to_sample, replace=False)
        return indices.tolist()

    def _power_of_choice_sample(self, num_to_sample: int) -> List[int]:
        client_ids = list(self.client_stats.keys())
        sample_pool_size = min(num_to_sample * 2, len(client_ids))

        sampled_pool = np.random.choice(
            client_ids, size=sample_pool_size, replace=False
        )

        sorted_clients = sorted(
            sampled_pool,
            key=lambda cid: self.client_stats[cid].compute_power,
            reverse=True,
        )

        return sorted_clients[:num_to_sample]

    def _bandwidth_aware_sample(self, num_to_sample: int) -> List[int]:
        client_ids = list(self.client_stats.keys())

        bandwidth_scores = {
            cid: self.client_stats[cid].network_bandwidth
            / (self.client_stats[cid].latency + 1e-6)
            for cid in client_ids
        }

        sorted_clients = sorted(
            client_ids, key=lambda cid: bandwidth_scores[cid], reverse=True
        )

        return sorted_clients[:num_to_sample]


class ClientSelector(ABC):
    @abstractmethod
    def select(
        self,
        client_stats: List[ClientStats],
        round_number: int,
        total_clients: int,
        sample_fraction: float,
    ) -> List[int]:
        pass


class RandomClientSelector(ClientSelector):
    def select(
        self,
        client_stats: List[ClientStats],
        round_number: int,
        total_clients: int,
        sample_fraction: float,
    ) -> List[int]:
        num_to_select = max(1, int(total_clients * sample_fraction))
        return np.random.choice(
            total_clients, size=num_to_select, replace=False
        ).tolist()


class OortClientSelector(ClientSelector):
    def __init__(self, exploration_factor: float = 0.2):
        self.exploration_factor = exploration_factor
        self.client_performance_history: Dict[int, List[float]] = {}

    def select(
        self,
        client_stats: List[ClientStats],
        round_number: int,
        total_clients: int,
        sample_fraction: float,
    ) -> List[int]:
        num_to_select = max(1, int(total_clients * sample_fraction))

        client_scores = []
        for cs in client_stats:
            historical_perf = self.client_performance_history.get(cs.client_id, [1.0])
            avg_perf = np.mean(historical_perf[-10:])

            utility = avg_perf * cs.num_samples
            client_scores.append((cs.client_id, utility))

        client_scores.sort(key=lambda x: x[1], reverse=True)

        exploit_clients = [c[0] for c in client_scores[:num_to_select]]

        if len(exploit_clients) < num_to_select:
            explore_clients = [c[0] for c in client_scores[num_to_select:]]
            needed = num_to_select - len(exploit_clients)
            explore_sample = np.random.choice(
                explore_clients, size=min(needed, len(explore_clients)), replace=False
            )
            exploit_clients.extend(explore_sample.tolist())

        return exploit_clients

    def update_performance(self, client_id: int, performance: float) -> None:
        if client_id not in self.client_performance_history:
            self.client_performance_history[client_id] = []
        self.client_performance_history[client_id].append(performance)


class GradientCompressor(ABC):
    @abstractmethod
    def compress(
        self, gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        pass

    @abstractmethod
    def decompress(
        self, compressed: Dict[str, torch.Tensor], metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        pass


class TopKCompressor(GradientCompressor):
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio

    def compress(
        self, gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        compressed = {}
        metadata = {}

        for name, grad in gradients.items():
            flat_grad = grad.flatten()
            k = max(1, int(len(flat_grad) * self.compression_ratio))

            abs_values = torch.abs(flat_grad)
            _, topk_indices = torch.topk(abs_values, k)

            compressed_values = flat_grad[topk_indices]
            compressed_indices = topk_indices.cpu()

            compressed[name] = {
                "values": compressed_values,
                "indices": compressed_indices,
                "shape": grad.shape,
            }

            metadata[name] = {
                "shape": grad.shape,
                "num_elements": grad.numel(),
                "compressed_size": k,
            }

        return compressed, metadata

    def decompress(
        self, compressed: Dict[str, torch.Tensor], metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        decompressed = {}

        for name, data in compressed.items():
            full_size = metadata[name]["num_elements"]
            indices = (
                data["indices"].to(self.device)
                if isinstance(data["indices"], torch.Tensor)
                else torch.tensor(data["indices"], device=self.device)
            )
            values = (
                data["values"].to(self.device)
                if isinstance(data["values"], torch.Tensor)
                else torch.tensor(data["values"], device=self.device)
            )

            grad = torch.zeros(full_size, device=self.device)
            grad[indices] = values

            shape = metadata[name]["shape"]
            decompressed[name] = grad.view(shape)

        return decompressed


class QuantizationCompressor(GradientCompressor):
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
        self.levels = 2**num_bits

    def compress(
        self, gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        compressed = {}
        metadata = {}

        for name, grad in gradients.items():
            min_val = grad.min()
            max_val = grad.max()

            normalized = (grad - min_val) / (max_val - min_val + 1e-8)
            quantized = torch.round(normalized * (self.levels - 1)).to(torch.uint8)

            compressed[name] = quantized.cpu()
            metadata[name] = {
                "min_val": min_val.item(),
                "max_val": max_val.item(),
                "shape": grad.shape,
            }

        return compressed, metadata

    def decompress(
        self, compressed: Dict[str, torch.Tensor], metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        decompressed = {}

        for name, quantized in compressed.items():
            meta = metadata[name]
            min_val = meta["min_val"]
            max_val = meta["max_val"]

            dequantized = quantized.float() / (self.levels - 1)
            dequantized = dequantized * (max_val - min_val) + min_val

            decompressed[name] = dequantized.view(meta["shape"])

        return decompressed


class SparsificationCompressor(GradientCompressor):
    def __init__(
        self, threshold: float = 0.01, device: torch.device = torch.device("cpu")
    ):
        self.threshold = threshold
        self.device = device

    def compress(
        self, gradients: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        compressed = {}
        metadata = {}

        for name, grad in gradients.items():
            mask = torch.abs(grad) > self.threshold

            compressed_grad = grad * mask
            non_zero_indices = torch.nonzero(mask.flatten()).squeeze()

            compressed[name] = {
                "values": compressed_grad[mask],
                "indices": non_zero_indices,
                "shape": grad.shape,
            }

            metadata[name] = {
                "shape": grad.shape,
                "sparsity": (1 - mask.sum().item() / mask.numel()),
            }

        return compressed, metadata

    def decompress(
        self, compressed: Dict[str, torch.Tensor], metadata: Dict
    ) -> Dict[str, torch.Tensor]:
        decompressed = {}

        for name, data in compressed.items():
            shape = metadata[name]["shape"]
            grad = torch.zeros(shape).to(self.device)

            indices = (
                data["indices"].to(self.device)
                if isinstance(data["indices"], torch.Tensor)
                else torch.tensor(data["indices"], device=self.device)
            )
            values = (
                data["values"].to(self.device)
                if isinstance(data["values"], torch.Tensor)
                else torch.tensor(data["values"], device=self.device)
            )

            grad.flatten()[indices] = values
            decompressed[name] = grad

        return decompressed
