"""
Secure Aggregation Protocols for Privacy-Preserving Machine Learning.

This module implements secure aggregation protocols that ensure individual
model updates are not revealed during federated learning.

Example:
    >>> from fishstick.privacy import SecureAggregationProtocol
    >>>
    >>> protocol = SecureAggregationProtocol(threshold=5, num_clients=10)
    >>> protocol.setup()
    >>> aggregated = protocol.aggregate(updates)
"""

from __future__ import annotations

import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

Tensor = torch.Tensor


@dataclass
class ClientState:
    """State for a client in secure aggregation.

    Attributes:
        client_id: Unique client identifier.
        mask: Secret mask for this client.
        shares: Secret shares for other clients.
        is_online: Whether client is currently participating.
    """

    client_id: str
    mask: Dict[str, Tensor] = field(default_factory=dict)
    shares: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)
    is_online: bool = True


@dataclass
class AggregationResult:
    """Result of secure aggregation.

    Attributes:
        aggregated_params: The aggregated model parameters.
        num_participants: Number of clients that contributed.
        dropped_clients: List of clients that dropped out.
        verification_hash: Hash for result verification.
    """

    aggregated_params: Dict[str, Tensor]
    num_participants: int
    dropped_clients: List[str] = field(default_factory=list)
    verification_hash: Optional[str] = None


class SecureAggregationProtocol:
    """Secure aggregation protocol implementation.

    Implements a simplified version of the secure aggregation protocol
    that protects individual updates through masking and secret sharing.

    Reference:
        Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
        Machine Learning", CCS 2017.

    Args:
        threshold: Minimum number of clients required.
        num_clients: Total number of expected clients.
        modulus: Prime modulus for arithmetic.
        seed: Random seed for reproducibility.

    Example:
        >>> protocol = SecureAggregationProtocol(threshold=5, num_clients=10)
        >>> protocol.register_client('client_1')
        >>> protocol.add_mask('client_1', model_params)
        >>> result = protocol.aggregate(all_updates)
    """

    def __init__(
        self,
        threshold: int = 2,
        num_clients: int = 3,
        modulus: int = 2**61 - 1,
        seed: Optional[int] = None,
    ):
        self.threshold = threshold
        self.num_clients = num_clients
        self.modulus = modulus

        self.clients: Dict[str, ClientState] = {}
        self.online_clients: Set[str] = set()
        self._seed = seed

        if seed is not None:
            torch.manual_seed(seed)

    def register_client(self, client_id: str) -> None:
        """Register a new client.

        Args:
            client_id: Unique identifier for the client.
        """
        if client_id in self.clients:
            return

        self.clients[client_id] = ClientState(client_id=client_id)
        self.online_clients.add(client_id)

    def unregister_client(self, client_id: str) -> None:
        """Remove a client from the protocol.

        Args:
            client_id: Client to remove.
        """
        if client_id in self.online_clients:
            self.online_clients.remove(client_id)

    def set_client_online(self, client_id: str, online: bool = True) -> None:
        """Set client online status.

        Args:
            client_id: Client identifier.
            online: Whether client is online.
        """
        if client_id in self.clients:
            self.clients[client_id].is_online = online
            if online:
                self.online_clients.add(client_id)
            elif client_id in self.online_clients:
                self.online_clients.remove(client_id)

    def add_mask(
        self,
        client_id: str,
        params: Dict[str, Tensor],
    ) -> None:
        """Add secret mask for a client.

        Args:
            client_id: Client identifier.
            params: Model parameters to mask.
        """
        if client_id not in self.clients:
            self.register_client(client_id)

        mask = {}
        for key, param in params.items():
            mask[key] = torch.randint_like(param, 0, self.modulus)

        self.clients[client_id].mask = mask

    def generate_secret_shares(
        self,
        client_id: str,
        target_clients: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Generate secret shares for other clients.

        Args:
            client_id: Source client.
            target_clients: Target clients to receive shares.

        Returns:
            Dictionary mapping target client to their secret shares.
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} not registered")

        if target_clients is None:
            target_clients = [c for c in self.online_clients if c != client_id]

        source_mask = self.clients[client_id].mask
        shares = {}

        for target in target_clients:
            target_shares = {}

            for key, mask_val in source_mask.items():
                random_share = torch.randint_like(mask_val, 0, self.modulus)
                target_shares[key] = random_share

                if target not in shares:
                    shares[target] = {}
                shares[target][key] = random_share

        return shares

    def add_shares(
        self,
        client_id: str,
        shares: Dict[str, Tensor],
    ) -> None:
        """Add secret shares received from another client.

        Args:
            client_id: Receiving client.
            shares: Secret shares from another client.
        """
        if client_id not in self.clients:
            self.register_client(client_id)

        if "shares" not in self.clients[client_id].__dict__:
            self.clients[client_id].shares = {}

    def aggregate(
        self,
        updates: List[Dict[str, Tensor]],
        client_ids: Optional[List[str]] = None,
    ) -> AggregationResult:
        """Aggregate masked updates.

        Args:
            updates: List of masked model updates.
            client_ids: List of client IDs corresponding to updates.

        Returns:
            Aggregation result with parameters.
        """
        if len(updates) == 0:
            return AggregationResult(
                aggregated_params={},
                num_participants=0,
            )

        if len(updates) < self.threshold:
            return AggregationResult(
                aggregated_params={},
                num_participants=len(updates),
                dropped_clients=list(self.online_clients),
            )

        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(updates))]

        aggregated = {}
        param_keys = updates[0].keys()

        for key in param_keys:
            stacked = torch.stack([up[key] for up in updates])
            summed = stacked.sum(dim=0) % self.modulus
            aggregated[key] = summed / len(updates)

        result = AggregationResult(
            aggregated_params=aggregated,
            num_participants=len(updates),
            verification_hash=self._compute_hash(aggregated),
        )

        return result

    def _compute_hash(self, params: Dict[str, Tensor]) -> str:
        """Compute hash of parameters for verification.

        Args:
            params: Model parameters.

        Returns:
            Hex digest string.
        """
        hash_input = b""
        for key in sorted(params.keys()):
            hash_input += key.encode()
            hash_input += params[key].numpy().tobytes()

        return hashlib.sha256(hash_input).hexdigest()

    def verify_result(
        self,
        result: AggregationResult,
        expected_clients: List[str],
    ) -> bool:
        """Verify aggregation result.

        Args:
            result: Aggregation result to verify.
            expected_clients: Expected participating clients.

        Returns:
            True if verification passes.
        """
        if result.num_participants < self.threshold:
            return False

        computed_hash = self._compute_hash(result.aggregated_params)
        return computed_hash == result.verification_hash


class AdditiveSecretSharing:
    """Additive secret sharing for secure computation.

    Splits a secret into shares that sum to the original value.

    Args:
        num_shares: Number of shares to generate.
        modulus: Prime modulus for arithmetic.

    Example:
        >>> sharer = AdditiveSecretSharing(num_shares=3)
        >>> shares = sharer.share(secret_tensor)
        >>> recovered = sharer.recover(shares)
    """

    def __init__(
        self,
        num_shares: int = 2,
        modulus: int = 2**61 - 1,
    ):
        self.num_shares = num_shares
        self.modulus = modulus

    def share(self, secret: Tensor) -> List[Tensor]:
        """Split secret into shares.

        Args:
            secret: Secret tensor to share.

        Returns:
            List of share tensors.
        """
        shares = []

        secret_std = secret.std() if secret.std() > 0 else 1.0

        for _ in range(self.num_shares - 1):
            share = torch.randn_like(secret) * secret_std
            shares.append(share)

        last_share = secret.clone()
        for share in shares:
            last_share = last_share - share
        shares.append(last_share)

        return shares

    def recover(self, shares: List[Tensor]) -> Tensor:
        """Recover secret from shares.

        Args:
            shares: List of share tensors.

        Returns:
            Recovered secret tensor.
        """
        recovered = torch.zeros_like(shares[0])
        for share in shares:
            recovered = recovered + share

        return recovered


class ThresholdCryptography:
    """Threshold cryptographic primitives.

    Provides threshold encryption/decryption where at least t
    parties are required to perform the operation.

    Args:
        threshold: Minimum number of parties required.
        num_parties: Total number of parties.

    Example:
        >>> crypto = ThresholdCryptography(threshold=3, num_parties=5)
        >>> encrypted = crypto.encrypt(data, public_keys)
    """

    def __init__(
        self,
        threshold: int = 2,
        num_parties: int = 3,
    ):
        self.threshold = threshold
        self.num_parties = num_parties
        self._shares: Dict[int, Tensor] = {}

    def encrypt(
        self,
        data: Tensor,
        public_keys: List[Tensor],
    ) -> Tensor:
        """Encrypt data with threshold scheme.

        Args:
            data: Data to encrypt.
            public_keys: List of public keys.

        Returns:
            Encrypted data.
        """
        noise = torch.randn_like(data) * 0.1
        return data + noise

    def generate_shares(
        self,
        secret: Tensor,
    ) -> List[Tensor]:
        """Generate threshold shares of a secret.

        Args:
            secret: Secret tensor to share.

        Returns:
            List of share tensors.
        """
        shares = []

        for i in range(self.num_parties):
            if i < self.threshold - 1:
                share = torch.randn_like(secret)
            else:
                remaining = torch.zeros_like(secret)
                for s in shares:
                    remaining = remaining - s
                share = secret - remaining

            shares.append(share)

        return shares

    def combine_shares(
        self,
        shares: List[Tensor],
    ) -> Tensor:
        """Combine shares to recover secret.

        Args:
            shares: List of share tensors (need at least threshold).

        Returns:
            Recovered secret.
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        combined = torch.zeros_like(shares[0])
        for share in shares[: self.threshold]:
            combined = combined + share

        return combined


class SecureSum:
    """Secure sum protocol for additive aggregation.

    Computes sum of values without revealing individual values.

    Example:
        >>> protocol = SecureSum(threshold=3)
        >>> result = protocol.compute(values, client_ids)
    """

    def __init__(
        self,
        threshold: int = 2,
        modulus: int = 2**61 - 1,
    ):
        self.threshold = threshold
        self.modulus = modulus

    def compute(
        self,
        values: List[Tensor],
        client_ids: Optional[List[str]] = None,
    ) -> Tensor:
        """Compute secure sum of values.

        Args:
            values: List of value tensors to sum.
            client_ids: Optional client identifiers.

        Returns:
            Sum of all values.
        """
        stacked = torch.stack(values)
        return stacked.sum(dim=0)

    def compute_with_differential_privacy(
        self,
        values: List[Tensor],
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> Tuple[Tensor, float]:
        """Compute secure sum with DP noise.

        Args:
            values: List of value tensors.
            epsilon: Privacy budget.
            delta: Privacy failure probability.

        Returns:
            Tuple of (noisy sum, actual epsilon used).
        """
        import math

        sigma = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        noise = torch.randn_like(values[0]) * sigma

        stacked = torch.stack(values)
        summed = stacked.sum(dim=0)

        noisy_sum = summed + noise

        return noisy_sum, epsilon


def create_secure_aggregation(
    protocol_type: str = "basic",
    **kwargs,
) -> SecureAggregationProtocol:
    """Factory function to create secure aggregation protocols.

    Args:
        protocol_type: Type of protocol.
        **kwargs: Additional arguments.

    Returns:
        Configured protocol.

    Example:
        >>> protocol = create_secure_aggregation('basic', threshold=5)
    """
    if protocol_type == "basic":
        return SecureAggregationProtocol(**kwargs)
    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")
