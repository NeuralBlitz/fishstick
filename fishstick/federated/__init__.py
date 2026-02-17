"""
fishstick Federated Learning Module

Federated learning algorithms and utilities.
"""

from fishstick.federated.clients import FederatedClient, ClientManager
from fishstick.federated.server import FederatedServer, AggregationStrategy
from fishstick.federated.strategies import FedAvg, FedProx, FedNova

# Advanced federated learning components
from fishstick.federated.advanced import (
    # Main classes
    FederatedServer as AdvancedFederatedServer,
    FederatedClient as AdvancedFederatedClient,
    FederatedTrainer,
    # Data partitioning
    DataPartitioner,
    PartitionConfig,
    PartitionStrategy,
    # Aggregation strategies
    AggregationStrategy as AdvancedAggregationStrategy,
    FedAvgStrategy,
    FedProxStrategy,
    ScaffoldStrategy,
    FedNovaStrategy,
    SecureAggregationStrategy,
    # Communication
    CommunicationManager,
    CommunicationConfig,
    # Compression
    GradientCompressor,
    CompressionConfig,
    CompressionMethod,
    # Privacy
    DifferentialPrivacy,
    PrivacyConfig,
    # Configuration
    ClientConfig,
    ServerConfig,
    TrainerConfig,
    # Selection
    ClientSelectionStrategy,
    # Exceptions
    AggregationError,
    CommunicationError,
    ClientNotAvailableError,
)

__all__ = [
    # Original exports
    "FederatedClient",
    "ClientManager",
    "FederatedServer",
    "AggregationStrategy",
    "FedAvg",
    "FedProx",
    "FedNova",
    # Advanced exports
    "AdvancedFederatedServer",
    "AdvancedFederatedClient",
    "FederatedTrainer",
    "DataPartitioner",
    "PartitionConfig",
    "PartitionStrategy",
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
