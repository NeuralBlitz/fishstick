"""
Geometric Deep Learning Module for fishstick.

Implements comprehensive geometric deep learning tools including:
- Group Convolutions (SO3, O3, SE3, Cn, Dn)
- Steerable CNNs
- Non-Euclidean (Hyperbolic/Riemannian) Convolutions
- Graph Embedding Methods
- Set Transformers

Based on state-of-the-art research in geometric deep learning and
equivariant neural networks.
"""

from typing import Tuple, List, Optional

from .group_convolution import (
    GroupEquivariantConv,
    SO3EquivariantConv,
    O3EquivariantConv,
    SE3EquivariantConv,
    CyclicGroupConv,
    DihedralGroupConv,
    SphericalHarmonics,
    GroupBatchNorm,
)

from .steerable_cnn import (
    ClebschGordan,
    IrrepRepresentations,
    SteerableFilter,
    SteerableConv2D,
    SteerableResBlock,
    SteerableCNN,
    EquivariantNonLinearity,
    GeometricFeatureAggregation,
    SteerableAttention,
    SteerablePool,
)

from .non_euclidean_conv import (
    PoincareEmbedding,
    LorentzEmbedding,
    HyperbolicGraphConv,
    HyperbolicMLP,
    HyperbolicAttention,
    RiemannianGNN,
    HyperbolicGraphPooling,
    HyperbolicDistance,
    LorentzModelConv,
    HyperbolicBatchNorm,
)

from .graph_embedding import (
    DeepWalkEmbedder,
    Node2VecEmbedder,
    GraphSAGEEmbedder,
    SAGEConv,
    AttributedGraphEmbedding,
    GraphAutoEncoder,
    SignPredictor,
    LaplacianEigenmap,
    HigherOrderProximity,
)

from .set_transformer import (
    SetAttentionBlock,
    InducedSetAttentionBlock,
    PoolingByMultiHeadAttention,
    SetTransformer,
    DeepSet,
    SetEncoder,
    SetResBlock,
    Set2SetPool,
    MultiSetAttention,
)

__all__ = [
    "GroupEquivariantConv",
    "SO3EquivariantConv",
    "O3EquivariantConv",
    "SE3EquivariantConv",
    "CyclicGroupConv",
    "DihedralGroupConv",
    "SphericalHarmonics",
    "GroupBatchNorm",
    "ClebschGordan",
    "IrrepRepresentations",
    "SteerableFilter",
    "SteerableConv2D",
    "SteerableResBlock",
    "SteerableCNN",
    "EquivariantNonLinearity",
    "GeometricFeatureAggregation",
    "SteerableAttention",
    "SteerablePool",
    "PoincareEmbedding",
    "LorentzEmbedding",
    "HyperbolicGraphConv",
    "HyperbolicMLP",
    "HyperbolicAttention",
    "RiemannianGNN",
    "HyperbolicGraphPooling",
    "HyperbolicDistance",
    "LorentzModelConv",
    "HyperbolicBatchNorm",
    "DeepWalkEmbedder",
    "Node2VecEmbedder",
    "GraphSAGEEmbedder",
    "SAGEConv",
    "AttributedGraphEmbedding",
    "GraphAutoEncoder",
    "SignPredictor",
    "LaplacianEigenmap",
    "HigherOrderProximity",
    "SetAttentionBlock",
    "InducedSetAttentionBlock",
    "PoolingByMultiHeadAttention",
    "SetTransformer",
    "DeepSet",
    "SetEncoder",
    "SetResBlock",
    "Set2SetPool",
    "MultiSetAttention",
]
