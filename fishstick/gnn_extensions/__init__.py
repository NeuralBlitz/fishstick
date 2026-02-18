"""
Graph Neural Network Extensions for fishstick.

Advanced GNN modules including:
- Graph Transformers
- Heterogeneous Graph Convolutions
- Advanced Pooling Methods
- Graph Generation Models
- Graph Matching Algorithms
"""

from .graph_transformer import (
    GraphTransformerLayer,
    GraphTransformer,
    GraphAttentionLayer,
    DirectionalGraphAttention,
)

from .positional_encoding import (
    LaplacianPositionalEncoding,
    RandomWalkPositionalEncoding,
    CentralityPositionalEncoding,
    RelativePositionalEncoding,
    SignNetPositionalEncoding,
)

from .heterogeneous_conv import (
    RelationGraphConv,
    HeterogeneousGraphConv,
    HANLayer,
    MetapathConv,
    RelationLearner,
    FastRGCNConv,
)

from .graph_pooling import (
    MinCutPool,
    DiffPool,
    TopKPool,
    SAGPool,
    AttentionPool,
    HierarchicalPool,
    Set2SetPool,
    SelfAttentionGraphPool,
)

from .graph_generation import (
    GraphVAE,
    GraphEncoder,
    GraphDecoder,
    GraphGAN,
    GraphGenerator,
    GraphDiscriminator,
    GraphAutoReggressiveGenerator,
    MoleculeGenerator,
)

from .graph_matching import (
    GraphMatchingNetwork,
    GraphMatchingLayer,
    CrossGraphAttention,
    GraphSimilarity,
    SubgraphMatching,
    GraphAlignment,
    GraphEditDistance,
    GraphKernel,
    WeisfeilerLehman,
)

__all__ = [
    # Graph Transformer
    "GraphTransformerLayer",
    "GraphTransformer",
    "GraphAttentionLayer",
    "DirectionalGraphAttention",
    # Positional Encoding
    "LaplacianPositionalEncoding",
    "RandomWalkPositionalEncoding",
    "CentralityPositionalEncoding",
    "RelativePositionalEncoding",
    "SignNetPositionalEncoding",
    # Heterogeneous Conv
    "RelationGraphConv",
    "HeterogeneousGraphConv",
    "HANLayer",
    "MetapathConv",
    "RelationLearner",
    "FastRGCNConv",
    # Pooling
    "MinCutPool",
    "DiffPool",
    "TopKPool",
    "SAGPool",
    "AttentionPool",
    "HierarchicalPool",
    "Set2SetPool",
    "SelfAttentionGraphPool",
    # Generation
    "GraphVAE",
    "GraphEncoder",
    "GraphDecoder",
    "GraphGAN",
    "GraphGenerator",
    "GraphDiscriminator",
    "GraphAutoReggressiveGenerator",
    "MoleculeGenerator",
    # Matching
    "GraphMatchingNetwork",
    "GraphMatchingLayer",
    "CrossGraphAttention",
    "GraphSimilarity",
    "SubgraphMatching",
    "GraphAlignment",
    "GraphEditDistance",
    "GraphKernel",
    "WeisfeilerLehman",
]
