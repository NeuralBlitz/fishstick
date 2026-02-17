"""
Graph Neural Networks with Geometric and Sheaf Structure.
"""

from .geometric_gnn import (
    EquivariantMessagePassing,
    SheafGraphConv,
    GeometricGraphTransformer,
    GeometricTransformerLayer,
    RiemannianGraphConv,
    MolecularGraphNetwork,
    CrystalGraphNetwork,
    GeometricEdge,
)

from .advanced_gnn import (
    # Data structures
    GraphData,
    # Message passing layers
    MessagePassing,
    GCNConv,
    GATConv,
    GraphSAGEConv,
    GINConv,
    TransformerConv,
    EdgeConv,
    # Pooling
    global_mean_pool,
    global_max_pool,
    global_add_pool,
    TopKPool,
    SAGPool,
    DiffPool,
    Set2Set,
    # Utilities
    batch_graphs,
    to_dense_batch,
    add_self_loops_fn,
    normalize_adj,
    gcn_norm,
    # Data loaders
    GraphDataset,
    GraphDataLoader,
    NeighborLoader,
    collate_graphs,
    # Models
    GraphClassifier,
    LinkPredictor,
    GraphAutoencoder,
    # Applications
    MolecularPropertyPredictor,
    SocialNetworkClassifier,
    RecommendationGNN,
    # Training
    GNNTrainer,
)
