# TODO: Geometric Deep Learning Module Development

## Overview
Build comprehensive Geometric Deep Learning tools for the fishstick AI framework under `/home/runner/workspace/fishstick/geometric_dl/`

## Module List (5+ substantial modules)

### 1. group_convolution.py ✅
- [x] GroupEquivariantConv: Base group equivariant convolution
- [x] SO3EquivariantConv: Rotation group convolutions (SO3)
- [x] O3EquivariantConv: Full orthogonal group convolutions
- [x] SE3EquivariantConv: Special Euclidean group convolutions
- [x] CyclicGroupConv: C_n group convolutions
- [x] DihedralGroupConv: D_n group convolutions

### 2. steerable_cnn.py ✅
- [x] SteerableFilter: Learnable steerable filters
- [x] SteerableConv2D: Steerable 2D convolution
- [x] SteerableResBlock: Steerable residual block
- [x] SteerableCNN: Complete steerable CNN architecture
- [x] ClebshGordan: Clebsch-Gordan coefficient computation
- [x] IrrepRepresentations: Irreducible representation handlers

### 3. non_euclidean_conv.py ✅
- [x] HyperbolicGraphConv: Graph convolutions in hyperbolic space
- [x] HyperbolicMLP: Multi-layer perceptron in hyperbolic space
- [x] RiemannianGNN: GNN on Riemannian manifolds
- [x] HyperbolicAttention: Attention in hyperbolic space
- [x] PoincareEmbedding: Embeddings in Poincaré ball model
- [x] LorentzEmbedding: Embeddings in Lorentz model

### 4. graph_embedding.py ✅
- [x] DeepWalkEmbedder: DeepWalk-based embeddings
- [x] Node2VecEmbedder: Node2Vec embeddings
- [x] GraphSAGEEmbedder: SAGE-style embeddings
- [x] AttributedGraphEmbedding: Embeddings for attributed graphs
- [x] SignPredictor: Sign prediction for graph edges
- [x] GraphAutoEncoder: Autoencoder-based graph embeddings

### 5. set_transformer.py ✅
- [x] SetAttentionBlock: Attention for sets
- [x] InducedSetAttentionBlock: Induced set attention
- [x] SetTransformer: Full set transformer architecture
- [x] PoolingByMultiHeadAttention: PBMA attention pooling
- [x] SetEncoder: Encoder for unordered sets
- [x] DeepSet: Deep Sets architecture

### 6. __init__.py ✅
- [x] Export all classes from all modules
- [x] Define __all__ with all exported symbols

## Code Requirements
- All modules must have comprehensive docstrings
- Use type hints throughout
- Follow fishstick code style (import torch, nn, F, Tensor)
- Include both forward() methods with detailed parameter documentation
- Each module should have practical utility

## Implementation Order
1. Create directory structure
2. Implement group_convolution.py (foundational)
3. Implement steerable_cnn.py (builds on group conv)
4. Implement non_euclidean_conv.py (different geometry)
5. Implement graph_embedding.py (complements existing GNN)
6. Implement set_transformer.py (transformers for sets)
7. Create __init__.py with exports

## Notes
- Focus on mathematically rigorous implementations
- Maintain consistency with existing fishstick modules
- Ensure PyTorch compatibility
