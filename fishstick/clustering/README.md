# Clustering Module

Comprehensive collection of clustering algorithms from classical to deep learning methods.

## Overview

This module provides:

- **K-Means**: Standard, Mini-batch, Bisecting
- **Hierarchical**: Agglomerative, Divisive, BIRCH
- **Density-Based**: DBSCAN, OPTICS, MeanShift
- **Spectral**: Normalized cuts, Kernel, Self-tuning
- **Deep Clustering**: DEC, IDEC, DCN, VaDE, ClusterGAN

## Installation

```bash
pip install torch numpy scipy scikit-learn
```

## Quick Start

### K-Means

```python
import numpy as np
from fishstick.clustering import KMeans

# Standard K-Means
kmeans = KMeans(n_clusters=5, n_init=10)
labels = kmeans.fit_predict(data)

# Mini-batch for large datasets
minibatch = MiniBatchKMeans(n_clusters=5, batch_size=1000)
labels = minibatch.fit_predict(large_data)
```

### DBSCAN

```python
from fishstick.clustering import DBSCAN

# Density-based clustering
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric='euclidean'
)
labels = dbscan.fit_predict(data)
# -1 label = noise/outlier
```

### Spectral Clustering

```python
from fishstick.clustering import SpectralClustering

spectral = SpectralClustering(
    n_clusters=5,
    affinity='rbf',
    n_neighbors=10
)
labels = spectral.fit_predict(data)
```

### Deep Embedded Clustering (DEC)

```python
import torch
from fishstick.clustering import DEC, AutoEncoder

# Autoencoder for representation learning
autoencoder = AutoEncoder(
    input_dim=784,
    hidden_dims=[256, 128, 32]
)

# DEC model
dec = DEC(
    encoder=autoencoder.encoder,
    n_clusters=10,
    hidden_dim=32
)

# Train
dec.fit(train_data, epochs=100)

# Predict
labels = dec.predict(test_data)
```

## API Reference

### K-Means

| Class | Description |
|-------|-------------|
| `KMeans` | Standard K-means |
| `MiniBatchKMeans` | Mini-batch variant |
| `BisectingKMeans` | Hierarchical K-means |
| `KMeansPlusPlus` | Smart initialization |

### Hierarchical

| Class | Description |
|-------|-------------|
| `AgglomerativeClustering` | Bottom-up clustering |
| `DivisiveClustering` | Top-down clustering |
| `BIRCH` | Memory-efficient |

### Density-Based

| Class | Description |
|-------|-------------|
| `DBSCAN` | Density-based |
| `OPTICS` | Ordering points |
| `MeanShift` | Mode seeking |

### Spectral

| Class | Description |
|-------|-------------|
| `SpectralClustering` | Standard spectral |
| `NormalizedCutSpectral` | Normalized cuts |
| `KernelSpectralClustering` | Kernel-based |

### Deep Clustering

| Class | Description |
|-------|-------------|
| `DEC` | Deep Embedded Clustering |
| `IDEC` | Improved DEC |
| `DCN` | Deep Clustering Network |
| `VaDE` | Variational Deep Embedding |
| `ClusterGAN` | GAN-based clustering |

## License

MIT License
