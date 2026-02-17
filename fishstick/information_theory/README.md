# Information Theory

Entropy, mutual information, and information-theoretic losses.

## Installation

```bash
pip install fishstick[information_theory]
```

## Overview

The `information_theory` module provides information-theoretic measures including entropy, mutual information, and compression algorithms.

## Usage

```python
from fishstick.information_theory import entropy, mutual_info, shannon_entropy

# Shannon entropy
h = shannon_entropy(probabilities)

# Mutual information
mi = mutual_info(X, Y)

# Differential entropy
diff_h = entropy.differential(x)
```

## Entropy

| Function | Description |
|----------|-------------|
| `shannon_entropy` | Shannon entropy |
| `differential_entropy` | Differential entropy |
| `renyi_entropy` | Renyi entropy |
| `tsallis_entropy` | Tsallis entropy |
| `sample_entropy` | Sample entropy |

## Mutual Information

| Function | Description |
|----------|-------------|
| `mutual_info` | Mutual information |
| `conditional_entropy` | Conditional entropy |
| `joint_entropy` | Joint entropy |

## Compression

| Class | Description |
|-------|-------------|
| `KLDivergence` | KL divergence |
| `CompressionAlgorithm` | Data compression |

## Losses

| Loss | Description |
|------|-------------|
| `InfoNCE` | InfoNCE loss |
| `IBLoss` | Information Bottleneck loss |

## Examples

See `examples/information_theory/` for complete examples.
