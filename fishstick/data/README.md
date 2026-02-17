# data - Data Processing Module

## Overview

The `data` module provides advanced data loading, augmentation, and preprocessing utilities for various data modalities including images, text, graphs, and time series.

## Purpose and Scope

This module enables:
- Augmented dataset support
- Caching for large datasets
- Streaming data loading
- MixUp and CutMix augmentation
- Schema validation

## Key Classes and Functions

### Dataset Classes

#### `AugmentedDataset`
Dataset with built-in augmentation support.

```python
from fishstick.data import AugmentedDataset
import torch

dataset = AugmentedDataset(
    data=torch.randn(1000, 3, 32, 32),
    labels=torch.randint(0, 10, (1000,)),
    transform=my_transform,
    augment=True
)

x, y = dataset[0]  # Returns augmented sample
```

#### `CachedDataset`
Dataset with disk/memory caching.

```python
from fishstick.data import CachedDataset

dataset = CachedDataset(
    data_source="large_dataset.npy",
    cache_dir="./cache",
    cache_in_memory=True
)

# Data loaded on first access, then cached
x = dataset[0]
```

#### `StreamingDataset`
Streaming for data larger than memory.

```python
from fishstick.data import StreamingDataset

def load_file(path):
    return torch.load(path)

dataset = StreamingDataset(
    file_list=["data1.pt", "data2.pt", ...],
    loader_fn=load_file,
    buffer_size=100
)
```

### Augmentation Classes

#### `MixUp`
MixUp augmentation for training.

```python
from fishstick.data import MixUp

mixup = MixUp(alpha=0.2)

# Mix two samples
x_mixed, y_mixed = mixup(x1, y1, x2, y2)
```

#### `CutMix`
CutMix augmentation for images.

```python
from fishstick.data import CutMix

cutmix = CutMix(alpha=1.0)

# Cut and mix two images
x_mixed, y_mixed = cutmix(x1, y1, x2, y2)
```

### Data Loading

#### `DataLoader`
Enhanced DataLoader with fishstick features.

```python
from fishstick.data import DataLoader

loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

#### `create_dataloader`
Factory function for optimized DataLoader.

```python
from fishstick.data import create_dataloader

loader = create_dataloader(
    dataset=dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8
)
```

### Validation

#### `DataSchema`
Schema validation for datasets.

```python
from fishstick.data import DataSchema

schema = DataSchema({
    "image": "tensor[3,224,224]",
    "label": "int"
})

sample = {"image": torch.randn(3, 224, 224), "label": 5}
is_valid = schema.validate(sample)
```

## Dependencies

- `torch`: PyTorch datasets and data loading
- `numpy`: Numerical operations

## Usage Examples

### Complete Data Pipeline

```python
from fishstick.data import (
    AugmentedDataset, DataLoader, MixUp, CutMix,
    create_dataloader
)
import torchvision.transforms as T

# Transforms
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
])

# Dataset
train_dataset = AugmentedDataset(
    data=train_images,
    labels=train_labels,
    transform=train_transform,
    augment=True
)

# DataLoader
train_loader = create_dataloader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# Training with MixUp
mixup = MixUp(alpha=0.2)

for x, y in train_loader:
    # Apply mixup
    x2, y2 = next(iter(train_loader))
    x_mixed, y_mixed = mixup(x, y, x2, y2)
    
    # Train step
    output = model(x_mixed)
    loss = loss_fn(output, y_mixed)
```

### Streaming Large Datasets

```python
from fishstick.data import StreamingDataset, DataLoader
from pathlib import Path

# Get all data files
files = list(Path("large_dataset/").glob("*.pt"))

def load_chunk(path):
    return torch.load(path)

dataset = StreamingDataset(
    file_list=[str(f) for f in files],
    loader_fn=load_chunk,
    buffer_size=50
)

loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    # Process batch
    pass
```
