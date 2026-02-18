"""
Medical Image Dataset Base Classes

Base classes for medical image datasets.
"""

from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MedicalImageDataset(Dataset):
    """Base dataset for medical images.
    
    Example:
        >>> dataset = MedicalImageDataset(image_paths, transform=transform)
        >>> image, metadata = dataset[0]
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_fn: Optional[Callable] = None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.target_transform = target_transform
        self.load_fn = load_fn or self._default_load

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        path = self.image_paths[idx]
        
        image = self.load_fn(path)
        
        metadata = {"path": str(path), "index": idx}
        
        if self.transform:
            image = self.transform(image)
        
        return image, metadata

    def _default_load(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        
        if suffix in ['.nii', '.gz']:
            import nibabel as nib
            img = nib.load(str(path))
            data = img.get_fdata()
            return torch.from_numpy(data.astype(np.float32))
        
        elif suffix in ['.png', '.jpg', '.jpeg']:
            img = Image.open(path)
            data = np.array(img)
            return torch.from_numpy(data).float()
        
        else:
            data = np.load(path)
            return torch.from_numpy(data).float()


class SegmentationDataset(MedicalImageDataset):
    """Dataset for medical image segmentation.
    
    Example:
        >>> dataset = SegmentationDataset(image_paths, mask_paths, transform=transform)
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        mask_paths: List[Union[str, Path]],
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        super().__init__(image_paths, transform)
        
        self.mask_paths = [Path(p) for p in mask_paths]
        self.mask_transform = mask_transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.load_fn(self.image_paths[idx])
        
        mask = self._load_mask(self.mask_paths[idx])
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask

    def _load_mask(self, path: Path) -> torch.Tensor:
        suffix = path.suffix.lower()
        
        if suffix in ['.nii', '.gz']:
            import nibabel as nib
            img = nib.load(str(path))
            data = img.get_fdata()
            return torch.from_numpy(data.astype(np.int64))
        
        else:
            data = np.load(path)
            return torch.from_numpy(data).long()


class ClassificationDataset(MedicalImageDataset):
    """Dataset for medical image classification.
    
    Example:
        >>> dataset = ClassificationDataset(image_paths, labels, transform=transform)
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: List[int],
        transform: Optional[Callable] = None,
        weights: Optional[List[float]] = None,
    ):
        super().__init__(image_paths, transform)
        
        self.labels = labels
        self.weights = weights

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.load_fn(self.image_paths[idx])
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.weights is not None:
            weight = self.weights[idx]
            return image, label, weight
        
        return image, label


class RandomAccessDataset(Dataset):
    """Dataset with random access for large medical images."""

    def __init__(
        self,
        data_source: Any,
        index_map: Dict[int, Any],
    ):
        self.data_source = data_source
        self.index_map = index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Any:
        key = self.index_map[idx]
        return self.data_source.get(key)


class CachedDataset(Dataset):
    """Dataset with caching for frequently accessed samples."""

    def __init__(
        self,
        dataset: Dataset,
        cache_size: int = 100,
    ):
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        if idx in self.cache:
            return self.cache[idx]
        
        item = self.dataset[idx]
        
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        
        return item
