"""
Pose Estimation Datasets

Dataset classes for pose estimation including COCO, MPII,
hand pose, and animal pose datasets.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import os

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


@dataclass
class PoseDataset(Dataset):
    """
    Base class for pose estimation datasets.

    Args:
        root: Root directory for dataset
        split: Dataset split ("train", "val", "test")
        transform: Optional transforms to apply
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        self.images = []
        self.annotations = []

        self._load_annotations()

    def _load_annotations(self):
        """Load dataset annotations."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        raise NotImplementedError

    def _get_image(self, path: str) -> Tensor:
        """Load and preprocess image."""
        img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img


class COCOPoseDataset(PoseDataset):
    """
    COCO keypoint detection dataset.

    Args:
        root: Root directory for COCO dataset
        split: Dataset split
        year: COCO year (2017, 2018)
    """

    def __init__(
        self,
        root: str = "./data/coco",
        split: str = "train",
        year: str = "2017",
        transform: Optional[Any] = None,
    ):
        self.year = year
        super().__init__(root, split, transform)

    def _load_annotations(self):
        """Load COCO annotations."""
        ann_file = os.path.join(
            self.root, f"annotations/person_keypoints_{self.split}{self.year}.json"
        )

        if os.path.exists(ann_file):
            try:
                import json

                with open(ann_file, "r") as f:
                    self.coco_data = json.load(f)

                self.images = [
                    img
                    for img in self.coco_data["images"]
                    if any(
                        ann["image_id"] == img["id"]
                        for ann in self.coco_data["annotations"]
                    )
                ]

                self.annotations = self.coco_data["annotations"]
            except Exception:
                self.images = []
                self.annotations = []
        else:
            self.images = []
            self.annotations = []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get COCO pose item."""
        img_info = self.images[idx]

        img_path = os.path.join(
            self.root, f"{self.split}{self.year}", img_info["file_name"]
        )

        image = self._get_image(img_path)

        img_anns = [
            ann for ann in self.annotations if ann["image_id"] == img_info["id"]
        ]

        keypoints = []
        bboxes = []

        for ann in img_anns:
            kp = np.array(ann["keypoints"]).reshape(-1, 3)
            keypoints.append(kp)

            bbox = ann["bbox"]
            bboxes.append(bbox)

        if len(keypoints) == 0:
            keypoints = np.zeros((17, 3))

        item = {
            "image": image,
            "keypoints": torch.tensor(keypoints[0])
            if len(keypoints) > 0
            else torch.zeros(17, 3),
            "image_id": img_info["id"],
            "file_name": img_info["file_name"],
        }

        if len(bboxes) > 0:
            item["bbox"] = torch.tensor(bboxes[0])

        if self.transform:
            item = self.transform(item)

        return item


class MPIIPoseDataset(PoseDataset):
    """
    MPII human pose dataset.

    Args:
        root: Root directory for MPII dataset
        split: Dataset split
    """

    def __init__(
        self,
        root: str = "./data/mpii",
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        super().__init__(root, split, transform)

    def _load_annotations(self):
        """Load MPII annotations."""
        ann_file = os.path.join(self.root, f"annotations/{self.split}.json")

        if os.path.exists(ann_file):
            try:
                import json

                with open(ann_file, "r") as f:
                    self.annotations = json.load(f)

                self.images = [ann["image"] for ann in self.annotations]
            except Exception:
                self.images = []
                self.annotations = []
        else:
            self.images = []
            self.annotations = []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get MPII pose item."""
        ann = self.annotations[idx]

        img_path = os.path.join(self.root, "images", ann["image"])

        image = self._get_image(img_path)

        keypoints = np.array(ann["joints"]).reshape(-1, 3)

        center = ann["center"]
        scale = ann["scale"]

        item = {
            "image": image,
            "keypoints": torch.tensor(keypoints),
            "center": torch.tensor(center),
            "scale": torch.tensor([scale]),
            "file_name": ann["image"],
        }

        if self.transform:
            item = self.transform(item)

        return item


class HandDataset(PoseDataset):
    """
    Hand pose estimation dataset.

    Args:
        root: Root directory for hand dataset
        split: Dataset split
        dataset_name: Name of hand dataset ("freihand", "interhand", "oham")
    """

    def __init__(
        self,
        root: str = "./data/hand",
        split: str = "train",
        dataset_name: str = "freihand",
        transform: Optional[Any] = None,
    ):
        self.dataset_name = dataset_name
        super().__init__(root, split, transform)

    def _load_annotations(self):
        """Load hand pose annotations."""
        ann_file = os.path.join(
            self.root, self.dataset_name, f"{self.split}_annotations.json"
        )

        if os.path.exists(ann_file):
            try:
                import json

                with open(ann_file, "r") as f:
                    self.annotations = json.load(f)

                self.images = [ann["image_path"] for ann in self.annotations]
            except Exception:
                self.images = []
                self.annotations = []
        else:
            self.images = []
            self.annotations = []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get hand pose item."""
        ann = self.annotations[idx]

        img_path = os.path.join(self.root, self.dataset_name, ann["image_path"])

        image = self._get_image(img_path)

        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)

        item = {
            "image": image,
            "keypoints": torch.tensor(keypoints),
            "hand_type": ann.get("hand_type", "right"),
            "file_name": ann["image_path"],
        }

        if self.transform:
            item = self.transform(item)

        return item


class AnimalPoseDataset(PoseDataset):
    """
    Animal pose estimation dataset.

    Args:
        root: Root directory for animal pose dataset
        split: Dataset split
        animal_type: Type of animal ("dog", "horse", "cat", "bird")
    """

    def __init__(
        self,
        root: str = "./data/animal",
        split: str = "train",
        animal_type: str = "dog",
        transform: Optional[Any] = None,
    ):
        self.animal_type = animal_type
        super().__init__(root, split, transform)

    def _load_annotations(self):
        """Load animal pose annotations."""
        ann_file = os.path.join(self.root, f"{self.animal_type}_{self.split}.json")

        if os.path.exists(ann_file):
            try:
                import json

                with open(ann_file, "r") as f:
                    self.annotations = json.load(f)

                self.images = [ann["image_path"] for ann in self.annotations]
            except Exception:
                self.images = []
                self.annotations = []
        else:
            self.images = []
            self.annotations = []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get animal pose item."""
        ann = self.annotations[idx]

        img_path = os.path.join(self.root, ann["image_path"])

        image = self._get_image(img_path)

        keypoints = np.array(ann["keypoints"]).reshape(-1, 3)

        item = {
            "image": image,
            "keypoints": torch.tensor(keypoints),
            "animal_id": ann.get("animal_id", 0),
            "file_name": ann["image_path"],
        }

        if "bbox" in ann:
            item["bbox"] = torch.tensor(ann["bbox"])

        if self.transform:
            item = self.transform(item)

        return item


class PoseDataLoader:
    """
    Data loader wrapper for pose datasets.

    Args:
        dataset: Pose dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        **kwargs: Additional DataLoader arguments
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ):
        from torch.utils.data import DataLoader

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

        self.dataset = dataset

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


def collate_pose(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for pose datasets.

    Args:
        batch: List of samples

    Returns:
        Collated batch
    """
    result = {}

    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch if key in item]

        if all(isinstance(v, torch.Tensor) for v in values):
            result[key] = torch.stack(values)
        elif all(isinstance(v, (int, float)) for v in values):
            result[key] = torch.tensor(values)
        elif all(isinstance(v, str) for v in values):
            result[key] = values
        elif all(isinstance(v, list) for v in values):
            result[key] = values
        elif all(isinstance(v, dict) for v in values):
            result[key] = values
        else:
            result[key] = values

    return result


__all__ = [
    "PoseDataset",
    "COCOPoseDataset",
    "MPIIPoseDataset",
    "HandDataset",
    "AnimalPoseDataset",
    "PoseDataLoader",
    "collate_pose",
]
