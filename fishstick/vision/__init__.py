"""
Computer Vision Module for fishstick

Advanced image processing, augmentation, and vision models with:
- Geometric augmentations (rotation, affine transforms)
- Sheaf-based image processing
- Vision transformers with attention
- Object detection utilities
- Image classification models
"""

from typing import Tuple, List, Optional, Union, Callable
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image


class GeometricAugmentation:
    """
    Geometrically meaningful augmentations preserving image structure.

    Unlike standard augmentations, these respect the geometric structure
    of images and can be composed functorially.
    """

    def __init__(self, rotation_range: Tuple[float, float] = (-15, 15)):
        self.rotation_range = rotation_range

    def random_rotation(self, image: Tensor) -> Tensor:
        """Apply random rotation preserving aspect ratio."""
        angle = np.random.uniform(*self.rotation_range)
        return TF.rotate(image, angle)

    def random_affine(
        self,
        image: Tensor,
        degrees: float = 10,
        translate: Tuple[float, float] = (0.1, 0.1),
        scale: Tuple[float, float] = (0.9, 1.1),
    ) -> Tensor:
        """Apply random affine transformation."""
        return TF.affine(
            image,
            angle=np.random.uniform(-degrees, degrees),
            translate=(
                np.random.uniform(-translate[0], translate[0]) * image.shape[-1],
                np.random.uniform(-translate[1], translate[1]) * image.shape[-2],
            ),
            scale=np.random.uniform(*scale),
            shear=0,
        )

    def random_perspective(
        self, image: Tensor, distortion_scale: float = 0.5
    ) -> Tensor:
        """Apply perspective transformation (3D viewpoint change)."""
        width, height = image.shape[-1], image.shape[-2]

        # Random perspective corners
        topleft = [
            np.random.uniform(0, distortion_scale * width),
            np.random.uniform(0, distortion_scale * height),
        ]
        topright = [
            np.random.uniform((1 - distortion_scale) * width, width),
            np.random.uniform(0, distortion_scale * height),
        ]
        botright = [
            np.random.uniform((1 - distortion_scale) * width, width),
            np.random.uniform((1 - distortion_scale) * height, height),
        ]
        botleft = [
            np.random.uniform(0, distortion_scale * width),
            np.random.uniform((1 - distortion_scale) * height, height),
        ]

        startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
        endpoints = [topleft, topright, botright, botleft]

        return TF.perspective(image, startpoints, endpoints)


class SheafImageProcessor(nn.Module):
    """
    Process images using sheaf theory concepts.

    Images are treated as sheaves over a base space (pixel grid),
    with stalks representing local features.
    """

    def __init__(self, in_channels: int = 3, stalk_dim: int = 16):
        super().__init__()
        self.stalk_dim = stalk_dim

        # Local feature extractors (stalks)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, stalk_dim, kernel_size=3, padding=1),
        )

        # Restriction maps between neighborhoods
        self.restriction = nn.Conv2d(stalk_dim, stalk_dim, kernel_size=3, padding=1)

        # Global consistency layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Process image through sheaf structure.

        Returns:
            local_features: Features per pixel (stalks)
            global_features: Pooled representation
        """
        # Extract local features (stalks)
        local_features = self.local_encoder(x)

        # Apply restriction maps (enforce consistency)
        consistent = self.restriction(local_features)

        # Global pooling
        global_features = self.global_pool(consistent).squeeze(-1).squeeze(-1)

        return consistent, global_features


class VisionTransformerBlock(nn.Module):
    """
    Vision Transformer block with geometric inductive bias.

    Combines standard ViT with position encoding that respects
    image geometry.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, num_patches, embed_dim]
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding via convolution
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional encoding with geometric structure
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            [batch, num_patches + 1, embed_dim]
        """
        batch_size = x.shape[0]

        # Create patches
        x = self.proj(x)  # [batch, embed_dim, h', w']
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        return x


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer with fishstick enhancements.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            [batch, num_classes]
        """
        # Patch embedding
        x = self.patch_embed(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling (using CLS token)
        x = self.norm(x)
        x = x[:, 0]  # CLS token

        # Classification head
        x = self.head(x)

        return x


class ObjectDetector:
    """
    Simple object detection wrapper using torchvision models.
    """

    def __init__(self, model_name: str = "fasterrcnn_resnet50_fpn"):
        if model_name == "fasterrcnn_resnet50_fpn":
            self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif model_name == "ssd300_vgg16":
            self.model = models.detection.ssd300_vgg16(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.eval()

    def detect(self, image: Tensor, threshold: float = 0.5) -> List[dict]:
        """
        Detect objects in image.

        Args:
            image: [channels, height, width] tensor
            threshold: Confidence threshold

        Returns:
            List of detections with 'boxes', 'labels', 'scores'
        """
        with torch.no_grad():
            predictions = self.model([image])

        # Filter by threshold
        pred = predictions[0]
        mask = pred["scores"] > threshold

        detections = []
        for box, label, score in zip(
            pred["boxes"][mask], pred["labels"][mask], pred["scores"][mask]
        ):
            detections.append(
                {"box": box.tolist(), "label": int(label), "score": float(score)}
            )

        return detections


class ImageAugmentationPipeline:
    """
    Complete augmentation pipeline for training.
    """

    def __init__(self, img_size: int = 224, augment: bool = True):
        self.img_size = img_size
        self.augment = augment

        # Geometric augmentations
        self.geometric = GeometricAugmentation()

        # Standard transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image: Union[Image.Image, Tensor]) -> Tensor:
        """Apply augmentation pipeline to image."""
        if isinstance(image, Image.Image):
            image = self.transform(image)

        if self.augment:
            # Apply geometric augmentations
            if np.random.rand() > 0.5:
                image = self.geometric.random_rotation(image)

            if np.random.rand() > 0.5:
                image = self.geometric.random_affine(image)

        return image


class SheafAugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset with sheaf-based image processing and augmentations.
    """

    def __init__(
        self,
        images: List[str],
        labels: List[int],
        img_size: int = 224,
        augment: bool = True,
    ):
        self.images = images
        self.labels = labels
        self.augment = augment

        self.pipeline = ImageAugmentationPipeline(img_size, augment)
        self.sheaf_processor = SheafImageProcessor(stalk_dim=16)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Load and process image."""
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        # Apply pipeline
        image = self.pipeline(image)

        # Optional: Apply sheaf processing
        if not self.augment:
            local_feat, global_feat = self.sheaf_processor(image.unsqueeze(0))
            # Could return features instead of raw image

        return image, self.labels[idx]


# COCO class names for object detection
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
