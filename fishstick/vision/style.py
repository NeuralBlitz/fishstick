"""
Neural Style Transfer module for Fishstick.

This module provides comprehensive implementations of various neural style transfer
techniques including Gatys-style optimization-based methods, fast feed-forward
networks, arbitrary style transfer, and video style transfer.

Classes:
    GatysStyleTransfer: Original optimization-based style transfer
    AdaIN: Adaptive Instance Normalization for arbitrary style transfer
    WCT: Whitening and Coloring Transform
    AvatarNet: Multimodal style transfer
    SANet: Style-attentional networks
    LinearStyleTransfer: Fast feed-forward style transfer
    JohnsonStyleNet: Fast feed-forward with residual blocks
    TransformerStyle: Transformer-based style transfer
    MobileStyleTransfer: Mobile-optimized style transfer
    StyleTransferNetwork: Real-time arbitrary style transfer
    MetaStyleTransfer: Meta-learning for style transfer
    ZeroShotStyleTransfer: Zero-shot style transfer
    VGGFeatures: VGG-based feature extraction
    ContentLoss: Perceptual content loss
    StyleLoss: Gram matrix style loss
    TotalVariationLoss: Smoothness regularization
    MultiScaleStyleLoss: Multi-resolution style loss
    StyleInterpolation: Interpolate between styles
    StyleBlending: Combine multiple styles
    SpatialControl: Region-based style control
    TemporalConsistency: Video style consistency
    HistogramMatching: Match color histograms
    ReinhardColorTransfer: Statistical color transfer
    DeepColorTransfer: Neural color transfer
    VideoStyleTransfer: Consistent video style transfer
    OpticalFlowGuided: Optical flow-guided video style transfer
    RecurrentStyle: LSTM-based temporal coherence
    StyleDataset: Dataset for style transfer training
    StyleTrainer: Specialized trainer for style models
    StyleEvaluation: Evaluation metrics for style transfer

Example:
    >>> from fishstick.vision.style import GatysStyleTransfer, VGGFeatures
    >>> model = GatysStyleTransfer()
    >>> stylized = model(content_image, style_image)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer models."""

    content_weight: float = 1.0
    style_weight: float = 1000.0
    tv_weight: float = 10.0
    learning_rate: float = 0.01
    num_iterations: int = 300
    image_size: int = 256
    content_layers: List[str] = None
    style_layers: List[str] = None

    def __post_init__(self):
        if self.content_layers is None:
            self.content_layers = ["relu4_2"]
        if self.style_layers is None:
            self.style_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]


# ============================================================================
# Utility Functions
# ============================================================================


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style representation.

    Args:
        features: Feature tensor of shape (B, C, H, W)

    Returns:
        Gram matrix of shape (B, C, C)
    """
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize batch using ImageNet statistics.

    Args:
        batch: Input batch tensor

    Returns:
        Normalized batch
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std


def unnormalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize batch using ImageNet statistics.

    Args:
        batch: Normalized batch tensor

    Returns:
        Unnormalized batch
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return batch * std + mean


def match_histogram(source: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """
    Match histogram of source to template.

    Args:
        source: Source image tensor (B, C, H, W)
        template: Template image tensor (B, C, H, W)

    Returns:
        Histogram-matched image
    """
    result = source.clone()
    for i in range(source.size(0)):
        for c in range(source.size(1)):
            s_values, s_indices = torch.sort(source[i, c].view(-1))
            t_values, t_indices = torch.sort(template[i, c].view(-1))
            s_quantiles = torch.cumsum(
                torch.ones_like(s_values), dim=0
            ) / s_values.size(0)
            t_quantiles = torch.cumsum(
                torch.ones_like(t_values), dim=0
            ) / t_values.size(0)
            interp = torch.interp(s_quantiles, t_quantiles, t_values)
            result[i, c].view(-1)[s_indices] = interp
    return result


# ============================================================================
# Feature Extraction
# ============================================================================


class VGGFeatures(nn.Module):
    """
    VGG-based feature extractor for style transfer.

    Extracts hierarchical features from VGG19 pretrained on ImageNet
    for content and style representation.

    Attributes:
        layers (nn.ModuleDict): Dictionary of VGG layers by name
        content_layers (List[str]): Layers to use for content representation
        style_layers (List[str]): Layers to use for style representation

    Example:
        >>> vgg = VGGFeatures(
        ...     content_layers=['relu4_2'],
        ...     style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        ... )
        >>> content_features = vgg(content_image, 'content')
        >>> style_features = vgg(style_image, 'style')
    """

    def __init__(
        self,
        content_layers: List[str] = None,
        style_layers: List[str] = None,
        pretrained: bool = True,
    ):
        """
        Initialize VGG feature extractor.

        Args:
            content_layers: Layer names for content features
            style_layers: Layer names for style features
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        vgg = models.vgg19(pretrained=pretrained).features

        self.content_layers = content_layers or ["relu4_2"]
        self.style_layers = style_layers or [
            "relu1_1",
            "relu2_1",
            "relu3_1",
            "relu4_1",
            "relu5_1",
        ]
        self.all_layers = list(set(self.content_layers + self.style_layers))

        # Build layer mapping
        layer_mapping = {
            "conv1_1": 0,
            "relu1_1": 1,
            "conv1_2": 2,
            "relu1_2": 3,
            "pool1": 4,
            "conv2_1": 5,
            "relu2_1": 6,
            "conv2_2": 7,
            "relu2_2": 8,
            "pool2": 9,
            "conv3_1": 10,
            "relu3_1": 11,
            "conv3_2": 12,
            "relu3_2": 13,
            "conv3_3": 14,
            "relu3_3": 15,
            "conv3_4": 16,
            "relu3_4": 17,
            "pool3": 18,
            "conv4_1": 19,
            "relu4_1": 20,
            "conv4_2": 21,
            "relu4_2": 22,
            "conv4_3": 23,
            "relu4_3": 24,
            "conv4_4": 25,
            "relu4_4": 26,
            "pool4": 27,
            "conv5_1": 28,
            "relu5_1": 29,
            "conv5_2": 30,
            "relu5_2": 31,
            "conv5_3": 32,
            "relu5_3": 33,
            "conv5_4": 34,
            "relu5_4": 35,
            "pool5": 36,
        }

        # Create sequential model up to max layer
        max_idx = max([layer_mapping[l] for l in self.all_layers])
        self.model = nn.Sequential(*list(vgg.children())[: max_idx + 1])

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.eval()

    def forward(
        self, x: torch.Tensor, mode: str = "all"
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Extract features from input.

        Args:
            x: Input image tensor (B, 3, H, W)
            mode: 'content', 'style', or 'all'

        Returns:
            Dictionary of features by layer name
        """
        features = {}
        out = x

        layer_names = []
        if mode == "content":
            target_layers = self.content_layers
        elif mode == "style":
            target_layers = self.style_layers
        else:
            target_layers = self.all_layers

        # Mapping from sequential index to layer name
        layer_map = {
            1: "relu1_1",
            6: "relu2_1",
            11: "relu3_1",
            20: "relu4_1",
            22: "relu4_2",
            29: "relu5_1",
        }

        for idx, layer in enumerate(self.model):
            out = layer(out)
            if idx in layer_map and layer_map[idx] in target_layers:
                features[layer_map[idx]] = out

        return features


# ============================================================================
# Loss Functions
# ============================================================================


class ContentLoss(nn.Module):
    """
    Perceptual content loss using VGG features.

    Measures content similarity at the feature level rather than
    pixel level for better perceptual quality.

    Example:
        >>> content_loss = ContentLoss()
        >>> loss = content_loss(generated_features, target_features)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute content loss.

        Args:
            generated_features: Features from generated image
            target_features: Features from content image

        Returns:
            Content loss value
        """
        loss = 0
        for layer in generated_features:
            loss += F.mse_loss(generated_features[layer], target_features[layer])
        return loss


class StyleLoss(nn.Module):
    """
    Gram matrix-based style loss.

    Captures texture and style information through correlations
    between feature maps at different layers.

    Example:
        >>> style_loss = StyleLoss()
        >>> loss = style_loss(generated_features, target_features)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute style loss using Gram matrices.

        Args:
            generated_features: Features from generated image
            target_features: Features from style image

        Returns:
            Style loss value
        """
        loss = 0
        for layer in generated_features:
            gen_gram = gram_matrix(generated_features[layer])
            target_gram = gram_matrix(target_features[layer])
            loss += F.mse_loss(gen_gram, target_gram)
        return loss


class TotalVariationLoss(nn.Module):
    """
    Total variation loss for smoothness regularization.

    Reduces noise and encourages smooth transitions in the
    generated image.

    Example:
        >>> tv_loss = TotalVariationLoss()
        >>> loss = tv_loss(generated_image)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Total variation loss value
        """
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


class MultiScaleStyleLoss(nn.Module):
    """
    Multi-resolution style loss.

    Computes style loss at multiple scales to capture style
    information at different frequencies.

    Example:
        >>> ms_loss = MultiScaleStyleLoss(scales=[1.0, 0.5, 0.25])
        >>> loss = ms_loss(generated_features, target_features)
    """

    def __init__(self, scales: List[float] = None):
        """
        Initialize multi-scale style loss.

        Args:
            scales: List of scales to compute loss at
        """
        super().__init__()
        self.scales = scales or [1.0, 0.5, 0.25]
        self.style_loss = StyleLoss()

    def forward(
        self,
        generated_features: Dict[str, torch.Tensor],
        target_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute multi-scale style loss.

        Args:
            generated_features: Features from generated image
            target_features: Features from style image

        Returns:
            Multi-scale style loss value
        """
        loss = 0
        for scale in self.scales:
            if scale == 1.0:
                loss += self.style_loss(generated_features, target_features)
            else:
                gen_scaled = {
                    k: F.interpolate(v, scale_factor=scale, mode="bilinear")
                    for k, v in generated_features.items()
                }
                tgt_scaled = {
                    k: F.interpolate(v, scale_factor=scale, mode="bilinear")
                    for k, v in target_features.items()
                }
                loss += self.style_loss(gen_scaled, tgt_scaled)
        return loss / len(self.scales)


# ============================================================================
# Style Transfer Models
# ============================================================================


class GatysStyleTransfer(nn.Module):
    """
    Original Gatys et al. neural style transfer.

    Optimization-based style transfer that iteratively optimizes
    an image to match content and style representations.

    Example:
        >>> model = GatysStyleTransfer()
        >>> stylized = model(content_image, style_image, num_iterations=300)
    """

    def __init__(self, config: StyleTransferConfig = None):
        """
        Initialize Gatys style transfer.

        Args:
            config: Style transfer configuration
        """
        super().__init__()
        self.config = config or StyleTransferConfig()
        self.vgg = VGGFeatures(
            content_layers=self.config.content_layers,
            style_layers=self.config.style_layers,
        )
        self.content_loss_fn = ContentLoss()
        self.style_loss_fn = StyleLoss()
        self.tv_loss_fn = TotalVariationLoss()

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        num_iterations: int = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Perform style transfer via optimization.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)
            num_iterations: Number of optimization steps
            verbose: Whether to print progress

        Returns:
            Stylized image (B, 3, H, W)
        """
        num_iterations = num_iterations or self.config.num_iterations

        # Extract target features
        content_features = self.vgg(normalize_batch(content), "content")
        style_features = self.vgg(normalize_batch(style), "style")

        # Initialize with content image
        generated = content.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([generated], lr=self.config.learning_rate)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Normalize generated image
            gen_norm = normalize_batch(generated)

            # Extract features
            gen_content_features = self.vgg(gen_norm, "content")
            gen_style_features = self.vgg(gen_norm, "style")

            # Compute losses
            content_loss = self.content_loss_fn(gen_content_features, content_features)
            style_loss = self.style_loss_fn(gen_style_features, style_features)
            tv_loss = self.tv_loss_fn(generated)

            total_loss = (
                self.config.content_weight * content_loss
                + self.config.style_weight * style_loss
                + self.config.tv_weight * tv_loss
            )

            total_loss.backward()
            optimizer.step()

            # Clamp to valid range
            with torch.no_grad():
                generated.clamp_(0, 1)

            if verbose and (i + 1) % 50 == 0:
                print(
                    f"Iteration {i + 1}/{num_iterations}: "
                    f"Content={content_loss.item():.4f}, "
                    f"Style={style_loss.item():.4f}"
                )

        return generated.detach()


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization for arbitrary style transfer.

    Transfers the channel-wise mean and variance from style features
    to content features, enabling real-time style transfer.

    Reference: Huang & Belongie (2017)

    Example:
        >>> adain = AdaIN()
        >>> stylized_features = adain(content_features, style_features)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Apply adaptive instance normalization.

        Args:
            content: Content features (B, C, H, W)
            style: Style features (B, C, H, W)
            alpha: Style strength (0 to 1)

        Returns:
            Normalized features
        """
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True) + 1e-6

        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True) + 1e-6

        normalized = (content - content_mean) / content_std
        stylized = normalized * style_std + style_mean

        return alpha * stylized + (1 - alpha) * content


class AdaINStyleTransfer(nn.Module):
    """
    Complete AdaIN style transfer network.

    Encoder-decoder architecture with AdaIN layer for fast
    arbitrary style transfer.

    Example:
        >>> model = AdaINStyleTransfer()
        >>> stylized = model(content_image, style_image)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize AdaIN style transfer network.

        Args:
            alpha: Style interpolation parameter
        """
        super().__init__()
        self.alpha = alpha
        self.encoder = VGGFeatures()
        self.adain = AdaIN()
        self.decoder = self._build_decoder()

    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        decoder = nn.Sequential(
            # 512 -> 256
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # 256 -> 128
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # 128 -> 64
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            # 64 -> 3
            nn.Conv2d(64, 3, 3, padding=1),
        )
        return decoder

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = None
    ) -> torch.Tensor:
        """
        Perform style transfer using AdaIN.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)
            alpha: Style strength (overrides initialization)

        Returns:
            Stylized image (B, 3, H, W)
        """
        alpha = alpha or self.alpha

        # Extract features
        content_norm = normalize_batch(content)
        style_norm = normalize_batch(style)

        content_features = self.encoder(content_norm, "content")["relu4_2"]
        style_features = self.encoder(style_norm, "style")["relu4_2"]

        # Apply AdaIN
        stylized_features = self.adain(content_features, style_features, alpha)

        # Decode
        output = self.decoder(stylized_features)
        return torch.sigmoid(output)


class WCT(nn.Module):
    """
    Whitening and Coloring Transform for style transfer.

    Transfers style by whitening content features and then
    coloring them with style statistics using eigendecomposition.

    Reference: Li et al. (2017)

    Example:
        >>> wct = WCT()
        >>> stylized = wct(content_features, style_features)
    """

    def __init__(self, eps: float = 1e-5):
        """
        Initialize WCT.

        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def whiten(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Whiten features using eigendecomposition.

        Args:
            x: Input features (C, H*W)

        Returns:
            Whitened features, eigenvalues, and eigenvectors
        """
        c, hw = x.size()
        mean = x.mean(dim=1, keepdim=True)
        x_centered = x - mean

        cov = torch.mm(x_centered, x_centered.t()) / hw

        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except:
            # Fallback for singular matrices
            eigenvalues = torch.ones(c, device=x.device)
            eigenvectors = torch.eye(c, device=x.device)

        # Sort in descending order
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Filter small eigenvalues
        eigenvalues = torch.where(
            eigenvalues > self.eps, eigenvalues, torch.ones_like(eigenvalues) * self.eps
        )

        # Whiten
        whitened = torch.mm(
            eigenvectors,
            torch.mm(
                torch.diag(1.0 / torch.sqrt(eigenvalues)),
                torch.mm(eigenvectors.t(), x_centered),
            ),
        )

        return whitened, eigenvalues, eigenvectors, mean

    def color(
        self,
        whitened: torch.Tensor,
        style_eigenvalues: torch.Tensor,
        style_eigenvectors: torch.Tensor,
        style_mean: torch.Tensor,
    ) -> torch.Tensor:
        """
        Color whitened features with style statistics.

        Args:
            whitened: Whitened content features
            style_eigenvalues: Style eigenvalues
            style_eigenvectors: Style eigenvectors
            style_mean: Style mean

        Returns:
            Colored features
        """
        colored = torch.mm(
            style_eigenvectors,
            torch.mm(
                torch.diag(torch.sqrt(style_eigenvalues)),
                torch.mm(style_eigenvectors.t(), whitened),
            ),
        )
        return colored + style_mean

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Apply WCT.

        Args:
            content: Content features (B, C, H, W)
            style: Style features (B, C, H, W)
            alpha: Style strength

        Returns:
            Transformed features
        """
        b, c, h, w = content.size()
        content_flat = content.view(c, -1)
        style_flat = style.view(c, -1)

        # Whiten content and get style stats
        content_whitened, _, _, _ = self.whiten(content_flat)
        _, style_eigenvalues, style_eigenvectors, style_mean = self.whiten(style_flat)

        # Color with style
        stylized = self.color(
            content_whitened, style_eigenvalues, style_eigenvectors, style_mean
        )

        stylized = stylized.view(b, c, h, w)
        return alpha * stylized + (1 - alpha) * content


class WCTStyleTransfer(nn.Module):
    """
    Complete WCT-based style transfer network.

    Multi-level WCT with encoder-decoder architecture.

    Example:
        >>> model = WCTStyleTransfer()
        >>> stylized = model(content_image, style_image)
    """

    def __init__(self, num_levels: int = 5):
        """
        Initialize WCT style transfer.

        Args:
            num_levels: Number of WCT levels
        """
        super().__init__()
        self.num_levels = num_levels
        self.encoder = VGGFeatures()
        self.wct = WCT()
        self.decoders = nn.ModuleList(
            [self._build_decoder() for _ in range(num_levels)]
        )
        self.levels = ["relu5_1", "relu4_1", "relu3_1", "relu2_1", "relu1_1"][
            :num_levels
        ]

    def _build_decoder(self) -> nn.Module:
        """Build decoder for each level."""
        return nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = 0.8
    ) -> torch.Tensor:
        """
        Perform WCT style transfer at multiple levels.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)
            alpha: Style strength

        Returns:
            Stylized image (B, 3, H, W)
        """
        content_norm = normalize_batch(content)
        style_norm = normalize_batch(style)

        result = content

        for level_idx in range(self.num_levels):
            level = self.levels[level_idx]

            # Extract features at this level
            content_features = self.encoder(result if level_idx == 0 else result, "all")
            style_features = self.encoder(style_norm, "all")

            if level in content_features and level in style_features:
                # Apply WCT
                stylized = self.wct(
                    content_features[level], style_features[level], alpha
                )

                # Decode
                if level_idx == 0:
                    result = self.decoders[level_idx](stylized)
                else:
                    # For subsequent levels, refine
                    result = self.decoders[level_idx](stylized)

        return result


class SANet(nn.Module):
    """
    Style-Attentional Network for style transfer.

    Uses attention mechanism to selectively apply style patterns
    based on content structure.

    Reference: Park & Lee (2019)

    Example:
        >>> sanet = SANet(in_channels=512)
        >>> output = sanet(content_features, style_features)
    """

    def __init__(self, in_channels: int = 512):
        """
        Initialize SANet.

        Args:
            in_channels: Number of input channels
        """
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply style attention.

        Args:
            content: Content features (B, C, H, W)
            style: Style features (B, C, H, W)

        Returns:
            Attended features
        """
        b, c, h, w = content.size()

        # Generate query, key, value
        query = self.query_conv(content).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(style).view(b, -1, h * w)
        value = self.value_conv(style).view(b, -1, h * w)

        # Compute attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        # Residual connection
        out = self.gamma * out + content
        return out


class AvatarNet(nn.Module):
    """
    Avatar-Net for multimodal style transfer.

    Enables transfer of multiple styles simultaneously and
    provides style control at different semantic levels.

    Reference: Sheng et al. (2018)

    Example:
        >>> model = AvatarNet()
        >>> stylized = model(content_image, [style1, style2])
    """

    def __init__(self, num_styles: int = 4):
        """
        Initialize AvatarNet.

        Args:
            num_styles: Maximum number of styles to blend
        """
        super().__init__()
        self.num_styles = num_styles
        self.encoder = VGGFeatures()
        self.style_fusion = nn.ModuleList(
            [nn.Conv2d(512 * num_styles, 512, 1) for _ in range(5)]
        )
        self.sanets = nn.ModuleList([SANet(512) for _ in range(5)])
        self.decoder = self._build_decoder()

    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        content: torch.Tensor,
        styles: Union[torch.Tensor, List[torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Perform multimodal style transfer.

        Args:
            content: Content image (B, 3, H, W)
            styles: Style image(s) - single tensor or list
            weights: Weights for style blending

        Returns:
            Stylized image (B, 3, H, W)
        """
        if isinstance(styles, torch.Tensor):
            styles = [styles]

        weights = weights or [1.0 / len(styles)] * len(styles)

        content_norm = normalize_batch(content)
        content_features = self.encoder(content_norm, "content")["relu4_2"]

        # Process styles
        style_features_list = []
        for style in styles:
            style_norm = normalize_batch(style)
            style_feat = self.encoder(style_norm, "style")["relu4_2"]
            style_features_list.append(style_feat)

        # Pad or truncate to num_styles
        while len(style_features_list) < self.num_styles:
            style_features_list.append(style_features_list[-1])
        style_features_list = style_features_list[: self.num_styles]

        # Concatenate and fuse styles
        style_concat = torch.cat(style_features_list, dim=1)
        fused_style = self.style_fusion[0](style_concat)

        # Apply SANet
        attended = self.sanets[0](content_features, fused_style)

        # Decode
        output = self.decoder(attended)
        return output


class ResidualBlock(nn.Module):
    """Residual block for style transfer networks."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class JohnsonStyleNet(nn.Module):
    """
    Johnson et al. fast feed-forward style transfer.

    Fast feed-forward network with residual blocks for real-time
    style transfer with fixed style.

    Reference: Johnson et al. (2016)

    Example:
        >>> model = JohnsonStyleNet()
        >>> stylized = model(content_image)
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, num_res_blocks: int = 5
    ):
        """
        Initialize Johnson style network.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_res_blocks: Number of residual blocks
        """
        super().__init__()

        # Downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 32, 9, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_res_blocks)]
        )

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform fast style transfer.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Stylized image (B, 3, H, W)
        """
        out = self.downsample(x)
        out = self.residuals(out)
        out = self.upsample(out)
        return (out + 1) / 2  # Scale from [-1, 1] to [0, 1]


class LinearStyleTransfer(nn.Module):
    """
    Linear style transfer network.

    Lightweight feed-forward network optimized for mobile devices.

    Example:
        >>> model = LinearStyleTransfer()
        >>> stylized = model(content_image, style_image)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.linear_transfer = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Perform linear style transfer.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)

        Returns:
            Stylized image (B, 3, H, W)
        """
        content_feat = self.encoder(content)
        style_feat = self.style_encoder(style)

        # Resize style features to match content
        if style_feat.size() != content_feat.size():
            style_feat = F.interpolate(
                style_feat, size=content_feat.shape[2:], mode="bilinear"
            )

        combined = torch.cat([content_feat, style_feat], dim=1)
        transferred = self.linear_transfer(combined)
        output = self.decoder(transferred)
        return output


class StyleAttention(nn.Module):
    """Multi-head style attention for transformer-based style transfer."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c, h, w = content.size()

        # Generate Q, K, V
        q = self.query(content).view(b, self.num_heads, self.head_dim, h * w)
        k = self.key(style).view(b, self.num_heads, self.head_dim, h * w)
        v = self.value(style).view(b, self.num_heads, self.head_dim, h * w)

        # Attention
        attn = torch.softmax(
            torch.matmul(q.transpose(-2, -1), k) / (self.head_dim**0.5), dim=-1
        )
        out = torch.matmul(v, attn.transpose(-2, -1))

        out = out.view(b, c, h, w)
        out = self.proj(out)
        return out


class TransformerStyle(nn.Module):
    """
    Transformer-based style transfer.

    Uses transformer attention mechanisms for global style
    and content understanding.

    Example:
        >>> model = TransformerStyle()
        >>> stylized = model(content_image, style_image)
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 6):
        """
        Initialize transformer style transfer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, 4, stride=4)
        self.style_embed = nn.Conv2d(3, embed_dim, 4, stride=4)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.content_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.style_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Style fusion
        self.style_attention = StyleAttention(embed_dim, num_heads)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Perform transformer-based style transfer.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)

        Returns:
            Stylized image (B, 3, H, W)
        """
        # Patch embedding
        content_tokens = self.patch_embed(content)
        style_tokens = self.style_embed(style)

        b, c, h, w = content_tokens.shape

        # Flatten for transformer
        content_seq = content_tokens.flatten(2).permute(0, 2, 1)
        style_seq = style_tokens.flatten(2).permute(0, 2, 1)

        # Encode
        content_encoded = self.content_encoder(content_seq)
        style_encoded = self.style_encoder(style_seq)

        # Reshape back
        content_encoded = content_encoded.permute(0, 2, 1).view(b, c, h, w)
        style_encoded = style_encoded.permute(0, 2, 1).view(b, c, h, w)

        # Apply style attention
        fused = self.style_attention(content_encoded, style_encoded)

        # Decode
        output = self.decoder(fused)
        return output


class MobileStyleTransfer(nn.Module):
    """
    Mobile-optimized style transfer.

    Uses depthwise separable convolutions and efficient
    architectures for real-time performance on mobile devices.

    Example:
        >>> model = MobileStyleTransfer()
        >>> stylized = model(content_image)
    """

    def __init__(self, num_res_blocks: int = 3):
        """
        Initialize mobile style transfer.

        Args:
            num_res_blocks: Number of residual blocks
        """
        super().__init__()

        # Efficient encoder with separable convolutions
        self.encoder = nn.Sequential(
            self._separable_conv(3, 32, 9, padding=4),
            nn.ReLU(inplace=True),
            self._separable_conv(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            self._separable_conv(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Lightweight residual blocks
        self.residuals = nn.Sequential(
            *[self._mobile_res_block(128) for _ in range(num_res_blocks)]
        )

        # Efficient decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 9, padding=4),
            nn.Sigmoid(),
        )

    def _separable_conv(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> nn.Module:
        """Create depthwise separable convolution."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
        )

    def _mobile_res_block(self, channels: int) -> nn.Module:
        """Create mobile-friendly residual block."""
        return nn.Sequential(
            self._separable_conv(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            self._separable_conv(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform mobile style transfer.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Stylized image (B, 3, H, W)
        """
        out = self.encoder(x)
        out = self.residuals(out)
        out = self.decoder(out)
        return out


# ============================================================================
# Arbitrary Style Transfer
# ============================================================================


class StyleTransferNetwork(nn.Module):
    """
    Real-time arbitrary style transfer network.

    Combines encoder-decoder architecture with adaptive normalization
    for fast arbitrary style transfer.

    Example:
        >>> model = StyleTransferNetwork()
        >>> stylized = model(content_image, style_image)
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGFeatures()

        # Style embedding network
        self.style_embedding = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        # Modulated decoder
        self.decoder = self._build_modulated_decoder()

    def _build_modulated_decoder(self) -> nn.Module:
        """Build style-conditioned decoder."""
        return nn.ModuleDict(
            {
                "up1": nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True)
                ),
                "up2": nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                "up3": nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                "up4": nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(64, 3, 3, padding=1),
                    nn.Sigmoid(),
                ),
            }
        )

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Perform real-time style transfer.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)

        Returns:
            Stylized image (B, 3, H, W)
        """
        # Extract features
        content_norm = normalize_batch(content)
        style_norm = normalize_batch(style)

        content_feat = self.encoder(content_norm, "content")["relu4_2"]
        style_feat = self.encoder(style_norm, "style")["relu4_2"]

        # Get style embedding
        style_embed = self.style_embedding(style_feat)

        # Apply AdaIN
        adain = AdaIN()
        stylized_feat = adain(content_feat, style_feat)

        # Decode
        out = self.decoder["up1"](stylized_feat)
        out = self.decoder["up2"](out)
        out = self.decoder["up3"](out)
        out = self.decoder["up4"](out)

        return out


class MetaStyleTransfer(nn.Module):
    """
    Meta-learning based style transfer.

    Uses Model-Agnostic Meta-Learning (MAML) to quickly adapt
    to new styles with few examples.

    Example:
        >>> model = MetaStyleTransfer()
        >>> # Adapt to new style
        >>> adapted_model = model.adapt(style_images, num_steps=5)
    """

    def __init__(self, inner_lr: float = 0.01):
        """
        Initialize meta style transfer.

        Args:
            inner_lr: Learning rate for inner loop adaptation
        """
        super().__init__()
        self.inner_lr = inner_lr
        self.base_model = StyleTransferNetwork()
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def adapt(
        self,
        style_images: torch.Tensor,
        content_images: torch.Tensor,
        num_steps: int = 5,
    ) -> nn.Module:
        """
        Adapt model to new style.

        Args:
            style_images: Style example images
            content_images: Content images for adaptation
            num_steps: Number of adaptation steps

        Returns:
            Adapted model
        """
        adapted_model = self._clone_model()
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        vgg = VGGFeatures()
        style_loss_fn = StyleLoss()
        content_loss_fn = ContentLoss()

        for _ in range(num_steps):
            optimizer.zero_grad()

            # Forward pass
            stylized = adapted_model(content_images, style_images)

            # Compute loss
            content_feat = vgg(normalize_batch(content_images), "content")
            stylized_feat = vgg(normalize_batch(stylized), "all")
            style_feat = vgg(normalize_batch(style_images), "style")

            loss = content_loss_fn(
                {k: stylized_feat[k] for k in content_feat}, content_feat
            ) + 1000 * style_loss_fn(stylized_feat, style_feat)

            loss.backward()
            optimizer.step()

        return adapted_model

    def _clone_model(self) -> nn.Module:
        """Create a copy of the base model."""
        cloned = StyleTransferNetwork()
        cloned.load_state_dict(self.base_model.state_dict())
        return cloned


class ZeroShotStyleTransfer(nn.Module):
    """
    Zero-shot style transfer.

    Transfers style without any training on the specific style,
    using pre-trained features and statistical matching.

    Example:
        >>> model = ZeroShotStyleTransfer()
        >>> stylized = model(content_image, style_image)
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGFeatures()
        self.wct = WCT()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, alpha: float = 0.6
    ) -> torch.Tensor:
        """
        Perform zero-shot style transfer.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)
            alpha: Style strength

        Returns:
            Stylized image (B, 3, H, W)
        """
        content_norm = normalize_batch(content)
        style_norm = normalize_batch(style)

        # Extract features
        content_feat = self.encoder(content_norm, "all")["relu4_2"]
        style_feat = self.encoder(style_norm, "all")["relu4_2"]

        # Apply WCT
        transferred = self.wct(content_feat, style_feat, alpha)

        # Decode
        output = self.decoder(transferred)
        return output


# ============================================================================
# Style Control
# ============================================================================


class StyleInterpolation(nn.Module):
    """
    Interpolate between multiple styles.

    Provides smooth transitions between different styles
    using learned interpolation.

    Example:
        >>> interp = StyleInterpolation()
        >>> stylized = interp(content, [style1, style2], [0.7, 0.3])
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGFeatures()
        self.decoder = self._build_decoder()
        self.interpolation_net = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 512)
        )

    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, content: torch.Tensor, styles: List[torch.Tensor], weights: List[float]
    ) -> torch.Tensor:
        """
        Interpolate between styles.

        Args:
            content: Content image (B, 3, H, W)
            styles: List of style images
            weights: Interpolation weights

        Returns:
            Interpolated stylized image
        """
        content_norm = normalize_batch(content)
        content_feat = self.encoder(content_norm, "content")["relu4_2"]

        # Extract and weight style features
        style_feats = []
        for style, weight in zip(styles, weights):
            style_norm = normalize_batch(style)
            style_feat = self.encoder(style_norm, "style")["relu4_2"]
            style_feats.append(style_feat * weight)

        # Interpolate
        interpolated_style = sum(style_feats)

        # Apply AdaIN
        adain = AdaIN()
        stylized_feat = adain(content_feat, interpolated_style)

        # Decode
        output = self.decoder(stylized_feat)
        return output


class StyleBlending(nn.Module):
    """
    Blend multiple styles spatially or semantically.

    Allows different regions of the image to have different styles.

    Example:
        >>> blender = StyleBlending()
        >>> stylized = blender(content, [style1, style2], masks)
    """

    def __init__(self):
        super().__init__()
        self.encoder = VGGFeatures()
        self.decoder = self._build_decoder()

    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        content: torch.Tensor,
        styles: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Blend styles using spatial masks.

        Args:
            content: Content image (B, 3, H, W)
            styles: List of style images
            masks: List of binary masks for each style

        Returns:
            Blended stylized image
        """
        content_norm = normalize_batch(content)
        content_feat = self.encoder(content_norm, "content")["relu4_2"]

        # Apply each style to masked regions
        stylized_regions = []
        for style, mask in zip(styles, masks):
            style_norm = normalize_batch(style)
            style_feat = self.encoder(style_norm, "style")["relu4_2"]

            adain = AdaIN()
            stylized_feat = adain(content_feat, style_feat)

            # Resize mask to feature size
            mask_resized = F.interpolate(
                mask, size=stylized_feat.shape[2:], mode="bilinear"
            )
            stylized_regions.append(stylized_feat * mask_resized)

        # Combine regions
        combined = sum(stylized_regions)
        mask_sum = sum(
            [F.interpolate(m, size=combined.shape[2:], mode="bilinear") for m in masks]
        )
        combined = combined / (mask_sum + 1e-6)

        # Decode
        output = self.decoder(combined)
        return output


class SpatialControl(nn.Module):
    """
    Region-based style control.

    Apply different styles to different regions of the image
    based on semantic segmentation.

    Example:
        >>> control = SpatialControl()
        >>> stylized = control(content, style, segmentation_mask)
    """

    def __init__(self, num_regions: int = 5):
        """
        Initialize spatial control.

        Args:
            num_regions: Number of semantic regions
        """
        super().__init__()
        self.num_regions = num_regions
        self.encoder = VGGFeatures()
        self.style_transforms = nn.ModuleList([AdaIN() for _ in range(num_regions)])
        self.decoder = self._build_decoder()

    def _build_decoder(self) -> nn.Module:
        """Build decoder network."""
        return nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, segmentation: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply region-based style control.

        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)
            segmentation: Segmentation mask (B, num_regions, H, W)

        Returns:
            Regionally stylized image
        """
        content_norm = normalize_batch(content)
        style_norm = normalize_batch(style)

        content_feat = self.encoder(content_norm, "content")["relu4_2"]
        style_feat = self.encoder(style_norm, "style")["relu4_2"]

        # Resize segmentation to feature size
        seg_resized = F.interpolate(
            segmentation, size=content_feat.shape[2:], mode="bilinear"
        )

        # Apply style transforms per region
        stylized_feat = torch.zeros_like(content_feat)
        for i in range(self.num_regions):
            region_mask = seg_resized[:, i : i + 1]
            region_style = self.style_transforms[i](content_feat, style_feat)
            stylized_feat += region_style * region_mask

        # Decode
        output = self.decoder(stylized_feat)
        return output


class TemporalConsistency(nn.Module):
    """
    Temporal consistency for video style transfer.

    Ensures frame-to-frame consistency in video style transfer
    using temporal constraints.

    Example:
        >>> temporal = TemporalConsistency()
        >>> loss = temporal(current_frame, previous_frame, flow)
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize temporal consistency.

        Args:
            weight: Weight for temporal loss
        """
        super().__init__()
        self.weight = weight

    def warp(self, frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp frame using optical flow.

        Args:
            frame: Frame to warp (B, C, H, W)
            flow: Optical flow (B, 2, H, W)

        Returns:
            Warped frame
        """
        b, c, h, w = frame.size()

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=frame.device),
            torch.linspace(-1, 1, w, device=frame.device),
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)

        # Normalize flow
        flow_norm = flow.clone()
        flow_norm[:, 0] = flow_norm[:, 0] / (w / 2)
        flow_norm[:, 1] = flow_norm[:, 1] / (h / 2)

        # Warp
        grid = grid + flow_norm
        warped = F.grid_sample(frame, grid.permute(0, 2, 3, 1), align_corners=True)

        return warped

    def forward(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        flow: torch.Tensor,
        occlusion_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            current: Current stylized frame
            previous: Previous stylized frame
            flow: Optical flow from previous to current
            occlusion_mask: Occlusion mask

        Returns:
            Temporal consistency loss
        """
        # Warp previous frame
        warped_previous = self.warp(previous, flow)

        # Compute loss
        if occlusion_mask is not None:
            diff = torch.abs(current - warped_previous) * occlusion_mask
        else:
            diff = torch.abs(current - warped_previous)

        loss = torch.mean(diff)
        return self.weight * loss


# ============================================================================
# Color Transfer
# ============================================================================


class HistogramMatching:
    """
    Histogram matching for color transfer.

    Matches the color histogram of source image to target image.

    Example:
        >>> matcher = HistogramMatching()
        >>> result = matcher.match(source, target)
    """

    def match(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Match histogram of source to target.

        Args:
            source: Source image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Histogram-matched image
        """
        return match_histogram(source, target)


class ReinhardColorTransfer:
    """
    Reinhard et al. statistical color transfer.

    Transfers color statistics in LAB color space.

    Example:
        >>> transfer = ReinhardColorTransfer()
        >>> result = transfer.transfer(source, target)
    """

    def rgb_to_lab(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space."""
        # Simplified conversion
        return rgb  # Placeholder

    def lab_to_rgb(self, lab: torch.Tensor) -> torch.Tensor:
        """Convert LAB to RGB color space."""
        # Simplified conversion
        return lab  # Placeholder

    def transfer(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Transfer color statistics.

        Args:
            source: Source image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Color-transferred image
        """
        # Convert to LAB
        source_lab = self.rgb_to_lab(source)
        target_lab = self.rgb_to_lab(target)

        # Compute statistics
        source_mean = source_lab.mean(dim=[2, 3], keepdim=True)
        source_std = source_lab.std(dim=[2, 3], keepdim=True)
        target_mean = target_lab.mean(dim=[2, 3], keepdim=True)
        target_std = target_lab.std(dim=[2, 3], keepdim=True)

        # Transfer
        normalized = (source_lab - source_mean) / (source_std + 1e-6)
        transferred = normalized * target_std + target_mean

        # Convert back to RGB
        result = self.lab_to_rgb(transferred)
        return result


class DeepColorTransfer(nn.Module):
    """
    Neural color transfer.

    Learns color transfer using deep networks.

    Example:
        >>> model = DeepColorTransfer()
        >>> result = model(source, target)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.color_transform = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Perform deep color transfer.

        Args:
            source: Source image (B, 3, H, W)
            target: Target color image (B, 3, H, W)

        Returns:
            Color-transferred image
        """
        combined = torch.cat([source, target], dim=1)
        features = self.encoder(combined)
        color_map = self.color_transform(features)
        return source * color_map


# ============================================================================
# Video Style Transfer
# ============================================================================


class VideoStyleTransfer(nn.Module):
    """
    Consistent video style transfer.

    Applies style transfer to video frames while maintaining
    temporal consistency.

    Example:
        >>> model = VideoStyleTransfer()
        >>> stylized_frames = model(frames, style_image)
    """

    def __init__(self, consistency_weight: float = 1.0):
        """
        Initialize video style transfer.

        Args:
            consistency_weight: Weight for temporal consistency
        """
        super().__init__()
        self.style_transfer = StyleTransferNetwork()
        self.temporal_consistency = TemporalConsistency(consistency_weight)

    def forward(
        self,
        frames: torch.Tensor,
        style: torch.Tensor,
        flows: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Apply style transfer to video frames.

        Args:
            frames: Video frames (T, B, 3, H, W)
            style: Style image (B, 3, H, W)
            flows: Optical flows between consecutive frames

        Returns:
            List of stylized frames
        """
        stylized_frames = []
        previous_frame = None

        for i, frame in enumerate(frames):
            # Apply style transfer
            stylized = self.style_transfer(frame, style)

            # Apply temporal consistency if available
            if previous_frame is not None and flows is not None and i > 0:
                # Compute temporal loss (would be used during training)
                temp_loss = self.temporal_consistency(
                    stylized, previous_frame, flows[i - 1]
                )

            stylized_frames.append(stylized)
            previous_frame = stylized.detach()

        return stylized_frames


class OpticalFlowGuided(nn.Module):
    """
    Optical flow-guided video style transfer.

    Uses optical flow to guide style propagation between frames
    for better temporal consistency.

    Example:
        >>> model = OpticalFlowGuided()
        >>> stylized_frames = model(frames, style_image, flows)
    """

    def __init__(self):
        super().__init__()
        self.style_transfer = StyleTransferNetwork()
        self.flow_warping = TemporalConsistency()

    def forward(
        self, frames: torch.Tensor, style: torch.Tensor, flows: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply flow-guided style transfer.

        Args:
            frames: Video frames (T, B, 3, H, W)
            style: Style image (B, 3, H, W)
            flows: Optical flows between frames

        Returns:
            List of stylized frames
        """
        stylized_frames = []

        # Style first frame
        first_stylized = self.style_transfer(frames[0], style)
        stylized_frames.append(first_stylized)

        # Propagate style using flow
        for i in range(1, len(frames)):
            # Warp previous stylized frame
            warped_style = self.flow_warping.warp(stylized_frames[-1], flows[i - 1])

            # Blend with current frame style
            current_stylized = self.style_transfer(frames[i], style)

            # Create occlusion mask (simplified)
            occlusion_mask = torch.ones_like(current_stylized)

            # Blend
            blended = warped_style * 0.3 + current_stylized * 0.7
            stylized_frames.append(blended)

        return stylized_frames


class RecurrentStyle(nn.Module):
    """
    Recurrent neural network for video style transfer.

    Uses LSTM to maintain temporal coherence across frames.

    Example:
        >>> model = RecurrentStyle()
        >>> stylized_frames = model(frames, style_image)
    """

    def __init__(self, hidden_size: int = 512):
        """
        Initialize recurrent style transfer.

        Args:
            hidden_size: LSTM hidden size
        """
        super().__init__()
        self.style_transfer = StyleTransferNetwork()

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512 * 16 * 16,  # Assuming 16x16 features
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )

        # Feature projection
        self.feature_proj = nn.Linear(hidden_size, 512 * 16 * 16)

    def forward(self, frames: torch.Tensor, style: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply recurrent style transfer.

        Args:
            frames: Video frames (T, B, 3, H, W)
            style: Style image (B, 3, H, W)

        Returns:
            List of stylized frames
        """
        stylized_frames = []
        hidden = None

        for frame in frames:
            # Get features
            stylized = self.style_transfer(frame, style)

            # Flatten for LSTM
            features = stylized.view(stylized.size(0), -1).unsqueeze(1)

            # LSTM forward
            lstm_out, hidden = self.lstm(features, hidden)

            # Project back
            refined = self.feature_proj(lstm_out.squeeze(1))
            refined = refined.view(stylized.size())

            stylized_frames.append(refined)

        return stylized_frames


# ============================================================================
# Training Utilities
# ============================================================================


class StyleDataset(torch.utils.data.Dataset):
    """
    Dataset for style transfer training.

    Provides content-style image pairs for training.

    Example:
        >>> dataset = StyleDataset(
        ...     content_dir='content_images/',
        ...     style_dir='style_images/'
        ... )
        >>> loader = DataLoader(dataset, batch_size=4)
    """

    def __init__(
        self,
        content_dir: Union[str, Path],
        style_dir: Union[str, Path],
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize style dataset.

        Args:
            content_dir: Directory containing content images
            style_dir: Directory containing style images
            image_size: Size to resize images
            transform: Additional transforms to apply
        """
        from pathlib import Path

        self.content_dir = Path(content_dir)
        self.style_dir = Path(style_dir)
        self.image_size = image_size

        self.content_images = list(self.content_dir.glob("*.jpg")) + list(
            self.content_dir.glob("*.png")
        )
        self.style_images = list(self.style_dir.glob("*.jpg")) + list(
            self.style_dir.glob("*.png")
        )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.content_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get content-style pair.

        Args:
            idx: Index

        Returns:
            Tuple of (content_image, style_image)
        """
        from PIL import Image

        # Load content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert("RGB")
        content_tensor = self.transform(content_img)

        # Random style
        style_idx = torch.randint(0, len(self.style_images), (1,)).item()
        style_path = self.style_images[style_idx]
        style_img = Image.open(style_path).convert("RGB")
        style_tensor = self.transform(style_img)

        return content_tensor, style_tensor


class StyleTrainer:
    """
    Specialized trainer for style transfer models.

    Provides training loop with appropriate loss functions
    and monitoring for style transfer.

    Example:
        >>> trainer = StyleTrainer(model, config)
        >>> trainer.train(train_loader, num_epochs=100)
    """

    def __init__(
        self, model: nn.Module, config: StyleTransferConfig = None, device: str = "cuda"
    ):
        """
        Initialize style trainer.

        Args:
            model: Style transfer model
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config or StyleTransferConfig()
        self.device = device

        self.vgg = VGGFeatures().to(device)
        self.content_loss_fn = ContentLoss()
        self.style_loss_fn = StyleLoss()
        self.tv_loss_fn = TotalVariationLoss()

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config.learning_rate
        )

    def train_step(
        self, content: torch.Tensor, style: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            content: Content images
            style: Style images

        Returns:
            Dictionary of losses
        """
        self.optimizer.zero_grad()

        # Forward pass
        stylized = self.model(content, style)

        # Extract features
        content_norm = normalize_batch(content)
        style_norm = normalize_batch(style)
        stylized_norm = normalize_batch(stylized)

        content_features = self.vgg(content_norm, "content")
        style_features = self.vgg(style_norm, "style")
        stylized_features = self.vgg(stylized_norm, "all")

        # Compute losses
        content_loss = self.content_loss_fn(
            {k: stylized_features[k] for k in content_features}, content_features
        )
        style_loss = self.style_loss_fn(stylized_features, style_features)
        tv_loss = self.tv_loss_fn(stylized)

        total_loss = (
            self.config.content_weight * content_loss
            + self.config.style_weight * style_loss
            + self.config.tv_weight * tv_loss
        )

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "content": content_loss.item(),
            "style": style_loss.item(),
            "tv": tv_loss.item(),
        }

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        save_path: str = None,
        log_interval: int = 10,
    ):
        """
        Train model.

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            save_path: Path to save checkpoints
            log_interval: Logging interval
        """
        self.model.train()

        for epoch in range(num_epochs):
            epoch_losses = {"total": 0, "content": 0, "style": 0, "tv": 0}

            for batch_idx, (content, style) in enumerate(train_loader):
                content = content.to(self.device)
                style = style.to(self.device)

                losses = self.train_step(content, style)

                for key in epoch_losses:
                    epoch_losses[key] += losses[key]

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}: Loss={losses['total']:.4f}"
                    )

            # Average losses
            avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
            print(f"Epoch {epoch} - Average losses: {avg_losses}")

            # Save checkpoint
            if save_path:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "losses": avg_losses,
                    },
                    f"{save_path}_epoch_{epoch}.pth",
                )


class StyleEvaluation:
    """
    Evaluation metrics for style transfer.

    Provides quantitative metrics for content preservation
    and style adherence.

    Example:
        >>> evaluator = StyleEvaluation()
        >>> metrics = evaluator.evaluate(stylized, content, style)
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize evaluator.

        Args:
            device: Device for computation
        """
        self.device = device
        self.vgg = VGGFeatures().to(device)
        self.content_loss_fn = ContentLoss()
        self.style_loss_fn = StyleLoss()

    def content_preservation(
        self, stylized: torch.Tensor, content: torch.Tensor
    ) -> float:
        """
        Measure content preservation.

        Args:
            stylized: Stylized image
            content: Content image

        Returns:
            Content preservation score (lower is better)
        """
        stylized_feat = self.vgg(normalize_batch(stylized), "content")
        content_feat = self.vgg(normalize_batch(content), "content")
        loss = self.content_loss_fn(stylized_feat, content_feat)
        return loss.item()

    def style_adherence(self, stylized: torch.Tensor, style: torch.Tensor) -> float:
        """
        Measure style adherence.

        Args:
            stylized: Stylized image
            style: Style image

        Returns:
            Style adherence score (lower is better)
        """
        stylized_feat = self.vgg(normalize_batch(stylized), "style")
        style_feat = self.vgg(normalize_batch(style), "style")
        loss = self.style_loss_fn(stylized_feat, style_feat)
        return loss.item()

    def lpips_score(self, stylized: torch.Tensor, content: torch.Tensor) -> float:
        """
        Compute LPIPS perceptual similarity.

        Args:
            stylized: Stylized image
            content: Content image

        Returns:
            LPIPS score
        """
        try:
            import lpips

            loss_fn = lpips.LPIPS(net="alex").to(self.device)
            with torch.no_grad():
                score = loss_fn(stylized, content)
            return score.mean().item()
        except ImportError:
            return 0.0

    def evaluate(
        self, stylized: torch.Tensor, content: torch.Tensor, style: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            stylized: Stylized image
            content: Content image
            style: Style image

        Returns:
            Dictionary of metrics
        """
        return {
            "content_preservation": self.content_preservation(stylized, content),
            "style_adherence": self.style_adherence(stylized, style),
            "lpips": self.lpips_score(stylized, content),
            "psnr": self.psnr(stylized, content),
        }

    def psnr(self, stylized: torch.Tensor, content: torch.Tensor) -> float:
        """
        Compute PSNR.

        Args:
            stylized: Stylized image
            content: Content image

        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(stylized, content)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


# ============================================================================
# Convenience Functions
# ============================================================================


def stylize_image(
    content_path: str,
    style_path: str,
    output_path: str,
    model_type: str = "gatys",
    **kwargs,
) -> torch.Tensor:
    """
    Convenience function to stylize an image.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Path to save output
        model_type: Type of model to use ('gatys', 'adain', 'johnson', etc.)
        **kwargs: Additional arguments for specific models

    Returns:
        Stylized image tensor
    """
    from PIL import Image

    # Load images
    content_img = Image.open(content_path).convert("RGB")
    style_img = Image.open(style_path).convert("RGB")

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()]
    )

    content_tensor = transform(content_img).unsqueeze(0)
    style_tensor = transform(style_img).unsqueeze(0)

    # Create model
    if model_type == "gatys":
        model = GatysStyleTransfer()
    elif model_type == "adain":
        model = AdaINStyleTransfer()
    elif model_type == "johnson":
        model = JohnsonStyleNet()
    elif model_type == "wct":
        model = WCTStyleTransfer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Stylize
    with torch.no_grad():
        if model_type == "gatys":
            stylized = model(content_tensor, style_tensor, **kwargs)
        else:
            stylized = model(content_tensor, style_tensor)

    # Save output
    output_img = transforms.ToPILImage()(stylized.squeeze(0))
    output_img.save(output_path)

    return stylized


def stylize_video(
    video_path: str,
    style_path: str,
    output_path: str,
    model_type: str = "video",
    fps: int = 30,
    **kwargs,
):
    """
    Convenience function to stylize a video.

    Args:
        video_path: Path to input video
        style_path: Path to style image
        output_path: Path to save output video
        model_type: Type of model to use
        fps: Output frame rate
        **kwargs: Additional arguments
    """
    import cv2
    from PIL import Image

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load style
    style_img = Image.open(style_path).convert("RGB")
    style_tensor = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])(
        style_img
    ).unsqueeze(0)

    # Create model
    if model_type == "video":
        model = VideoStyleTransfer()
    else:
        model = StyleTransferNetwork()

    # Process frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame_rgb)
        frame_tensor = transforms.Compose(
            [transforms.Resize(256), transforms.ToTensor()]
        )(frame_img).unsqueeze(0)

        # Stylize
        with torch.no_grad():
            stylized = model(frame_tensor, style_tensor)

        # Convert back to video frame
        stylized_img = transforms.ToPILImage()(stylized.squeeze(0))
        stylized_array = np.array(stylized_img.resize((width, height)))
        stylized_bgr = cv2.cvtColor(stylized_array, cv2.COLOR_RGB2BGR)

        out.write(stylized_bgr)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Configuration
    "StyleTransferConfig",
    # Feature extraction
    "VGGFeatures",
    # Loss functions
    "ContentLoss",
    "StyleLoss",
    "TotalVariationLoss",
    "MultiScaleStyleLoss",
    # Style transfer models
    "GatysStyleTransfer",
    "AdaIN",
    "AdaINStyleTransfer",
    "WCT",
    "WCTStyleTransfer",
    "SANet",
    "AvatarNet",
    "JohnsonStyleNet",
    "LinearStyleTransfer",
    "TransformerStyle",
    "MobileStyleTransfer",
    # Arbitrary style transfer
    "StyleTransferNetwork",
    "MetaStyleTransfer",
    "ZeroShotStyleTransfer",
    # Style control
    "StyleInterpolation",
    "StyleBlending",
    "SpatialControl",
    "TemporalConsistency",
    # Color transfer
    "HistogramMatching",
    "ReinhardColorTransfer",
    "DeepColorTransfer",
    # Video style transfer
    "VideoStyleTransfer",
    "OpticalFlowGuided",
    "RecurrentStyle",
    # Training utilities
    "StyleDataset",
    "StyleTrainer",
    "StyleEvaluation",
    # Utility functions
    "gram_matrix",
    "normalize_batch",
    "unnormalize_batch",
    "match_histogram",
    "stylize_image",
    "stylize_video",
]
