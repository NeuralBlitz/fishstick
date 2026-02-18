"""
Surface Normal Estimation Module

Estimating surface normals from single images for scene understanding.
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SurfaceNormalEncoder(nn.Module):
    """
    Encoder for surface normal estimation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
        """
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
    ) -> nn.Module:
        """Create a layer with residual blocks."""
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Multi-scale features
        """
        x = self.initial(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return [c1, c2, c3, c4]


class NormalDecoder(nn.Module):
    """
    Decoder for surface normal prediction.
    """

    def __init__(
        self,
        encoder_channels: List[int],
    ):
        """
        Args:
            encoder_channels: Channel dimensions from encoder
        """
        super().__init__()

        encoder_channels = encoder_channels[::-1]

        self.up_convs = nn.ModuleList()
        self.refine_convs = nn.ModuleList()

        for i in range(len(encoder_channels) - 1):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]

            self.up_convs.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

            self.refine_convs.append(nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

    def forward(
        self,
        encoder_features: List[Tensor],
    ) -> Tensor:
        """
        Args:
            encoder_features: Multi-scale features

        Returns:
            Decoded features
        """
        encoder_features = encoder_features[::-1]

        x = encoder_features[0]

        for i, (up_conv, refine_conv) in enumerate(
            zip(self.up_convs, self.refine_convs)
        ):
            x = up_conv(x)

            skip = encoder_features[i + 1]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

            x = torch.cat([x, skip], dim=1)
            x = refine_conv(x)

        return x


class SurfaceNormalEstimator(nn.Module):
    """
    Complete surface normal estimation network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
        """
        super().__init__()

        self.encoder = SurfaceNormalEncoder(in_channels, base_channels)

        encoder_channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        self.decoder = NormalDecoder(encoder_channels)

        self.normal_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Surface normals [B, 3, H, W] (normalized)
        """
        features = self.encoder(x)
        decoded = self.decoder(features)

        normal = self.normal_head(decoded)

        normal = F.normalize(normal, p=2, dim=1)

        return normal


class NormalRefinementModule(nn.Module):
    """
    Refine surface normals using edge-aware smoothing.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels (normals + image)
            hidden_channels: Hidden layer channels
        """
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )

    def forward(
        self,
        normals: Tensor,
        image: Tensor,
    ) -> Tensor:
        """
        Args:
            normals: Initial normal prediction [B, 3, H, W]
            image: Input image [B, 3, H, W]

        Returns:
            Refined normals
        """
        if image.shape[2:] != normals.shape[2:]:
            image = F.interpolate(
                image,
                size=normals.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        combined = torch.cat([normals, image], dim=1)
        residual = self.refine(combined)

        refined = normals + residual
        refined = F.normalize(refined, p=2, dim=1)

        return refined


class ConfidenceWeightedNormals(nn.Module):
    """
    Predict surface normals with confidence weighting.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
        """
        super().__init__()

        self.encoder = SurfaceNormalEncoder(in_channels, base_channels)

        encoder_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.decoder = NormalDecoder(encoder_channels)

        self.normal_head = nn.Conv2d(base_channels * 2, 3, 1)
        self.confidence_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Tuple of (normals, confidence)
        """
        features = self.encoder(x)
        decoded = self.decoder(features)

        normals = self.normal_head(decoded)
        normals = F.normalize(normals, p=2, dim=1)

        confidence = self.confidence_head(decoded)

        return normals, confidence


class NormalFromDepthConsistency(nn.Module):
    """
    Derive surface normals from depth using consistency constraints.
    """

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        """
        Args:
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, depth: Tensor, focal_length: float = 1.0) -> Tensor:
        """
        Compute surface normals from depth map.

        Args:
            depth: Depth map [B, 1, H, W]
            focal_length: Camera focal length

        Returns:
            Surface normals [B, 3, H, W]
        """
        B, _, H, W = depth.shape

        dx, dy = self._compute_gradients(depth)

        normal_x = -dx * focal_length
        normal_y = -dy * focal_length
        normal_z = torch.ones_like(depth)

        normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
        normals = F.normalize(normals, p=2, dim=1)

        return normals

    def _compute_gradients(self, depth: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute depth gradients."""
        grad_x = depth[:, :, :, :-1] - depth[:, :, :, 1:]
        grad_y = depth[:, :, :-1, :] - depth[:, :, 1:, :]

        pad_x = nn.functional.pad(grad_x, (1, 0, 0, 0), mode='replicate')
        pad_y = nn.functional.pad(grad_y, (0, 0, 1, 0), mode='replicate')

        return pad_x, pad_y


class EdgeAwareNormalSmoothing(nn.Module):
    """
    Smooth normals while preserving edges using bilateral filtering.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma_color: float = 0.1,
        sigma_spatial: float = 1.0,
    ):
        """
        Args:
            kernel_size: Size of bilateral filter kernel
            sigma_color: Color space sigma
            sigma_spatial: Spatial sigma
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial

    def forward(self, normals: Tensor, image: Tensor) -> Tensor:
        """
        Args:
            normals: Surface normals [B, 3, H, W]
            image: Input image [B, 3, H, W]

        Returns:
            Smoothed normals
        """
        B, C, H, W = normals.shape
        pad = self.kernel_size // 2

        padded_normals = F.pad(normals, [pad] * 4, mode='replicate')
        padded_image = F.pad(image, [pad] * 4, mode='replicate')

        x_range = torch.arange(-pad, pad + 1, device=normals.device, dtype=torch.float32)
        spatial_kernel = torch.exp(
            -(x_range**2 + x_range**2.unsqueeze(0)) / (2 * self.sigma_spatial**2)
        )
        spatial_kernel = spatial_kernel / spatial_kernel.sum()

        smoothed = torch.zeros_like(normals)
        weight_sum = torch.zeros(B, 1, H, W, device=normals.device)

        for i in range(H):
            for j in range(W):
                patch_normals = padded_normals[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                patch_image = padded_image[:, :, i:i + self.kernel_size, j:j + self.kernel_size]

                center_image = image[:, :, i:i + 1, j:j + 1]
                color_diff = (patch_image - center_image).pow(2).sum(dim=1, keepdim=True)
                color_kernel = torch.exp(-color_diff / (2 * self.sigma_color**2))

                kernel = spatial_kernel.unsqueeze(0).unsqueeze(0) * color_kernel
                kernel = kernel / (kernel.sum(dim=(2, 3), keepdim=True) + 1e-8)

                smoothed[:, :, i:i + 1, j:j + 1] = (patch_normals * kernel).sum(dim=(2, 3), keepdim=True)
                weight_sum[:, :, i:i + 1, j:j + 1] = kernel.sum(dim=(2, 3), keepdim=True)

        smoothed = smoothed / (weight_sum + 1e-8)
        smoothed = F.normalize(smoothed, p=2, dim=1)

        return smoothed


class MultiScaleNormalPrediction(nn.Module):
    """
    Predict surface normals at multiple scales with feature fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
        """
        super().__init__()

        self.encoder = SurfaceNormalEncoder(in_channels, base_channels)

        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels * 2, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1),
            )
            for _ in range(4)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * 4, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Fused normal prediction [B, 3, H, W]
        """
        features = self.encoder(x)
        features = features[::-1]

        target_size = x.shape[2:]

        scale_preds = []
        for i, head in enumerate(self.scale_heads):
            pred = head(features[i])

            if pred.shape[2:] != target_size:
                pred = F.interpolate(
                    pred,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )

            scale_preds.append(pred)

        combined = torch.cat(scale_preds, dim=1)
        fused = self.fusion(combined)
        fused = F.normalize(fused, p=2, dim=1)

        return fused


class NormalizationLoss(nn.Module):
    """
    Loss function that encourages normalized normals.
    """

    def __init__(self):
        super().__init__()

    def forward(self, normals: Tensor) -> Tensor:
        """
        Args:
            normals: Predicted normals [B, 3, H, W]

        Returns:
            Normalization loss
        """
        norm = torch.norm(normals, p=2, dim=1, keepdim=True)
        return F.mse_loss(norm, torch.ones_like(norm))


def create_normal_estimator(
    model_type: str = 'standard',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create normal estimators.

    Args:
        model_type: Type of model ('standard', 'confident', 'multiscale')
        **kwargs: Additional model arguments

    Returns:
        Normal estimator model
    """
    if model_type == 'standard':
        return SurfaceNormalEstimator(
            in_channels=kwargs.get('in_channels', 3),
            base_channels=kwargs.get('base_channels', 64),
        )
    elif model_type == 'confident':
        return ConfidenceWeightedNormals(
            in_channels=kwargs.get('in_channels', 3),
            base_channels=kwargs.get('base_channels', 64),
        )
    elif model_type == 'multiscale':
        return MultiScaleNormalPrediction(
            in_channels=kwargs.get('in_channels', 3),
            base_channels=kwargs.get('base_channels', 64),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
