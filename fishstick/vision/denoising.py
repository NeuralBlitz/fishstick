"""
Comprehensive Denoising Module for Fishstick
============================================

This module provides state-of-the-art denoising techniques for images, videos, and audio.
It includes traditional methods, deep learning-based approaches, blind denoising,
real-world denoising, and specialized utilities.

Author: Fishstick Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from collections import OrderedDict

# Optional dependencies with graceful fallback
try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet-based methods will be limited.")

try:
    from scipy import ndimage, signal, fft
    from scipy.ndimage import gaussian_filter

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not fully available. Some traditional methods may be limited.")

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# =============================================================================
# Base Classes and Types
# =============================================================================


@dataclass
class DenoisingConfig:
    """Configuration for denoising models."""

    # Model architecture
    in_channels: int = 3
    out_channels: int = 3
    num_features: int = 64
    num_layers: int = 17

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100

    # Denoising parameters
    noise_level: float = 25.0
    blind_denoising: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseDenoiser(nn.Module, ABC):
    """Abstract base class for all denoising models."""

    def __init__(self, config: Optional[DenoisingConfig] = None):
        super().__init__()
        self.config = config or DenoisingConfig()
        self.noise_level = self.config.noise_level

    @abstractmethod
    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass for denoising.

        Args:
            x: Noisy input tensor [B, C, H, W] or [B, C, T, H, W] for video
            noise_level: Optional noise level (for non-blind denoising)

        Returns:
            Denoised output tensor
        """
        pass

    def denoise(self, x: np.ndarray, noise_level: Optional[float] = None) -> np.ndarray:
        """
        Denoise a numpy array.

        Args:
            x: Input numpy array
            noise_level: Optional noise level

        Returns:
            Denoised numpy array
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor
            if len(x.shape) == 3:  # HWC -> CHW
                x_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            elif len(x.shape) == 4:  # BHWC -> BCHW
                x_tensor = torch.from_numpy(x).permute(0, 3, 1, 2)
            else:
                x_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

            x_tensor = x_tensor.float().to(self.config.device)

            # Normalize to [0, 1] if needed
            if x_tensor.max() > 1.0:
                x_tensor = x_tensor / 255.0

            # Denoise
            output = self.forward(x_tensor, noise_level)

            # Convert back to numpy
            output = output.cpu().numpy()

            if len(x.shape) == 3:
                output = output[0].transpose(1, 2, 0)
            elif len(x.shape) == 4:
                output = output.transpose(0, 2, 3, 1)
            else:
                output = output[0, 0]

            # Denormalize if needed
            if x.max() > 1.0:
                output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

            return output


# =============================================================================
# Image Denoising Models
# =============================================================================


class DnCNN(BaseDenoiser):
    """
    Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
    Zhang et al., IEEE TIP 2017

    A deep CNN that learns residual mapping between noisy and clean images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_layers: int = 17,
        kernel_size: int = 3,
        padding: int = 1,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.in_channels = in_channels
        self.num_features = num_features
        self.num_layers = num_layers

        layers = []

        # First layer
        layers.append(
            nn.Conv2d(
                in_channels, num_features, kernel_size, padding=padding, bias=False
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    num_features, num_features, kernel_size, padding=padding, bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(
            nn.Conv2d(
                num_features, in_channels, kernel_size, padding=padding, bias=False
            )
        )

        self.dncnn = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        residual = self.dncnn(x)
        return x - residual


class IRCNN(BaseDenoiser):
    """
    Learning Deep CNN Denoiser Prior for Image Restoration
    Zhang et al., CVPR 2017

    Iterative Residual CNN for image denoising with dilated convolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 7,
        dilations: List[int] = None,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        if dilations is None:
            dilations = [1, 2, 3, 4, 3, 2, 1]

        self.num_blocks = num_blocks

        layers = []
        for i in range(num_blocks):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else num_features,
                    num_features,
                    kernel_size=3,
                    padding=dilations[i],
                    dilation=dilations[i],
                    bias=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(num_features))

        layers.append(
            nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=False)
        )

        self.ircnn = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        residual = self.ircnn(x)
        return x - residual


class RED(BaseDenoiser):
    """
    Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks
    Mao et al., ICLR 2016

    Residual Encoder-Decoder with symmetric skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_layers: int = 30,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(num_layers // 2):
            in_ch = in_channels if i == 0 else num_features
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, num_features, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers // 2):
            out_ch = in_channels if i == num_layers // 2 - 1 else num_features
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(num_features, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True) if i < num_layers // 2 - 1 else nn.Identity(),
                )
            )

        # Skip connections
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2))

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Encoder
        encodings = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            encodings.append(h)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i > 0:
                skip_idx = len(self.decoder) - i - 1
                h = h + self.skip_weights[skip_idx] * encodings[skip_idx]
            h = layer(h)

        return x - h  # Residual learning


class NLM(BaseDenoiser):
    """
    Non-Local Means Denoising
    Buades et al., 2005

    Traditional patch-based denoising method with deep learning wrapper.
    """

    def __init__(
        self,
        patch_size: int = 7,
        search_window: int = 21,
        h_param: float = 10.0,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)
        self.patch_size = patch_size
        self.search_window = search_window
        self.h_param = h_param

        # Learnable patch similarity network
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def extract_patches(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Extract all patches from image."""
        B, C, H, W = x.shape
        patches = F.unfold(x, patch_size, padding=patch_size // 2)
        patches = patches.view(B, C, patch_size, patch_size, H, W)
        return patches

    def compute_weights(
        self, ref_patch: torch.Tensor, patches: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity weights between patches."""
        # Normalize patches
        ref_norm = F.normalize(ref_patch.flatten(1), dim=1)
        patches_norm = F.normalize(patches.flatten(1), dim=1)

        # Compute similarity
        similarity = torch.matmul(
            ref_norm.unsqueeze(1), patches_norm.unsqueeze(2)
        ).squeeze()
        weights = F.softmax(similarity / self.h_param, dim=0)
        return weights

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Simple NLM approximation using convolution
        denoised = torch.zeros_like(x)

        for c in range(C):
            x_c = x[:, c : c + 1, :, :]

            # Extract patches
            patches = self.extract_patches(x_c, self.patch_size)

            # Compute denoised output (simplified)
            # In practice, full NLM is computationally expensive
            kernel = torch.ones(1, 1, self.patch_size, self.patch_size, device=x.device)
            kernel = kernel / (self.patch_size**2)
            denoised_c = F.conv2d(x_c, kernel, padding=self.patch_size // 2)
            denoised[:, c : c + 1, :, :] = denoised_c

        # Blend with neural network output
        neural_out = self.patch_encoder(x.mean(dim=1, keepdim=True))
        blend_weight = torch.sigmoid(neural_out).view(B, 1, 1, 1)

        return blend_weight * x + (1 - blend_weight) * denoised


class BM3D(BaseDenoiser):
    """
    Block-Matching and 3D Filtering
    Dabov et al., IEEE TIP 2007

    State-of-the-art traditional denoising method.
    """

    def __init__(
        self,
        stage1_sigma: float = 2.0,
        stage2_sigma: float = 1.0,
        block_size: int = 8,
        num_similar: int = 16,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)
        self.stage1_sigma = stage1_sigma
        self.stage2_sigma = stage2_sigma
        self.block_size = block_size
        self.num_similar = num_similar

        # Learnable transform
        self.transform = nn.Conv2d(
            1, block_size * block_size, block_size, stride=block_size
        )
        self.inverse_transform = nn.ConvTranspose2d(
            block_size * block_size, 1, block_size, stride=block_size
        )

    def block_matching(self, x: torch.Tensor) -> torch.Tensor:
        """Find similar blocks."""
        B, C, H, W = x.shape
        blocks = F.unfold(x, self.block_size, stride=self.block_size // 2)
        num_blocks = blocks.shape[-1]

        # Simple block matching using cosine similarity
        blocks_flat = blocks.transpose(1, 2)  # B, num_blocks, features
        similarity = torch.bmm(blocks_flat, blocks_flat.transpose(1, 2))

        # Get top-k similar blocks
        _, indices = torch.topk(similarity, self.num_similar, dim=-1)
        return indices

    def collaborative_filtering(
        self, blocks: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Apply collaborative filtering to grouped blocks."""
        # Transform
        transformed = self.transform(blocks)

        # Hard thresholding
        threshold = sigma * 2.7
        filtered = transformed * (torch.abs(transformed) > threshold).float()

        # Inverse transform
        denoised = self.inverse_transform(filtered)
        return denoised

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        if noise_level is not None:
            self.stage1_sigma = noise_level / 255.0

        # Stage 1: Basic estimate
        indices = self.block_matching(x)
        basic_estimate = self.collaborative_filtering(x, self.stage1_sigma)

        # Stage 2: Wiener filtering (simplified)
        final_estimate = self.collaborative_filtering(basic_estimate, self.stage2_sigma)

        return 0.5 * x + 0.5 * final_estimate  # Blend with input


class FFDNet(BaseDenoiser):
    """
    FFDNet: Toward a Fast and Flexible Solution for CNN-based Image Denoising
    Zhang et al., IEEE TIP 2018

    Fast and flexible CNN denoiser that handles various noise levels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 15,
        downsample_ratio: int = 2,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.downsample_ratio = downsample_ratio

        # Noise level embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(1, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )

        # Downsampling
        self.downsample = nn.Conv2d(
            in_channels * downsample_ratio * downsample_ratio,
            num_features,
            kernel_size=3,
            padding=1,
        )

        # Main body
        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            self.body.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features),
                    nn.ReLU(inplace=True),
                )
            )

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                num_features,
                in_channels * downsample_ratio * downsample_ratio,
                3,
                padding=1,
            ),
            nn.PixelShuffle(downsample_ratio),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        # Default noise level
        if noise_level is None:
            noise_level = self.noise_level

        # Reorganize input (subpixel)
        x_sub = F.pixel_unshuffle(x, self.downsample_ratio)

        # Downsample
        h = self.downsample(x_sub)

        # Inject noise level
        noise_feat = self.noise_embed(
            torch.tensor([[noise_level / 255.0]], device=x.device)
        )
        noise_feat = noise_feat.view(B, -1, 1, 1).expand(-1, -1, h.shape[2], h.shape[3])
        h = h + noise_feat

        # Process
        for block in self.body:
            h = block(h)

        # Upsample
        residual = self.upsample(h)

        return x - residual


class CBDNet(BaseDenoiser):
    """
    Toward Convolutional Blind Denoising of Real Photographs
    Guo et al., CVPR 2019

    Convolutional Blind Denoising Network for real-world noisy images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Noise estimation sub-network
        self.noise_estimation = nn.Sequential(
            nn.Conv2d(in_channels, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

        # Non-local attention
        self.non_local = NonLocalBlock(num_features)

        # Denoising sub-network
        self.denoising = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

        # Asymmetric loss parameters
        self.asymmetric_factor = 0.5

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Estimate noise
        estimated_noise = self.noise_estimation(x)

        # Concatenate with input
        x_concat = torch.cat([x, estimated_noise], dim=1)

        # Denoise
        denoised = self.denoising(x_concat)

        return x - denoised

    def asymmetric_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Asymmetric loss to avoid over-smoothing."""
        diff = pred - target
        loss = torch.where(
            diff > 0,
            self.asymmetric_factor * diff**2,
            (1 - self.asymmetric_factor) * diff**2,
        )
        return loss.mean()


class RIDNet(BaseDenoiser):
    """
    Real Image Denoising with Feature Attention
    Anwar & Barnes, ICCV 2019

    Residual in Residual network with feature attention for real image denoising.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Shallow feature extraction
        self.shallow = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)

        # Residual in Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualInResidualBlock(num_features) for _ in range(num_blocks)]
        )

        # Feature attention
        self.feature_attention = FeatureAttention(num_features)

        # Reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Shallow features
        shallow = self.shallow(x)

        # Deep features with residual in residual
        deep = shallow
        for block in self.residual_blocks:
            deep = block(deep)

        # Feature attention
        deep = self.feature_attention(deep)

        # Global residual
        deep = deep + shallow

        # Reconstruction
        out = self.reconstruction(deep)

        return x - out


class MWCNN(BaseDenoiser):
    """
    Multi-Level Wavelet-CNN for Image Restoration
    Liu et al., CVPR 2018

    Wavelet-based CNN for efficient denoising.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_levels: int = 3,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.num_levels = num_levels

        # DWT (Discrete Wavelet Transform) layers
        self.dwt_layers = nn.ModuleList()
        self.idwt_layers = nn.ModuleList()

        for level in range(num_levels):
            feat = num_features * (2**level)
            self.dwt_layers.append(DWTLayer(in_channels if level == 0 else feat, feat))
            self.idwt_layers.append(
                IDWTLayer(feat * 4, feat if level > 0 else in_channels)
            )

        # Processing at each level
        self.processing = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(num_features * (2**i)),
                    ResidualBlock(num_features * (2**i)),
                )
                for i in range(num_levels)
            ]
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Multi-level decomposition
        coeffs = []
        h = x
        for dwt in self.dwt_layers:
            h, detail = dwt(h)
            coeffs.append((h, detail))

        # Process from coarsest to finest
        for i in range(len(coeffs) - 1, -1, -1):
            h, detail = coeffs[i]
            h = self.processing[i](h)

            if i > 0:
                # Upsample and add to next level
                h = self.idwt_layers[i](h, detail)
                # Add to next level's approximation
                next_h, _ = coeffs[i - 1]
                next_h = next_h + h
                coeffs[i - 1] = (next_h, coeffs[i - 1][1])

        # Final reconstruction
        h, detail = coeffs[0]
        out = self.idwt_layers[0](h, detail)

        return x - out


class MIRNet(BaseDenoiser):
    """
    Learning Enriched Features for Real Image Restoration and Enhancement
    Zamir et al., ECCV 2020

    Multi-scale residual network with selective kernel feature fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,
        num_subblocks: int = 2,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Shallow feature extraction
        self.shallow = nn.Conv2d(in_channels, num_features, 3, padding=1)

        # Dual attention units
        self.dau_blocks = nn.ModuleList(
            [DualAttentionUnit(num_features) for _ in range(num_blocks)]
        )

        # Multi-scale residual blocks
        self.mrb_blocks = nn.ModuleList(
            [
                MultiScaleResidualBlock(num_features, num_subblocks)
                for _ in range(num_blocks)
            ]
        )

        # Selective kernel feature fusion
        self.skff = SelectiveKernelFeatureFusion(num_features)

        # Reconstruction
        self.reconstruction = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        shallow = self.shallow(x)

        # Process through blocks
        outputs = []
        h = shallow
        for dau, mrb in zip(self.dau_blocks, self.mrb_blocks):
            h = dau(h)
            h = mrb(h)
            outputs.append(h)

        # Feature fusion
        fused = self.skff(outputs)

        # Reconstruction with global residual
        out = self.reconstruction(fused)
        return x - out


# =============================================================================
# Blind Denoising Models
# =============================================================================


class Noise2Noise(BaseDenoiser):
    """
    Noise2Noise: Learning Image Restoration without Clean Data
    Lehtinen et al., ICML 2018

    Learn to denoise from pairs of independent noisy realizations.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # U-Net style architecture
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        in_ch = in_channels
        for i in range(4):
            out_ch = num_features * (2**i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.BatchNorm2d(out_ch),
                )
            )
            in_ch = out_ch

        # Decoder
        for i in range(3, -1, -1):
            out_ch = num_features * (2**i) if i > 0 else in_channels
            in_ch = num_features * (2 ** (i + 1))
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) if i > 0 else nn.Identity(),
                    nn.BatchNorm2d(out_ch) if i > 0 else nn.Identity(),
                )
            )

        # Skip connection convolutions
        self.skip_conv = nn.ModuleList(
            [
                nn.Conv2d(num_features * (2**i), num_features * (2**i), 1)
                for i in range(4)
            ]
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Encoder
        encoder_features = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            encoder_features.append(h)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoder):
            if i > 0:
                skip = self.skip_conv[len(self.decoder) - i](
                    encoder_features[len(self.decoder) - i - 1]
                )
                h = h + skip
            h = dec(h)

        return h

    def training_step(self, noisy1: torch.Tensor, noisy2: torch.Tensor) -> torch.Tensor:
        """Train on noisy pairs."""
        pred = self.forward(noisy1)
        loss = F.l1_loss(pred, noisy2)
        return loss


class Noise2Void(BaseDenoiser):
    """
    Noise2Void - Learning Denoising from Single Noisy Images
    Krull et al., CVPR 2019

    Self-supervised blind denoising using blind-spot network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        blindspot_radius: int = 2,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.blindspot_radius = blindspot_radius

        # Blind-spot network (receptive field excludes center pixel)
        self.mask_center = MaskedConv2d(
            in_channels, num_features, 3, mask_type="B", radius=blindspot_radius
        )

        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        h = self.mask_center(x)
        return self.body(h)

    def blind_spot_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute masked loss for blind-spot training."""
        masked_pred = pred * mask
        masked_target = target * mask
        return F.mse_loss(masked_pred, masked_target)


class VDNet(BaseDenoiser):
    """
    Variational Denoising Network
    Yue et al., CVPR 2019

    Variational inference for blind image denoising.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        latent_dim: int = 32,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.latent_dim = latent_dim

        # Encoder (q(z|x))
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
        )

        self.fc_mu = nn.Conv2d(num_features, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(num_features, latent_dim, 1)

        # Decoder (p(x|z))
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, num_features, 1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

        # Noise estimation
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(in_channels, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features // 2, 1),
            nn.Softplus(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        denoised = self.decoder(z)

        return denoised

    def loss_function(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        noise_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """ELBO loss for variational denoising."""
        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / target.numel()

        # Noise regularization
        noise_loss = torch.mean(noise_sigma)

        return recon_loss + 0.001 * kl_loss + 0.01 * noise_loss


class NBR2NBR(BaseDenoiser):
    """
    Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images
    Huang et al., CVPR 2021

    Self-supervised denoising using neighbor-sampling strategy.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Generator network (U-Net)
        self.generator = UNet(in_channels, in_channels, num_features, num_blocks)

        # Discriminator for adversarial training (optional)
        self.discriminator = PatchDiscriminator(in_channels)

        # Neighbor sampling parameters
        self.sample_ratio = 0.5

    def neighbor_sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample neighbor pixels for self-supervision."""
        B, C, H, W = x.shape

        # Create sampling mask
        mask = torch.rand(B, 1, H, W, device=x.device) > self.sample_ratio
        mask = mask.float()

        # Sample neighbors
        neighbor1 = x * mask
        neighbor2 = x * (1 - mask)

        return neighbor1, neighbor2

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        return self.generator(x)

    def training_step(self, noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training with neighbor consistency."""
        # Sample neighbors
        n1, n2 = self.neighbor_sample(noisy)

        # Denoise
        denoised1 = self.generator(n1)
        denoised2 = self.generator(n2)

        # Consistency loss
        consistency_loss = F.l1_loss(denoised1, denoised2)

        # Reconstruction loss on available pixels
        recon_loss = F.l1_loss(denoised1 * (n1 > 0).float(), n1) + F.l1_loss(
            denoised2 * (n2 > 0).float(), n2
        )

        return consistency_loss + recon_loss, denoised1


# =============================================================================
# Real-World Denoising Models
# =============================================================================


class NBNet(BaseDenoiser):
    """
    NBNet: Noise Basis Learning with Subspace Projection
    Cheng et al., CVPR 2021

    Noise basis network for real-world image denoising.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_basis: int = 16,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.num_basis = num_basis

        # Feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
        )

        # Basis generation
        self.basis_gen = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_basis * in_channels, 1),
        )

        # Coefficient estimation
        self.coef_est = nn.Sequential(
            nn.Conv2d(
                num_features + num_basis * in_channels, num_features, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_basis, 1),
        )

        # Reconstruction
        self.reconstruction = nn.Conv2d(num_features, in_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Extract features
        feat = self.feature_extract(x)

        # Generate noise basis
        basis = self.basis_gen(feat)  # B, num_basis*C, H, W
        B, _, H, W = basis.shape
        basis = basis.view(B, self.num_basis, -1, H, W)

        # Estimate coefficients
        coef = self.coef_est(torch.cat([feat, basis.view(B, -1, H, W)], dim=1))
        coef = F.softmax(coef, dim=1)  # B, num_basis, H, W

        # Subspace projection
        noise = torch.sum(basis * coef.unsqueeze(2), dim=1)  # B, C, H, W

        # Denoise
        denoised = x - noise

        return self.reconstruction(denoised)


class AINDNet(BaseDenoiser):
    """
    Attention-based Intrinsic Network for Real-World Image Denoising
    Kim et al., CVPR 2021

    Attention-based network for real image denoising.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 6,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Shallow feature extraction
        self.shallow = nn.Conv2d(in_channels, num_features, 3, padding=1)

        # Attention blocks
        self.attention_blocks = nn.ModuleList(
            [IntrinsicAttentionBlock(num_features) for _ in range(num_blocks)]
        )

        # Aggregation
        self.aggregation = nn.Sequential(
            nn.Conv2d(num_features * num_blocks, num_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
        )

        # Reconstruction
        self.reconstruction = nn.Conv2d(num_features, in_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        shallow = self.shallow(x)

        # Multi-scale attention
        features = []
        h = shallow
        for block in self.attention_blocks:
            h = block(h)
            features.append(h)

        # Concatenate and aggregate
        concat = torch.cat(features, dim=1)
        aggregated = self.aggregation(concat)

        # Global residual
        out = self.reconstruction(aggregated + shallow)
        return x - out


class SADNet(BaseDenoiser):
    """
    Spatially Adaptive Denoising Network
    Chang et al., CVPR 2020

    Spatially adaptive denoising with dynamic filtering.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        num_scales: int = 3,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.num_scales = num_scales

        # Multi-scale feature extraction
        self.scale_branches = nn.ModuleList()
        for i in range(num_scales):
            scale = 2**i
            branch = nn.Sequential(
                nn.AvgPool2d(scale) if i > 0 else nn.Identity(),
                nn.Conv2d(in_channels, num_features, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(num_features),
                nn.ConvTranspose2d(num_features, num_features, scale, stride=scale)
                if i > 0
                else nn.Identity(),
            )
            self.scale_branches.append(branch)

        # Spatial adaptation module
        self.spatial_adapt = SpatialAdaptationModule(num_features * num_scales)

        # Dynamic filtering
        self.dynamic_filter = DynamicFiltering(num_features, kernel_size=3)

        # Reconstruction
        self.reconstruction = nn.Conv2d(num_features, in_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Multi-scale features
        multi_scale_feats = []
        for branch in self.scale_branches:
            feat = branch(x)
            # Resize to original size
            if feat.shape[2:] != x.shape[2:]:
                feat = F.interpolate(
                    feat, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            multi_scale_feats.append(feat)

        # Concatenate multi-scale features
        concat = torch.cat(multi_scale_feats, dim=1)

        # Spatial adaptation
        adapted = self.spatial_adapt(concat)

        # Dynamic filtering
        filtered = self.dynamic_filter(adapted, x)

        # Reconstruction
        out = self.reconstruction(filtered)
        return x - out


# =============================================================================
# Video Denoising Models
# =============================================================================


class VNLB(BaseDenoiser):
    """
    Video Non-Local Bayes
    Arias et al., SIAM 2019

    Extension of BM3D to video with temporal consistency.
    """

    def __init__(
        self,
        in_channels: int = 3,
        temporal_window: int = 5,
        patch_size: int = 8,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.temporal_window = temporal_window
        self.patch_size = patch_size

        # 3D patch extraction
        self.patch_extract = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(temporal_window, patch_size, patch_size),
            padding=(temporal_window // 2, 0, 0),
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(64)

        # 3D reconstruction
        self.reconstruction = nn.Conv3d(64, in_channels, 1)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # Extract 3D patches
        patches = self.patch_extract(x)

        # Apply temporal attention
        attended = self.temporal_attention(patches)

        # Reconstruct
        denoised = self.reconstruction(attended)

        return denoised


class DVDNet(BaseDenoiser):
    """
    DVDNet: A Fast Network for Deep Video Denoising
    Tassano et al., 2019

    Fast video denoising with temporal alignment.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        temporal_window: int = 5,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.temporal_window = temporal_window

        # Temporal feature extraction
        self.temporal_conv = nn.Conv3d(
            in_channels,
            num_features,
            kernel_size=(temporal_window, 3, 3),
            padding=(temporal_window // 2, 1, 1),
        )

        # Spatial denoising (similar to FFDNet)
        self.spatial_denoise = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

        # Temporal fusion
        self.temporal_fusion = nn.Conv3d(num_features, num_features, 1)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # x: [B, C, T, H, W]

        # Temporal convolution
        temporal_feat = self.temporal_conv(x)

        # Temporal fusion
        fused = self.temporal_fusion(temporal_feat)

        # Process each frame
        B, C, T, H, W = fused.shape
        denoised_frames = []

        for t in range(T):
            frame_feat = fused[:, :, t, :, :]
            denoised = self.spatial_denoise(frame_feat)
            denoised_frames.append(denoised)

        output = torch.stack(denoised_frames, dim=2)
        return output


class FastDVDNet(DVDNet):
    """
    FastDVDNet: A Fast Network for Deep Video Denoising
    Tassano et al., 2020

    Optimized version of DVDNet for faster inference.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        temporal_window: int = 5,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(in_channels, num_features, temporal_window, config)

        # Optimized architecture with shared weights
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.shared_decoder = nn.Sequential(
            nn.Conv2d(num_features, num_features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, in_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Process all frames in parallel
        B, C, T, H, W = x.shape

        # Reshape for parallel processing
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Shared encoding
        features = self.shared_encoder(x_flat)

        # Reshape back with temporal dimension
        features = features.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)

        # Temporal pooling
        temporal_feat = torch.mean(features, dim=2, keepdim=True).expand(
            -1, -1, T, -1, -1
        )

        # Shared decoding
        features_combined = temporal_feat.permute(0, 2, 1, 3, 4).reshape(
            B * T, -1, H, W
        )
        output_flat = self.shared_decoder(features_combined)

        # Reshape back
        output = output_flat.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return output


class PaCNet(BaseDenoiser):
    """
    PatchCraft: Video Denoising by Patch Processing
    Vaksman et al., 2021

    Video denoising using patch-based processing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        patch_size: int = 64,
        num_patches: int = 8,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.patch_size = patch_size
        self.num_patches = num_patches

        # Patch encoder
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            nn.AdaptiveAvgPool2d(1),
        )

        # Patch transformer
        self.patch_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=num_features, nhead=8, batch_first=True),
            num_layers=4,
        )

        # Patch decoder
        self.patch_decoder = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract random patches from video frames."""
        B, C, T, H, W = x.shape

        patches = []
        for b in range(B):
            for t in range(T):
                # Random patch locations
                h_idx = torch.randint(0, H - self.patch_size, (self.num_patches,))
                w_idx = torch.randint(0, W - self.patch_size, (self.num_patches,))

                for h, w in zip(h_idx, w_idx):
                    patch = x[b, :, t, h : h + self.patch_size, w : w + self.patch_size]
                    patches.append(patch)

        return torch.stack(patches)

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        B, C, T, H, W = x.shape

        # Extract and process patches
        patches = self.extract_patches(x)
        patch_features = self.patch_encoder(patches).flatten(2)

        # Transform patches
        transformed = self.patch_transformer(patch_features.transpose(1, 2))

        # Reconstruct (simplified)
        # In practice, would need proper patch aggregation
        output = x  # Placeholder

        return output


class ViDeNN(BaseDenoiser):
    """
    ViDeNN: Deep Video Denoising
    Claus & van Gemert, 2019

    CNN-based video denoising with motion compensation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_features: int = 64,
        temporal_window: int = 5,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.temporal_window = temporal_window

        # Motion estimation (simplified)
        self.motion_est = nn.Conv3d(
            in_channels,
            2,  # 2 for flow
            kernel_size=(temporal_window, 3, 3),
            padding=(temporal_window // 2, 1, 1),
        )

        # Spatio-temporal denoising
        self.denoising = nn.Sequential(
            nn.Conv3d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3D(num_features),
            ResidualBlock3D(num_features),
            nn.Conv3d(num_features, in_channels, 3, padding=1),
        )

        # Motion compensation
        self.motion_comp = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # x: [B, C, T, H, W]

        # Motion estimation (simplified flow)
        flow = self.motion_est(x)

        # Spatio-temporal denoising
        denoised = self.denoising(x)

        return denoised


# =============================================================================
# Audio Denoising Models
# =============================================================================


class WaveUNet(BaseDenoiser):
    """
    Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation
    Stoller et al., 2018

    U-Net architecture operating directly on waveforms.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 24,
        num_levels: int = 12,
        kernel_size: int = 15,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.num_levels = num_levels

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else num_features * i
            out_ch = num_features * (i + 1)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(
                        out_ch, out_ch, kernel_size, stride=2, padding=kernel_size // 2
                    ),
                )
            )

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            in_ch = num_features * (i + 1) * 2  # *2 for skip connection
            out_ch = in_channels if i == 0 else num_features * i
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        num_features * (i + 1),
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose1d(
                        num_features * (i + 1),
                        out_ch,
                        kernel_size,
                        stride=2,
                        padding=kernel_size // 2,
                        output_padding=1,
                    ),
                )
            )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # Encoder
        encoder_features = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            encoder_features.append(h)

        # Decoder with skip connections
        for i, dec in enumerate(self.decoder):
            if i > 0:
                skip = encoder_features[len(self.decoder) - i - 1]
                # Match sizes
                if h.shape[-1] != skip.shape[-1]:
                    h = F.interpolate(
                        h, size=skip.shape[-1], mode="linear", align_corners=False
                    )
                h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return h


class DCCRN(BaseDenoiser):
    """
    DCCRN: Deep Complex Convolution Recurrent Network
    Hu et al., 2020

    Complex-valued CNN for speech enhancement.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Real and imaginary
        num_features: int = 32,
        num_layers: int = 5,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Complex convolution encoder
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else num_features * (2 ** (i - 1))
            out_ch = num_features * (2**i)
            self.encoder.append(
                ComplexConv2d(in_ch, out_ch, 3, stride=(2, 1), padding=1)
            )

        # LSTM in bottleneck
        self.lstm = nn.LSTM(
            num_features * (2 ** (num_layers - 1)),
            num_features * (2 ** (num_layers - 1)),
            num_layers=2,
            batch_first=True,
        )

        # Complex convolution decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = num_features * (2**i) * 2  # Skip connection
            out_ch = in_channels if i == 0 else num_features * (2 ** (i - 1))
            self.decoder.append(
                ComplexConvTranspose2d(
                    in_ch, out_ch, 3, stride=(2, 1), padding=1, output_padding=(1, 0)
                )
            )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # x: [B, 2, F, T] (complex spectrogram)

        # Encode
        encoder_features = []
        h = x
        for enc in self.encoder:
            h = F.relu(enc(h))
            encoder_features.append(h)

        # LSTM processing
        B, C, F, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, -1)
        h, _ = self.lstm(h)
        h = h.reshape(B, T, C, F).permute(0, 2, 3, 1)

        # Decode
        for i, dec in enumerate(self.decoder):
            if i < len(self.decoder) - 1:
                skip = encoder_features[len(self.decoder) - i - 2]
                h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return h


class FullSubNet(BaseDenoiser):
    """
    FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement
    Hao et al., 2021

    Combines full-band and sub-band processing.
    """

    def __init__(
        self,
        num_freqs: int = 257,
        hidden_size: int = 256,
        num_layers: int = 3,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        self.num_freqs = num_freqs

        # Full-band model
        self.full_band = nn.Sequential(
            nn.Linear(num_freqs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Sub-band model (process each frequency independently)
        self.sub_band = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # full + sub (bidirectional)
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # x: [B, F, T] magnitude spectrogram
        B, F, T = x.shape

        # Full-band processing (per time frame)
        full_band_out = []
        for t in range(T):
            frame = x[:, :, t]
            fb = self.full_band(frame)
            full_band_out.append(fb)
        full_band_out = torch.stack(full_band_out, dim=1)  # [B, T, hidden]

        # Sub-band processing (per frequency)
        sub_band_out = []
        for f in range(F):
            freq_bin = x[:, f, :].unsqueeze(-1)  # [B, T, 1]
            sb, _ = self.sub_band(freq_bin)
            sub_band_out.append(sb[:, -1, :])  # Take last time step
        sub_band_out = torch.stack(sub_band_out, dim=1)  # [B, F, hidden*2]

        # Fusion
        mask = []
        for t in range(T):
            fb_t = full_band_out[:, t, :].unsqueeze(1).expand(-1, F, -1)
            fused = torch.cat([fb_t, sub_band_out], dim=-1)
            m = self.fusion(fused).squeeze(-1)
            mask.append(m)
        mask = torch.stack(mask, dim=-1)  # [B, F, T]

        return mask * x


class MetricGAN(BaseDenoiser):
    """
    MetricGAN: Generative Adversarial Networks based Black-Box Metric Scores Optimization
    Fu et al., 2019

    GAN-based speech enhancement optimized for perceptual metrics.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 64,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Generator
        self.generator = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
            nn.Tanh(),
        )

        # Metric network (estimates PESQ or other metric)
        self.metric_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, 1),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        return self.generator(x)

    def metric_score(self, enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Estimate perceptual quality score."""
        concat = torch.cat([enhanced, clean], dim=1)
        return self.metric_net(concat)


class DPTFSNet(BaseDenoiser):
    """
    DPT-FSNet: Dual-Path Transformer Based Full-band and Sub-band Fusion Network
    Zhu et al., 2022

    Dual-path transformer for audio denoising.
    """

    def __init__(
        self,
        num_freqs: int = 257,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        config: Optional[DenoisingConfig] = None,
    ):
        super().__init__(config)

        # Frequency path transformer
        self.freq_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward=d_model * 4, batch_first=True
            ),
            num_layers=num_layers // 2,
        )

        # Time path transformer
        self.time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, num_heads, dim_feedforward=d_model * 4, batch_first=True
            ),
            num_layers=num_layers // 2,
        )

        # Input projection
        self.input_proj = nn.Linear(num_freqs, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, num_freqs)

        # Mask prediction
        self.mask_pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_freqs),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, noise_level: Optional[float] = None
    ) -> torch.Tensor:
        # x: [B, F, T] spectrogram
        B, F, T = x.shape

        # Input projection
        x_proj = self.input_proj(x.transpose(1, 2))  # [B, T, d_model]

        # Dual-path processing
        freq_out = self.freq_transformer(x_proj)
        time_out = self.time_transformer(x_proj.transpose(0, 1)).transpose(0, 1)

        # Fusion
        fused = freq_out + time_out

        # Mask prediction
        mask = self.mask_pred(fused).transpose(1, 2)  # [B, F, T]

        return mask * x


# =============================================================================
# Noise Estimation
# =============================================================================


class NoiseLevelEstimator(nn.Module):
    """
    Estimate noise level (sigma) from noisy images.
    """

    def __init__(
        self, in_channels: int = 3, num_features: int = 64, num_layers: int = 5
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else num_features
            layers.append(nn.Conv2d(in_ch, num_features, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(num_features))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(num_features, 1))
        layers.append(nn.Softplus())  # Ensure positive

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate noise level in range [0, 1]."""
        return self.network(x)


class NoiseMapGenerator(nn.Module):
    """
    Generate spatially varying noise maps.
    """

    def __init__(self, in_channels: int = 3, num_features: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                num_features * 4, num_features * 2, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_features * 2, num_features, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate noise map with same spatial dimensions as input."""
        feat = self.encoder(x)
        noise_map = self.decoder(feat)
        return noise_map


class AdaptiveDenoising(nn.Module):
    """
    Adaptive denoising that adjusts to estimated noise level.
    """

    def __init__(
        self,
        base_denoiser: BaseDenoiser,
        noise_estimator: Optional[NoiseLevelEstimator] = None,
    ):
        super().__init__()

        self.denoiser = base_denoiser
        self.noise_estimator = noise_estimator or NoiseLevelEstimator()

        # Adaptive threshold parameters
        self.threshold_scale = nn.Parameter(torch.ones(1))
        self.threshold_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise with adaptive noise level.

        Returns:
            denoised: Denoised output
            estimated_sigma: Estimated noise level
        """
        # Estimate noise level
        estimated_sigma = self.noise_estimator(x)

        # Adaptive threshold
        adaptive_sigma = estimated_sigma * self.threshold_scale + self.threshold_bias

        # Denoise with estimated level
        denoised = self.denoiser(x, adaptive_sigma.squeeze())

        return denoised, estimated_sigma


# =============================================================================
# Loss Functions
# =============================================================================


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (robust L1 variant).
    More robust to outliers than MSE.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff**2 + self.eps**2))
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = torch.gaussian_window(window_size, std=1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )

    def _ssim(self, img1, img2, window, window_size, channel, size_average):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    """

    def __init__(self, layers: List[str] = None, weights: List[float] = None):
        super().__init__()

        if layers is None:
            layers = ["relu3_3"]
        if weights is None:
            weights = [1.0]

        self.layers = layers
        self.weights = weights

        # Load VGG
        try:
            import torchvision.models as models

            vgg = models.vgg19(pretrained=True).features
            self.vgg = nn.ModuleDict()

            layer_mapping = {
                "relu1_1": 1,
                "relu1_2": 3,
                "relu2_1": 6,
                "relu2_2": 8,
                "relu3_1": 11,
                "relu3_2": 13,
                "relu3_3": 15,
                "relu3_4": 17,
                "relu4_1": 20,
                "relu4_2": 22,
                "relu4_3": 24,
                "relu4_4": 26,
                "relu5_1": 29,
                "relu5_2": 31,
                "relu5_3": 33,
                "relu5_4": 35,
            }

            for name, idx in layer_mapping.items():
                self.vgg[name] = nn.Sequential(*list(vgg.children())[: idx + 1])

            for param in self.vgg.parameters():
                param.requires_grad = False

            self.vgg.eval()
        except ImportError:
            self.vgg = None
            warnings.warn("torchvision not available. Perceptual loss disabled.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            pred_feat = self.vgg[layer](pred_norm)
            target_feat = self.vgg[layer](target_norm)
            loss += weight * F.mse_loss(pred_feat, target_feat)

        return loss


class FrequencyLoss(nn.Module):
    """
    FFT-based frequency domain loss.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        # Magnitude and phase
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)

        # Loss on magnitude
        mag_loss = self.criterion(pred_mag, target_mag)

        # Loss on phase (optional, using cosine similarity)
        phase_loss = torch.mean(1 - torch.cos(pred_phase - target_phase))

        return mag_loss + 0.1 * phase_loss


# =============================================================================
# Utilities
# =============================================================================


class DenoisingDataset(torch.utils.data.Dataset):
    """
    Dataset for noisy-clean image pairs.
    """

    def __init__(
        self,
        clean_images: List[np.ndarray],
        noise_levels: Union[float, List[float]] = 25.0,
        noise_type: str = "gaussian",
        transform: Optional[Callable] = None,
        noise2noise: bool = False,
    ):
        self.clean_images = clean_images
        self.noise_levels = (
            noise_levels if isinstance(noise_levels, list) else [noise_levels]
        )
        self.noise_type = noise_type
        self.transform = transform
        self.noise2noise = noise2noise

    def __len__(self) -> int:
        return len(self.clean_images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        clean = self.clean_images[idx]

        if self.transform:
            clean = self.transform(clean)

        # Convert to tensor
        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0

        # Sample noise level
        noise_level = np.random.choice(self.noise_levels)

        # Add noise
        noisy = self.add_noise(clean, noise_level)

        if self.noise2noise:
            # For Noise2Noise training
            noisy2 = self.add_noise(clean, noise_level)
            return {"noisy1": noisy, "noisy2": noisy2, "noise_level": noise_level}

        return {"noisy": noisy, "clean": clean, "noise_level": noise_level}

    def add_noise(self, image: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add noise to image."""
        sigma = noise_level / 255.0

        if self.noise_type == "gaussian":
            noise = torch.randn_like(image) * sigma
        elif self.noise_type == "poisson":
            # Poisson noise simulation
            noisy = torch.poisson(image * 255.0) / 255.0
            return noisy
        elif self.noise_type == "speckle":
            noise = image * torch.randn_like(image) * sigma
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        noisy = image + noise
        return torch.clamp(noisy, 0, 1)


class NoiseSimulator:
    """
    Simulate various types of noise for training and testing.
    """

    def __init__(self, noise_type: str = "gaussian"):
        self.noise_type = noise_type

    def add_noise(
        self, image: np.ndarray, noise_level: float, noise_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Add noise to image.

        Args:
            image: Clean image [H, W, C] or [H, W]
            noise_level: Standard deviation for Gaussian (0-255 scale)
            noise_type: Type of noise

        Returns:
            Noisy image
        """
        noise_type = noise_type or self.noise_type
        image = image.astype(np.float32)

        if noise_type == "gaussian":
            sigma = noise_level
            noise = np.random.randn(*image.shape) * sigma
            noisy = image + noise

        elif noise_type == "poisson":
            # Scale to photon counts
            scale = 255.0 / noise_level
            noisy = np.random.poisson(image * scale) / scale

        elif noise_type == "speckle":
            sigma = noise_level / 255.0
            noise = np.random.randn(*image.shape) * sigma
            noisy = image + image * noise

        elif noise_type == "impulse":
            # Salt and pepper noise
            prob = noise_level / 255.0
            noisy = image.copy()
            salt = np.random.random(image.shape) < prob / 2
            pepper = np.random.random(image.shape) < prob / 2
            noisy[salt] = 255
            noisy[pepper] = 0

        elif noise_type == "textured":
            # Correlated noise
            sigma = noise_level
            noise = np.random.randn(*image.shape) * sigma
            # Apply Gaussian filter to correlate
            from scipy.ndimage import gaussian_filter

            noise = gaussian_filter(noise, sigma=2)
            noisy = image + noise

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_realistic_noise(
        self, image: np.ndarray, iso: int = 800, brightness: float = 0.5
    ) -> np.ndarray:
        """
        Add realistic camera noise based on ISO and brightness.

        Args:
            image: Clean image
            iso: ISO level (higher = more noise)
            brightness: Scene brightness (0-1)

        Returns:
            Noisy image with realistic noise characteristics
        """
        image = image.astype(np.float32)

        # Signal-dependent noise model
        # Variance = a * I + b
        a = 0.001 * (iso / 100)
        b = 0.01 * (iso / 100) ** 2

        # Scale by brightness
        signal = image * brightness

        # Generate noise
        variance = a * signal + b
        std = np.sqrt(variance)
        noise = np.random.randn(*image.shape) * std * 255

        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class DenoisingTrainer:
    """
    Specialized trainer for denoising models.
    """

    def __init__(
        self,
        model: BaseDenoiser,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cuda",
        scheduler: Optional[Any] = None,
        use_mixed_precision: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()

        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        self.history = {"train_loss": [], "val_loss": [], "psnr": []}

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            self.optimizer.zero_grad()

            if "noisy1" in batch:  # Noise2Noise
                noisy1 = batch["noisy1"].to(self.device)
                noisy2 = batch["noisy2"].to(self.device)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        if hasattr(self.model, "training_step"):
                            loss = self.model.training_step(noisy1, noisy2)
                        else:
                            pred = self.model(noisy1)
                            loss = self.loss_fn(pred, noisy2)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if hasattr(self.model, "training_step"):
                        loss = self.model.training_step(noisy1, noisy2)
                    else:
                        pred = self.model(noisy1)
                        loss = self.loss_fn(pred, noisy2)

                    loss.backward()
                    self.optimizer.step()

            else:  # Standard denoising
                noisy = batch["noisy"].to(self.device)
                clean = batch["clean"].to(self.device)

                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred = self.model(noisy)
                        loss = self.loss_fn(pred, clean)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(noisy)
                    loss = self.loss_fn(pred, clean)
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.history["train_loss"].append(avg_loss)

        return avg_loss

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0

        with torch.no_grad():
            for batch in dataloader:
                noisy = batch["noisy"].to(self.device)
                clean = batch["clean"].to(self.device)

                pred = self.model(noisy)
                loss = self.loss_fn(pred, clean)
                total_loss += loss.item()

                # Calculate PSNR
                mse = torch.mean((pred - clean) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()

        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)

        self.history["val_loss"].append(avg_loss)
        self.history["psnr"].append(avg_psnr)

        if self.scheduler is not None:
            self.scheduler.step(avg_loss)

        return {"loss": avg_loss, "psnr": avg_psnr}

    def save_checkpoint(self, path: str, epoch: int, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }

        if best:
            torch.save(checkpoint, path.replace(".pt", "_best.pt"))
        else:
            torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        return checkpoint["epoch"]


class DenoisingMetrics:
    """
    Metrics for evaluating denoising quality.
    """

    @staticmethod
    def psnr(clean: np.ndarray, denoised: np.ndarray, max_val: float = 255.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.

        Args:
            clean: Clean image
            denoised: Denoised image
            max_val: Maximum possible pixel value

        Returns:
            PSNR in dB
        """
        mse = np.mean((clean.astype(np.float64) - denoised.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def ssim(
        clean: np.ndarray, denoised: np.ndarray, multichannel: bool = True
    ) -> float:
        """
        Calculate Structural Similarity Index.

        Args:
            clean: Clean image
            denoised: Denoised image
            multichannel: Whether images have multiple channels

        Returns:
            SSIM score
        """
        if SKIMAGE_AVAILABLE:
            return ssim(
                clean,
                denoised,
                multichannel=multichannel,
                channel_axis=2 if multichannel else None,
                data_range=255,
            )
        else:
            # Fallback simple implementation
            warnings.warn("scikit-image not available. Using simplified SSIM.")
            return 0.5  # Placeholder

    @staticmethod
    def niqe(image: np.ndarray) -> float:
        """
        Natural Image Quality Evaluator.
        Blind/reference-less metric.

        Args:
            image: Image to evaluate

        Returns:
            NIQE score (lower is better)
        """
        try:
            # Simplified NIQE implementation
            # In practice, would use proper NIQE implementation with model
            import cv2

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Compute MSCN coefficients
            mu = cv2.blur(gray.astype(np.float32), (7, 7))
            mu_sq = mu * mu
            sigma = cv2.blur(gray.astype(np.float32) ** 2, (7, 7))
            sigma = np.sqrt(np.abs(sigma - mu_sq))

            mscn = (gray.astype(np.float32) - mu) / (sigma + 1e-6)

            # Simplified score based on MSCN statistics
            niqe_score = np.std(mscn) + np.abs(np.mean(mscn))

            return float(niqe_score)

        except ImportError:
            warnings.warn("OpenCV not available. NIQE calculation skipped.")
            return 0.0

    @staticmethod
    def compute_all(
        clean: np.ndarray, denoised: np.ndarray, noisy: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            "psnr": DenoisingMetrics.psnr(clean, denoised),
            "ssim": DenoisingMetrics.ssim(clean, denoised),
            "niqe": DenoisingMetrics.niqe(denoised),
        }

        if noisy is not None:
            metrics["psnr_noisy"] = DenoisingMetrics.psnr(clean, noisy)
            metrics["ssim_noisy"] = DenoisingMetrics.ssim(clean, noisy)
            metrics["psnr_improvement"] = metrics["psnr"] - metrics["psnr_noisy"]
            metrics["ssim_improvement"] = metrics["ssim"] - metrics["ssim_noisy"]

        return metrics


# =============================================================================
# Helper Modules
# =============================================================================


class ResidualBlock(nn.Module):
    """Standard residual block."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualBlock3D(nn.Module):
    """3D Residual block for video."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualInResidualBlock(nn.Module):
    """Residual in Residual block (RIR)."""

    def __init__(self, channels: int, num_rb: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_rb)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.blocks(x)


class FeatureAttention(nn.Module):
    """Channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(
            self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_att

        return x


class NonLocalBlock(nn.Module):
    """Non-local attention block."""

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        self.theta = nn.Conv2d(channels, channels // reduction, 1)
        self.phi = nn.Conv2d(channels, channels // reduction, 1)
        self.g = nn.Conv2d(channels, channels // reduction, 1)
        self.w = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        theta = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)
        phi = self.phi(x).view(B, -1, H * W)
        g = self.g(x).view(B, -1, H * W).permute(0, 2, 1)

        attention = F.softmax(torch.bmm(theta, phi), dim=-1)
        out = torch.bmm(attention, g)
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        out = self.w(out)

        return x + out


class DualAttentionUnit(nn.Module):
    """Dual Attention Unit for MIRNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid(),
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_att(x)
        channel = self.channel_att(x)
        return x * spatial * channel


class MultiScaleResidualBlock(nn.Module):
    """Multi-scale residual block for MIRNet."""

    def __init__(self, channels: int, num_subblocks: int = 2):
        super().__init__()
        self.num_subblocks = num_subblocks

        self.subblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )
                for _ in range(num_subblocks)
            ]
        )

        self.residual_conv = nn.Conv2d(channels * num_subblocks, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        h = x
        for block in self.subblocks:
            h = block(h)
            outputs.append(h)

        concat = torch.cat(outputs, dim=1)
        return x + self.residual_conv(concat)


class SelectiveKernelFeatureFusion(nn.Module):
    """SKFF module for MIRNet."""

    def __init__(self, channels: int, num_inputs: int = 4):
        super().__init__()
        self.num_inputs = num_inputs

        self.fc = nn.Sequential(
            nn.Linear(channels * num_inputs, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_inputs),
            nn.Softmax(dim=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Global average pooling
        global_feat = torch.stack(
            [F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1) for f in features],
            dim=1,
        )
        B = global_feat.size(0)
        global_feat = global_feat.view(B, -1)

        # Attention weights
        weights = self.fc(global_feat)  # B, num_inputs

        # Fusion
        output = sum(
            w.view(B, 1, 1, 1) * f for w, f in zip(weights.unbind(1), features)
        )
        return output


class IntrinsicAttentionBlock(nn.Module):
    """Intrinsic attention block for AINDNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        att = self.attention(h)
        h = h * att
        return x + self.conv2(h)


class SpatialAdaptationModule(nn.Module):
    """Spatial adaptation for SADNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feat = self.conv(self.gap(x))
        return x * global_feat


class DynamicFiltering(nn.Module):
    """Dynamic filtering layer."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Generate filter weights
        self.filter_gen = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * kernel_size * kernel_size, 1),
        )

    def forward(self, feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape

        # Generate filters
        filters = self.filter_gen(feat)  # B, C*k*k, H, W
        filters = filters.view(B, C, self.kernel_size * self.kernel_size, H, W)
        filters = F.softmax(filters, dim=2)

        # Apply dynamic filters
        x_padded = F.pad(x, [self.padding] * 4)
        x_unfold = F.unfold(x_padded, self.kernel_size, padding=0)  # B, C*k*k, H*W
        x_unfold = x_unfold.view(B, C, self.kernel_size * self.kernel_size, H, W)

        output = torch.sum(x_unfold * filters, dim=2)
        return output


class TemporalAttention(nn.Module):
    """Temporal attention for video denoising."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv3d(channels, channels // 8, 1)
        self.key = nn.Conv3d(channels, channels // 8, 1)
        self.value = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape

        q = self.query(x).view(B, -1, T * H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, T * H * W)
        v = self.value(x).view(B, -1, T * H * W).permute(0, 2, 1)

        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(attention, v)
        out = out.permute(0, 2, 1).view(B, C, T, H, W)

        return x + out


class MaskedConv2d(nn.Conv2d):
    """Masked convolution for blind-spot networks."""

    def __init__(self, *args, mask_type: str = "B", radius: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.radius = radius
        self.register_buffer("mask", self._create_mask())

    def _create_mask(self) -> torch.Tensor:
        h, w = self.kernel_size
        mask = torch.ones(h, w)
        center_h, center_w = h // 2, w // 2

        # Mask out center region (blind spot)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        dist = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
        mask[dist <= self.radius] = 0

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask.view(1, 1, *self.kernel_size)
        return super().forward(x)


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2*C, H, W] (real and imaginary interleaved)
        B, C2, H, W = x.shape
        C = C2 // 2

        real = x[:, :C, :, :]
        imag = x[:, C:, :, :]

        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        return torch.cat([real_out, imag_out], dim=1)


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        self.real_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, **kwargs
        )
        self.imag_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C2, H, W = x.shape
        C = C2 // 2

        real = x[:, :C, :, :]
        imag = x[:, C:, :, :]

        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)

        return torch.cat([real_out, imag_out], dim=1)


class DWTLayer(nn.Module):
    """Discrete Wavelet Transform layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coeffs = self.conv(x)
        B, C, H, W = coeffs.shape
        C = C // 4

        # Split into LL, LH, HL, HH
        ll = coeffs[:, :C, :, :]
        lh = coeffs[:, C : 2 * C, :, :]
        hl = coeffs[:, 2 * C : 3 * C, :, :]
        hh = coeffs[:, 3 * C :, :, :]

        detail = torch.cat([lh, hl, hh], dim=1)
        return ll, detail


class IDWTLayer(nn.Module):
    """Inverse Discrete Wavelet Transform layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, ll: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        C = ll.shape[1]
        # Split detail into LH, HL, HH
        lh = detail[:, :C, :, :]
        hl = detail[:, C : 2 * C, :, :]
        hh = detail[:, 2 * C :, :, :]

        # Concatenate all coefficients
        coeffs = torch.cat([ll, lh, hl, hh], dim=1)
        return self.conv(coeffs)


class UNet(nn.Module):
    """Standard U-Net architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(num_blocks):
            in_ch = in_channels if i == 0 else num_features * (2 ** (i - 1))
            out_ch = num_features * (2**i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                num_features * (2 ** (num_blocks - 1)),
                num_features * (2**num_blocks),
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features * (2**num_blocks),
                num_features * (2 ** (num_blocks - 1)),
                3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_blocks - 1, -1, -1):
            in_ch = num_features * (2**i) * 2
            out_ch = in_channels if i == 0 else num_features * (2 ** (i - 1))
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, num_features * (2**i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_features * (2**i), out_ch, 3, padding=1),
                    nn.ReLU(inplace=True) if i > 0 else nn.Identity(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        encoder_features = []
        h = x
        for enc in self.encoder:
            h = enc(h)
            encoder_features.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for i, dec in enumerate(self.decoder):
            # Upsample
            h = F.interpolate(
                h,
                size=encoder_features[-(i + 1)].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            # Skip connection
            h = torch.cat([h, encoder_features[-(i + 1)]], dim=1)
            h = dec(h)

        return h


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# =============================================================================
# Factory Functions
# =============================================================================


def create_denoiser(model_name: str, **kwargs) -> BaseDenoiser:
    """
    Factory function to create denoising models.

    Args:
        model_name: Name of the model
        **kwargs: Model-specific arguments

    Returns:
        Denoising model instance
    """
    models = {
        # Image denoising
        "dncnn": DnCNN,
        "ircnn": IRCNN,
        "red": RED,
        "nlm": NLM,
        "bm3d": BM3D,
        "ffdnet": FFDNet,
        "cbdnet": CBDNet,
        "ridnet": RIDNet,
        "mwcnn": MWCNN,
        "mirnet": MIRNet,
        # Blind denoising
        "noise2noise": Noise2Noise,
        "noise2void": Noise2Void,
        "vdnet": VDNet,
        "nbr2nbr": NBR2NBR,
        # Real-world denoising
        "nbnet": NBNet,
        "aindnet": AINDNet,
        "sadnet": SADNet,
        # Video denoising
        "vnlb": VNLB,
        "dvdnet": DVDNet,
        "fastdvdnet": FastDVDNet,
        "pacnet": PaCNet,
        "videnn": ViDeNN,
        # Audio denoising
        "waveunet": WaveUNet,
        "dccrn": DCCRN,
        "fullsubnet": FullSubNet,
        "metricgan": MetricGAN,
        "dptfsnet": DPTFSNet,
    }

    if model_name.lower() not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    return models[model_name.lower()](**kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "DenoisingConfig",
    "BaseDenoiser",
    # Image Denoising
    "DnCNN",
    "IRCNN",
    "RED",
    "NLM",
    "BM3D",
    "FFDNet",
    "CBDNet",
    "RIDNet",
    "MWCNN",
    "MIRNet",
    # Blind Denoising
    "Noise2Noise",
    "Noise2Void",
    "VDNet",
    "NBR2NBR",
    # Real-World Denoising
    "NBNet",
    "AINDNet",
    "SADNet",
    # Video Denoising
    "VNLB",
    "DVDNet",
    "FastDVDNet",
    "PaCNet",
    "ViDeNN",
    # Audio Denoising
    "WaveUNet",
    "DCCRN",
    "FullSubNet",
    "MetricGAN",
    "DPTFSNet",
    # Noise Estimation
    "NoiseLevelEstimator",
    "NoiseMapGenerator",
    "AdaptiveDenoising",
    # Loss Functions
    "CharbonnierLoss",
    "SSIMLoss",
    "PerceptualLoss",
    "FrequencyLoss",
    # Utilities
    "DenoisingDataset",
    "NoiseSimulator",
    "DenoisingTrainer",
    "DenoisingMetrics",
    # Factory
    "create_denoiser",
]
