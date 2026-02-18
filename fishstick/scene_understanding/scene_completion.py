"""
Scene Completion & Inpainting Module

Provides models for completing occluded scene regions by predicting
missing geometry and semantics. Includes partial convolutions, gated
convolutions, contextual attention, and semantics-guided completion.
"""

from typing import Tuple, List, Optional, Union, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class PartialConv2d(nn.Module):
    """
    Partial convolution for masked image regions.

    Convolves only valid (unmasked) pixels and re-normalises the output
    to compensate for the varying number of contributing inputs, as
    described in *Image Inpainting for Irregular Holes Using Partial
    Convolutions* (Liu et al., 2018).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            bias: Whether to include a bias term.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.mask_conv.weight.requires_grad = False

        self.kernel_area = kernel_size * kernel_size

    def forward(self, x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input features ``[B, C, H, W]``.
            mask: Binary mask ``[B, 1, H, W]`` (1 = valid, 0 = hole).

        Returns:
            Tuple of (output features, updated mask).
        """
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)
            update_mask = (mask_sum > 0).float()
            ratio = self.kernel_area / (mask_sum + 1e-8)
            ratio = ratio * update_mask

        output = self.conv(x * mask)
        output = output * ratio

        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1, 1)
            output = output - bias
            output = output * update_mask + bias

        return output, update_mask


class GatedConv2d(nn.Module):
    """
    Gated convolution with learned soft attention mask.

    Learns a gating signal that automatically attends to valid vs.
    invalid regions, removing the need for explicit binary masks
    (*Free-Form Image Inpainting with Gated Convolution*, Yu et al., 2019).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: str = "elu",
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            activation: Activation for the feature branch.
        """
        super().__init__()
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == "elu":
            self.activation: nn.Module = nn.ELU(inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features ``[B, C, H, W]``.

        Returns:
            Gated output ``[B, C_out, H', W']``.
        """
        feat = self.activation(self.bn(self.feature_conv(x)))
        gate = torch.sigmoid(self.gate_conv(x))
        return feat * gate


class ContextualAttention(nn.Module):
    """
    Patch-swap contextual attention for texture synthesis.

    For each patch inside the masked region, finds the most similar
    patch from the known region and propagates its content to fill
    the hole with plausible textures.
    """

    def __init__(
        self,
        channels: int = 128,
        patch_size: int = 3,
        stride: int = 1,
        softmax_scale: float = 10.0,
    ):
        """
        Args:
            channels: Feature channel dimension.
            patch_size: Attention patch size.
            stride: Stride for extracting patches.
            softmax_scale: Temperature for softmax attention.
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.softmax_scale = softmax_scale

        self.query_proj = nn.Conv2d(channels, channels, 1)
        self.key_proj = nn.Conv2d(channels, channels, 1)
        self.value_proj = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, foreground: Tensor, background: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            foreground: Features in the hole region ``[B, C, H, W]``.
            background: Features in the known region ``[B, C, H, W]``.
            mask: Binary mask ``[B, 1, H, W]`` (1 = hole).

        Returns:
            Attended features ``[B, C, H, W]``.
        """
        B, C, H, W = foreground.shape

        q = self.query_proj(foreground)
        k = self.key_proj(background)
        v = self.value_proj(background)

        q_flat = q.view(B, C, -1).permute(0, 2, 1)
        k_flat = k.view(B, C, -1)
        v_flat = v.view(B, C, -1).permute(0, 2, 1)

        attn = torch.bmm(q_flat, k_flat) * (self.softmax_scale / math.sqrt(C))

        bg_mask = (1 - mask).view(B, 1, -1)
        attn = attn.masked_fill(bg_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v_flat).permute(0, 2, 1).view(B, C, H, W)

        return self.out_proj(out)


class SemanticGuidedCompletion(nn.Module):
    """
    Semantics-conditioned scene completion.

    Uses a semantic label map to guide the generation of missing
    content, ensuring that completed regions are consistent with
    the surrounding semantic layout.
    """

    def __init__(
        self,
        num_semantic_classes: int = 150,
        feature_dim: int = 128,
        embed_dim: int = 64,
    ):
        """
        Args:
            num_semantic_classes: Number of semantic categories.
            feature_dim: Dimension of appearance features.
            embed_dim: Semantic embedding dimension.
        """
        super().__init__()
        self.semantic_embed = nn.Embedding(num_semantic_classes, embed_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim + embed_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Tensor,
        semantic_map: Tensor,
    ) -> Tensor:
        """
        Args:
            features: Appearance features ``[B, C, H, W]``.
            semantic_map: Semantic label map ``[B, H, W]`` (long tensor).

        Returns:
            Completed RGB image ``[B, 3, H, W]``.
        """
        B, C, H, W = features.shape
        sem_embed = self.semantic_embed(semantic_map)
        sem_embed = sem_embed.permute(0, 3, 1, 2)

        fused = self.fusion(torch.cat([features, sem_embed], dim=1))
        return self.decoder(fused)


class SceneCompletionNetwork(nn.Module):
    """
    End-to-end scene completion and inpainting network.

    Combines gated convolution encoding, contextual attention for
    texture propagation, and optional semantic guidance to fill
    masked image regions with plausible content.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_stages: int = 4,
        use_contextual_attention: bool = True,
        num_semantic_classes: Optional[int] = None,
    ):
        """
        Args:
            in_channels: Input channels (typically 3 RGB + 1 mask).
            base_channels: Base encoder channel width.
            num_stages: Number of encoder/decoder stages.
            use_contextual_attention: Whether to use contextual attention.
            num_semantic_classes: If given, enable semantic guidance.
        """
        super().__init__()
        self.use_contextual_attention = use_contextual_attention
        self.use_semantic = num_semantic_classes is not None

        enc_layers: List[nn.Module] = []
        ch_in = in_channels
        for i in range(num_stages):
            ch_out = base_channels * (2**i)
            enc_layers.append(GatedConv2d(ch_in, ch_out, 3, stride=2, padding=1))
            ch_in = ch_out
        self.encoder = nn.ModuleList(enc_layers)

        if use_contextual_attention:
            self.attn = ContextualAttention(ch_in)

        dec_layers: List[nn.Module] = []
        for i in range(num_stages - 1, -1, -1):
            ch_out = base_channels * (2 ** max(i - 1, 0))
            if i == 0:
                ch_out = base_channels
            dec_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    GatedConv2d(ch_in, ch_out, 3, padding=1),
                )
            )
            ch_in = ch_out
        self.decoder = nn.ModuleList(dec_layers)

        self.output_head = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        if self.use_semantic and num_semantic_classes is not None:
            self.semantic_guide = SemanticGuidedCompletion(
                num_semantic_classes, base_channels, 64
            )

    def forward(
        self,
        image: Tensor,
        mask: Tensor,
        semantic_map: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            image: Input image ``[B, 3, H, W]``.
            mask: Binary hole mask ``[B, 1, H, W]`` (1 = hole).
            semantic_map: Optional semantic labels ``[B, H, W]``.

        Returns:
            Dictionary containing:
                - ``completed``: Completed image ``[B, 3, H, W]``.
                - ``coarse``: Coarse output before semantic refinement.
        """
        masked_img = image * (1 - mask)
        x = torch.cat([masked_img, mask], dim=1)

        enc_feats: List[Tensor] = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            enc_feats.append(h)

        if self.use_contextual_attention:
            mask_down = F.interpolate(mask, size=h.shape[2:], mode="nearest")
            h = h + self.attn(h, h, mask_down)

        for layer in self.decoder:
            h = layer(h)

        coarse = self.output_head(h)
        if coarse.shape[2:] != image.shape[2:]:
            coarse = F.interpolate(
                coarse, size=image.shape[2:], mode="bilinear", align_corners=False
            )

        completed = image * (1 - mask) + coarse * mask

        if self.use_semantic and semantic_map is not None:
            refined = self.semantic_guide(h, semantic_map)
            if refined.shape[2:] != image.shape[2:]:
                refined = F.interpolate(
                    refined, size=image.shape[2:], mode="bilinear", align_corners=False
                )
            completed = image * (1 - mask) + refined * mask

        return {"completed": completed, "coarse": coarse}


def create_completion_model(
    model_type: str = "default",
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create scene completion models.

    Args:
        model_type: Model variant (``"default"``).
        **kwargs: Forwarded to the model constructor.

    Returns:
        Scene completion network instance.
    """
    if model_type == "default":
        return SceneCompletionNetwork(
            in_channels=kwargs.get("in_channels", 4),
            base_channels=kwargs.get("base_channels", 64),
            num_stages=kwargs.get("num_stages", 4),
            use_contextual_attention=kwargs.get("use_contextual_attention", True),
            num_semantic_classes=kwargs.get("num_semantic_classes", None),
        )
    raise ValueError(f"Unknown completion model type: {model_type}")
