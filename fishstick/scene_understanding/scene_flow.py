"""
Scene Flow Estimation Module

Provides models for estimating 3D motion fields (scene flow) from
pairs of images or stereo frames. Includes cost volume construction,
multi-scale flow decoding, occlusion reasoning, and rigid/non-rigid
motion decomposition.
"""

from typing import Tuple, List, Optional, Union, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class CostVolumeBuilder(nn.Module):
    """
    Build a 4D correlation cost volume from two feature maps.

    For each spatial position in the source features, computes the
    dot-product similarity against a local neighbourhood in the
    target features, producing a ``[B, D, H, W]`` cost volume where
    ``D = (2 * max_disp + 1) ** 2``.
    """

    def __init__(
        self,
        max_displacement: int = 4,
        stride: int = 1,
    ):
        """
        Args:
            max_displacement: Maximum displacement to search.
            stride: Stride of the correlation window.
        """
        super().__init__()
        self.max_displacement = max_displacement
        self.stride = stride

    def forward(self, feat1: Tensor, feat2: Tensor) -> Tensor:
        """
        Args:
            feat1: Source features ``[B, C, H, W]``.
            feat2: Target features ``[B, C, H, W]``.

        Returns:
            Cost volume ``[B, D, H, W]`` where
            ``D = (2 * max_displacement // stride + 1) ** 2``.
        """
        B, C, H, W = feat1.shape
        d = self.max_displacement
        s = self.stride

        feat2_pad = F.pad(feat2, [d, d, d, d])

        cost_list: List[Tensor] = []
        for dy in range(-d, d + 1, s):
            for dx in range(-d, d + 1, s):
                offset_y = dy + d
                offset_x = dx + d
                patch = feat2_pad[
                    :, :, offset_y : offset_y + H, offset_x : offset_x + W
                ]
                cost = (feat1 * patch).sum(dim=1, keepdim=True) / math.sqrt(C)
                cost_list.append(cost)

        return torch.cat(cost_list, dim=1)


class FlowDecoder(nn.Module):
    """
    Multi-scale iterative flow regression decoder.

    Takes encoder features and a cost volume at the coarsest scale
    and progressively upsamples and refines the predicted flow field.
    """

    def __init__(
        self,
        cost_channels: int = 81,
        feature_channels: int = 128,
        num_levels: int = 4,
        flow_dim: int = 3,
    ):
        """
        Args:
            cost_channels: Number of cost volume channels.
            feature_channels: Encoder feature channels per level.
            num_levels: Number of decoder levels (coarse-to-fine).
            flow_dim: Dimensionality of flow (2 for optical, 3 for scene).
        """
        super().__init__()
        self.num_levels = num_levels
        self.flow_dim = flow_dim

        decoders: List[nn.Module] = []
        for i in range(num_levels):
            in_ch = (
                cost_channels + feature_channels + flow_dim
                if i == 0
                else feature_channels + flow_dim
            )
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, 128, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(64, flow_dim, 3, padding=1),
                )
            )
        self.decoders = nn.ModuleList(decoders)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        cost_volume: Tensor,
        features: List[Tensor],
    ) -> List[Tensor]:
        """
        Args:
            cost_volume: Correlation volume ``[B, D, H_c, W_c]``.
            features: Multi-scale encoder features, coarsest first.

        Returns:
            List of flow predictions at each scale, finest last.
        """
        flows: List[Tensor] = []
        flow = torch.zeros(
            cost_volume.shape[0],
            self.flow_dim,
            cost_volume.shape[2],
            cost_volume.shape[3],
            device=cost_volume.device,
        )

        for i in range(self.num_levels):
            if i == 0:
                inp = torch.cat([cost_volume, features[i], flow], dim=1)
            else:
                flow = self.upsample(flow) * 2.0
                if flow.shape[2:] != features[i].shape[2:]:
                    flow = F.interpolate(
                        flow,
                        size=features[i].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                inp = torch.cat([features[i], flow], dim=1)

            residual = self.decoders[i](inp)
            flow = flow + residual
            flows.append(flow)

        return flows


class OcclusionEstimator(nn.Module):
    """
    Predict forward and backward occlusion masks.

    Uses forward-backward flow consistency checking augmented with
    a lightweight CNN to produce soft occlusion probability masks.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels (typically 2 * flow_dim).
            hidden_channels: Hidden layer width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        flow_fwd: Tensor,
        flow_bwd: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            flow_fwd: Forward flow ``[B, D, H, W]``.
            flow_bwd: Backward flow ``[B, D, H, W]``.

        Returns:
            Tuple of (forward_occlusion, backward_occlusion) masks
            each ``[B, 1, H, W]`` in ``[0, 1]``.
        """
        fwd_occ = self.net(torch.cat([flow_fwd, flow_bwd], dim=1))
        bwd_occ = self.net(torch.cat([flow_bwd, flow_fwd], dim=1))
        return fwd_occ, bwd_occ


class RigidFlowDecomposition(nn.Module):
    """
    Decompose scene flow into rigid and non-rigid components.

    Predicts a per-pixel rigidity mask and a global rigid transformation
    (rotation + translation), then computes the non-rigid residual as
    the difference between full flow and the rigid component.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        flow_dim: int = 3,
    ):
        """
        Args:
            feature_dim: Encoder feature dimension.
            flow_dim: Flow dimensionality (2 or 3).
        """
        super().__init__()
        self.flow_dim = flow_dim

        self.rigidity_head = nn.Sequential(
            nn.Conv2d(feature_dim + flow_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        self.rigid_motion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )

    def forward(
        self,
        features: Tensor,
        flow: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Args:
            features: Encoder features ``[B, C, H, W]``.
            flow: Full scene flow ``[B, D, H, W]``.

        Returns:
            Dictionary with keys:
                - ``rigidity_mask``: ``[B, 1, H, W]``
                - ``rigid_params``: ``[B, 6]`` (rotation + translation)
                - ``rigid_flow``: ``[B, D, H, W]``
                - ``nonrigid_flow``: ``[B, D, H, W]``
        """
        rigidity = self.rigidity_head(torch.cat([features, flow], dim=1))

        params = self.rigid_motion_head(features)
        rotation = params[:, :3]
        translation = params[:, 3:]

        rigid_flow = translation.unsqueeze(-1).unsqueeze(-1).expand_as(flow)
        nonrigid_flow = flow - rigidity * rigid_flow

        return {
            "rigidity_mask": rigidity,
            "rigid_params": params,
            "rigid_flow": rigid_flow,
            "nonrigid_flow": nonrigid_flow,
        }


class SceneFlowEncoder(nn.Module):
    """
    Shared feature encoder for scene flow estimation.

    Produces multi-scale features from an input image using a simple
    convolutional pyramid.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_levels: int = 4,
    ):
        """
        Args:
            in_channels: Input image channels.
            base_channels: Channels in the first level.
            num_levels: Number of pyramid levels.
        """
        super().__init__()
        levels: List[nn.Module] = []
        ch_in = in_channels
        for i in range(num_levels):
            ch_out = base_channels * (2**i)
            levels.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )
            ch_in = ch_out
        self.levels = nn.ModuleList(levels)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x: Input image ``[B, C, H, W]``.

        Returns:
            List of feature maps from coarsest to finest resolution.
        """
        feats: List[Tensor] = []
        h = x
        for level in self.levels:
            h = level(h)
            feats.append(h)
        return feats


class SceneFlowEstimator(nn.Module):
    """
    End-to-end 3D scene flow estimation from image pairs.

    Combines shared feature encoding, cost volume correlation,
    multi-scale flow decoding, occlusion estimation, and
    rigid / non-rigid decomposition.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_levels: int = 4,
        max_displacement: int = 4,
        flow_dim: int = 3,
    ):
        """
        Args:
            in_channels: Number of input image channels.
            base_channels: Base encoder channel width.
            num_levels: Number of pyramid levels.
            max_displacement: Cost volume search range.
            flow_dim: Flow dimensionality (3 for full scene flow).
        """
        super().__init__()
        self.encoder = SceneFlowEncoder(in_channels, base_channels, num_levels)
        self.cost_volume = CostVolumeBuilder(max_displacement)

        d = max_displacement
        cost_ch = (2 * d + 1) ** 2
        feat_ch = base_channels * (2 ** (num_levels - 1))

        self.decoder = FlowDecoder(cost_ch, feat_ch, num_levels, flow_dim)
        self.occlusion = OcclusionEstimator(2 * flow_dim)
        self.decompose = RigidFlowDecomposition(feat_ch, flow_dim)

    def forward(
        self,
        img1: Tensor,
        img2: Tensor,
    ) -> Dict[str, Any]:
        """
        Args:
            img1: First image ``[B, 3, H, W]``.
            img2: Second image ``[B, 3, H, W]``.

        Returns:
            Dictionary containing:
                - ``flows``: Multi-scale flow predictions (list).
                - ``occlusion_fwd``: Forward occlusion mask.
                - ``occlusion_bwd``: Backward occlusion mask.
                - ``rigid``: Rigid decomposition dict.
        """
        feats1 = self.encoder(img1)
        feats2 = self.encoder(img2)

        cost = self.cost_volume(feats1[-1], feats2[-1])

        feats_coarse_first = list(reversed(feats1))
        flows = self.decoder(cost, feats_coarse_first)

        final_flow = flows[-1]
        final_flow_up = F.interpolate(
            final_flow, size=img1.shape[2:], mode="bilinear", align_corners=False
        )

        feats2_rev = list(reversed(feats2))
        flows_bwd = self.decoder(
            self.cost_volume(feats2[-1], feats1[-1]),
            feats2_rev,
        )
        bwd_flow_up = F.interpolate(
            flows_bwd[-1], size=img1.shape[2:], mode="bilinear", align_corners=False
        )

        occ_fwd, occ_bwd = self.occlusion(final_flow_up, bwd_flow_up)

        rigid = self.decompose(
            F.interpolate(
                feats1[-1], size=img1.shape[2:], mode="bilinear", align_corners=False
            ),
            final_flow_up,
        )

        return {
            "flows": flows,
            "flow": final_flow_up,
            "flow_bwd": bwd_flow_up,
            "occlusion_fwd": occ_fwd,
            "occlusion_bwd": occ_bwd,
            "rigid": rigid,
        }


def create_scene_flow_model(
    model_type: str = "default",
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create scene flow models.

    Args:
        model_type: Model variant (``"default"``).
        **kwargs: Forwarded to the model constructor.

    Returns:
        Scene flow estimator instance.
    """
    if model_type == "default":
        return SceneFlowEstimator(
            in_channels=kwargs.get("in_channels", 3),
            base_channels=kwargs.get("base_channels", 32),
            num_levels=kwargs.get("num_levels", 4),
            max_displacement=kwargs.get("max_displacement", 4),
            flow_dim=kwargs.get("flow_dim", 3),
        )
    raise ValueError(f"Unknown scene flow model type: {model_type}")
