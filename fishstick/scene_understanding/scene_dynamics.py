"""
Scene Dynamics Prediction Module

Provides models for predicting future scene states from video
observations. Includes spatio-temporal encoding, motion field
prediction, recurrent scene evolution, and physics-aware
regularisation for temporally coherent predictions.
"""

from typing import Tuple, List, Optional, Union, Dict, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math


class SpatioTemporalEncoder(nn.Module):
    """
    Encode spatial and temporal features from a video clip.

    Processes a sequence of frames with 2D spatial convolutions
    followed by temporal 1D convolutions to capture motion context
    across time.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_spatial_stages: int = 3,
        temporal_kernel: int = 3,
    ):
        """
        Args:
            in_channels: Input image channels.
            base_channels: Base spatial channel width.
            num_spatial_stages: Number of spatial downsampling stages.
            temporal_kernel: Kernel size for temporal convolutions.
        """
        super().__init__()

        spatial: List[nn.Module] = []
        ch_in = in_channels
        for i in range(num_spatial_stages):
            ch_out = base_channels * (2**i)
            spatial.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, 3, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                )
            )
            ch_in = ch_out
        self.spatial_stages = nn.ModuleList(spatial)
        self.spatial_out_ch = ch_in

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_in, temporal_kernel, padding=temporal_kernel // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_in, ch_in, temporal_kernel, padding=temporal_kernel // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            frames: Video clip ``[B, T, C, H, W]``.

        Returns:
            Tuple of (spatial_features ``[B, T, C', h, w]``,
            temporal_features ``[B, C', T]``).
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)

        for stage in self.spatial_stages:
            x = stage(x)

        _, Cp, h, w = x.shape
        spatial = x.view(B, T, Cp, h, w)

        pooled = spatial.mean(dim=[3, 4])
        temporal = self.temporal_conv(pooled.permute(0, 2, 1))

        return spatial, temporal


class MotionFieldPredictor(nn.Module):
    """
    Predict per-pixel future motion field.

    Given the current spatial features and temporal context, produces
    a dense 2D motion field representing how each pixel is expected to
    move in the next time step.
    """

    def __init__(
        self,
        spatial_channels: int = 256,
        temporal_channels: int = 256,
        hidden_channels: int = 128,
    ):
        """
        Args:
            spatial_channels: Channels of spatial feature maps.
            temporal_channels: Channels of temporal context.
            hidden_channels: Internal hidden channel width.
        """
        super().__init__()
        self.temporal_proj = nn.Linear(temporal_channels, hidden_channels)

        self.predictor = nn.Sequential(
            nn.Conv2d(
                spatial_channels + hidden_channels, hidden_channels, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1),
        )

    def forward(
        self,
        spatial_feat: Tensor,
        temporal_ctx: Tensor,
    ) -> Tensor:
        """
        Args:
            spatial_feat: Last-frame spatial features ``[B, C, h, w]``.
            temporal_ctx: Temporal context vector ``[B, C_t]``.

        Returns:
            Motion field ``[B, 2, h, w]``.
        """
        B, C, h, w = spatial_feat.shape
        t_proj = self.temporal_proj(temporal_ctx)
        t_proj = t_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

        combined = torch.cat([spatial_feat, t_proj], dim=1)
        return self.predictor(combined)


class SceneEvolutionGRU(nn.Module):
    """
    Recurrent module for temporal scene evolution.

    A convolutional GRU that maintains a hidden state representing the
    evolving scene and updates it at each time step to propagate
    information across the prediction horizon.
    """

    def __init__(
        self,
        channels: int = 256,
    ):
        """
        Args:
            channels: Feature channel dimension (input and hidden).
        """
        super().__init__()
        self.channels = channels

        self.reset_gate = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.update_gate = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        Args:
            x: Input features ``[B, C, h, w]``.
            h: Previous hidden state ``[B, C, h, w]``.

        Returns:
            Updated hidden state ``[B, C, h, w]``.
        """
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))

        reset_h = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.candidate(reset_h))

        return (1 - z) * h + z * h_tilde

    def init_hidden(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """
        Initialise a zero hidden state.

        Args:
            batch_size: Batch size.
            height: Spatial height.
            width: Spatial width.
            device: Target device.

        Returns:
            Zero tensor ``[B, C, h, w]``.
        """
        return torch.zeros(batch_size, self.channels, height, width, device=device)


class PhysicsAwarePrediction(nn.Module):
    """
    Physics-informed regularisation for plausible scene dynamics.

    Applies soft constraints based on simplified physical priors
    (mass conservation, smoothness, and inertia) to predicted motion
    fields and future frames to encourage physically plausible outputs.
    """

    def __init__(
        self,
        lambda_smooth: float = 1.0,
        lambda_divergence: float = 0.5,
        lambda_inertia: float = 0.3,
    ):
        """
        Args:
            lambda_smooth: Weight for spatial smoothness loss.
            lambda_divergence: Weight for divergence-free penalty.
            lambda_inertia: Weight for temporal inertia consistency.
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_divergence = lambda_divergence
        self.lambda_inertia = lambda_inertia

    def smoothness_loss(self, flow: Tensor) -> Tensor:
        """
        Penalise spatial gradients in the motion field.

        Args:
            flow: Motion field ``[B, 2, H, W]``.

        Returns:
            Scalar smoothness loss.
        """
        dx = flow[:, :, :, :-1] - flow[:, :, :, 1:]
        dy = flow[:, :, :-1, :] - flow[:, :, 1:, :]
        return dx.abs().mean() + dy.abs().mean()

    def divergence_loss(self, flow: Tensor) -> Tensor:
        """
        Penalise non-zero divergence (approximate mass conservation).

        Args:
            flow: Motion field ``[B, 2, H, W]``.

        Returns:
            Scalar divergence loss.
        """
        du_dx = flow[:, 0:1, :, :-1] - flow[:, 0:1, :, 1:]
        dv_dy = flow[:, 1:2, :-1, :] - flow[:, 1:2, 1:, :]

        min_h = min(du_dx.shape[2], dv_dy.shape[2])
        min_w = min(du_dx.shape[3], dv_dy.shape[3])
        du_dx = du_dx[:, :, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :min_h, :min_w]

        div = du_dx + dv_dy
        return div.pow(2).mean()

    def inertia_loss(self, flow_prev: Tensor, flow_curr: Tensor) -> Tensor:
        """
        Penalise abrupt changes in motion between time steps.

        Args:
            flow_prev: Previous-step motion ``[B, 2, H, W]``.
            flow_curr: Current-step motion ``[B, 2, H, W]``.

        Returns:
            Scalar inertia loss.
        """
        return F.mse_loss(flow_curr, flow_prev)

    def forward(
        self,
        flows: List[Tensor],
    ) -> Tensor:
        """
        Compute combined physics-aware regularisation loss.

        Args:
            flows: List of predicted motion fields over time.

        Returns:
            Scalar regularisation loss.
        """
        loss = torch.tensor(0.0, device=flows[0].device)

        for flow in flows:
            loss = loss + self.lambda_smooth * self.smoothness_loss(flow)
            loss = loss + self.lambda_divergence * self.divergence_loss(flow)

        for i in range(1, len(flows)):
            loss = loss + self.lambda_inertia * self.inertia_loss(
                flows[i - 1], flows[i]
            )

        return loss / max(len(flows), 1)


class SceneDynamicsPredictor(nn.Module):
    """
    End-to-end scene dynamics predictor.

    Given a short video clip, predicts future frames and motion fields
    by combining spatio-temporal encoding, recurrent scene evolution,
    motion prediction, and physics-aware regularisation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_spatial_stages: int = 3,
        prediction_steps: int = 5,
        lambda_smooth: float = 1.0,
        lambda_divergence: float = 0.5,
        lambda_inertia: float = 0.3,
    ):
        """
        Args:
            in_channels: Input frame channels.
            base_channels: Base encoder width.
            num_spatial_stages: Spatial encoder stages.
            prediction_steps: Number of future steps to predict.
            lambda_smooth: Smoothness regularisation weight.
            lambda_divergence: Divergence penalty weight.
            lambda_inertia: Inertia consistency weight.
        """
        super().__init__()
        self.prediction_steps = prediction_steps

        self.encoder = SpatioTemporalEncoder(
            in_channels, base_channels, num_spatial_stages
        )
        feat_ch = base_channels * (2 ** (num_spatial_stages - 1))

        self.gru = SceneEvolutionGRU(feat_ch)
        self.motion = MotionFieldPredictor(feat_ch, feat_ch, feat_ch // 2)

        self.frame_decoder = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feat_ch // 2, feat_ch // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feat_ch // 4, feat_ch // 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feat_ch // 8, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        self.physics = PhysicsAwarePrediction(
            lambda_smooth, lambda_divergence, lambda_inertia
        )

    def forward(
        self,
        frames: Tensor,
    ) -> Dict[str, Any]:
        """
        Args:
            frames: Input video clip ``[B, T, 3, H, W]``.

        Returns:
            Dictionary containing:
                - ``predicted_frames``: ``[B, prediction_steps, 3, H, W]``
                - ``motion_fields``: ``[B, prediction_steps, 2, h, w]``
                - ``physics_loss``: Scalar regularisation loss.
        """
        B, T, C, H, W = frames.shape

        spatial, temporal = self.encoder(frames)

        last_feat = spatial[:, -1]
        temporal_ctx = temporal[:, :, -1]

        h = self.gru.init_hidden(
            B, last_feat.shape[2], last_feat.shape[3], frames.device
        )

        for t in range(T):
            h = self.gru(spatial[:, t], h)

        pred_frames: List[Tensor] = []
        motion_fields: List[Tensor] = []

        for _ in range(self.prediction_steps):
            motion = self.motion(h, temporal_ctx)
            motion_fields.append(motion)

            h = self.gru(h, h)

            frame = self.frame_decoder(h)
            if frame.shape[2:] != (H, W):
                frame = F.interpolate(
                    frame, size=(H, W), mode="bilinear", align_corners=False
                )
            pred_frames.append(frame)

        physics_loss = self.physics(motion_fields)

        return {
            "predicted_frames": torch.stack(pred_frames, dim=1),
            "motion_fields": torch.stack(motion_fields, dim=1),
            "physics_loss": physics_loss,
        }


def create_dynamics_model(
    model_type: str = "default",
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to create scene dynamics models.

    Args:
        model_type: Model variant (``"default"``).
        **kwargs: Forwarded to the model constructor.

    Returns:
        Scene dynamics predictor instance.
    """
    if model_type == "default":
        return SceneDynamicsPredictor(
            in_channels=kwargs.get("in_channels", 3),
            base_channels=kwargs.get("base_channels", 64),
            num_spatial_stages=kwargs.get("num_spatial_stages", 3),
            prediction_steps=kwargs.get("prediction_steps", 5),
            lambda_smooth=kwargs.get("lambda_smooth", 1.0),
            lambda_divergence=kwargs.get("lambda_divergence", 0.5),
            lambda_inertia=kwargs.get("lambda_inertia", 0.3),
        )
    raise ValueError(f"Unknown dynamics model type: {model_type}")
