"""
Temporal Action Localization Module for fishstick

Comprehensive temporal action localization tools including:
- Actionness scoring
- Boundary detection
- Proposal generation (BMN, BSN)
- Action refinement

Author: Fishstick Team
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LocalizationConfig:
    """Configuration for temporal action localization."""

    feature_dim: int = 2048
    hidden_dim: int = 256
    num_classes: int = 1
    temporal_stride: int = 1
    num_proposals: int = 100
    min_duration: float = 0.1
    max_duration: float = 10.0
    iou_threshold: float = 0.5


@dataclass
class ActionSegment:
    """Represents a detected action segment."""

    start_time: float
    end_time: float
    label: Optional[int] = None
    score: float = 0.0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def center(self) -> float:
        return (self.start_time + self.end_time) / 2


class ActionnessScorer(nn.Module):
    """
    Actionness Scorer Network.

    Scores each temporal position for action presence.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute actionness scores.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask for valid frames

        Returns:
            Actionness scores (B, T)
        """
        encoded, _ = self.encoder(features)

        scores = self.scorer(encoded).squeeze(-1)

        return scores


class BoundaryDetector(nn.Module):
    """
    Boundary Detection Network.

    Detects action boundaries (start and end) in videos.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.start_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.end_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Detect boundaries.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Tuple of (start_scores, end_scores)
        """
        encoded, _ = self.encoder(features)

        start_scores = self.start_predictor(encoded).squeeze(-1)
        end_scores = self.end_predictor(encoded).squeeze(-1)

        return start_scores, end_scores


class ProposalGenerator(nn.Module):
    """
    Base Proposal Generator.

    Generates temporal action proposals from features.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_proposals: Number of proposals to generate
        min_duration: Minimum proposal duration
        max_duration: Maximum proposal duration
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_proposals: int = 100,
        min_duration: float = 0.1,
        max_duration: float = 10.0,
    ):
        super().__init__()

        self.num_proposals = num_proposals
        self.min_duration = min_duration
        self.max_duration = max_duration

        self.center_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.duration_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generate proposals.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Tuple of (centers, durations, scores)
        """
        center_scores = self.center_predictor(features).squeeze(-1)

        durations = self.duration_predictor(features).squeeze(-1)
        durations = durations.clamp(min=self.min_duration, max=self.max_duration)

        if mask is not None:
            center_scores = center_scores.masked_fill(~mask, float("-inf"))

        scores = torch.sigmoid(center_scores)

        return center_scores, durations, scores

    def generate_segments(
        self,
        centers: Tensor,
        durations: Tensor,
        scores: Tensor,
        video_duration: float,
    ) -> List[List[ActionSegment]]:
        """
        Convert predictions to segments.

        Args:
            centers: Center predictions
            durations: Duration predictions
            scores: Confidence scores
            video_duration: Total video duration

        Returns:
            List of action segments per video
        """
        B, T = centers.shape

        segments = []

        for b in range(B):
            video_segments = []

            for t in range(T):
                center_time = centers[b, t].item() * video_duration
                duration = durations[b, t].item() * self.max_duration
                score = scores[b, t].item()

                start_time = max(0, center_time - duration / 2)
                end_time = min(video_duration, center_time + duration / 2)

                video_segments.append(
                    ActionSegment(
                        start_time=start_time,
                        end_time=end_time,
                        score=score,
                        confidence=score,
                    )
                )

            segments.append(video_segments)

        return segments


class BMNModule(nn.Module):
    """
    BMN (Boundary-Matching Network) Module.

    Generates temporal proposals using boundary matching.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        temporal_scale: Temporal scale factor
        num_samples: Number of boundary samples
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        temporal_scale: int = 100,
        num_samples: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.temporal_scale = temporal_scale
        self.num_samples = num_samples

        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.prop_boundary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        self.prop_start = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, temporal_scale),
        )

        self.prop_end = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, temporal_scale),
        )

        self.boundary_matching = BoundaryMatching(temporal_scale, num_samples)

        self.confidence_scorer = nn.Sequential(
            nn.Linear(num_samples * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Tuple of (boundary_scores, start_scores, end_scores)
        """
        B, T, D = features.shape

        encoded = self.base_encoder(features)

        boundary = self.prop_boundary(encoded)

        start_scores = self.prop_start(encoded)
        end_scores = self.prop_end(encoded)

        bm_features = self.boundary_matching(start_scores, end_scores)

        confidence = self.confidence_scorer(bm_features)

        return boundary, start_scores, end_scores

    def generate_proposals(
        self,
        start_scores: Tensor,
        end_scores: Tensor,
        confidence: Tensor,
        video_duration: float,
    ) -> List[List[ActionSegment]]:
        """Generate proposals from BMN outputs."""
        B = start_scores.size(0)
        num_segments = start_scores.size(1)

        segments = []

        for b in range(B):
            proposals = []

            start_probs = F.softmax(start_scores[b], dim=-1)
            end_probs = F.softmax(end_scores[b], dim=-1)

            for i in range(num_segments):
                for j in range(i, num_segments):
                    start_idx = i
                    end_idx = j

                    start_time = (start_idx / self.temporal_scale) * video_duration
                    end_time = (end_idx / self.temporal_scale) * video_duration

                    duration = end_time - start_time

                    if duration < 0.1:
                        continue

                    score = (
                        start_probs[start_idx]
                        * end_probs[end_idx]
                        * confidence[b, i, j]
                    ).item()

                    proposals.append(
                        ActionSegment(
                            start_time=start_time,
                            end_time=end_time,
                            score=score,
                            confidence=score,
                        )
                    )

            proposals = sorted(proposals, key=lambda x: x.score, reverse=True)
            segments.append(proposals[:100])

        return segments


class BoundaryMatching(nn.Module):
    """
    Boundary Matching Module for BMN.

    Creates boundary matching features from start/end scores.
    """

    def __init__(
        self,
        temporal_scale: int = 100,
        num_samples: int = 32,
    ):
        super().__init__()

        self.temporal_scale = temporal_scale
        self.num_samples = num_samples

    def forward(
        self,
        start_scores: Tensor,
        end_scores: Tensor,
    ) -> Tensor:
        """
        Compute boundary matching features.

        Args:
            start_scores: Start boundary scores (B, T, S)
            end_scores: End boundary scores (B, T, S)

        Returns:
            Boundary matching features (B, S, S, 2*num_samples)
        """
        B, T, S = start_scores.shape

        start_expanded = start_scores.unsqueeze(2).expand(-1, -1, S, -1)
        end_expanded = end_scores.unsqueeze(1).expand(-1, S, -1, -1)

        matches = torch.cat([start_expanded, end_expanded], dim=-1)

        return matches


class BSNModule(nn.Module):
    """
    BSN (Boundary Sensitive Network) Module.

    Generates proposals using boundary-sensitive features.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_proposals: Number of proposals to generate
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_proposals: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_proposals = num_proposals

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.actionness = ActionnessScorer(input_dim, hidden_dim, 2, dropout)

        self.boundary = BoundaryDetector(input_dim, hidden_dim, 2, dropout)

        self.proposal_generator = ProposalGenerator(
            input_dim,
            hidden_dim,
            num_proposals,
        )

        self.proposal_refiner = ProposalRefiner(hidden_dim)

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Dictionary of outputs
        """
        encoded = self.feature_encoder(features)

        actionness_scores = self.actionness(features, mask)

        start_scores, end_scores = self.boundary(features, mask)

        center_scores, durations, prop_scores = self.proposal_generator(features, mask)

        proposals = self._generate_proposals(
            center_scores, durations, prop_scores, mask
        )

        refined = self.proposal_refiner(proposals, encoded)

        return {
            "actionness": actionness_scores,
            "start": start_scores,
            "end": end_scores,
            "proposals": proposals,
            "refined": refined,
        }

    def _generate_proposals(
        self,
        center_scores: Tensor,
        durations: Tensor,
        prop_scores: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Generate initial proposals."""
        B, T = center_scores.shape

        scores = center_scores * prop_scores

        _, top_indices = torch.topk(scores, min(self.num_proposals, T), dim=-1)

        proposals = []

        batch_indices = torch.arange(B, device=scores.device).unsqueeze(-1)

        start_scores_idx = top_indices
        duration_vals = durations[batch_indices, top_indices]

        start_times = start_scores_idx.float() / T
        end_times = start_times + duration_vals

        proposals = torch.stack(
            [start_times, end_times, scores[batch_indices, top_indices]], dim=-1
        )

        return proposals


class ProposalRefiner(nn.Module):
    """
    Proposal Refinement Module.

    Refines generated proposals using feature aggregation.

    Args:
        hidden_dim: Hidden dimension
        num_layers: Number of refinement layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.proposal_encoder = nn.Linear(3, hidden_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        proposals: Tensor,
        features: Tensor,
    ) -> Tensor:
        """
        Refine proposals.

        Args:
            proposals: Initial proposals (B, N, 3) [start, end, score]
            features: Video features (B, T, D)

        Returns:
            Refined proposals
        """
        B, N, _ = proposals.shape
        T = features.size(1)

        prop_features = self.proposal_encoder(proposals)

        for layer in self.layers:
            prop_features = layer(prop_features)

        scores = self.scorer(prop_features).squeeze(-1)

        refined = proposals.clone()
        refined[:, :, 2] = scores

        return refined


class TemporalLocalizer(nn.Module):
    """
    Complete Temporal Action Localizer.

    Combines actionness, boundary detection, and proposal generation.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_classes: Number of action classes
        num_proposals: Number of proposals to generate
        config: Localization configuration
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 1,
        num_proposals: int = 100,
        config: Optional[LocalizationConfig] = None,
    ):
        super().__init__()

        if config is None:
            config = LocalizationConfig()

        self.num_proposals = num_proposals

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.actionness_scorer = ActionnessScorer(input_dim, hidden_dim, 2, 0.3)

        self.boundary_detector = BoundaryDetector(input_dim, hidden_dim, 2, 0.3)

        self.proposal_generator = ProposalGenerator(
            input_dim,
            hidden_dim,
            num_proposals,
            config.min_duration,
            config.max_duration,
        )

        self.bsn = BSNModule(input_dim, hidden_dim, num_proposals)

        self.nms_threshold = config.iou_threshold

    def forward(
        self,
        features: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            features: Video features (B, T, D)
            mask: Optional mask

        Returns:
            Dictionary with proposals and scores
        """
        actionness = self.actionness_scorer(features, mask)

        start_scores, end_scores = self.boundary_detector(features, mask)

        center_scores, durations, prop_scores = self.proposal_generator(features, mask)

        bsn_outputs = self.bsn(features, mask)

        proposals = self._generate_proposals(
            start_scores, end_scores, actionness, features.size(1)
        )

        return {
            "actionness": actionness,
            "start_scores": start_scores,
            "end_scores": end_scores,
            "proposals": proposals,
            "bsn_outputs": bsn_outputs,
        }

    def _generate_proposals(
        self,
        start_scores: Tensor,
        end_scores: Tensor,
        actionness: Tensor,
        num_timesteps: int,
    ) -> List[List[ActionSegment]]:
        """Generate proposals from scores."""
        B = start_scores.size(0)

        segments = []

        for b in range(B):
            proposals = []

            start_probs = F.softmax(start_scores[b], dim=-1)
            end_probs = F.softmax(end_scores[b], dim=-1)

            for i in range(num_timesteps):
                for j in range(i, num_timesteps):
                    start_time = i / num_timesteps
                    end_time = j / num_timesteps

                    duration = end_time - start_time

                    if duration < 0.05:
                        continue

                    score = (
                        start_probs[i] * end_probs[j] * actionness[b, i : j + 1].mean()
                    ).item()

                    proposals.append(
                        ActionSegment(
                            start_time=start_time,
                            end_time=end_time,
                            score=score,
                            confidence=score,
                        )
                    )

            proposals = sorted(proposals, key=lambda x: x.score, reverse=True)
            proposals = self._nms(proposals)

            segments.append(proposals[: self.num_proposals])

        return segments

    def _nms(
        self,
        proposals: List[ActionSegment],
    ) -> List[ActionSegment]:
        """Apply Non-Maximum Suppression."""
        if len(proposals) == 0:
            return []

        proposals = sorted(proposals, key=lambda x: x.score, reverse=True)

        keep = []

        while len(proposals) > 0:
            current = proposals[0]
            keep.append(current)

            proposals = [
                p for p in proposals[1:] if self._iou(current, p) < self.nms_threshold
            ]

        return keep

    def _iou(self, seg1: ActionSegment, seg2: ActionSegment) -> float:
        """Compute IoU between two segments."""
        start = max(seg1.start_time, seg2.start_time)
        end = min(seg1.end_time, seg2.end_time)

        if start >= end:
            return 0.0

        intersection = end - start

        union = (
            (seg1.end_time - seg1.start_time)
            + (seg2.end_time - seg2.start_time)
            - intersection
        )

        return intersection / union if union > 0 else 0.0

    def detect(
        self,
        features: Tensor,
        video_duration: float,
        threshold: float = 0.5,
        mask: Optional[Tensor] = None,
    ) -> List[List[ActionSegment]]:
        """
        Detect actions in video.

        Args:
            features: Video features
            video_duration: Total video duration in seconds
            threshold: Confidence threshold
            mask: Optional mask

        Returns:
            List of detected action segments
        """
        outputs = self.forward(features, mask)

        proposals = outputs["proposals"]

        for video_proposals in proposals:
            for proposal in video_proposals:
                proposal.start_time *= video_duration
                proposal.end_time *= video_duration

            video_proposals = [p for p in video_proposals if p.score >= threshold]

        return proposals


def create_bmn(
    input_dim: int = 2048,
    hidden_dim: int = 256,
    temporal_scale: int = 100,
    **kwargs,
) -> BMNModule:
    """
    Create BMN (Boundary-Matching Network) model.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        temporal_scale: Temporal scale
        **kwargs: Additional arguments

    Returns:
        BMN model
    """
    num_samples = kwargs.get("num_samples", 32)
    dropout = kwargs.get("dropout", 0.3)

    return BMNModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        temporal_scale=temporal_scale,
        num_samples=num_samples,
        dropout=dropout,
    )


def create_bsn(
    input_dim: int = 2048,
    hidden_dim: int = 256,
    num_proposals: int = 100,
    **kwargs,
) -> BSNModule:
    """
    Create BSN (Boundary Sensitive Network) model.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        num_proposals: Number of proposals
        **kwargs: Additional arguments

    Returns:
        BSN model
    """
    dropout = kwargs.get("dropout", 0.3)

    return BSNModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_proposals=num_proposals,
        dropout=dropout,
    )
