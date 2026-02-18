"""
Video Understanding Module for fishstick

Comprehensive video understanding and analysis tools including:
- Video feature extraction (3D CNNs, Video Transformers)
- Action recognition models (Two-stream, TSM, TPN)
- Video captioning (Encoder-decoder with attention)
- Video summarization (Supervised and unsupervised)
- Temporal action localization (Proposal generation, boundary detection)

Author: Fishstick Team
"""

from typing import (
    Optional,
    Union,
    List,
    Dict,
    Tuple,
    Any,
)

import torch
from torch import Tensor, nn

# Feature Extraction
from fishstick.videoUnderstanding.feature_extraction import (
    I3DFeatureExtractor,
    C3DFeatureExtractor,
    R3DFeatureExtractor,
    SlowFastFeatureExtractor,
    TimeSformerEncoder,
    ViViTEncoder,
    SpatioTemporalAttention,
    create_i3d,
    create_c3d,
    create_r3d,
    create_slowfast,
)

# Action Recognition
from fishstick.videoUnderstanding.action_recognition import (
    TwoStreamNetwork,
    TSMHead,
    TPNHead,
    ActionRecognitionModel,
    create_tsn,
    create_tsm,
    create_tpn,
)

# Video Captioning
from fishstick.videoUnderstanding.video_captioning import (
    VideoCaptionEncoder,
    CaptionDecoder,
    VideoCaptionModel,
    BeamSearchDecoder,
    SoftAttention,
    HierarchicalAttention,
    create_captions_model,
)

# Video Summarization
from fishstick.videoUnderstanding.video_summarization import (
    SupervisedSummarizer,
    UnsupervisedSummarizer,
    KeyFrameSelector,
    SummaryGenerator,
    SummarizationLoss,
    create_supervised_summarizer,
    create_unsupervised_summarizer,
)

# Temporal Action Localization
from fishstick.videoUnderstanding.temporal_localization import (
    ActionnessScorer,
    BoundaryDetector,
    ProposalGenerator,
    BMNModule,
    BSNModule,
    TemporalLocalizer,
    ActionSegment,
    create_bmn,
    create_bsn,
)

__all__ = [
    # Feature Extraction
    "I3DFeatureExtractor",
    "C3DFeatureExtractor",
    "R3DFeatureExtractor",
    "SlowFastFeatureExtractor",
    "TimeSformerEncoder",
    "ViViTEncoder",
    "SpatioTemporalAttention",
    "create_i3d",
    "create_c3d",
    "create_r3d",
    "create_slowfast",
    # Action Recognition
    "TwoStreamNetwork",
    "TSMHead",
    "TPNHead",
    "ActionRecognitionModel",
    "create_tsn",
    "create_tsm",
    "create_tpn",
    # Video Captioning
    "VideoCaptionEncoder",
    "CaptionDecoder",
    "VideoCaptionModel",
    "BeamSearchDecoder",
    "SoftAttention",
    "HierarchicalAttention",
    "create_captions_model",
    # Video Summarization
    "SupervisedSummarizer",
    "UnsupervisedSummarizer",
    "KeyFrameSelector",
    "SummaryGenerator",
    "SummarizationLoss",
    "create_supervised_summarizer",
    "create_unsupervised_summarizer",
    # Temporal Action Localization
    "ActionnessScorer",
    "BoundaryDetector",
    "ProposalGenerator",
    "BMNModule",
    "BSNModule",
    "TemporalLocalizer",
    "ActionSegment",
    "create_bmn",
    "create_bsn",
]

# Version
__version__ = "1.0.0"
