"""
Scene Understanding Module for fishstick

Comprehensive scene understanding toolkit including:
- Scene classification
- Scene segmentation
- Depth estimation
- Surface normal estimation
- Scene graph generation

Built with state-of-the-art architectures including:
- ResNet and Vision Transformer backbones
- Multi-scale feature fusion
- Graph neural networks for scene reasoning
"""

from typing import Tuple, List, Optional, Union, Dict, Any

import torch
from torch import Tensor, nn

from fishstick.scene_understanding.utils import (
    compute_psnr,
    compute_ssim,
    normalize_tensor,
    safe_divide,
    gradient_x,
    gradient_y,
    IntermediateLayerGetter,
    meshgrid,
    get_gaussian_kernel,
    apply_bilateral_filter,
)

from fishstick.scene_understanding.scene_classifier import (
    SceneClassifier,
    ResNetSceneClassifier,
    MultiScaleSceneFeatures,
    VisionTransformerSceneClassifier,
    TransformerBlock,
    SceneContextEncoder,
    create_scene_classifier,
)

from fishstick.scene_understanding.scene_segmentation import (
    SemanticSegmentationHead,
    PSPModule,
    SceneSegmentationNetwork,
    BoundaryAwareSegmentation,
    BoundaryRefinementModule,
    PanopticSegmentationHead,
    DeepLabV3Plus,
    ASPP,
    create_segmentation_model,
)

from fishstick.scene_understanding.depth_estimator import (
    DepthEncoder,
    DepthDecoder,
    MonocularDepthEstimator,
    ConfidenceDepthEstimator,
    DepthRefinementModule,
    MultiScaleDepthFusion,
    DispNet,
    ResidualDepthRefinement,
    ResidualBlock,
    create_depth_estimator,
)

from fishstick.scene_understanding.surface_normal import (
    SurfaceNormalEncoder,
    NormalDecoder,
    SurfaceNormalEstimator,
    NormalRefinementModule,
    ConfidenceWeightedNormals,
    NormalFromDepthConsistency,
    EdgeAwareNormalSmoothing,
    MultiScaleNormalPrediction,
    NormalizationLoss,
    create_normal_estimator,
)

from fishstick.scene_understanding.scene_graph import (
    ObjectDetector,
    RegionProposalNetwork,
    RCNNHead,
    RoIAlignPooling,
    RelationshipPredictor,
    SceneGraphBuilder,
    SceneGraphNode,
    SceneGraphEdge,
    SceneGraph,
    GraphConvolutionLayer,
    SceneGraphReasoning,
    RELATIONSHIP_CLASSES,
    create_scene_graph_model,
)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "normalize_tensor",
    "safe_divide",
    "gradient_x",
    "gradient_y",
    "IntermediateLayerGetter",
    "meshgrid",
    "get_gaussian_kernel",
    "apply_bilateral_filter",
    "SceneClassifier",
    "ResNetSceneClassifier",
    "MultiScaleSceneFeatures",
    "VisionTransformerSceneClassifier",
    "TransformerBlock",
    "SceneContextEncoder",
    "create_scene_classifier",
    "SemanticSegmentationHead",
    "PSPModule",
    "SceneSegmentationNetwork",
    "BoundaryAwareSegmentation",
    "BoundaryRefinementModule",
    "PanopticSegmentationHead",
    "DeepLabV3Plus",
    "ASPP",
    "create_segmentation_model",
    "DepthEncoder",
    "DepthDecoder",
    "MonocularDepthEstimator",
    "ConfidenceDepthEstimator",
    "DepthRefinementModule",
    "MultiScaleDepthFusion",
    "DispNet",
    "ResidualDepthRefinement",
    "ResidualBlock",
    "create_depth_estimator",
    "SurfaceNormalEncoder",
    "NormalDecoder",
    "SurfaceNormalEstimator",
    "NormalRefinementModule",
    "ConfidenceWeightedNormals",
    "NormalFromDepthConsistency",
    "EdgeAwareNormalSmoothing",
    "MultiScaleNormalPrediction",
    "NormalizationLoss",
    "create_normal_estimator",
    "ObjectDetector",
    "RegionProposalNetwork",
    "RCNNHead",
    "RoIAlignPooling",
    "RelationshipPredictor",
    "SceneGraphBuilder",
    "SceneGraphNode",
    "SceneGraphEdge",
    "SceneGraph",
    "GraphConvolutionLayer",
    "SceneGraphReasoning",
    "RELATIONSHIP_CLASSES",
    "create_scene_graph_model",
]
