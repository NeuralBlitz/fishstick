# TODO: Scene Understanding V2 - Advanced Modules

## Overview
Extend the fishstick scene_understanding package with 5 new advanced modules
that add capabilities not covered by the existing base modules.

## New Modules

### 1. `scene_layout.py` - 3D Scene Layout Estimation
- [x] RoomLayoutEstimator: Estimate 3D room layout (floor, walls, ceiling) from single image
- [x] LayoutEncoder: Feature encoder with horizon line prediction
- [x] LayoutDecoder: Decode layout corners and edges
- [x] PerspectiveFieldEstimator: Estimate vanishing points and perspective field
- [x] LayoutRefinementModule: Iterative refinement via geometric consistency
- [x] create_layout_estimator() factory function

### 2. `scene_flow.py` - Scene Flow Estimation
- [x] SceneFlowEstimator: 3D motion field estimation from frame pairs
- [x] CostVolumeBuilder: Build 4D correlation cost volume
- [x] FlowDecoder: Multi-scale iterative flow regression
- [x] OcclusionEstimator: Predict forward/backward occlusion masks
- [x] RigidFlowDecomposition: Decompose flow into rigid/non-rigid components
- [x] create_scene_flow_model() factory function

### 3. `material_recognition.py` - Material & Surface Recognition
- [x] MaterialClassifier: Per-pixel material classification (metal, wood, fabric, etc.)
- [x] MaterialEncoder: BRDF-aware feature encoder
- [x] TextureDescriptor: Local texture pattern descriptor network
- [x] ReflectanceEstimator: Estimate diffuse/specular reflectance properties
- [x] MaterialSegmentationHead: Dense material segmentation
- [x] create_material_model() factory function

### 4. `scene_completion.py` - Scene Completion & Inpainting
- [x] SceneCompletionNetwork: Predict occluded geometry and semantics
- [x] PartialConv2d: Partial convolution for masked regions
- [x] GatedConv2d: Gated convolution with learned attention mask
- [x] ContextualAttention: Patch-swap attention for texture synthesis
- [x] SemanticGuidedCompletion: Semantics-conditioned hole filling
- [x] create_completion_model() factory function

### 5. `scene_dynamics.py` - Scene Dynamics Prediction
- [x] SceneDynamicsPredictor: Predict future scene states from observation
- [x] SpatioTemporalEncoder: Encode spatial + temporal features from video
- [x] MotionFieldPredictor: Predict per-pixel future motion
- [x] SceneEvolutionGRU: Recurrent module for temporal scene evolution
- [x] PhysicsAwarePrediction: Physics-informed regularization for plausible dynamics
- [x] create_dynamics_model() factory function

## Integration
- [x] Update __init__.py with all new exports
