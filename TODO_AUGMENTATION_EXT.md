# TODO: augmentation_ext - Data Augmentation Extensions for fishstick

## Phase 1: Project Setup
- [ ] Create directory: /home/runner/workspace/fishstick/augmentation_ext/
- [ ] Create __init__.py with base exports
- [ ] Create base augmentation classes

## Phase 2: Image Augmentation Module
- [ ] Create image_augmentation.py with:
  - MixUp (enhanced version)
  - CutMix (enhanced version)
  - RandAugment
  - AutoAugment (Randaugment policy learning)
  - GridMask
  - RandomErasing
  - Mosaic
  - Blend

## Phase 3: Video Augmentation Module
- [ ] Create video_augmentation.py with:
  - TemporalDropout
  - RandomCropResize
  - ColorJitterVideo
  - RandomRotation3D
  - MixUpVideo
  - CutMixVideo
  - FrameShuffle

## Phase 4: Tabular Data Augmentation Module
- [ ] Create tabular_augmentation.py with:
  - SMOTE (Synthetic Minority Over-sampling)
  - RandomNoiseInjection
  - FeatureShuffle
  - RowMixing
  - SMOTETomek
  - ADASYN

## Phase 5: Graph Augmentation Module
- [ ] Create graph_augmentation.py with:
  - NodeDrop
  - EdgeDrop
  - AttributeMasking
  - SubgraphExtraction
  - NodeFeatureNoise
  - EdgeWeightPerturbation
  - GraphMixup

## Phase 6: Audio Augmentation Module
- [ ] Create audio_augmentation.py with:
  - TimeStretch
  - PitchShift
  - AddBackgroundNoise
  - TimeShift
  - VolumePerturbation
  - SpecAugment
  - AudioMixUp

## Phase 7: Pipeline and Utilities
- [ ] Create pipeline.py with:
  - AugmentationScheduler
  - ConditionalAugmentation
  - AdaptiveAugmentation
  - AugmentationCache

## Phase 8: Finalize Exports
- [ ] Update __init__.py with all module exports
- [ ] Add comprehensive documentation
