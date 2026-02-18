# TODO: Audio Source Separation Module for fishstick

## Overview
Build comprehensive audio source separation tools for the fishstick AI framework, implementing state-of-the-art methods for:
- Time-frequency masking
- Deep clustering methods
- Waveform-based separation
- Speaker separation
- Music source separation

## Requirements
- Create in NEW directory: /home/runner/workspace/fishstick/audio_separation/
- Each module should be well-documented with docstrings
- Follow fishstick's code style (type hints, torch-based, etc.)
- Create __init__.py with proper exports
- Create at least 5 substantial new modules

## Module Breakdown

### 1. Base Classes (core/base.py)
- SeparationModel: Base class for all separation models
- Separator: High-level interface for audio separation
- SeparationMetrics: Metrics for evaluating separation quality

### 2. Time-Frequency Masking (time_frequency_masking.py)
- IdealBinaryMask: Binary mask based on source dominance
- IdealRatioMask: Ratio mask based on power spectral density
- PhaseSensitiveMask: Complex ratio mask preserving phase
- ComplexMask: Complex-valued mask for STFT domain
- WienerFilter: Wiener filtering for optimal MMSE estimation
- IBMEstimator: Estimator for IBM from mixtures

### 3. Deep Clustering Methods (deep_clustering.py)
- DeepClusteringLoss: Permutation-invariant clustering loss
- ClusteringNetwork: Network producing embeddings for clustering
- EmbeddingExtractor: Extracts embeddings from audio
- AgglomerativeClustering: Post-processing clustering

### 4. Waveform-based Separation (waveform_separation.py)
- WaveUNet: U-Net architecture for waveform domain
- SudoRM-RF: Sudo Random Matrix Refinement
- ConvTasNet: Convolutional time-domain audio separation network
- DCCRN: Deep Complex Convolution Reconstructor

### 5. Speaker Separation (speaker_separation.py)
- SpeakerBeam: Speaker-aware beamforming
- Tar-VAE: Target Speaker VAE for extraction
- SpeakerEncoder: Speaker embedding network
- SpeakerVerification: Speaker verification module

### 6. Music Source Separation (music_separation.py)
- Demucs: Facebook's music separation model
- OpenUnmix: Open source music separation
- MusicSourceSeparator: Unified interface for music separation
- SourceDecoder: Decodes separated sources

### 7. Losses (losses.py)
- PITLoss: Permutation Invariant Training loss
- SISDRLoss: Scale-Invariant Source-to-Distortion Ratio
- SDRLoss: Source-to-Distortion Ratio
- SARLoss: Source-to-Artifact Ratio
- SNRLoss: Signal-to-Noise Ratio loss

### 8. Utilities (utils.py)
- STFT/iSTFT transforms
- Audio loading utilities
- Mixing utilities for training
- Evaluation utilities

## Implementation Priority
1. Base classes and utilities (essential for all modules)
2. Time-frequency masking (foundational for other methods)
3. Losses (needed for training any model)
4. Waveform-based separation (most general approach)
5. Deep clustering (important research method)
6. Speaker separation (specialized application)
7. Music separation (specialized application)

## Code Style Requirements
- Use torch.nn.Module as base class
- Type hints for all functions
- Docstrings with examples
- Follow fishstick naming conventions
- Use @classmethod and @staticmethod where appropriate
