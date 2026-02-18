# Signal Processing Module

Advanced signal processing tools for machine learning built for the fishstick AI framework.

## Features

### Wavelet Transforms (`wavelet_transform.py`)
- **ContinuousWaveletTransform**: Continuous wavelet transform with Morlet and Ricker wavelets
- **DiscreteWaveletTransform**: Discrete wavelet transform with perfect reconstruction
- **WaveletPacketTransform**: Full wavelet packet decomposition tree
- **WaveletScatteringTransform**: Scattering transform for invariant feature extraction

### Fourier Neural Operators (`fourier_operators.py`)
- **FourierLayer**: Spectral convolution using FFT
- **SpectralConv1D**: 1D spectral convolution layer
- **GlobalFourierOperator**: Global context via frequency domain
- **FourierNeuralOperatorBlock**: FNO block with residual connections
- **FrequencyDomainAttention**: Attention in frequency domain
- **FNO1D**: Complete Fourier Neural Operator for 1D signals

### Time-Frequency Analysis (`time_frequency.py`)
- **ShortTimeFourierTransform**: STFT for time-frequency representation
- **InverseSTFT**: Inverse STFT for signal reconstruction
- **ConstantQTransform**: CQT for logarithmic frequency resolution
- **SynchrosqueezingTransform**: Enhanced time-frequency analysis
- **GaborTransform**: Gabor transform with Gaussian windows

### Filter Banks (`filter_banks.py`)
- **GaborFilterBank**: Bank of Gabor filters
- **MorletWaveletBank**: Bank of Morlet wavelets
- **ComplexWaveletBank**: Dual-tree complex wavelets
- **HalfBandFilterBank**: Perfect reconstruction half-band filters
- **PolynomialFilterBank**: Lagrange interpolation filters
- **LearnableFilterBank**: Trainable filter coefficients

### Spectral Pooling (`spectral_pooling.py`)
- **SpectralPooling**: Keep low-frequency components
- **FourierPooling**: Pooling in Fourier domain
- **WaveletPooling**: Wavelet-based pooling
- **AdaptiveSpectralPooling**: Learnable spectral pooling
- **MultiResolutionSpectralPooling**: Combine multiple resolutions

### Utilities (`utils.py`)
- **WindowFunctions**: Hann, Hamming, Blackman, Kaiser, Nuttall, Gaussian, Tukey
- **SignalNormalizer**: Standard, min-max, L2 normalization
- **SignalPreprocessor**: Complete preprocessing pipeline
- **SignalAugmentation**: Time stretch, pitch shift, noise
- **SignalGenerator**: Sine, square, sawtooth, chirp, noise signals
- **SignalQualityMetrics**: SNR, spectral flatness, zero-crossing rate
- **Resample1D**: Signal resampling

## Usage

```python
import torch
from fishstick.signal_processing import (
    ContinuousWaveletTransform,
    FNO1D,
    ShortTimeFourierTransform,
    GaborFilterBank,
    SpectralPooling,
    SignalGenerator,
)

# Generate test signal
signal = SignalGenerator.sine(1024, frequency=0.1)

# Compute CWT
cwt = ContinuousWaveletTransform(n_scales=32)
scaleogram = cwt.get_scaleogram(signal.unsqueeze(0))

# STFT
stft = ShortTimeFourierTransform(n_fft=512, hop_length=128)
spectrogram = stft(signal.unsqueeze(0))

# Filter bank
gabor = GaborFilterBank(num_filters=16)
filtered = gabor(signal.unsqueeze(0))

# Spectral pooling
pool = SpectralPooling(keep_ratio=0.5)
pooled = pool(torch.randn(2, 64, 256))

# FNO
fno = FNO1D(in_channels=1, out_channels=1, hidden_channels=64)
output = fno(torch.randn(2, 128, 1))
```

## Requirements

- PyTorch >= 1.9
- NumPy
- PyWavelets (for discrete wavelet transforms)

## Author

Created by Agent 12 for the fishstick AI framework.
