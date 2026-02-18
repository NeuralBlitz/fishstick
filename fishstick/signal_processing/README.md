# Signal Processing Module

Advanced signal processing tools for machine learning built for the fishstick AI framework.

## Features

### Wavelet Transforms (`wavelet_transform.py`)
- **ContinuousWaveletTransform**: Continuous wavelet transform with Morlet and Ricker wavelets
- **DiscreteWaveletTransform**: Discrete wavelet transform with perfect reconstruction
- **WaveletPacketTransform**: Full wavelet packet decomposition tree
- **WaveletScatteringTransform**: Scattering transform for invariant feature extraction

### Learnable Wavelets (`wavelet_learnable.py`)
- **LearnableWavelet**: Wavelet with learnable frequency and bandwidth
- **AdaptiveWaveletBank**: Bank of learnable wavelets with adaptive scales
- **LearnableWaveletLayer**: End-to-end learnable wavelet layer
- **WaveletScattering1D**: 1D scattering transform with learnable paths
- **StationaryWaveletTransform**: Undecimated DWT for shift invariance
- **DualTreeWaveletTransform**: Nearly shift-invariant complex wavelet transform
- **AdaptiveWaveletSynthesis**: Learnable synthesis from wavelet coefficients

### 2D Wavelet Scattering (`wavelet_scattering2d.py`)
- **GaborWavelet2D**: 2D Gabor wavelet for image features
- **WaveletScattering2D**: Translation-invariant scattering for images
- **ScatteringResNet**: ResNet-style architecture using scattering features
- **ConvScattering2D**: Convolutional scattering network
- **WaveletAttention2D**: Attention over wavelet coefficients

### Fourier Neural Operators (`fourier_operators.py`)
- **FourierLayer**: Spectral convolution using FFT
- **SpectralConv1D**: 1D spectral convolution layer
- **GlobalFourierOperator**: Global context via frequency domain
- **FourierNeuralOperatorBlock**: FNO block with residual connections
- **FrequencyDomainAttention**: Attention in frequency domain
- **FNO1D**: Complete Fourier Neural Operator for 1D signals

### Fourier Attention (`fourier_attention.py`)
- **SpectralAttention**: Attention for frequency component recalibration
- **GlobalFrequencyContext**: Global context via frequency pooling
- **FrequencyTokenAttention**: Token-based attention for frequency bins
- **MultiScaleFrequencyAttention**: Multi-scale FFT-based attention
- **PhaseAwareAttention**: Attention considering phase information
- **FrequencyResponseAttention**: Learned frequency response attention

### Adaptive FNO (`adaptive_fno.py`)
- **AdaptiveSpectralConv**: Spectral conv with adaptive mode selection
- **AdaptiveFNO1D**: Adaptive Fourier Neural Operator
- **LearnableSpectralConv**: Learnable complex weight spectral convolution
- **FactorizedSpectralConv**: Efficient factorized spectral convolution
- **MultiScaleFNO**: Multi-scale Fourier Neural Operator
- **KernelizedFNO**: FNO with learnable convolution kernels
- **TokenFNO**: FNO with frequency token learning

### Time-Frequency Analysis (`time_frequency.py`)
- **ShortTimeFourierTransform**: STFT for time-frequency representation
- **InverseSTFT**: Inverse STFT for signal reconstruction
- **ConstantQTransform**: CQT for logarithmic frequency resolution
- **SynchrosqueezingTransform**: Enhanced time-frequency analysis
- **GaborTransform**: Gabor transform with Gaussian windows

### Hilbert Transform (`hilbert_transform.py`)
- **HilbertTransform**: Analytic signal computation
- **InstantaneousFrequency**: Frequency from Hilbert transform
- **EnvelopeExtraction**: Learnable envelope extraction
- **PhaseExtraction**: Phase encoding
- **InstantaneousAttributes**: Complete instantaneous analysis
- **CyclicSpectrumAnalysis**: Cyclostationary signal analysis

### Empirical Mode Decomposition (`emd.py`)
- **EmpiricalModeDecomposition**: Classic EMD implementation
- **LearnableEMD**: EMD with learnable envelopes
- **EMDToIMF**: Fixed-output IMF extraction layer
- **EMDConvNet**: CNN using IMF decomposition
- **EMDLSTM**: LSTM on IMF features
- **EMDAttention**: Attention over IMFs

### Filter Banks (`filter_banks.py`)
- **GaborFilterBank**: Bank of Gabor filters
- **MorletWaveletBank**: Bank of Morlet wavelets
- **ComplexWaveletBank**: Dual-tree complex wavelets
- **HalfBandFilterBank**: Perfect reconstruction half-band filters
- **PolynomialFilterBank**: Lagrange interpolation filters
- **LearnableFilterBank**: Trainable filter coefficients

### Filter Design (`filter_design.py`)
- **FIRFilterDesign**: FIR filters using windowing
- **IIRFilterDesign**: IIR filters (Butterworth, Chebyshev)
- **LearnableFIRFilter**: FIR with learnable coefficients
- **AdaptiveFilter**: LMS/RLS adaptive filtering
- **WindowedSincFilter**: Optimal FIR design
- **GaborFilterDesign**: Time-frequency Gabor filters
- **ComplexGaborFilter**: Complex Gabor for analytic signals
- **PolyphaseFilterBank**: Efficient multirate processing
- **QuadratureMirrorFilter**: Perfect reconstruction QMFs

### Sparse Coding (`sparse_coding.py`)
- **DictionaryLearning**: Learnable dictionary for sparse coding
- **SparseCodingLayer**: ISTA-based sparse coding
- **KSVDLearner**: K-SVD inspired dictionary learning
- **SparseAutoencoder**: Sparse autoencoder architecture
- **OnlineDictionaryLearning**: Streaming dictionary learning
- **MultiScaleSparseCoding**: Multi-scale sparse representations

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
    AdaptiveFNO1D,
    ShortTimeFourierTransform,
    GaborFilterBank,
    SpectralPooling,
    SignalGenerator,
    HilbertTransform,
    EmpiricalModeDecomposition,
    DictionaryLearning,
)

# Generate test signal
signal = SignalGenerator.sine(1024, frequency=0.1)

# Learnable wavelet transform
learnable_wavelet = AdaptiveWaveletBank(num_scales=16)
coeffs = learnable_wavelet(signal.unsqueeze(0))

# Adaptive FNO
fno = AdaptiveFNO1D(in_channels=1, out_channels=1, hidden_channels=64)
output = fno(torch.randn(2, 128, 1))

# Hilbert transform
hilbert = HilbertTransform(dim=-1)
analytic, envelope, phase = hilbert(signal)

# Empirical Mode Decomposition
emd = EmpiricalModeDecomposition(num_imfs=5)
imfs, residual = emd(signal.unsqueeze(0))

# Sparse coding
_dictlearn = DictionaryLearning(input_dim=256, num_atoms=32)
reconstructed, code = dict_learn(signal[:256])

# STFT
stft = ShortTimeFourierTransform(n_fft=512, hop_length=128)
spectrogram = stft(signal.unsqueeze(0))

# Filter bank
gabor = GaborFilterBank(num_filters=16)
filtered = gabor(signal.unsqueeze(0))

# Spectral pooling
pool = SpectralPooling(keep_ratio=0.5)
pooled = pool(torch.randn(2, 64, 256))
```

## Requirements

- PyTorch >= 1.9
- NumPy
- PyWavelets (for discrete wavelet transforms)
- SciPy (for filter design)

## Author

Created by Agent 12 for the fishstick AI framework.
