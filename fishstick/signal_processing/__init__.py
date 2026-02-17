"""
fishstick Signal Processing Module

Advanced signal processing tools for machine learning including:
- Wavelet transforms (continuous, discrete, scattering)
- Fourier neural operators
- Time-frequency analysis
- Filter banks
- Spectral pooling

Author: Agent 12
"""

from fishstick.signal_processing.wavelet_transform import (
    MorletWavelet,
    RickerWavelet,
    ContinuousWaveletTransform,
    DiscreteWaveletTransform,
    WaveletPacketTransform,
    WaveletScatteringTransform,
    InverseWaveletTransform,
)

from fishstick.signal_processing.fourier_operators import (
    FourierLayer,
    SpectralConv1D,
    GlobalFourierOperator,
    FourierNeuralOperatorBlock,
    FrequencyDomainAttention,
    FNO1D,
    SpectralResNetBlock,
)

from fishstick.signal_processing.time_frequency import (
    ShortTimeFourierTransform,
    InverseSTFT,
    ConstantQTransform,
    SynchrosqueezingTransform,
    ReassignmentMethod,
    GaborTransform,
    AdaptiveTimeFrequency,
)

from fishstick.signal_processing.filter_banks import (
    GaborFilterBank,
    MorletWaveletBank,
    ComplexWaveletBank,
    HalfBandFilterBank,
    PolynomialFilterBank,
    LearnableFilterBank,
    BiorthogonalFilterBank,
    FilterBankLayer,
)

from fishstick.signal_processing.spectral_pooling import (
    SpectralPooling,
    FourierPooling,
    WaveletPooling,
    AdaptiveSpectralPooling,
    LearnableFilterBankPooling,
    SpectralPooling1D,
    ScaledSpectralPooling,
    StridedSpectralConv,
    FrequencyAttentionPooling,
    MultiResolutionSpectralPooling,
)

from fishstick.signal_processing.utils import (
    WindowFunctions,
    SignalNormalizer,
    SignalPreprocessor,
    SignalAugmentation,
    SignalGenerator,
    SignalQualityMetrics,
    Resample1D,
    SignalProcessor,
)

__all__ = [
    # Wavelet transforms
    "MorletWavelet",
    "RickerWavelet",
    "ContinuousWaveletTransform",
    "DiscreteWaveletTransform",
    "WaveletPacketTransform",
    "WaveletScatteringTransform",
    "InverseWaveletTransform",
    # Fourier operators
    "FourierLayer",
    "SpectralConv1D",
    "GlobalFourierOperator",
    "FourierNeuralOperatorBlock",
    "FrequencyDomainAttention",
    "FNO1D",
    "SpectralResNetBlock",
    # Time-frequency
    "ShortTimeFourierTransform",
    "InverseSTFT",
    "ConstantQTransform",
    "SynchrosqueezingTransform",
    "ReassignmentMethod",
    "GaborTransform",
    "AdaptiveTimeFrequency",
    # Filter banks
    "GaborFilterBank",
    "MorletWaveletBank",
    "ComplexWaveletBank",
    "HalfBandFilterBank",
    "PolynomialFilterBank",
    "LearnableFilterBank",
    "BiorthogonalFilterBank",
    "FilterBankLayer",
    # Spectral pooling
    "SpectralPooling",
    "FourierPooling",
    "WaveletPooling",
    "AdaptiveSpectralPooling",
    "LearnableFilterBankPooling",
    "SpectralPooling1D",
    "ScaledSpectralPooling",
    "StridedSpectralConv",
    "FrequencyAttentionPooling",
    "MultiResolutionSpectralPooling",
    # Utilities
    "WindowFunctions",
    "SignalNormalizer",
    "SignalPreprocessor",
    "SignalAugmentation",
    "SignalGenerator",
    "SignalQualityMetrics",
    "Resample1D",
    "SignalProcessor",
]
