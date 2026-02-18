"""
fishstick Signal Processing Module

Advanced signal processing tools for machine learning including:
- Wavelet transforms (continuous, discrete, scattering)
- Fourier neural operators
- Time-frequency analysis
- Filter banks
- Spectral pooling
- Learnable wavelets
- Sparse coding
- Filter design

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

from fishstick.signal_processing.wavelet_learnable import (
    LearnableWavelet,
    AdaptiveWaveletBank,
    LearnableWaveletLayer,
    WaveletScattering1D,
    StationaryWaveletTransform,
    DualTreeWaveletTransform,
    WaveletReconstructionLoss,
    AdaptiveWaveletSynthesis,
)

from fishstick.signal_processing.wavelet_scattering2d import (
    GaborWavelet2D,
    WaveletScattering2D,
    ScatteringResNet,
    ConvScattering2D,
    InvariantScatteringLoss,
    ScatteringBatchNorm,
    WaveletAttention2D,
    MultiscaleScattering,
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

from fishstick.signal_processing.fourier_attention import (
    FrequencyDomainAttention,
    SpectralAttention,
    GlobalFrequencyContext,
    FrequencyTokenAttention,
    MultiScaleFrequencyAttention,
    PhaseAwareAttention,
    FrequencyResponseAttention,
    ChannelFrequencyAttention,
    CrossFrequencyAttention,
    FrequencyLinear,
)

from fishstick.signal_processing.adaptive_fno import (
    AdaptiveSpectralConv,
    AdaptiveFNOBlock,
    AdaptiveFNO1D,
    LearnableSpectralConv,
    FactorizedSpectralConv,
    MultiScaleFNO,
    KernelizedFNO,
    TokenFNO,
    FNOWithPositionalEncoding,
    FNODynamicModeDecomposition,
    HierarchicalFNO,
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

from fishstick.signal_processing.hilbert_transform import (
    HilbertTransform,
    InstantaneousFrequency,
    AnalyticSignalConv,
    EnvelopeExtraction,
    PhaseExtraction,
    InstantaneousAttributes,
    HilbertEnvelopeLoss,
    ComplexConv1D,
    AnalyticFilterBank,
    FrequencyDomainPhase,
    EnvelopeConsistencyLoss,
    CyclicSpectrumAnalysis,
    TimeDomainHilbertLayer,
)

from fishstick.signal_processing.emd import (
    EmpiricalModeDecomposition,
    LearnableEMD,
    EMDToIMF,
    IMFSynthesis,
    IMFSelectiveReconstruction,
    IntrinsicModeFunctionFeatures,
    EMDConvNet,
    VariationalEMD,
    EMDLSTM,
    EMDAttention,
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

from fishstick.signal_processing.filter_design import (
    FIRFilterDesign,
    IIRFilterDesign,
    LearnableFIRFilter,
    AdaptiveFilter,
    WindowedSincFilter,
    FilterBankLayer,
    GaborFilterDesign,
    ComplexGaborFilter,
    PolyphaseFilterBank,
    QuadratureMirrorFilter,
    AllpassFilter,
    GradientFilter,
)

from fishstick.signal_processing.sparse_coding import (
    DictionaryLearning,
    SparseCodingLayer,
    KSVDLearner,
    ConvSparseCoding,
    SparseAutoencoder,
    OnlineDictionaryLearning,
    DictionaryConv1D,
    SparseCodingLoss,
    ScatteringSparseCoding,
    LearnedSparsify,
    MultiScaleSparseCoding,
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
    # Wavelet transforms (original)
    "MorletWavelet",
    "RickerWavelet",
    "ContinuousWaveletTransform",
    "DiscreteWaveletTransform",
    "WaveletPacketTransform",
    "WaveletScatteringTransform",
    "InverseWaveletTransform",
    # Learnable wavelets
    "LearnableWavelet",
    "AdaptiveWaveletBank",
    "LearnableWaveletLayer",
    "WaveletScattering1D",
    "StationaryWaveletTransform",
    "DualTreeWaveletTransform",
    "WaveletReconstructionLoss",
    "AdaptiveWaveletSynthesis",
    # 2D wavelet scattering
    "GaborWavelet2D",
    "WaveletScattering2D",
    "ScatteringResNet",
    "ConvScattering2D",
    "InvariantScatteringLoss",
    "ScatteringBatchNorm",
    "WaveletAttention2D",
    "MultiscaleScattering",
    # Fourier operators
    "FourierLayer",
    "SpectralConv1D",
    "GlobalFourierOperator",
    "FourierNeuralOperatorBlock",
    "FrequencyDomainAttention",
    "FNO1D",
    "SpectralResNetBlock",
    # Fourier attention
    "SpectralAttention",
    "GlobalFrequencyContext",
    "FrequencyTokenAttention",
    "MultiScaleFrequencyAttention",
    "PhaseAwareAttention",
    "FrequencyResponseAttention",
    "ChannelFrequencyAttention",
    "CrossFrequencyAttention",
    "FrequencyLinear",
    # Adaptive FNO
    "AdaptiveSpectralConv",
    "AdaptiveFNOBlock",
    "AdaptiveFNO1D",
    "LearnableSpectralConv",
    "FactorizedSpectralConv",
    "MultiScaleFNO",
    "KernelizedFNO",
    "TokenFNO",
    "FNOWithPositionalEncoding",
    "FNODynamicModeDecomposition",
    "HierarchicalFNO",
    # Time-frequency
    "ShortTimeFourierTransform",
    "InverseSTFT",
    "ConstantQTransform",
    "SynchrosqueezingTransform",
    "ReassignmentMethod",
    "GaborTransform",
    "AdaptiveTimeFrequency",
    # Hilbert transform
    "InstantaneousFrequency",
    "AnalyticSignalConv",
    "EnvelopeExtraction",
    "PhaseExtraction",
    "InstantaneousAttributes",
    "HilbertEnvelopeLoss",
    "ComplexConv1D",
    "AnalyticFilterBank",
    "FrequencyDomainPhase",
    "EnvelopeConsistencyLoss",
    "CyclicSpectrumAnalysis",
    "TimeDomainHilbertLayer",
    # EMD
    "EmpiricalModeDecomposition",
    "LearnableEMD",
    "EMDToIMF",
    "IMFSynthesis",
    "IMFSelectiveReconstruction",
    "IntrinsicModeFunctionFeatures",
    "EMDConvNet",
    "VariationalEMD",
    "EMDLSTM",
    "EMDAttention",
    # Filter banks
    "GaborFilterBank",
    "MorletWaveletBank",
    "ComplexWaveletBank",
    "HalfBandFilterBank",
    "PolynomialFilterBank",
    "LearnableFilterBank",
    "BiorthogonalFilterBank",
    # Filter design
    "FIRFilterDesign",
    "IIRFilterDesign",
    "LearnableFIRFilter",
    "AdaptiveFilter",
    "WindowedSincFilter",
    "GaborFilterDesign",
    "ComplexGaborFilter",
    "PolyphaseFilterBank",
    "QuadratureMirrorFilter",
    "AllpassFilter",
    "GradientFilter",
    # Sparse coding
    "DictionaryLearning",
    "SparseCodingLayer",
    "KSVDLearner",
    "ConvSparseCoding",
    "SparseAutoencoder",
    "OnlineDictionaryLearning",
    "DictionaryConv1D",
    "SparseCodingLoss",
    "ScatteringSparseCoding",
    "LearnedSparsify",
    "MultiScaleSparseCoding",
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
