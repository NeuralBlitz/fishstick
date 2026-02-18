"""
fishstick Sequence Models Module

Sequence model extensions including LSTM/GRU variants, attention-based seq2seq,
beam search implementations, position-wise feedforward networks, and
positional encoding utilities.
"""

from fishstick.sequence_models.lstm_gru_variants import (
    VariationalLSTM,
    ZoneoutGRU,
    LayerNormalizedLSTM,
    LayerNormalizedGRU,
    BiLSTMAttention,
    ConvLSTM,
    IndRNNCell,
    IndRNN,
)

from fishstick.sequence_models.seq2seq_attention import (
    Seq2SeqEncoder,
    BahdanauAttention,
    LuongAttention,
    MultiHeadAttention,
    Seq2SeqDecoder,
    AttentionSeq2SeqDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    Seq2SeqModel,
)

from fishstick.sequence_models.beam_search import (
    BeamSearchScorer,
    BeamSearchDecoder,
    DiverseBeamSearchDecoder,
    LengthPenalty,
    IterativeRefinementDecoder,
)

from fishstick.sequence_models.positionwise_ffn import (
    PositionwiseFeedForward,
    GatedLinearUnit,
    GatedResidualNetwork,
    SwitchTransformerFFN,
    Conv1dFFN,
    PositionwiseFFNWithConv,
    FeedForwardChunk,
    MLPMixerFFN,
    FastformerFFN,
)

from fishstick.sequence_models.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RelativePositionalEncoding,
    RotaryPositionalEmbedding,
    ALiBiPositionalEncoding,
    CoherentPositionalEncoding,
    T5RelativePositionalBias,
    StreamingPositionalEncoding,
    MultiScalePositionalEncoding,
)

__all__ = [
    # LSTM/GRU variants
    "VariationalLSTM",
    "ZoneoutGRU",
    "LayerNormalizedLSTM",
    "LayerNormalizedGRU",
    "BiLSTMAttention",
    "ConvLSTM",
    "IndRNNCell",
    "IndRNN",
    # Seq2Seq with attention
    "Seq2SeqEncoder",
    "BahdanauAttention",
    "LuongAttention",
    "MultiHeadAttention",
    "Seq2SeqDecoder",
    "AttentionSeq2SeqDecoder",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "Seq2SeqModel",
    # Beam search
    "BeamSearchScorer",
    "BeamSearchDecoder",
    "DiverseBeamSearchDecoder",
    "LengthPenalty",
    "IterativeRefinementDecoder",
    # Position-wise FFN
    "PositionwiseFeedForward",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "SwitchTransformerFFN",
    "Conv1dFFN",
    "PositionwiseFFNWithConv",
    "FeedForwardChunk",
    "MLPMixerFFN",
    "FastformerFFN",
    # Positional encoding
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RelativePositionalEncoding",
    "RotaryPositionalEmbedding",
    "ALiBiPositionalEncoding",
    "CoherentPositionalEncoding",
    "T5RelativePositionalBias",
    "StreamingPositionalEncoding",
    "MultiScalePositionalEncoding",
]
