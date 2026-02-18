"""
fishstick Machine Translation Module

Advanced machine translation tools including:
- Seq2Seq translation models with attention
- Transformer-based translation models
- Specialized attention mechanisms for MT
- Beam search and advanced decoding strategies
- Evaluation metrics (BLEU, METEOR, TER, chrF, COMET)
- Data processing utilities
- Training utilities
"""

from fishstick.machine_translation.seq2seq_models import (
    Seq2SeqEncoder,
    Seq2SeqDecoder,
    AttentionSeq2Seq,
    ConditionalRNNSequence,
)
from fishstick.machine_translation.transformer_mt import (
    TransformerMTEncoder,
    TransformerMTDecoder,
    TransformerTranslationModel,
    RelativePositionTransformer,
)
from fishstick.machine_translation.attention_mechanisms import (
    BahdanauAttention,
    LuongAttention,
    MultiHeadAttention,
    ConvolutionalAttention,
    LinearizedAttention,
)
from fishstick.machine_translation.beam_search import (
    BeamSearchDecoder,
    DiverseBeamSearch,
    GreedyDecoder,
    SamplingDecoder,
    LengthPenaltyBeamSearch,
)
from fishstick.machine_translation.metrics import (
    BLEUScore,
    METEORScore,
    TERScore,
    chrFScore,
    COMETScore,
    TranslationMetrics,
)
from fishstick.machine_translation.data_utils import (
    Vocabulary,
    MTDataset,
    MTCollator,
    build_vocab_from_data,
    preprocess_parallel_corpus,
)
from fishstick.machine_translation.training import (
    MTTrainer,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Seq2Seq Models
    "Seq2SeqEncoder",
    "Seq2SeqDecoder",
    "AttentionSeq2Seq",
    "ConditionalRNNSequence",
    # Transformer MT
    "TransformerMTEncoder",
    "TransformerMTDecoder",
    "TransformerTranslationModel",
    "RelativePositionTransformer",
    # Attention Mechanisms
    "BahdanauAttention",
    "LuongAttention",
    "MultiHeadAttention",
    "ConvolutionalAttention",
    "LinearizedAttention",
    # Beam Search
    "BeamSearchDecoder",
    "DiverseBeamSearch",
    "GreedyDecoder",
    "SamplingDecoder",
    "LengthPenaltyBeamSearch",
    # Metrics
    "BLEUScore",
    "METEORScore",
    "TERScore",
    "chrFScore",
    "COMETScore",
    "TranslationMetrics",
    # Data Utils
    "Vocabulary",
    "MTDataset",
    "MTCollator",
    "build_vocab_from_data",
    "preprocess_parallel_corpus",
    # Training
    "MTTrainer",
    "create_optimizer",
    "create_scheduler",
    "save_checkpoint",
    "load_checkpoint",
]
