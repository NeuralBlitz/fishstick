"""
Named Entity Recognition (NER) Module for fishstick AI Framework.

This module provides comprehensive NER tools including:
- Token classification with BiLSTM-CRF
- Sequence labeling with various tagging schemes
- Span-based NER
- Multi-lingual NER support
- CRF layers with constraints
"""

from .token_classifier import (
    BiLSTMTokenClassifier,
    TransformerTokenClassifier,
    NERTokenClassifier,
    build_token_classifier,
)

from .sequence_labeling import (
    SequenceLabeler,
    LSTMSequenceLabeler,
    TransformerSequenceLabeler,
    TaggingScheme,
    BIOESTaggingScheme,
    BIOLUETaggingScheme,
    BIEOS taggingScheme,
)

from .crf_layer import (
    CRF,
    CRFWithConstraints,
    ConditionalRandomField,
    ViterbiDecoder,
)

from .span_ner import (
    SpanNERModel,
    SpanExtractor,
    BoundaryDetector,
    EntityTypeClassifier,
    MergeAndLabel,
    TokenReduction,
)

from .multilingual_ner import (
    CrossLingualNER,
    MultiLingualEncoder,
    LanguageAdapter,
    ZeroShotNER,
    TranslationBasedNER,
)

from .utils import (
    NERLabelMapper,
    NEREvaluator,
    NERDataset,
    convert_bio_to_entities,
    convert_spans_to_bio,
    compute_ner_metrics,
    get_entity_spans,
)

__all__ = [
    # Token Classifier
    "BiLSTMTokenClassifier",
    "TransformerTokenClassifier",
    "NERTokenClassifier",
    "build_token_classifier",
    # Sequence Labeling
    "SequenceLabeler",
    "LSTMSequenceLabeler",
    "TransformerSequenceLabeler",
    "TaggingScheme",
    "BIOESTaggingScheme",
    "BIOLUETaggingScheme",
    "BIEOS_TAGGING_SCHEME",
    # CRF
    "CRF",
    "CRFWithConstraints",
    "ConditionalRandomField",
    "ViterbiDecoder",
    # Span NER
    "SpanNERModel",
    "SpanExtractor",
    "BoundaryDetector",
    "EntityTypeClassifier",
    "MergeAndLabel",
    "TokenReduction",
    # Multi-lingual
    "CrossLingualNER",
    "MultiLingualEncoder",
    "LanguageAdapter",
    "ZeroShotNER",
    "TranslationBasedNER",
    # Utils
    "NERLabelMapper",
    "NEREvaluator",
    "NERDataset",
    "convert_bio_to_entities",
    "convert_spans_to_bio",
    "compute_ner_metrics",
    "get_entity_spans",
]
