"""
Sentiment analysis module for fishstick AI framework.

Provides comprehensive sentiment analysis capabilities including:
- Document-level sentiment classification
- Aspect-based sentiment analysis
- Emotion detection
- Opinion mining
- Multi-class sentiment classification
"""

from .document_sentiment import (
    AttentionPooling,
    BiLSTMDocumentSentiment,
    TransformerDocumentSentiment,
    CNNDocumentSentiment,
    HierarchicalDocumentSentiment,
    SentimentBERTEncoder,
    MultiModalSentiment,
    build_document_sentiment_model,
)

from .aspect_sentiment import (
    AspectExtractor,
    AspectSentimentClassifier,
    RelationAwareAspectExtractor,
    EndToEndAspectSentiment,
    AspectOpinionPairExtractor,
    CategoryAspectSentiment,
    build_aspect_model,
)

from .emotion_detection import (
    EMOTION_LABELS_EKMAN,
    EMOTION_LABELS_PLUTCHIK,
    EMOTION_LABELS_BASIC,
    EmotionClassifier,
    TransformerEmotionClassifier,
    MultiLabelEmotionClassifier,
    DimensionalEmotionModel,
    EmotionGraphNetwork,
    ContextAwareEmotionClassifier,
    MultiTaskEmotionModel,
    build_emotion_model,
)

from .opinion_mining import (
    OpinionTriple,
    OpinionExtractor,
    OpinionRelationClassifier,
    SubjectivityClassifier,
    SentimentTupleExtractor,
    AspectOpinionSentimentClassifier,
    ComparativeOpinionExtractor,
    IntensityClassifier,
    OpinionSentimentScorer,
    SentimentLexiconLearner,
    build_opinion_model,
)

from .multiclass_sentiment import (
    FineGrainedSentimentClassifier,
    StarRatingClassifier,
    RegressionSentimentRegressor,
    MultiLabelSentimentClassifier,
    EnsembleSentimentClassifier,
    GaussianProcessSentimentClassifier,
    ContrastiveSentimentClassifier,
    LabelSmoothingLoss,
    build_multiclass_sentiment_model,
)

from .sentiment_utils import (
    SentimentLexicon,
    SentimentPreprocessor,
    SentimentMetrics,
    AspectTermExtractor,
    NegationHandler,
    SentiWordNetLexicon,
    VADELexicon,
    EmoLexicon,
    SentimentAugmenter,
    SentimentVocabulary,
    build_default_lexicon,
)

__all__ = [
    "AttentionPooling",
    "BiLSTMDocumentSentiment",
    "TransformerDocumentSentiment",
    "CNNDocumentSentiment",
    "HierarchicalDocumentSentiment",
    "SentimentBERTEncoder",
    "MultiModalSentiment",
    "build_document_sentiment_model",
    "AspectExtractor",
    "AspectSentimentClassifier",
    "RelationAwareAspectExtractor",
    "EndToEndAspectSentiment",
    "AspectOpinionPairExtractor",
    "CategoryAspectSentiment",
    "build_aspect_model",
    "EMOTION_LABELS_EKMAN",
    "EMOTION_LABELS_PLUTCHIK",
    "EMOTION_LABELS_BASIC",
    "EmotionClassifier",
    "TransformerEmotionClassifier",
    "MultiLabelEmotionClassifier",
    "DimensionalEmotionModel",
    "EmotionGraphNetwork",
    "ContextAwareEmotionClassifier",
    "MultiTaskEmotionModel",
    "build_emotion_model",
    "OpinionTriple",
    "OpinionExtractor",
    "OpinionRelationClassifier",
    "SubjectivityClassifier",
    "SentimentTupleExtractor",
    "AspectOpinionSentimentClassifier",
    "ComparativeOpinionExtractor",
    "IntensityClassifier",
    "OpinionSentimentScorer",
    "SentimentLexiconLearner",
    "build_opinion_model",
    "FineGrainedSentimentClassifier",
    "StarRatingClassifier",
    "RegressionSentimentRegressor",
    "MultiLabelSentimentClassifier",
    "EnsembleSentimentClassifier",
    "GaussianProcessSentimentClassifier",
    "ContrastiveSentimentClassifier",
    "LabelSmoothingLoss",
    "build_multiclass_sentiment_model",
    "SentimentLexicon",
    "SentimentPreprocessor",
    "SentimentMetrics",
    "AspectTermExtractor",
    "NegationHandler",
    "SentiWordNetLexicon",
    "VADELexicon",
    "EmoLexicon",
    "SentimentAugmenter",
    "SentimentVocabulary",
    "build_default_lexicon",
]
