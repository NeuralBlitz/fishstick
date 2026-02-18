"""
Fishstick Summarization Module
==============================

Advanced text summarization tools for the fishstick AI framework.

Modules:
- base: Abstract base classes and data structures
- extractive: Extractive summarization methods (TF-IDF, TextRank, LexRank, MMR)
- abstractive: Abstractive summarization with seq2seq models
- lead_based: Lead-based and headline extraction methods
- neural: Transformer-based neural summarization (BART, T5, Pegasus)
- multi_document: Multi-document summarization techniques
- evaluation: Summarization evaluation metrics
- utils: Common utilities and preprocessing

Author: Fishstick Team
"""

from .base import (
    SummarizerBase,
    SummaryResult,
    Document,
    SummaryConfig,
    Preprocessor,
)
from .extractive import (
    TFIDFSummarizer,
    TextRankSummarizer,
    LexRankSummarizer,
    MMRSummarizer,
    LSAExtractor,
    ClusterExtractor,
)
from .abstractive import (
    Seq2SeqSummarizer,
    PointerGeneratorSummarizer,
    CopyMechanism,
)
from .lead_based import (
    LeadBasedSummarizer,
    LeadKSummarizer,
    HeadlineExtractor,
    LeadWithScoring,
)
from .neural import (
    TransformerSummarizer,
    BARTSummarizer,
    T5Summarizer,
    PegasusSummarizer,
    SummarizationPipeline,
)
from .multi_document import (
    MultiDocSummarizer,
    ClusterBasedSummarizer,
    GraphBasedMultiDoc,
    HierarchicalSummarizer,
    ExtractiveMDS,
)
from .evaluation import (
    RougeEvaluator,
    BleuEvaluator,
    BertScoreEvaluator,
    SummarizationEvaluator,
)
from .utils import (
    TextPreprocessor,
    SentenceTokenizer,
    WordTokenizer,
    StopwordFilter,
)

__all__ = [
    # Base
    "SummarizerBase",
    "SummaryResult",
    "Document",
    "SummaryConfig",
    "Preprocessor",
    # Extractive
    "TFIDFSummarizer",
    "TextRankSummarizer",
    "LexRankSummarizer",
    "MMRSummarizer",
    "LSAExtractor",
    "ClusterExtractor",
    # Abstractive
    "Seq2SeqSummarizer",
    "PointerGeneratorSummarizer",
    "CopyMechanism",
    # Lead-based
    "LeadBasedSummarizer",
    "LeadKSummarizer",
    "HeadlineExtractor",
    "LeadWithScoring",
    # Neural
    "TransformerSummarizer",
    "BARTSummarizer",
    "T5Summarizer",
    "PegasusSummarizer",
    "SummarizationPipeline",
    # Multi-document
    "MultiDocSummarizer",
    "ClusterBasedSummarizer",
    "GraphBasedMultiDoc",
    "HierarchicalSummarizer",
    "ExtractiveMDS",
    # Evaluation
    "RougeEvaluator",
    "BleuEvaluator",
    "BertScoreEvaluator",
    "SummarizationEvaluator",
    # Utils
    "TextPreprocessor",
    "SentenceTokenizer",
    "WordTokenizer",
    "StopwordFilter",
]
