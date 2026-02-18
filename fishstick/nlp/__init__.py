"""
fishstick NLP Module

Natural language processing tools and models.
"""

from fishstick.nlp.tokenization import (
    BytePairEncoding,
    WordPieceTokenizer,
)
from fishstick.nlp.embeddings import (
    WordEmbedding,
    PositionalEncoding,
)
from fishstick.nlp.models import (
    TextClassifier,
    SequenceTagger,
    LanguageModel,
)
from fishstick.nlp.qa import (
    # Extractive QA
    BiDAF,
    DrQA,
    BERTQA,
    RoBERTaQA,
    DistilBERTQA,
    SpanBERTQA,
    SplinterQA,
    FiDQA,
    # Generative QA
    Seq2SeqQA,
    T5QA,
    BARTQA,
    FusionDecoder,
    # Open-Domain QA
    DenseRetriever,
    SparseRetriever,
    HybridRetriever,
    IterativeRetriever,
    RAG,
    REALM,
    # Multi-hop QA
    DecompRC,
    HotpotQAReader,
    HGN,
    BeamRetriever,
    # Conversational QA
    QuACReader,
    CoQABaseline,
    HistoryEncoder,
    CoreferenceQA,
    # Knowledge Base QA
    KBQA,
    SemanticParser,
    GraphQA,
    ComplexWebQuestions,
    # Evaluation
    ExactMatch,
    F1Score,
    QA_METRICS,
    HumanEvaluation,
    # Training Utilities
    QADataset,
    QATrainer,
    NegativeSampling,
    # Data structures
    QAExample,
    QAPrediction,
    RetrievalResult,
)

__all__ = [
    "BytePairEncoding",
    "WordPieceTokenizer",
    "WordEmbedding",
    "PositionalEncoding",
    "TextClassifier",
    "SequenceTagger",
    "LanguageModel",
    # QA Models
    "BiDAF",
    "DrQA",
    "BERTQA",
    "RoBERTaQA",
    "DistilBERTQA",
    "SpanBERTQA",
    "SplinterQA",
    "FiDQA",
    "Seq2SeqQA",
    "T5QA",
    "BARTQA",
    "FusionDecoder",
    # Retrievers
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "IterativeRetriever",
    "RAG",
    "REALM",
    # Multi-hop
    "DecompRC",
    "HotpotQAReader",
    "HGN",
    "BeamRetriever",
    # Conversational
    "QuACReader",
    "CoQABaseline",
    "HistoryEncoder",
    "CoreferenceQA",
    # KB QA
    "KBQA",
    "SemanticParser",
    "GraphQA",
    "ComplexWebQuestions",
    # Evaluation
    "ExactMatch",
    "F1Score",
    "QA_METRICS",
    "HumanEvaluation",
    # Training
    "QADataset",
    "QATrainer",
    "NegativeSampling",
    # Data structures
    "QAExample",
    "QAPrediction",
    "RetrievalResult",
]
