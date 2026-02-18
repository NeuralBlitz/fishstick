"""
Question Answering Module for Fishstick

This module provides comprehensive implementations for various Question Answering
systems including extractive QA, generative QA, multi-hop QA, reading comprehension,
and domain-specific QA.

Author: Fishstick AI Framework
"""

from fishstick.question_answering.types import (
    QAExample,
    QAPrediction,
    Answer,
    AnswerType,
    Context,
    Question,
    QAConfig,
    QATaskType,
    RetrievalResult,
    EvaluationResult,
    MultiHopStep,
    ReasoningChain,
    DomainConfig,
    load_jsonl,
    save_jsonl,
)

from fishstick.question_answering.base import (
    QABase,
    ExtractiveQABase,
    GenerativeQABase,
    MultiHopQABase,
    ReadingComprehensionBase,
    DomainSpecificQABase,
    RetrieverBase,
    QAPipeline,
    create_qa_system,
)

from fishstick.question_answering.extractive import (
    SpanExtractor,
    BERTExtractiveQA,
    BiDAFPlus,
    DocumentRanker,
    CrossAttentionQA,
)

from fishstick.question_answering.generative import (
    AnswerGenerator,
    CopyMechanism,
    GenerativeQAModel,
    T5GenerativeQA,
    BARTGenerativeQA,
    FusionInDecoder,
)

from fishstick.question_answering.multi_hop import (
    HopAttention,
    DecompositionReasoner,
    GraphReasoningLayer,
    EntityLinking,
    MultiHopReasoner,
    IterativeRetrieval,
)

from fishstick.question_answering.reading_comprehension import (
    CoherenceAttention,
    ArgumentExtractor,
    ContextGraph,
    ReadingComprehensionModel,
    MultiDocumentRC,
)

from fishstick.question_answering.domain_specific import (
    DomainVocabulary,
    MedicalVocabulary,
    LegalVocabulary,
    ScientificVocabulary,
    DomainAdaptationQA,
    MedicalQASystem,
    LegalQASystem,
    ScientificQASystem,
)

from fishstick.question_answering.retrieval import (
    DenseRetriever,
    HybridRetriever,
    KnowledgeAugmentedQA,
    RAGIntegration,
    CompleteQAPipeline,
)

from fishstick.question_answering.training import (
    QADataset,
    QATrainer,
    DataAugmentation,
    NegativeSampler,
)

__all__ = [
    "QAExample",
    "QAPrediction",
    "Answer",
    "AnswerType",
    "Context",
    "Question",
    "QAConfig",
    "QATaskType",
    "RetrievalResult",
    "EvaluationResult",
    "MultiHopStep",
    "ReasoningChain",
    "DomainConfig",
    "load_jsonl",
    "save_jsonl",
    "QABase",
    "ExtractiveQABase",
    "GenerativeQABase",
    "MultiHopQABase",
    "ReadingComprehensionBase",
    "DomainSpecificQABase",
    "RetrieverBase",
    "QAPipeline",
    "create_qa_system",
    "SpanExtractor",
    "BERTExtractiveQA",
    "BiDAFPlus",
    "DocumentRanker",
    "CrossAttentionQA",
    "AnswerGenerator",
    "CopyMechanism",
    "GenerativeQAModel",
    "T5GenerativeQA",
    "BARTGenerativeQA",
    "FusionInDecoder",
    "HopAttention",
    "DecompositionReasoner",
    "GraphReasoningLayer",
    "EntityLinking",
    "MultiHopReasoner",
    "IterativeRetrieval",
    "CoherenceAttention",
    "ArgumentExtractor",
    "ContextGraph",
    "ReadingComprehensionModel",
    "MultiDocumentRC",
    "DomainVocabulary",
    "MedicalVocabulary",
    "LegalVocabulary",
    "ScientificVocabulary",
    "DomainAdaptationQA",
    "MedicalQASystem",
    "LegalQASystem",
    "ScientificQASystem",
    "DenseRetriever",
    "HybridRetriever",
    "KnowledgeAugmentedQA",
    "RAGIntegration",
    "CompleteQAPipeline",
    "QADataset",
    "QATrainer",
    "DataAugmentation",
    "NegativeSampler",
]
