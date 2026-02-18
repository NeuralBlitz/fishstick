from dataclasses import dataclass
from typing import Optional

from fishstick.neural_search.dense import (
    BiEncoder,
    BiEncoderConfig,
    ColBERT,
    ColBERTConfig,
    CrossEncoder,
    CrossEncoderConfig,
    DenseRetriever,
    DPR,
    DPRConfig,
)
from fishstick.neural_search.rerank import (
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
    DistillReranker,
    PairwiseReranker,
    PointwiseReranker,
)
from fishstick.neural_search.sparse import (
    BM25,
    BM25Config,
    LearnedSparseRetrieval,
    LearnedSparseConfig,
    SPLADE,
    SPLADEConfig,
    SparseRetriever,
)

__all__ = [
    "DenseRetriever",
    "BiEncoder",
    "BiEncoderConfig",
    "CrossEncoder",
    "CrossEncoderConfig",
    "ColBERT",
    "ColBERTConfig",
    "DPR",
    "DPRConfig",
    "SparseRetriever",
    "BM25",
    "BM25Config",
    "LearnedSparseRetrieval",
    "LearnedSparseConfig",
    "SPLADE",
    "SPLADEConfig",
    "CrossEncoderReranker",
    "CrossEncoderRerankerConfig",
    "DistillReranker",
    "PointwiseReranker",
    "PairwiseReranker",
]
