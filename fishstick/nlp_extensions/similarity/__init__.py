"""
Similarity Module

Text similarity measures including cosine similarity, Levenshtein, BLEU, ROUGE, and BERTScore.
"""

from fishstick.nlp_extensions.similarity.cosine_similarity import (
    cosine_similarity,
    cosine_distance,
    batch_cosine_similarity,
    CosineSimilarityScorer,
    SemanticSimilarity,
)
from fishstick.nlp_extensions.similarity.levenshtein import (
    levenshtein_distance,
    normalized_levenshtein,
    levenshtein_ratio,
    partial_ratio,
    token_sort_ratio,
    token_set_ratio,
    LevenshteinScorer,
    FuzzyMatcher,
)
from fishstick.nlp_extensions.similarity.bleu_score import (
    bleu_score,
    sentence_bleu,
    corpus_bleu,
    BLEUScorer,
    compute_bleu_details,
)
from fishstick.nlp_extensions.similarity.rouge_score import (
    rouge_1_score,
    rouge_2_score,
    rouge_l_score,
    rouge_lsum_score,
    rouge_score,
    ROUGEScorer,
    rouge_n_score,
    compute_rouge_details,
)
from fishstick.nlp_extensions.similarity.bertscore import (
    BERTScorer,
    bertscore,
    bertscore_corpus,
)

__all__ = [
    "cosine_similarity",
    "cosine_distance",
    "batch_cosine_similarity",
    "CosineSimilarityScorer",
    "SemanticSimilarity",
    "levenshtein_distance",
    "normalized_levenshtein",
    "levenshtein_ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "LevenshteinScorer",
    "FuzzyMatcher",
    "bleu_score",
    "sentence_bleu",
    "corpus_bleu",
    "BLEUScorer",
    "compute_bleu_details",
    "rouge_1_score",
    "rouge_2_score",
    "rouge_l_score",
    "rouge_lsum_score",
    "rouge_score",
    "ROUGEScorer",
    "rouge_n_score",
    "compute_rouge_details",
    "BERTScorer",
    "bertscore",
    "bertscore_corpus",
]
