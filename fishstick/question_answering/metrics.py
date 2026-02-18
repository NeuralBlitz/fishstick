"""
Evaluation Metrics for Question Answering Systems

This module provides comprehensive evaluation metrics for QA systems,
including exact match, F1, BLEU, ROUGE, and domain-specific metrics.

Author: Fishstick AI Framework
"""

from __future__ import annotations

import re
import string
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
import math

import torch
from torch import Tensor
import numpy as np


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation.

    This function performs several normalization steps:
    - Converts to lowercase
    - Removes articles (a, an, the)
    - Removes punctuation
    - Normalizes whitespace

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text.strip()


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if answers match exactly (after normalization)
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_token_f1(
    prediction_tokens: List[str],
    ground_truth_tokens: List[str],
) -> float:
    """Compute token-level F1 score.

    Args:
        prediction_tokens: Predicted tokens
        ground_truth_tokens: Ground truth tokens

    Returns:
        F1 score
    """
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute word-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return compute_token_f1(pred_tokens, gt_tokens)


def compute_char_f1(prediction: str, ground_truth: str) -> float:
    """Compute character-level F1 score.

    Useful for languages without clear word boundaries.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Character-level F1 score
    """
    pred_chars = list(normalize_answer(prediction))
    gt_chars = list(normalize_answer(ground_truth))

    if len(pred_chars) == 0 or len(gt_chars) == 0:
        return 0.0

    common = Counter(pred_chars) & Counter(gt_chars)
    num_same = sum(common.values())

    precision = num_same / len(pred_chars)
    recall = num_same / len(gt_chars)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_em_with_squad_v2(
    prediction: str,
    ground_truths: List[str],
    na_prob_threshold: float = 1.0,
) -> Tuple[bool, float]:
    """Compute exact match for SQuAD 2.0 style evaluation.

    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        na_prob_threshold: Threshold for predicting unanswerable

    Returns:
        Tuple of (is_correct, best_score)
    """
    if not ground_truths:
        return prediction == "", 1.0 if prediction == "" else 0.0

    best_score = 0.0
    for gt in ground_truths:
        score = compute_exact_match(prediction, gt)
        best_score = max(best_score, score)

    return bool(best_score), best_score


def compute_f1_with_squad_v2(
    prediction: str,
    ground_truths: List[str],
    na_prob_threshold: float = 1.0,
) -> Tuple[float, float]:
    """Compute F1 for SQuAD 2.0 style evaluation.

    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        na_prob_threshold: Threshold for predicting unanswerable

    Returns:
        Tuple of (f1_score, best_score)
    """
    if not ground_truths:
        return 0.0, 1.0 if prediction == "" else 0.0

    best_score = 0.0
    for gt in ground_truths:
        score = compute_f1(prediction, gt)
        best_score = max(best_score, score)

    return best_score, best_score


def compute_bleu(
    prediction: str,
    reference: str,
    n: int = 4,
    smooth: bool = True,
) -> float:
    """Compute BLEU score.

    Args:
        prediction: Predicted text
        reference: Reference text
        n: Maximum n-gram order
        smooth: Apply smoothing

    Returns:
        BLEU score
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    precisions = []
    for i in range(1, n + 1):
        pred_ngrams = [
            tuple(pred_tokens[j : j + i]) for j in range(len(pred_tokens) - i + 1)
        ]
        ref_ngrams = [
            tuple(ref_tokens[j : j + i]) for j in range(len(ref_tokens) - i + 1)
        ]

        if len(pred_ngrams) == 0:
            precision = 0.0
        else:
            common = Counter(pred_ngrams) & Counter(ref_ngrams)
            matches = sum(common.values())
            precision = matches / len(pred_ngrams)

        if smooth and precision == 0 and i > 1:
            precisions.append(precisions[-1] / 2)
        else:
            precisions.append(precision)

    if all(p == 0 for p in precisions):
        return 0.0

    brevity_penalty = min(
        1.0,
        math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens))),
    )

    log_precisions = [math.log(p) if p > 0 else float("-inf") for p in precisions]
    avg_log_precision = sum(log_precisions) / n

    bleu = brevity_penalty * math.exp(avg_log_precision)
    return bleu


def compute_rouge_l(
    prediction: str,
    reference: str,
    beta: float = 1.2,
) -> float:
    """Compute ROUGE-L (Longest Common Subsequence) score.

    Args:
        prediction: Predicted text
        reference: Reference text
        beta: Beta parameter for F-measure

    Returns:
        ROUGE-L score
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    m = len(pred_tokens)
    n = len(ref_tokens)

    lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])

    lcs_length = lcs_matrix[m][n]

    precision = lcs_length / m
    recall = lcs_length / n

    if precision + recall == 0:
        return 0.0

    rouge_l = (1 + beta**2) * precision * recall / (recall + beta**2 * precision)

    return rouge_l


def compute_rouge_n(
    prediction: str,
    reference: str,
    n: int = 2,
) -> float:
    """Compute ROUGE-N score.

    Args:
        prediction: Predicted text
        reference: Reference text
        n: N-gram size

    Returns:
        ROUGE-N score
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    pred_ngrams = [
        tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
    ]
    ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]

    common = Counter(pred_ngrams) & Counter(ref_ngrams)
    matches = sum(common.values())

    precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
    recall = matches / len(ref_ngrams) if ref_ngrams else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_mean_reciprocal_rank(
    predictions: List[str],
    ground_truths: List[List[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        predictions: List of predictions
        ground_truths: List of lists of acceptable answers

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for pred, gts in zip(predictions, ground_truths):
        pred_normalized = normalize_answer(pred)

        rank = 1
        found = False
        for gt in gts:
            if normalize_answer(gt) == pred_normalized:
                found = True
                break
            rank += 1

        if found:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_recall_at_k(
    predictions: List[str],
    ground_truths: List[List[str]],
    k: int = 5,
) -> float:
    """Compute Recall@K for multiple ground truths.

    Args:
        predictions: List of predictions
        ground_truths: List of lists of acceptable answers
        k: Number of predictions to consider

    Returns:
        Recall@K score
    """
    recalls = []

    for pred, gts in zip(predictions[:k], ground_truths[:k]):
        pred_normalized = normalize_answer(pred)

        for gt in gts:
            if normalize_answer(gt) == pred_normalized:
                recalls.append(1.0)
                break
        else:
            recalls.append(0.0)

    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_precision_at_k(
    predictions: List[str],
    ground_truths: List[List[str]],
    k: int = 5,
) -> float:
    """Compute Precision@K for multiple ground truths.

    Args:
        predictions: List of predictions
        ground_truths: List of lists of acceptable answers
        k: Number of predictions to consider

    Returns:
        Precision@K score
    """
    precisions = []

    for pred, gts in zip(predictions[:k], ground_truths[:k]):
        pred_normalized = normalize_answer(pred)

        for gt in gts:
            if normalize_answer(gt) == pred_normalized:
                precisions.append(1.0)
                break
        else:
            precisions.append(0.0)

    return sum(precisions) / len(precisions) if precisions else 0.0


def compute_average_precision(
    prediction: str,
    ground_truths: List[str],
    ranked_predictions: Optional[List[str]] = None,
) -> float:
    """Compute Average Precision.

    Args:
        prediction: Primary prediction
        ground_truths: List of acceptable answers
        ranked_predictions: Optional ranked list of predictions

    Returns:
        Average precision score
    """
    if ranked_predictions is None:
        ranked_predictions = [prediction]

    num_hits = 0
    sum_precisions = 0.0

    for i, pred in enumerate(ranked_predictions):
        pred_normalized = normalize_answer(pred)

        for gt in ground_truths:
            if normalize_answer(gt) == pred_normalized:
                num_hits += 1
                precision = num_hits / (i + 1)
                sum_precisions += precision
                break

    if not ground_truths:
        return 0.0

    return sum_precisions / len(ground_truths)


def compute_map(
    predictions: List[str],
    ground_truths: List[List[str]],
    ranked_predictions: Optional[List[List[str]]] = None,
) -> float:
    """Compute Mean Average Precision (MAP).

    Args:
        predictions: List of primary predictions
        ground_truths: List of lists of acceptable answers
        ranked_predictions: Optional ranked lists of predictions

    Returns:
        MAP score
    """
    aps = []

    for i, (pred, gts) in enumerate(zip(predictions, ground_truths)):
        ranked = ranked_predictions[i] if ranked_predictions else None
        ap = compute_average_precision(pred, gts, ranked)
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def compute_ndcg(
    predictions: List[str],
    ground_truths: List[List[str]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG).

    Args:
        predictions: List of predictions (ranked)
        ground_truths: List of lists of acceptable answers with relevance scores
        k: Cutoff position

    Returns:
        NDCG score
    """
    dcg = 0.0

    for i, (pred, gts) in enumerate(zip(predictions[:k], ground_truths[:k])):
        pred_normalized = normalize_answer(pred)

        relevance = 0.0
        for j, gt in enumerate(gts):
            if isinstance(gt, tuple):
                gt_text, rel = gt
            else:
                gt_text = gt
                rel = 1.0 if normalize_answer(gt_text) == pred_normalized else 0.0

            if normalize_answer(gt_text) == pred_normalized:
                relevance = rel
                break

        dcg += relevance / math.log2(i + 2)

    idcg = 0.0
    for i in range(min(k, len(ground_truths))):
        gts = ground_truths[i]
        max_rel = 0.0
        for gt in gts:
            if isinstance(gt, tuple):
                _, rel = gt
            else:
                rel = 1.0
            max_rel = max(max_rel, rel)
        idcg += max_rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


class QAMetrics:
    """Comprehensive QA metrics calculator.

    This class provides a unified interface for computing multiple
    QA evaluation metrics.
    """

    def __init__(
        self,
        include_exact_match: bool = True,
        include_f1: bool = True,
        include_bleu: bool = False,
        include_rouge: bool = False,
        include_retrieval: bool = False,
    ):
        """Initialize QA metrics.

        Args:
            include_exact_match: Include exact match metric
            include_f1: Include F1 metric
            include_bleu: Include BLEU metric
            include_rouge: Include ROUGE metrics
            include_retrieval: Include retrieval metrics
        """
        self.include_exact_match = include_exact_match
        self.include_f1 = include_f1
        self.include_bleu = include_bleu
        self.include_rouge = include_rouge
        self.include_retrieval = include_retrieval

        self.predictions: List[str] = []
        self.ground_truths: List[List[str]] = []

    def reset(self) -> None:
        """Reset accumulated predictions and ground truths."""
        self.predictions = []
        self.ground_truths = []

    def update(
        self,
        prediction: str,
        ground_truths: Union[str, List[str]],
    ) -> None:
        """Update metrics with a single example.

        Args:
            prediction: Predicted answer
            ground_truths: Single or list of acceptable answers
        """
        self.predictions.append(prediction)
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        self.ground_truths.append(ground_truths)

    def compute(self) -> Dict[str, float]:
        """Compute all requested metrics.

        Returns:
            Dictionary of metric names and scores
        """
        results = {}

        if not self.predictions:
            return results

        if self.include_exact_match:
            exact_matches = [
                compute_exact_match(pred, gt)
                for pred, gts in zip(self.predictions, self.ground_truths)
                for gt in gts
            ]
            results["exact_match"] = np.mean(exact_matches)

        if self.include_f1:
            f1_scores = [
                compute_f1(pred, gt)
                for pred, gts in zip(self.predictions, self.ground_truths)
                for gt in gts
            ]
            results["f1"] = np.mean(f1_scores)

        if self.include_bleu:
            bleu_scores = [
                compute_bleu(pred, gt)
                for pred, gts in zip(self.predictions, self.ground_truths)
                for gt in gts
            ]
            results["bleu"] = np.mean(bleu_scores)

        if self.include_rouge:
            rouge_l_scores = [
                compute_rouge_l(pred, gt)
                for pred, gts in zip(self.predictions, self.ground_truths)
                for gt in gts
            ]
            results["rouge_l"] = np.mean(rouge_l_scores)

            rouge_1_scores = [
                compute_rouge_n(pred, gt, 1)
                for pred, gts in zip(self.predictions, self.ground_truths)
                for gt in gts
            ]
            results["rouge_1"] = np.mean(rouge_1_scores)

            rouge_2_scores = [
                compute_rouge_n(pred, gt, 2)
                for pred, gts in zip(self.predictions, self.ground_truths)
                for gt in gts
            ]
            results["rouge_2"] = np.mean(rouge_2_scores)

        if self.include_retrieval:
            results["mrr"] = compute_mean_reciprocal_rank(
                self.predictions, self.ground_truths
            )
            results["recall_at_5"] = compute_recall_at_k(
                self.predictions, self.ground_truths, 5
            )

        return results


class SpanMetrics:
    """Metrics for span-based extractive QA.

    Provides metrics specifically designed for evaluating span
    extraction tasks.
    """

    def __init__(self):
        """Initialize span metrics."""
        self.predicted_spans: List[Tuple[int, int]] = []
        self.gold_spans: List[Tuple[int, int]] = []

    def reset(self) -> None:
        """Reset accumulated spans."""
        self.predicted_spans = []
        self.gold_spans = []

    def update(
        self,
        pred_start: int,
        pred_end: int,
        gold_start: int,
        gold_end: int,
    ) -> None:
        """Update with predicted and gold spans.

        Args:
            pred_start: Predicted start position
            pred_end: Predicted end position
            gold_start: Gold start position
            gold_end: Gold end position
        """
        self.predicted_spans.append((pred_start, pred_end))
        self.gold_spans.append((gold_start, gold_end))

    def compute(self) -> Dict[str, float]:
        """Compute span metrics.

        Returns:
            Dictionary of metrics
        """
        if not self.predicted_spans:
            return {}

        exact = sum(p == g for p, g in zip(self.predicted_spans, self.gold_spans))
        exact_match = exact / len(self.predicted_spans)

        partial_overlaps = []
        for (ps, pe), (gs, ge) in zip(self.predicted_spans, self.gold_spans):
            overlap_start = max(ps, gs)
            overlap_end = min(pe, ge)

            if overlap_start < overlap_end:
                overlap_len = overlap_end - overlap_start
                pred_len = pe - ps
                gold_len = ge - gs

                precision = overlap_len / pred_len if pred_len > 0 else 0.0
                recall = overlap_len / gold_len if gold_len > 0 else 0.0

                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
            else:
                f1 = 0.0

            partial_overlaps.append(f1)

        partial_f1 = sum(partial_overlaps) / len(partial_overlaps)

        return {
            "span_exact_match": exact_match,
            "span_partial_f1": partial_f1,
        }


def compute_coherence_score(
    predictions: List[str],
    contexts: List[str],
) -> float:
    """Compute coherence score for generated answers.

    Measures how coherent the answer is with the context.

    Args:
        predictions: Generated answers
        contexts: Source contexts

    Returns:
        Average coherence score
    """
    scores = []

    for pred, ctx in zip(predictions, contexts):
        pred_set = set(normalize_answer(pred).split())
        ctx_set = set(normalize_answer(ctx).split())

        if len(pred_set) == 0:
            scores.append(0.0)
            continue

        overlap = len(pred_set & ctx_set)
        score = overlap / len(pred_set)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def compute_answer_relevance(
    prediction: str,
    question: str,
    nlp_model: Optional[Any] = None,
) -> float:
    """Compute relevance between answer and question.

    Args:
        prediction: Predicted answer
        question: Original question
        nlp_model: Optional NLP model for semantic similarity

    Returns:
        Relevance score
    """
    pred_tokens = set(normalize_answer(prediction).split())
    question_tokens = set(normalize_answer(question).split())

    if len(pred_tokens) == 0:
        return 0.0

    overlap = len(pred_tokens & question_tokens)
    return overlap / len(pred_tokens)


def evaluate_unanswerable(
    predictions: List[str],
    questions: List[str],
    confidence_scores: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate predictions for unanswerable questions.

    Args:
        predictions: Predicted answers
        questions: Original questions
        confidence_scores: Model confidence scores
        threshold: Threshold for predicting unanswerable

    Returns:
        Dictionary of metrics
    """
    null_predictions = 0
    correct_null = 0
    incorrect_null = 0

    for q, pred, conf in zip(questions, predictions, confidence_scores):
        is_null_predicted = pred == "" or conf < threshold
        is_truly_null = q in ["", None]

        if is_null_predicted:
            null_predictions += 1
            if is_truly_null:
                correct_null += 1
            else:
                incorrect_null += 1

    precision = correct_null / null_predictions if null_predictions > 0 else 0.0

    return {
        "null_precision": precision,
        "null_recall": correct_null / len(questions) if questions else 0.0,
        "null_f1": (
            2
            * precision
            * (correct_null / len(questions))
            / (precision + correct_null / len(questions))
            if (precision + correct_null / len(questions)) > 0
            else 0.0
        ),
    }
