"""
Evaluation Metrics for Recommender Systems.

Comprehensive evaluation metrics for ranking, rating prediction,
and business metrics for recommender systems.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from .base import InteractionMatrix


def precision_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
) -> float:
    """Calculate Precision@K.

    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of relevant item IDs
        k: Number of items to consider

    Returns:
        Precision@K score
    """
    if k <= 0:
        return 0.0

    top_k = recommendations[:k]

    relevant = len(set(top_k) & set(ground_truth))

    return relevant / k


def recall_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
) -> float:
    """Calculate Recall@K.

    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of relevant item IDs
        k: Number of items to consider

    Returns:
        Recall@K score
    """
    if k <= 0 or len(ground_truth) == 0:
        return 0.0

    top_k = recommendations[:k]

    relevant = len(set(top_k) & set(ground_truth))

    return relevant / len(ground_truth)


def ndcg_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
    gains: Optional[str] = "binary",
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG)@K.

    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of relevant item IDs
        k: Number of items to consider
        gains: Type of gains ('binary' or 'graded')

    Returns:
        NDCG@K score
    """
    if k <= 0:
        return 0.0

    top_k = recommendations[:k]

    if gains == "binary":
        relevance = [1.0 if item in ground_truth else 0.0 for item in top_k]
    else:
        relevance = [1.0 for _ in top_k]

    dcg = 0.0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)

    ideal_relevance = [1.0] * min(k, len(ground_truth))
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr_at_k(
    recommendations: List[int],
    ground_truth: List[int],
    k: int,
) -> float:
    """Calculate Mean Reciprocal Rank (MRR)@K.

    Args:
        recommendations: List of recommended item IDs
        ground_truth: List of relevant item IDs
        k: Number of items to consider

    Returns:
        MRR@K score
    """
    if k <= 0 or len(ground_truth) == 0:
        return 0.0

    ground_truth_set = set(ground_truth)

    for i, item in enumerate(recommendations[:k]):
        if item in ground_truth_set:
            return 1.0 / (i + 1)

    return 0.0


def rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate Root Mean Squared Error.

    Args:
        predictions: Predicted ratings
        targets: Actual ratings

    Returns:
        RMSE score
    """
    if len(predictions) == 0:
        return 0.0

    return np.sqrt(np.mean((predictions - targets) ** 2))


def mae(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calculate Mean Absolute Error.

    Args:
        predictions: Predicted ratings
        targets: Actual ratings

    Returns:
        MAE score
    """
    if len(predictions) == 0:
        return 0.0

    return np.mean(np.abs(predictions - targets))


def hit_rate(
    recommendations: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """Calculate Hit Rate@K.

    Args:
        recommendations: List of recommendation lists
        ground_truth: List of ground truth lists
        k: Number of items to consider

    Returns:
        Hit Rate@K score
    """
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return 0.0

    hits = 0

    for recs, truth in zip(recommendations, ground_truth):
        top_k = recs[:k]
        if len(set(top_k) & set(truth)) > 0:
            hits += 1

    return hits / len(recommendations)


def coverage(
    recommendations: List[List[int]],
    n_items: int,
) -> float:
    """Calculate catalog coverage.

    Fraction of items that appear in recommendations.

    Args:
        recommendations: List of recommendation lists
        n_items: Total number of items

    Returns:
        Coverage score
    """
    all_recommended = set()

    for recs in recommendations:
        all_recommended.update(recs)

    return len(all_recommended) / n_items


def diversity(
    recommendations: List[List[int]],
    similarity_matrix: Optional[np.ndarray] = None,
) -> float:
    """Calculate recommendation list diversity.

    Args:
        recommendations: List of recommendation lists
        similarity_matrix: Optional item-item similarity matrix

    Returns:
        Diversity score
    """
    if similarity_matrix is None:
        diversity_scores = []

        for recs in recommendations:
            n = len(recs)
            if n <= 1:
                diversity_scores.append(1.0)
            else:
                avg_sim = 0.0
                for i in range(n):
                    for j in range(i + 1, n):
                        avg_sim += 1.0
                avg_sim /= n * (n - 1) / 2
                diversity_scores.append(1.0 - avg_sim)

        return np.mean(diversity_scores) if diversity_scores else 0.0
    else:
        diversity_scores = []

        for recs in recommendations:
            n = len(recs)
            if n <= 1:
                diversity_scores.append(1.0)
            else:
                avg_sim = 0.0
                for i in range(n):
                    for j in range(i + 1, n):
                        avg_sim += similarity_matrix[recs[i], recs[j]]
                avg_sim /= n * (n - 1) / 2
                diversity_scores.append(1.0 - avg_sim)

        return np.mean(diversity_scores) if diversity_scores else 0.0


def novelty(
    recommendations: List[List[int]],
    item_popularity: Dict[int, float],
    n_users: int,
) -> float:
    """Calculate novelty of recommendations.

    Measures how much recommended items are unknown to users
    based on item popularity.

    Args:
        recommendations: List of recommendation lists
        item_popularity: Dictionary mapping item IDs to popularity scores
        n_users: Total number of users

    Returns:
        Novelty score
    """
    novelty_scores = []

    for recs in recommendations:
        novelty = 0.0

        for item in recs:
            popularity = item_popularity.get(item, 1.0 / n_users)
            novelty += -np.log2(popularity + 1e-10)

        novelty_scores.append(novelty / len(recs))

    return np.mean(novelty_scores) if novelty_scores else 0.0


def map_at_k(
    recommendations: List[List[int]],
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """Calculate Mean Average Precision@K.

    Args:
        recommendations: List of recommendation lists
        ground_truth: List of ground truth lists
        k: Number of items to consider

    Returns:
        MAP@K score
    """
    average_precisions = []

    for recs, truth in zip(recommendations, ground_truth):
        if len(truth) == 0:
            continue

        top_k = recs[:k]

        relevant = 0
        precision_sum = 0.0

        for i, item in enumerate(top_k):
            if item in truth:
                relevant += 1
                precision_sum += relevant / (i + 1)

        if relevant > 0:
            ap = precision_sum / min(k, len(truth))
            average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0.0


def f1_at_k(
    precision: float,
    recall: float,
) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


@dataclass
class RankingMetrics:
    """Container for ranking metrics results."""

    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    map_at_5: float = 0.0
    map_at_10: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision@1": self.precision_at_1,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "mrr": self.mrr,
            "map@5": self.map_at_5,
            "map@10": self.map_at_10,
        }


@dataclass
class RatingMetrics:
    """Container for rating prediction metrics results."""

    rmse: float = 0.0
    mae: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
        }


@dataclass
class DiversityMetrics:
    """Container for diversity metrics results."""

    coverage: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "coverage": self.coverage,
            "diversity": self.diversity,
            "novelty": self.novelty,
        }


class RecommenderMetrics:
    """Comprehensive metrics calculator for recommender systems.

    Supports both offline and online evaluation.

    Attributes:
        k_values: List of K values for ranking metrics
    """

    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.k_values = k_values

    def evaluate_ranking(
        self,
        model,
        test_interactions: InteractionMatrix,
        train_interactions: Optional[InteractionMatrix] = None,
        n_items: Optional[int] = None,
    ) -> RankingMetrics:
        """Evaluate ranking metrics.

        Args:
            model: Recommender model
            test_interactions: Test interaction matrix
            train_interactions: Optional training interactions for exclusion
            n_items: Number of items to consider

        Returns:
            RankingMetrics object
        """
        n_items = n_items or test_interactions.n_items

        all_recommendations = []
        all_ground_truth = []

        for user_idx in range(test_interactions.n_users):
            test_items = test_interactions.get_positive_items(user_idx)

            if len(test_items) == 0:
                continue

            known_items = []
            if train_interactions is not None:
                known_items = train_interactions.get_positive_items(user_idx).tolist()

            recommendations = model.recommend(
                user_idx,
                max(self.k_values),
                exclude_known=True,
            )

            rec_items = [item for item, _ in recommendations]

            all_recommendations.append(rec_items)
            all_ground_truth.append(test_items.tolist())

        metrics = RankingMetrics()

        for k in self.k_values:
            precisions = [
                precision_at_k(recs, truth, k)
                for recs, truth in zip(all_recommendations, all_ground_truth)
            ]

            recalls = [
                recall_at_k(recs, truth, k)
                for recs, truth in zip(all_recommendations, all_ground_truth)
            ]

            ndcgs = [
                ndcg_at_k(recs, truth, k)
                for recs, truth in zip(all_recommendations, all_ground_truth)
            ]

            if k == 1:
                metrics.precision_at_1 = np.mean(precisions)
            if k == 5:
                metrics.precision_at_5 = np.mean(precisions)
                metrics.recall_at_5 = np.mean(recalls)
                metrics.ndcg_at_5 = np.mean(ndcgs)
            if k == 10:
                metrics.precision_at_10 = np.mean(precisions)
                metrics.recall_at_10 = np.mean(recalls)
                metrics.ndcg_at_10 = np.mean(ndcgs)

        mrrs = [
            mrr_at_k(recs, truth, max(self.k_values))
            for recs, truth in zip(all_recommendations, all_ground_truth)
        ]
        metrics.mrr = np.mean(mrrs)

        metrics.map_at_5 = map_at_k(all_recommendations, all_ground_truth, 5)
        metrics.map_at_10 = map_at_k(all_recommendations, all_ground_truth, 10)

        return metrics

    def evaluate_rating(
        self,
        model,
        test_interactions: InteractionMatrix,
    ) -> RatingMetrics:
        """Evaluate rating prediction metrics.

        Args:
            model: Recommender model
            test_interactions: Test interaction matrix

        Returns:
            RatingMetrics object
        """
        coo = test_interactions.ratings.tocoo()

        predictions = []
        targets = []

        for user_idx, item_idx, rating in zip(coo.row, coo.col, coo.data):
            pred = model.predict(user_idx, item_idx)
            predictions.append(pred)
            targets.append(rating)

        predictions = np.array(predictions)
        targets = np.array(targets)

        metrics = RatingMetrics()
        metrics.rmse = rmse(predictions, targets)
        metrics.mae = mae(predictions, targets)

        return metrics

    def evaluate_diversity(
        self,
        model,
        test_interactions: InteractionMatrix,
        n_recommendations: int = 10,
    ) -> DiversityMetrics:
        """Evaluate diversity metrics.

        Args:
            model: Recommender model
            test_interactions: Test interaction matrix
            n_recommendations: Number of recommendations per user

        Returns:
            DiversityMetrics object
        """
        all_recommendations = []

        item_counts = defaultdict(int)

        for user_idx in range(test_interactions.n_users):
            recommendations = model.recommend(
                user_idx,
                n_recommendations,
                exclude_known=True,
            )

            rec_items = [item for item, _ in recommendations]
            all_recommendations.append(rec_items)

            for item in rec_items:
                item_counts[item] += 1

        n_users = test_interactions.n_users
        n_items = test_interactions.n_items

        item_popularity = {
            item: count / (n_users * n_recommendations)
            for item, count in item_counts.items()
        }

        metrics = DiversityMetrics()
        metrics.coverage = coverage(all_recommendations, n_items)
        metrics.diversity = diversity(all_recommendations)
        metrics.novelty = novelty(all_recommendations, item_popularity, n_users)

        return metrics

    def evaluate_all(
        self,
        model,
        test_interactions: InteractionMatrix,
        train_interactions: Optional[InteractionMatrix] = None,
        eval_rating: bool = True,
        eval_diversity: bool = True,
    ) -> Dict[str, float]:
        """Run all evaluation metrics.

        Args:
            model: Recommender model
            test_interactions: Test interaction matrix
            train_interactions: Optional training interactions
            eval_rating: Whether to evaluate rating metrics
            eval_diversity: Whether to evaluate diversity metrics

        Returns:
            Dictionary of all metrics
        """
        results = {}

        ranking_metrics = self.evaluate_ranking(
            model,
            test_interactions,
            train_interactions,
        )
        results.update(ranking_metrics.to_dict())

        if eval_rating:
            try:
                rating_metrics = self.evaluate_rating(model, test_interactions)
                results.update(rating_metrics.to_dict())
            except Exception:
                pass

        if eval_diversity:
            try:
                diversity_metrics = self.evaluate_diversity(model, test_interactions)
                results.update(diversity_metrics.to_dict())
            except Exception:
                pass

        return results


class OfflineEvaluator:
    """Offline evaluator for recommender models.

    Provides utilities for evaluating models on held-out test data.

    Attributes:
        metrics: RecommenderMetrics calculator
    """

    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.metrics = RecommenderMetrics(k_values)

    def evaluate(
        self,
        model,
        test_interactions: InteractionMatrix,
        train_interactions: Optional[InteractionMatrix] = None,
    ) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            model: Recommender model
            test_interactions: Test interaction matrix
            train_interactions: Optional training interactions

        Returns:
            Dictionary of evaluation metrics
        """
        return self.metrics.evaluate_all(
            model,
            test_interactions,
            train_interactions,
        )

    def compare_models(
        self,
        models: Dict[str, any],
        test_interactions: InteractionMatrix,
        train_interactions: Optional[InteractionMatrix] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models.

        Args:
            models: Dictionary of model_name -> model
            test_interactions: Test interaction matrix
            train_interactions: Optional training interactions

        Returns:
            Dictionary of model_name -> metrics
        """
        results = {}

        for name, model in models.items():
            print(f"Evaluating {name}...")
            results[name] = self.evaluate(model, test_interactions, train_interactions)

        return results


class OnlineEvaluator:
    """Online evaluator for A/B testing and live evaluation.

    Provides utilities for tracking online metrics.

    Attributes:
        metrics: Metrics to track
    """

    def __init__(self):
        self.impressions: List[int] = []
        self.clicks: List[int] = []
        self.conversions: List[int] = []
        self.user_satisfaction: List[float] = []

    def log_impression(self, user_id: int, item_id: int):
        """Log an impression."""
        self.impressions.append((user_id, item_id))

    def log_click(self, user_id: int, item_id: int):
        """Log a click."""
        self.clicks.append((user_id, item_id))

    def log_conversion(self, user_id: int, item_id: int):
        """Log a conversion."""
        self.conversions.append((user_id, item_id))

    def log_satisfaction(self, user_id: int, score: float):
        """Log user satisfaction score."""
        self.user_satisfaction.append((user_id, score))

    def click_through_rate(self) -> float:
        """Calculate CTR."""
        if len(self.impressions) == 0:
            return 0.0
        return len(self.clicks) / len(self.impressions)

    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        if len(self.clicks) == 0:
            return 0.0
        return len(self.conversions) / len(self.clicks)

    def average_satisfaction(self) -> float:
        """Calculate average satisfaction score."""
        if len(self.user_satisfaction) == 0:
            return 0.0
        return np.mean([s for _, s in self.user_satisfaction])

    def get_report(self) -> Dict[str, float]:
        """Get evaluation report."""
        return {
            "n_impressions": len(self.impressions),
            "n_clicks": len(self.clicks),
            "n_conversions": len(self.conversions),
            "ctr": self.click_through_rate(),
            "conversion_rate": self.conversion_rate(),
            "avg_satisfaction": self.average_satisfaction(),
        }


def create_recommendation_ground_truth(
    interactions: InteractionMatrix,
    n_items: int,
    exclude_items: Optional[List[int]] = None,
) -> Dict[int, List[int]]:
    """Create ground truth dictionary for evaluation.

    Args:
        interactions: Interaction matrix
        n_items: Number of items to include in ground truth
        exclude_items: Items to exclude

    Returns:
        Dictionary mapping user IDs to list of relevant items
    """
    ground_truth = {}

    for user_idx in range(interactions.n_users):
        items = interactions.get_positive_items(user_idx).tolist()

        if exclude_items:
            items = [i for i in items if i not in exclude_items]

        if items:
            ground_truth[user_idx] = items[:n_items]

    return ground_truth


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, List[int]],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """Evaluate pre-computed recommendations.

    Args:
        recommendations: Dictionary of user_id -> list of recommended items
        ground_truth: Dictionary of user_id -> list of relevant items
        k_values: List of K values

    Returns:
        Dictionary of metrics
    """
    all_recommendations = []
    all_ground_truth = []

    for user_id in recommendations.keys():
        if user_id in ground_truth:
            all_recommendations.append(recommendations[user_id])
            all_ground_truth.append(ground_truth[user_id])

    metrics = {}

    for k in k_values:
        precisions = [
            precision_at_k(recs, truth, k)
            for recs, truth in zip(all_recommendations, all_ground_truth)
        ]
        recalls = [
            recall_at_k(recs, truth, k)
            for recs, truth in zip(all_recommendations, all_ground_truth)
        ]
        ndcgs = [
            ndcg_at_k(recs, truth, k)
            for recs, truth in zip(all_recommendations, all_ground_truth)
        ]

        metrics[f"precision@{k}"] = np.mean(precisions)
        metrics[f"recall@{k}"] = np.mean(recalls)
        metrics[f"ndcg@{k}"] = np.mean(ndcgs)

    mrrs = [
        mrr_at_k(recs, truth, max(k_values))
        for recs, truth in zip(all_recommendations, all_ground_truth)
    ]
    metrics["mrr"] = np.mean(mrrs)

    return metrics
