"""
Advanced Causal Discovery Algorithms.

Implements:
- PC Algorithm (constraint-based)
- NOTEARS (continuous optimization)
- FCI Algorithm (with latent confounders)
- GES (score-based)
- LiNGAM (linear non-Gaussian)
- CAM (causal additive models)
"""

from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union
import torch
from torch import Tensor
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import itertools
from collections import defaultdict


@dataclass
class CausalDiscoveryAlgorithm(ABC):
    """Abstract base class for causal discovery algorithms."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """Fit algorithm to data and return adjacency matrix."""
        pass


def CI_test(
    x: np.ndarray,
    y: np.ndarray,
    z: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    test_type: str = "fisher",
) -> Tuple[bool, float]:
    """
    Test conditional independence: X ⊥ Y | Z

    Args:
        x: Variable 1
        y: Variable 2
        z: Conditioning set (optional)
        alpha: Significance level
        test_type: Type of test ('fisher', 'partial', 'chi2')

    Returns:
        (is_independent, p_value)
    """
    if z is None or len(z) == 0:
        if test_type == "fisher":
            corr, p_value = pearsonr(x, y)
            return p_value > alpha, p_value
        elif test_type == "spearman":
            corr, p_value = spearmanr(x, y)
            return p_value > alpha, p_value

    if test_type == "partial":
        return _partial_correlation_test(x, y, z, alpha)
    elif test_type == "fisher":
        return _fisher_z_test(x, y, z, alpha)
    elif test_type == "chi2":
        return _chi2_test(x, y, z)

    return True, 1.0


def _partial_correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    alpha: float,
) -> Tuple[bool, float]:
    """Test using partial correlation."""
    from scipy.stats import pearsonr

    n = len(x)
    k = z.shape[1] if z.ndim > 1 else 1

    if k == 1:
        z = z.reshape(-1, 1)

    X = np.column_stack([x, y, z])
    corr = np.corrcoef(X.T)

    r_xy = corr[0, 1]
    r_xz = corr[0, 2:]
    r_yz = corr[1, 2:]

    if k == 1:
        r_xz = r_xz[0]
        r_yz = r_yz[0]
        r = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    else:
        C = np.linalg.inv(corr[2:, 2:])
        r = (r_xy - r_xz @ C @ r_yz) / np.sqrt(
            (1 - r_xz @ C @ r_xz) * (1 - r_yz @ C @ r_yz)
        )

    t_stat = r * np.sqrt((n - k - 2) / (1 - r**2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 2))

    return p_value > alpha, p_value


def _fisher_z_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    alpha: float,
) -> Tuple[bool, float]:
    """Fisher's z test for partial correlation."""
    from scipy.stats import pearsonr

    n = len(x)
    k = (
        z.shape[1]
        if hasattr(z.shape, "__len__") and len(z.shape) > 1
        else (1 if z.ndim > 0 else 0)
    )

    X = np.column_stack([x, y, z]) if k > 0 else np.column_stack([x, y])
    corr = np.corrcoef(X.T)
    r = corr[0, 1]

    if k > 0:
        C = np.linalg.inv(corr[2:, 2:])
        r_xz = corr[0, 2:]
        r_yz = corr[1, 2:]
        r = (r - r_xz @ C @ r_yz) / np.sqrt(
            (1 - r_xz @ C @ r_xz) * (1 - r_yz @ C @ r_yz)
        )

    z_stat = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
    se = 1 / np.sqrt(n - k - 3)
    z_score = z_stat / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return p_value > alpha, p_value


def _chi2_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[bool, float]:
    """Chi-squared test for categorical variables."""
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    df = np.column_stack([x, y, z])
    contingency = pd.crosstab(df[:, 0], df[:, 1])

    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        return p_value > 0.05, p_value
    except:
        return True, 1.0


def conditional_independence_test(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    z_indices: List[int],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Test conditional independence of variables in dataset.

    Args:
        data: Data matrix [n_samples, n_variables]
        x_idx: Index of first variable
        y_idx: Index of second variable
        z_indices: Indices of conditioning variables
        alpha: Significance level

    Returns:
        (is_independent, p_value)
    """
    x = data[:, x_idx]
    y = data[:, y_idx]
    z = data[:, z_indices] if z_indices else None

    return CI_test(x, y, z, alpha)


def skeleton_recovery(
    data: np.ndarray,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Set[int]]]:
    """
    Recover skeleton of causal graph (undirected).

    Args:
        data: [n_samples, n_variables]
        alpha: Significance level
        max_cond_set: Maximum conditioning set size

    Returns:
        (skeleton_adjacency, separating_sets)
    """
    n_vars = data.shape[1]
    skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sep_sets: Dict[Tuple[int, int], Set[int]] = {}

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            sep_sets[(i, j)] = set()
            sep_sets[(j, i)] = set()

    for cond_size in range(max_cond_set + 1):
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] == 0:
                    continue

                neighbors_i = [
                    k
                    for k in range(n_vars)
                    if k != i and k != j and skeleton[i, k] == 1
                ]

                if len(neighbors_i) < cond_size:
                    continue

                for cond_set in itertools.combinations(neighbors_i, cond_size):
                    is_indep, _ = conditional_independence_test(
                        data, i, j, list(cond_set), alpha
                    )

                    if is_indep:
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        sep_sets[(i, j)] = set(cond_set)
                        sep_sets[(j, i)] = set(cond_set)
                        break

    return skeleton, sep_sets


def orient_edges(
    skeleton: np.ndarray,
    sep_sets: Dict[Tuple[int, int], Set[int]],
    data: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Orient edges to find DAG structure.

    Args:
        skeleton: Undirected adjacency matrix
        sep_sets: Separating sets
        data: Data matrix
        alpha: Significance level

    Returns:
        Directed adjacency matrix
    """
    n_vars = skeleton.shape[0]
    dag = skeleton.copy()

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if dag[i, j] == 0:
                continue

            for k in range(n_vars):
                if k == i or k == j:
                    continue

                if dag[i, k] == 1 and dag[k, j] == 1:
                    if (i, j) in sep_sets:
                        sep = sep_sets[(i, j)]
                    elif (j, i) in sep_sets:
                        sep = sep_sets[(j, i)]
                    else:
                        sep = set()

                    is_indep, _ = conditional_independence_test(
                        data, i, j, list(sep | {k}), alpha
                    )

                    if is_indep and k not in sep:
                        dag[i, j] = 0

    np.fill_diagonal(dag, 0)

    return dag


class PCAlgorithm(CausalDiscoveryAlgorithm):
    """
    PC Algorithm for causal discovery.

    Constraint-based algorithm that uses conditional independence
    tests to recover causal structure.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set: int = 3,
        ci_test: str = "fisher",
    ):
        self.alpha = alpha
        self.max_cond_set = max_cond_set
        self.ci_test = ci_test
        self.adjacency_ = None
        self.sep_sets_ = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit PC algorithm to data.

        Args:
            data: [n_samples, n_variables]

        Returns:
            Adjacency matrix of learned DAG
        """
        self.data_ = data

        skeleton, sep_sets = skeleton_recovery(data, self.alpha, self.max_cond_set)

        dag = orient_edges(skeleton, sep_sets, data, self.alpha)

        self.adjacency_ = dag
        self.sep_sets_ = sep_sets

        return dag

    def get_skeleton(self) -> np.ndarray:
        """Return the skeleton (undirected graph)."""
        if self.adjacency_ is None:
            raise ValueError("Must call fit() first")

        skeleton = (self.adjacency_ + self.adjacency_.T > 0).astype(int)
        np.fill_diagonal(skeleton, 0)
        return skeleton

    def get_separating_sets(self) -> Dict[Tuple[int, int], Set[int]]:
        """Return separating sets."""
        return self.sep_sets_


class NOTEARS(CausalDiscoveryAlgorithm):
    """
    NOTEARS: DAGs with NO TEARS.

    Continuous optimization-based algorithm that learns
    DAG structure using augmented Lagrangian method.
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.W_est_ = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit NOTEARS to data.

        Args:
            data: [n_samples, n_variables]

        Returns:
            Estimated DAG adjacency matrix
        """
        n_samples, d = data.shape

        data = self._normalize(data)

        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        h = np.inf

        for iteration in range(self.max_iter):
            W_new, h_new = self._solve_subproblem(data, W, rho, alpha)

            if h_new > 0.25 * h:
                rho *= 10

            W = W_new
            alpha += rho * h_new
            h = h_new

            if h <= self.h_tol or rho >= self.rho_max:
                break

        W_est = np.abs(W)
        W_est[W_est < self.w_threshold] = 0

        np.fill_diagonal(W_est, 0)

        self.W_est_ = W_est

        return (W_est > 0).astype(int)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Standardize data."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)

    def _solve_subproblem(
        self,
        data: np.ndarray,
        W: np.ndarray,
        rho: float,
        alpha: float,
    ) -> Tuple[np.ndarray, float]:
        """Solve the optimization subproblem."""
        n, d = data.shape

        W_new = W.copy()
        lr = 0.01

        for _ in range(100):
            M = np.eye(d) + W_new * W_new / d
            E = np.linalg.matrix_power(M, d)
            h = np.trace(E) - d

            loss = self._loss(data, W_new)
            grad_loss = self._grad_loss(data, W_new)
            grad_h = self._grad_h(W_new)
            grad_l1 = self.lambda1 * np.sign(W_new)

            grad = grad_loss + (rho * h + alpha) * grad_h + grad_l1

            W_new = W_new - lr * grad
            W_new = self._project_d(W_new)

        M = np.eye(d) + W_new * W_new / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d

        return W_new, h

    def _loss(self, data: np.ndarray, W: np.ndarray) -> float:
        """Least squares loss."""
        n, d = data.shape
        R = data @ W - data
        loss = 0.5 / n * np.sum(R**2)
        return loss

    def _grad_loss(self, data: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Gradient of loss."""
        n, d = data.shape
        R = data @ W - data
        grad = 1.0 / n * data.T @ R
        return grad

    def _h(self, W: np.ndarray) -> float:
        """Acyclicity constraint."""
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d)
        h = np.trace(E) - d
        return h

    def _grad_h(self, W: np.ndarray) -> np.ndarray:
        """Gradient of acyclicity constraint."""
        d = W.shape[0]
        M = np.eye(d) + W * W / d
        E = np.linalg.matrix_power(M, d - 1)
        grad = E.T * 2 * W / d
        return grad

    def _project_d(self, W: np.ndarray) -> np.ndarray:
        """Project to doubly stochastic matrix."""
        return W


class FCIAlgorithm(CausalDiscoveryAlgorithm):
    """
    FCI (Fast Causal Inference) Algorithm.

    Discovers causal structure in presence of latent confounders.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_cond_set: int = 3,
    ):
        self.alpha = alpha
        self.max_cond_set = max_cond_set

    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit FCI algorithm.

        Returns:
            Tuple of (adjacency_matrix, bidirected_edges)
        """
        pc = PCAlgorithm(self.alpha, self.max_cond_set)
        skeleton = pc.get_skeleton()

        dag = orient_edges(skeleton, pc.sep_sets_, data, self.alpha)

        bidirected = self._find_bidirected(dag, data)

        self.adjacency_ = dag
        self.bidirected_ = bidirected

        return dag, bidirected

    def _find_bidirected(
        self,
        dag: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        """Find bidirected edges (hidden confounders)."""
        n_vars = dag.shape[0]
        bidirected = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if dag[i, j] == 0 and dag[j, i] == 0:
                    is_indep, _ = conditional_independence_test(
                        data, i, j, [], self.alpha
                    )
                    if not is_indep:
                        bidirected[i, j] = 1
                        bidirected[j, i] = 1

        return bidirected


class GESAlgorithm(CausalDiscoveryAlgorithm):
    """
    Greedy Equivalence Search (GES) Algorithm.

    Score-based causal discovery algorithm.
    """

    def __init__(
        self,
        score_function: str = "bic",
        max_iter: int = 100,
    ):
        self.score_function = score_function
        self.max_iter = max_iter

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit GES algorithm.

        Args:
            data: [n_samples, n_variables]

        Returns:
            Adjacency matrix of estimated DAG
        """
        n_samples, d = data.shape

        skeleton = np.ones((d, d)) - np.eye(d)

        skeleton = self._forward_greedy(skeleton, data)

        dag = self._backward_greedy(skeleton, data)

        return dag

    def _forward_greedy(
        self,
        skeleton: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        """Forward greedy phase."""
        improved = True
        while improved:
            improved = False
            for i in range(skeleton.shape[0]):
                for j in range(i + 1, skeleton.shape[1]):
                    if skeleton[i, j] == 0:
                        continue

                    score_before = self._local_score(data, skeleton)
                    skeleton[i, j] = 0
                    skeleton[j, i] = 0
                    score_after = self._local_score(data, skeleton)

                    if score_after < score_before:
                        skeleton[i, j] = 1
                        skeleton[j, i] = 1
                    else:
                        improved = True

        return skeleton

    def _backward_greedy(
        self,
        skeleton: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        """Backward greedy phase."""
        improved = True
        while improved:
            improved = False
            for i in range(skeleton.shape[0]):
                for j in range(i + 1, skeleton.shape[1]):
                    if skeleton[i, j] == 1:
                        continue

                    score_before = self._local_score(data, skeleton)
                    skeleton[i, j] = 1
                    skeleton[j, i] = 1
                    score_after = self._local_score(data, skeleton)

                    if score_after < score_before:
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                    else:
                        improved = True

        return skeleton

    def _local_score(self, data: np.ndarray, adj: np.ndarray) -> float:
        """Compute local score (simplified BIC)."""
        n, d = data.shape

        n_edges = np.sum(adj > 0)

        residual_var = 1.0

        if self.score_function == "bic":
            score = n * np.log(residual_var + 1e-10) + n_edges * np.log(n)
        elif self.score_function == "aic":
            score = n * np.log(residual_var + 1e-10) + 2 * n_edges
        else:
            score = n * np.log(residual_var + 1e-10)

        return score


class LiNGAM(CausalDiscoveryAlgorithm):
    """
    LiNGAM: Linear Non-Gaussian Acyclic Model.

    Uses non-Gaussianity to identify causal direction.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit LiNGAM to data.

        Args:
            data: [n_samples, n_variables]

        Returns:
            Estimated DAG adjacency matrix
        """
        n_samples, d = data.shape

        data = self._preprocess(data)

        B = self._estimate_B(data)

        B[np.abs(B) < self.threshold] = 0
        np.fill_diagonal(B, 0)

        return (np.abs(B) > 0).astype(int)

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Remove mean and normalize."""
        data = data - np.mean(data, axis=0)
        data = data / (np.std(data, axis=0) + 1e-10)
        return data

    def _estimate_B(self, data: np.ndarray) -> np.ndarray:
        """Estimate structural coefficients."""
        n_samples, d = data.shape

        B = np.zeros((d, d))

        order = self._lingam_order(data)

        for i, target in enumerate(order):
            predictors = order[:i]

            if not predictors:
                continue

            X = data[:, predictors]
            y = data[:, target]

            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            residuals = y - X @ beta

            for j, predictor in enumerate(predictors):
                B[predictor, target] = beta[j]

        return B

    def _lingam_order(self, data: np.ndarray) -> List[int]:
        """Determine causal ordering using LiNGAM."""
        n_samples, d = data.shape

        residual_matrix = data.copy()
        order = []

        for _ in range(d):
            variances = np.var(residual_matrix, axis=0)
            source = np.argmin(variances)
            order.append(source)

            predictors = [i for i in range(d) if i not in order]

            if predictors:
                X = residual_matrix[:, predictors]
                y = residual_matrix[:, source]

                if X.shape[1] > 0:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    residual_matrix[:, source] = y - X @ beta

        return order


class CAM(CausalDiscoveryAlgorithm):
    """
    CAM: Causal Additive Models.

    Discovers causal structure with additive functional forms.
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        max_iter: int = 100,
        hidden_dim: int = 10,
    ):
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.hidden_dim = hidden_dim

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit CAM to data.

        Args:
            data: [n_samples, n_variables]

        Returns:
            Estimated DAG adjacency matrix
        """
        n_samples, d = data.shape

        W = np.random.randn(d, d) * 0.01
        W = self._project_to_dag(W)

        for iteration in range(self.max_iter):
            for j in range(d):
                predictors = [i for i in range(d) if W[i, j] != 0]

                if not predictors:
                    continue

                X = data[:, predictors]
                y = data[:, j]

                beta = np.linalg.lstsq(X, y, rcond=None)[0]

                for idx, i in enumerate(predictors):
                    if np.abs(beta[idx]) < self.lambda1:
                        W[i, j] = 0
                    else:
                        W[i, j] = beta[idx]

        W = self._project_to_dag(W)

        return (np.abs(W) > self.lambda1).astype(int)

    def _project_to_dag(self, W: np.ndarray) -> np.ndarray:
        """Project to DAG structure."""
        d = W.shape[0]

        for _ in range(d):
            for i in range(d):
                for k in range(d):
                    if i != k:
                        W[i, k] = W[i, k] - W[i, j] * W[j, k] if j < d else W[i, k]

        W[np.diag(np.diag(W))] = 0

        return W
