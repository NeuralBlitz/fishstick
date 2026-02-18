import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import itertools


@dataclass
class PCAlgorithm:
    data: torch.Tensor
    node_names: List[str]
    alpha: float = 0.05
    stable: bool = True
    orient_by_background_knowledge: bool = True

    def __post_init__(self):
        self.n_samples, self.n_vars = self.data.shape
        self.corr_matrix = self._compute_correlation_matrix()
        self.separation_sets: Dict[Tuple[int, int], Set[int]] = {}
        self.graph: List[List[int]] = [[-1] * self.n_vars for _ in range(self.n_vars)]

    def _compute_correlation_matrix(self) -> torch.Tensor:
        data_centered = self.data - self.data.mean(dim=0)
        std = data_centered.std(dim=0)
        std[std == 0] = 1
        data_normalized = data_centered / std
        corr = data_normalized.T @ data_normalized / self.n_samples
        return corr

    def _partial_correlation(self, i: int, j: int, cond_set: Set[int]) -> float:
        if len(cond_set) == 0:
            return self.corr_matrix[i, j].item()

        cond_list = list(cond_set)
        sub_corr = self.corr_matrix[np.ix_([i, j] + cond_list, [i, j] + cond_list)]

        try:
            precision = torch.linalg.inv(torch.tensor(sub_corr, dtype=torch.float32))
            pcorr = -precision[0, 1] / torch.sqrt(precision[0, 0] * precision[1, 1])
            return pcorr.item()
        except:
            return 0.0

    def _conditional_independence_test(
        self, i: int, j: int, cond_set: Set[int]
    ) -> Tuple[bool, float]:
        pcorr = self._partial_correlation(i, j, cond_set)

        n = self.n_samples
        k = len(cond_set)

        if abs(pcorr) < 1e-10:
            return True, 1.0

        try:
            z = 0.5 * torch.log(torch.tensor((1 + pcorr) / (1 - pcorr)))
            std_error = 1.0 / torch.sqrt(torch.tensor(n - k - 3).float())
            z_stat = abs(z / std_error).item()
            p_value = 2 * (1 - stats.norm.cdf(z_stat))
        except:
            p_value = 1.0

        return p_value > self.alpha, p_value

    def learn_structure(
        self,
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], Set[int]]]:
        undirected_graph = self._skeleton_phase()
        self._orientation_phase(undirected_graph)
        return self.graph, self.separation_sets

    def _skeleton_phase(self) -> List[List[int]]:
        skeleton = [[True] * self.n_vars for _ in range(self.n_vars)]

        for i in range(self.n_vars):
            skeleton[i][i] = False

        for depth in range(self.n_vars):
            for i in range(self.n_vars):
                for j in range(i + 1, self.n_vars):
                    if not skeleton[i][j]:
                        continue

                    neighbors_i = self._get_neighbors(i, skeleton, j)

                    if len(neighbors_i) < depth:
                        continue

                    for cond_set in itertools.combinations(neighbors_i, depth):
                        is_indep, p_value = self._conditional_independence_test(
                            i, j, set(cond_set)
                        )

                        if is_indep:
                            skeleton[i][j] = False
                            skeleton[j][i] = False
                            self.separation_sets[(i, j)] = set(cond_set)
                            self.separation_sets[(j, i)] = set(cond_set)
                            break

        return skeleton

    def _get_neighbors(
        self, node: int, skeleton: List[List[int]], exclude: int
    ) -> List[int]:
        neighbors = []
        for j in range(self.n_vars):
            if j != node and j != exclude and skeleton[node][j]:
                neighbors.append(j)
        return neighbors

    def _orientation_phase(self, skeleton: List[List[int]]) -> None:
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if skeleton[i][j]:
                    self.graph[i][j] = 1
                    self.graph[j][i] = 0
                else:
                    self.graph[i][j] = -1
                    self.graph[j][i] = -1

        self._meek_rules()

    def _meek_rules(self) -> None:
        changed = True
        while changed:
            changed = False

            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if i == j:
                        continue

                    if self.graph[i][j] == 1:
                        for k in range(self.n_vars):
                            if k == i or k == j:
                                continue

                            if self.graph[j][k] == 1 and self.graph[k][i] == -1:
                                self.graph[k][i] = 0
                                changed = True

                            for l in range(self.n_vars):
                                if l in [i, j, k]:
                                    continue
                                if (
                                    self.graph[j][l] == 1
                                    and self.graph[l][i] == 1
                                    and self.graph[k][l] == -1
                                ):
                                    self.graph[k][j] = 0
                                    changed = True

    def get_edges(self) -> List[Tuple[str, str]]:
        edges = []
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if self.graph[i][j] == 1:
                    edges.append((self.node_names[i], self.node_names[j]))
                elif self.graph[j][i] == 1:
                    edges.append((self.node_names[j], self.node_names[i]))
        return edges


@dataclass
class FCI:
    data: torch.Tensor
    node_names: List[str]
    alpha: float = 0.05
    orient_collider_bias: bool = True

    def __post_init__(self):
        self.n_samples, self.n_vars = self.data.shape
        self.pc = PCAlgorithm(self.data, self.node_names, self.alpha)
        self.possible_dsep: Dict[Tuple[int, int], Set[int]] = {}
        self.graph: List[List[int]] = []

    def learn_structure(
        self,
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], Set[int]]]:
        skeleton = self.pc._skeleton_phase()
        self._orient_edges(skeleton)
        self._rule_orient_edges()
        return self.graph, self.possible_dsep

    def _orient_edges(self, skeleton: List[List[int]]) -> None:
        self.graph = [[0] * self.n_vars for _ in range(self.n_vars)]

        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if skeleton[i][j]:
                    self.graph[i][j] = 1
                    self.graph[j][i] = -1
                else:
                    self.graph[i][j] = 2
                    self.graph[j][i] = 2

    def _rule_orient_edges(self) -> None:
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                if self.graph[i][j] == 2:
                    self._orient_by_collider(i, j)

    def _orient_by_collider(self, i: int, j: int) -> None:
        for k in range(self.n_vars):
            if k == i or k == j:
                continue

            if self.graph[i][k] == 1 and self.graph[k][j] == 1:
                if self._is_not_cond_set(j, i, k):
                    self.graph[k][j] = 0

    def _is_not_cond_set(self, node: int, i: int, k: int) -> bool:
        sep_set = self.pc.separation_sets.get((i, k), set())
        return node not in sep_set


@dataclass
class GES:
    data: torch.Tensor
    node_names: List[str]
    score_func: str = "bic"
    max_iter: int = 1000

    def __post_init__(self):
        self.n_samples, self.n_vars = self.data.shape
        self.graph: List[List[int]] = [[0] * self.n_vars for _ in range(self.n_vars)]

    def _local_score(self, parent_set: Set[int], child: int) -> float:
        k = len(parent_set)
        if k == 0:
            return -np.log(self.n_samples)

        child_data = self.data[:, child]
        parent_data = self.data[:, list(parent_set)]

        try:
            if parent_data.shape[1] > 0:
                X = torch.cat([torch.ones(self.n_samples, 1), parent_data], dim=1)
                beta = torch.linalg.lstsq(X, child_data).solution
                residuals = child_data - X @ beta
                rss = (residuals**2).sum().item()
                sigma2 = rss / self.n_samples

                if self.score_func == "bic":
                    return self.n_samples * np.log(sigma2) + k * np.log(self.n_samples)
                elif self.score_func == "aic":
                    return self.n_samples * np.log(sigma2) + 2 * k
                elif self.score_func == "mdl":
                    return (
                        self.n_samples * np.log(sigma2) + k * np.log(self.n_samples) / 2
                    )
            else:
                sigma2 = child_data.var().item()
                return np.log(sigma2) if sigma2 > 0 else 0
        except:
            return 0

    def _score_diff(
        self, i: int, j: int, parents_i: Set[int], parents_j: Set[int]
    ) -> float:
        score_i = self._local_score(parents_i, i)
        score_j = self._local_score(parents_j, j)
        return score_j - score_i

    def learn_structure(self) -> List[List[int]]:
        for i in range(self.n_vars):
            self.graph[i][i] = 0

        self._forward_phase()
        self._backward_phase()

        return self.graph

    def _forward_phase(self) -> None:
        for _ in range(self.max_iter):
            improved = False

            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if i == j:
                        continue

                    if self.graph[i][j] == 0:
                        parents_i = self._get_parents(i)
                        new_parents_i = parents_i.union({j})

                        if not self._creates_cycle(j, i):
                            diff = self._score_diff(i, j, parents_i, new_parents_i)

                            if diff > 0:
                                self.graph[j][i] = 1
                                self.graph[i][j] = 0
                                improved = True

            if not improved:
                break

    def _backward_phase(self) -> None:
        for _ in range(self.max_iter):
            improved = False

            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    if i == j or self.graph[j][i] == 0:
                        continue

                    parents_i = self._get_parents(i)

                    if j in parents_i:
                        new_parents_i = parents_i - {j}
                        diff = self._score_diff(i, j, parents_i, new_parents_i)

                        if diff >= 0:
                            self.graph[j][i] = 0
                            improved = True

            if not improved:
                break

    def _get_parents(self, node: int) -> Set[int]:
        parents = set()
        for j in range(self.n_vars):
            if self.graph[j][node] == 1:
                parents.add(j)
        return parents

    def _creates_cycle(self, i: int, j: int) -> bool:
        visited = set()
        stack = [j]

        while stack:
            node = stack.pop()
            if node == i:
                return True
            if node in visited:
                continue
            visited.add(node)

            for k in range(self.n_vars):
                if self.graph[node][k] == 1:
                    stack.append(k)

        return False


@dataclass
class NOTEARS:
    data: torch.Tensor
    node_names: List[str]
    lambda1: float = 0.1
    max_iter: int = 100
    h_tol: float = 1e-8
    rho_max: float = 1e16
    w_threshold: float = 0.3

    def __post_init__(self):
        self.n_samples, self.n_vars = self.data.shape
        self.graph: Optional[torch.Tensor] = None

    def _loss(self, W: torch.Tensor) -> torch.Tensor:
        M = torch.sigmoid(W)
        loss = 0.5 / self.n_samples * torch.sum((self.data - self.data @ M.T) ** 2)
        return loss

    def _gradient(self, W: torch.Tensor) -> torch.Tensor:
        M = torch.sigmoid(W)
        G_loss = (
            -1.0 / self.n_samples * (self.data.t() @ (self.data - self.data @ M.t()))
        )
        G_loss = G_loss.t()

        M_grade = M * (1 - M)
        G_h = M_grade.t() @ M + M.t() @ M_grade

        return G_loss + self.lambda1 * G_h

    def _constraint(self, W: torch.Tensor) -> torch.Tensor:
        d = W.shape[0]
        M = torch.sigmoid(W)
        E = torch.eye(d, device=W.device)
        h = torch.trace(torch.matrix_exp(M * M)) - d
        return h

    def _grad_h(self, W: torch.Tensor) -> torch.Tensor:
        d = W.shape[0]
        M = torch.sigmoid(W)
        G_h = M.t() * M
        for i in range(d):
            G_h[i, i] += torch.trace(M * M)
        G_h = G_h * (M * (1 - M))
        return G_h.t()

    def learn_structure(self) -> torch.Tensor:
        W = torch.zeros(
            self.n_vars, self.n_vars, requires_grad=True, device=self.data.device
        )
        rho = 1.0
        h = float("inf")
        alpha = 1.0

        for _ in range(self.max_iter):
            h_new = self._constraint(W)

            if h_new > 0.5 * h:
                rho *= 10
            else:
                break

            h = h_new

            optimizer = torch.optim.LBFGS([W], lr=1, max_iter=20)

            def closure():
                optimizer.zero_grad()
                loss = self._loss(W)
                h_val = self._constraint(W)
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                total_loss = loss + penalty
                total_loss.backward()
                return total_loss

            optimizer.step(closure)

            h = self._constraint(W).item()
            if h < self.h_tol:
                break

        W_np = W.detach().abs() > self.w_threshold
        W_np = W_np.float()

        W_dag = self._to_dag(W_np)
        self.graph = W_dag

        return W_dag

    def _to_dag(self, W: torch.Tensor) -> torch.Tensor:
        d = W.shape[0]
        W_new = W.clone()

        for _ in range(d):
            for i in range(d):
                for j in range(d):
                    if i != j and W_new[i, j] > 0:
                        path_exists = False
                        for k in range(d):
                            if (
                                k != i
                                and k != j
                                and W_new[i, k] > 0
                                and W_new[k, j] > 0
                            ):
                                path_exists = True
                                break

                        if path_exists:
                            min_weight = min(W_new[i, j], W_new[i, :].max())
                            W_new[i, j] = 0

        return W_new

    def get_adjacency_matrix(self) -> torch.Tensor:
        if self.graph is None:
            self.learn_structure()
        return self.graph

    def get_edges(self) -> List[Tuple[str, str]]:
        if self.graph is None:
            self.learn_structure()

        edges = []
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                if self.graph[i, j] > 0:
                    edges.append((self.node_names[i], self.node_names[j]))

        return edges
