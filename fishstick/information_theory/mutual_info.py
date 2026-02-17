"""
Mutual Information Estimators Module

Provides various MI estimation methods:
- KNN-based estimators
- Kernel-based estimators
- Variational bounds
- InfoNCE and contrastive estimators
"""

from typing import Optional, Callable, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.special import digamma


def mutual_information(
    x: Tensor,
    y: Tensor,
    method: str = "knn",
    k: int = 5,
) -> Tensor:
    """
    Estimate mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).

    Args:
        x: Random variable X (n, d_x)
        y: Random variable Y (n, d_y)
        method: Estimation method ("knn", "kernel", "gaussian")
        k: Number of neighbors for KNN

    Returns:
        Mutual information I(X;Y)
    """
    if method == "knn":
        from .entropy import differential_entropy, joint_entropy

        h_x = differential_entropy(x, method="knn", k=k)
        h_y = differential_entropy(y, method="knn", k=k)
        h_xy = joint_entropy(torch.cat([x, y], dim=-1), method="knn")
        return torch.relu(h_x + h_y - h_xy)

    elif method == "gaussian":
        from .entropy import differential_entropy

        h_x = differential_entropy(x, method="gaussian")
        h_y = differential_entropy(y, method="gaussian")
        xy = torch.cat([x, y], dim=-1)
        h_xy = differential_entropy(xy, method="gaussian")
        return torch.relu(h_x + h_y - h_xy)

    else:
        raise ValueError(f"Unknown method: {method}")


def knn_mi_estimator(
    x: Tensor,
    y: Tensor,
    k: int = 5,
    normalized: bool = True,
) -> Tensor:
    """
    KNN-based mutual information estimator.

    Based on Kraskov et al. (2004) - the "撤" estimator.

    Args:
        x: Samples from X (n, d_x)
        y: Samples from Y (n, d_y)
        k: Number of neighbors
        normalized: Whether to return normalized MI

    Returns:
        Mutual information estimate
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    n = x.shape[0]

    xy = np.concatenate([x, y], axis=1)

    nn_xy = NearestNeighbors(n_neighbors=k + 1)
    nn_xy.fit(xy)
    distances_xy, _ = nn_xy.kneighbors(xy)
    distances_xy = distances_xy[:, k]

    nn_x = NearestNeighbors(n_neighbors=k + 1)
    nn_x.fit(x)
    distances_x, _ = nn_x.kneighbors(x)
    distances_x = distances_x[:, k]

    nn_y = NearestNeighbors(n_neighbors=k + 1)
    nn_y.fit(y)
    distances_y, _ = nn_y.kneighbors(y)
    distances_y = distances_y[:, k]

    d_x = x.shape[1]
    d_y = y.shape[1]

    log_term = np.log(distances_xy + 1e-10) - np.log(np.maximum(distances_x, 1e-10))
    log_term = np.log(np.maximum(distances_y, 1e-10)) - log_term

    mi = (
        digamma(n)
        + digamma(k)
        - np.mean(
            digamma(np.sum(distances_xy[:, None] >= distances_x[None, :], axis=1) + 1)
            + digamma(np.sum(distances_xy[:, None] >= distances_y[None, :], axis=1) + 1)
        )
    )

    mi_value = (
        digamma(k)
        + np.log(n)
        - np.mean(np.log(distances_x * distances_y / (distances_xy + 1e-10) + 1e-10))
    )
    mi_value = (
        digamma(n)
        + digamma(k)
        - np.mean(
            np.log(distances_x + 1e-10)
            + np.log(distances_y + 1e-10)
            - np.log(distances_xy + 1e-10)
        )
    )

    mi_tensor = torch.tensor(mi_value)

    if normalized:
        h_x = _estimate_entropy(x, k)
        h_y = _estimate_entropy(y, k)
        mi_tensor = mi_tensor / torch.sqrt(h_x * h_y + 1e-10)

    return mi_tensor


def _estimate_entropy(x: np.ndarray, k: int) -> Tensor:
    """Helper to estimate entropy."""
    n, d = x.shape

    nn_model = NearestNeighbors(n_neighbors=k + 1)
    nn_model.fit(x)
    distances, _ = nn_model.kneighbors(x)
    distances = distances[:, k]

    volume_sphere = np.pi ** (d / 2) / np.math.gamma(d / 2 + 1)

    h = np.log(n * volume_sphere * distances**d + 1e-10).mean()
    return torch.tensor(h)


def kernel_mi_estimator(
    x: Tensor,
    y: Tensor,
    sigma: Optional[float] = None,
) -> Tensor:
    """
    Kernel-based mutual information estimator.

    Uses Gaussian kernels and the formula:
    I(X;Y) = E[log(k(x,y)/k(x)k(y))]

    Args:
        x: Samples from X (n, d_x)
        y: Samples from Y (n, d_y)
        sigma: Kernel bandwidth (median heuristic if None)

    Returns:
        MI estimate
    """
    x = x.detach()
    y = y.detach()
    n = x.shape[0]

    if sigma is None:
        x_dist = torch.cdist(x, x)
        y_dist = torch.cdist(y, y)
        sigma_x = torch.median(x_dist[x_dist > 0])
        sigma_y = torch.median(y_dist[y_dist > 0])
    else:
        sigma_x = sigma_y = sigma

    k_x = torch.exp(-0.5 * torch.cdist(x, x) ** 2 / sigma_x**2)
    k_y = torch.exp(-0.5 * torch.cdist(y, y) ** 2 / sigma_y**2)
    k_xy = torch.exp(
        -0.5
        * (torch.cdist(x, x) ** 2 / sigma_x**2 + torch.cdist(y, y) ** 2 / sigma_y**2)
    )

    k_x_mean = k_x.mean()
    k_y_mean = k_y.mean()
    k_xy_mean = k_xy.mean()

    mi = torch.log(k_xy_mean / (k_x_mean * k_y_mean + 1e-10))
    return torch.relu(mi)


def info_nce(
    x: Tensor,
    y: Tensor,
    hidden_dim: int = 64,
    temperature: float = 0.1,
) -> Tensor:
    """
    InfoNCE lower bound on mutual information.

    I(x,y) >= E[log(similarity(x_i, y_i))] - E[log(mean_j(similarity(x_i, y_j)))]

    Args:
        x: Samples from X (n, d)
        y: Samples from Y (n, d)
        hidden_dim: Projection dimension
        temperature: Temperature for softmax

    Returns:
        InfoNCE lower bound
    """
    batch_size = x.shape[0]

    proj_x = nn.Linear(x.shape[1], hidden_dim).to(x.device)
    proj_y = nn.Linear(y.shape[1], hidden_dim).to(y.device)

    h_x = torch.tanh(proj_x(x))
    h_y = torch.tanh(proj_y(y))

    logits = torch.mm(h_x, h_y.T) / temperature

    labels = torch.arange(batch_size, device=x.device)

    loss_x_y = nn.functional.cross_entropy(logits, labels)
    loss_y_x = nn.functional.cross_entropy(logits.T, labels)

    mi = -(loss_x_y + loss_y_x) / 2
    return mi


def variational_mi_bound(
    x: Tensor,
    y: Tensor,
    encoder: Optional[nn.Module] = None,
    classifier: Optional[nn.Module] = None,
    n_samples: int = 5,
) -> Tensor:
    """
    Variational bound on mutual information using Donsker-Varadhan representation.

    I(X;Y) >= sup_{f} E[f(x,y)] - log(E[e^{f(x',y')}])

    Args:
        x: Samples from X
        y: Samples from Y
        encoder: Optional encoder network
        classifier: Discriminator network f(x,y)
        n_samples: Number of samples for expectation

    Returns:
        Variational MI lower bound
    """
    batch_size = x.shape[0]

    if encoder is not None:
        x = encoder(x)
        y = encoder(y)

    if classifier is None:
        classifier = nn.Sequential(
            nn.Linear(x.shape[1] + y.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(x.device)

    xy = torch.cat([x, y], dim=-1)
    shuffled_indices = torch.randperm(batch_size)
    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    xy_shuffled = torch.cat([x_shuffled, y_shuffled], dim=-1)

    f_xy = classifier(xy)
    f_shuffled = classifier(xy_shuffled)

    mi = f_xy.mean() - torch.log(torch.exp(f_shuffled).mean() + 1e-10)

    return torch.relu(mi)


def mine_estimator(
    x: Tensor,
    y: Tensor,
    hidden_dim: int = 128,
    learning_rate: float = 1e-3,
    n_iterations: int = 500,
) -> Tensor:
    """
    MINE (Mutual Information Neural Estimation) estimator.

    Args:
        x: Samples from X
        y: Samples from Y
        hidden_dim: Hidden dimension for network
        learning_rate: Learning rate
        n_iterations: Training iterations

    Returns:
        MI estimate
    """
    x = x.detach()
    y = y.detach()
    batch_size = x.shape[0]

    class MINE(nn.Module):
        def __init__(self, dim_x, dim_y, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim_x + dim_y, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, y):
            xy = torch.cat([x, y], dim=-1)
            return self.net(xy)

    mine = MINE(x.shape[1], y.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(mine.parameters(), lr=learning_rate)

    for _ in range(n_iterations):
        shuffled_idx = torch.randperm(batch_size)
        x_shuffled = x[shuffled_idx]

        f_xy = mine(x, y)
        f_x_y = mine(x, x_shuffled)

        loss = -f_xy.mean() + torch.log(f_x_y.exp().mean() + 1e-10)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        f_xy = mine(x, y)
        shuffled_idx = torch.randperm(batch_size)
        f_x_y = mine(x, x[shuffled_idx])
        mi = f_xy.mean() - torch.log(f_x_y.exp().mean() + 1e-10)

    return torch.relu(mi)


def conditional_mutual_information(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    method: str = "knn",
    k: int = 5,
) -> Tensor:
    """
    Compute conditional mutual information I(X;Y|Z).

    I(X;Y|Z) = H(X|Z) - H(X|Y,Z)

    Args:
        x: Random variable X
        y: Random variable Y
        z: Random variable Z (conditioning)
        method: Estimation method
        k: KNN parameter

    Returns:
        Conditional mutual information
    """
    from .entropy import differential_entropy, joint_entropy

    xz = torch.cat([x, z], dim=-1)
    yz = torch.cat([y, z], dim=-1)
    xyz = torch.cat([x, y, z], dim=-1)

    h_xz = differential_entropy(xz, method=method)
    h_yz = differential_entropy(yz, method=method)
    h_z = differential_entropy(z, method=method)
    h_xyz = differential_entropy(xyz, method=method)

    return torch.relu(h_xz + h_yz - h_z - h_xyz)


def total_correlation(
    variables: Tensor,
    method: str = "knn",
) -> Tensor:
    """
    Compute total correlation (multi-information) for multiple variables.

    TC(X_1,...,X_k) = H(X_1,...,X_k) - sum_i H(X_i)

    Args:
        variables: Stack of variables (n, k, d_i)
        method: Estimation method

    Returns:
        Total correlation
    """
    from .entropy import differential_entropy, joint_entropy

    if variables.dim() == 2:
        variables = variables.unsqueeze(1)

    joint = torch.cat([variables[:, i] for i in range(variables.shape[1])], dim=-1)
    h_joint = differential_entropy(joint, method=method)

    marginals = [
        differential_entropy(variables[:, i], method=method)
        for i in range(variables.shape[1])
    ]
    h_marginals = sum(marginals)

    return torch.relu(h_joint - h_marginals)


def pairwise_mutual_information(
    x: Tensor,
    method: str = "knn",
    k: int = 5,
) -> Tensor:
    """
    Compute pairwise mutual information for all pairs of variables.

    Args:
        x: Variables (n, k, d)
        method: Estimation method
        k: KNN parameter

    Returns:
        Pairwise MI matrix (k, k)
    """
    n_vars = x.shape[1]
    mi_matrix = torch.zeros(n_vars, n_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            mi = mutual_information(x[:, i], x[:, j], method=method, k=k)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

    return mi_matrix


class MutualInformationEstimator:
    """
    Flexible mutual information estimator.
    """

    def __init__(
        self,
        method: str = "knn",
        k: int = 5,
        temperature: float = 0.1,
    ):
        """
        Initialize MI estimator.

        Args:
            method: Estimation method
            k: KNN parameter
            temperature: Temperature for InfoNCE
        """
        self.method = method
        self.k = k
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """Estimate I(X;Y)."""
        if self.method == "knn":
            return knn_mi_estimator(x, y, k=self.k)
        elif self.method == "kernel":
            return kernel_mi_estimator(x, y)
        elif self.method == "info_nce":
            return info_nce(x, y, temperature=self.temperature)
        elif self.method == "gaussian":
            return mutual_information(x, y, method="gaussian")
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def conditional(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        """Estimate I(X;Y|Z)."""
        return conditional_mutual_information(x, y, z, method=self.method, k=self.k)

    def total_correlation(self, variables: Tensor) -> Tensor:
        """Estimate total correlation."""
        return total_correlation(variables, method=self.method)
