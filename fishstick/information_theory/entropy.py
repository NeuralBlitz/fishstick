"""
Entropy Estimators Module

Provides various entropy estimation methods including:
- Shannon entropy (discrete and continuous)
- Differential entropy
- Renyi entropy
- Tsallis entropy
- Sample entropy and approximate entropy
"""

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import numpy as np
from scipy.special import digamma, gamma as gamma_func
from sklearn.neighbors import NearestNeighbors


def shannon_entropy(
    p: Tensor,
    base: float = 2.0,
    epsilon: float = 1e-10,
) -> Tensor:
    """
    Compute Shannon entropy H(X) = -sum(p * log(p)).

    Args:
        p: Probability distribution (sums to 1)
        base: Logarithm base (2 for bits, e for nats)
        epsilon: Small constant to avoid log(0)

    Returns:
        Entropy value
    """
    p = torch.clamp(p, min=epsilon)
    p = p / p.sum(dim=-1, keepdim=True)
    return -torch.sum(p * torch.log(p) / torch.log(torch.tensor(base)), dim=-1)


def differential_entropy(
    x: Tensor,
    method: str = "gaussian",
    bandwidth: Optional[float] = None,
) -> Tensor:
    """
    Estimate differential entropy h(X) for continuous distributions.

    Uses plugin estimator with bias correction.

    Args:
        x: Samples (n, d)
        method: Estimation method ("gaussian", "kde", "knn")
        bandwidth: Bandwidth for KDE (Scott's rule if None)

    Returns:
        Differential entropy estimate
    """
    x = x.detach()
    n, d = x.shape

    if method == "gaussian":
        cov = torch.cov(x.T)
        sign, logdet = torch.slogdet(cov + 1e-8 * torch.eye(d, device=x.device))
        return 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet

    elif method == "knn":
        x_np = x.cpu().numpy()
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(x_np)
        distances, _ = nn.kneighbors(x_np)
        rho = distances[:, 1]

        volume_unit_ball = np.pi ** (d / 2) / gamma_func(d / 2 + 1)
        h = (
            d * torch.log(torch.tensor(rho.mean())).item()
            - torch.log(torch.tensor(digamma(n) - digamma(2))).item()
            + torch.log(torch.tensor(volume_unit_ball)).item()
        )
        return torch.tensor(h)

    else:
        raise ValueError(f"Unknown method: {method}")


def renyi_entropy(
    x: Tensor,
    alpha: float = 2.0,
    method: str = "knn",
    k: int = 5,
) -> Tensor:
    """
    Compute Renyi entropy of order alpha.

    H_alpha(X) = (1/(1-alpha)) * log(sum(p^alpha))

    For alpha=1, approaches Shannon entropy.

    Args:
        x: Samples
        alpha: Order of Renyi entropy (alpha != 1)
        method: Estimation method
        k: Number of neighbors for KNN

    Returns:
        Renyi entropy estimate
    """
    if abs(alpha - 1.0) < 1e-6:
        return shannon_entropy(x)

    x = x.detach()
    n, d = x.shape

    if method == "knn":
        x_np = x.cpu().numpy()
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(x_np)
        distances, _ = nn.kneighbors(x_np)
        volumes = (np.pi ** (d / 2) / gamma_func(d / 2 + 1)) * (distances[:, 1:] ** d)

        rho = volumes.mean(axis=1)
        h_alpha = (
            d * torch.log(torch.tensor(k / (n - 1))).item()
            - torch.log(torch.tensor(1 - alpha)).item()
            + (1 / (1 - alpha)) * torch.log(torch.tensor(rho.mean())).item()
        )
        return torch.tensor(h_alpha)

    raise ValueError(f"Unknown method: {method}")


def tsallis_entropy(
    p: Tensor,
    q: float = 1.0,
    epsilon: float = 1e-10,
) -> Tensor:
    """
    Compute Tsallis entropy of order q.

    S_q = (1 - sum(p^q)) / (q - 1)

    For q -> 1, approaches Shannon entropy.

    Args:
        p: Probability distribution
        q: Order of Tsallis entropy (q != 1)
        epsilon: Small constant

    Returns:
        Tsallis entropy
    """
    if abs(q - 1.0) < 1e-6:
        return shannon_entropy(p)

    p = torch.clamp(p, min=epsilon)
    p = p / p.sum(dim=-1, keepdim=True)
    p_q = torch.pow(p, q)
    return (1 - p_q.sum(dim=-1)) / (q - 1)


def conditional_entropy(
    x: Tensor,
    y: Tensor,
    method: str = "knn",
    k: int = 5,
) -> Tensor:
    """
    Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).

    Args:
        x: Random variable X (n, d_x)
        y: Random variable Y (n, d_y)
        method: Estimation method
        k: Number of neighbors

    Returns:
        Conditional entropy H(X|Y)
    """
    xy = torch.cat([x, y], dim=-1)
    h_xy = (
        differential_entropy(xy, method="knn")
        if method == "knn"
        else differential_entropy(xy)
    )
    h_y = (
        differential_entropy(y, method="knn")
        if method == "knn"
        else differential_entropy(y)
    )
    return h_xy - h_y


def joint_entropy(
    x: Tensor,
    y: Tensor,
    method: str = "knn",
) -> Tensor:
    """
    Compute joint entropy H(X,Y).

    Args:
        x: Random variable X
        y: Random variable Y
        method: Estimation method

    Returns:
        Joint entropy H(X,Y)
    """
    xy = torch.cat([x, y], dim=-1)
    return differential_entropy(xy, method=method)


def entropy_rate(
    x: Tensor,
    order: int = 1,
    method: str = "knn",
) -> Tensor:
    """
    Compute entropy rate of a sequence.

    H_rate = lim_{n->inf} H(X_n | X_1, ..., X_{n-1}) / n

    Args:
        x: Sequence (n, d)
        order: Markov order
        method: Estimation method

    Returns:
        Entropy rate
    """
    if x.shape[0] <= order + 1:
        return torch.tensor(0.0)

    h_total = torch.tensor(0.0)
    for i in range(order, x.shape[0]):
        x_context = x[i - order : i]
        x_curr = x[i : i + 1]
        h_cond = conditional_entropy(x_curr, x_context, method=method)
        h_total = h_total + h_cond

    return h_total / (x.shape[0] - order)


def sample_entropy(
    x: Tensor,
    m: int = 2,
    r: float = 0.2,
) -> Tensor:
    """
    Compute sample entropy ( SampEn ) of a time series.

    Measures regularity/complexity of the signal.

    Args:
        x: Time series (n,) or (n, d)
        m: Embedding dimension
        r: Tolerance (usually 0.2 * std)

    Returns:
        Sample entropy (lower = more regular)
    """
    if x.dim() == 2:
        x = x.mean(dim=-1)

    x = x.detach().cpu().numpy()
    n = len(x)

    if r is None:
        r = 0.2 * np.std(x)

    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([x[i : i + m] for i in range(n - m)])
        count = np.zeros(n - m)

        for i in range(n - m):
            for j in range(n - m):
                if i != j and _maxdist(patterns[i], patterns[j]) < r:
                    count[i] += 1

        return np.sum(count) / (n - m)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return torch.tensor(float("inf"))

    return torch.tensor(-np.log(phi_m1 / phi_m))


def approximate_entropy(
    x: Tensor,
    m: int = 2,
    r: float = 0.2,
) -> Tensor:
    """
    Compute approximate entropy (ApEn).

    Args:
        x: Time series
        m: Embedding dimension
        r: Tolerance

    Returns:
        Approximate entropy
    """
    if x.dim() == 2:
        x = x.mean(dim=-1)

    x = x.detach().cpu().numpy()
    n = len(x)

    if r is None:
        r = 0.2 * np.std(x)

    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])

    def _phi(m):
        patterns = np.array([x[i : i + m] for i in range(n - m)])
        count = np.zeros(n - m)

        for i in range(n - m):
            for j in range(n - m):
                if _maxdist(patterns[i], patterns[j]) < r:
                    count[i] += 1

        return np.sum(np.log(count)) / (n - m)

    return torch.tensor(_phi(m) - _phi(m + 1))


def bispectrum_entropy(
    x: Tensor,
    n: int = 256,
) -> Tensor:
    """
    Compute bispectrum-based entropy for signal analysis.

    Args:
        x: Signal
        n: FFT length

    Returns:
        Bispectral entropy
    """
    x = x.detach()
    if x.dim() == 1:
        x = x.unsqueeze(0)

    n = min(n, x.shape[1])
    x = x[:, :n]

    x_fft = torch.fft.rfft(x, dim=-1)
    bispec = x_fft * torch.conj(x_fft) * x_fft

    bispec_abs = torch.abs(bispec)
    bispec_norm = bispec_abs / (bispec_abs.sum() + 1e-10)

    return -torch.sum(bispec_norm * torch.log(bispec_norm + 1e-10))


class EntropyEstimator:
    """
    Flexible entropy estimator with multiple methods.
    """

    def __init__(
        self,
        method: str = "gaussian",
        alpha: Optional[float] = None,
        q: Optional[float] = None,
    ):
        """
        Initialize entropy estimator.

        Args:
            method: Estimation method
            alpha: Renyi entropy order
            q: Tsallis entropy order
        """
        self.method = method
        self.alpha = alpha
        self.q = q

    def __call__(self, x: Tensor) -> Tensor:
        """Estimate entropy of input."""
        if self.method == "shannon":
            return shannon_entropy(x)
        elif self.method == "gaussian":
            return differential_entropy(x, method="gaussian")
        elif self.method == "knn":
            return differential_entropy(x, method="knn")
        elif self.method == "renyi":
            return renyi_entropy(x, alpha=self.alpha or 2.0)
        elif self.method == "tsallis":
            return tsallis_entropy(x, q=self.q or 1.5)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def estimate_rate(
        self,
        x: Tensor,
        order: int = 1,
    ) -> Tensor:
        """Estimate entropy rate."""
        return entropy_rate(x, order=order, method=self.method)
