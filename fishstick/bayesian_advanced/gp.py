import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RBFKernel(nn.Module):
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
        self.variance = nn.Parameter(torch.tensor(variance))

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x2 is None:
            x2 = x1

        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale

        x1_sq = (x1**2).sum(dim=-1, keepdim=True)
        x2_sq = (x2**2).sum(dim=-1, keepdim=True)

        dist_sq = x1_sq + x2_sq.t() - 2 * x1 @ x2.t()
        dist_sq = torch.clamp(dist_sq, min=0)

        return self.variance * torch.exp(-0.5 * dist_sq)


class GaussianProcess(nn.Module):
    def __init__(
        self,
        kernel: Optional[nn.Module] = None,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_std = noise_std

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        compute_log_marginal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        K = self.kernel(x_train)
        K_noise = K + (self.noise_std**2) * torch.eye(
            x_train.shape[0], device=x_train.device
        )

        L = torch.linalg.cholesky(K_noise)
        alpha = torch.cholesky_solve(y_train.unsqueeze(-1), L)

        k_star = self.kernel(x_test, x_train)
        mean = k_star @ alpha

        v = torch.cholesky_solve(k_star.t(), L)
        var_diag = self.kernel(x_test).diagonal()
        var = var_diag - (k_star * v).sum(dim=-1)

        var = torch.clamp(var, min=1e-6)

        log_marginal = None
        if compute_log_marginal:
            log_marginal = (
                -0.5 * y_train.unsqueeze(-1).t() @ alpha
                - torch.log(L.diagonal()).sum()
                - 0.5 * x_train.shape[0] * torch.log(torch.tensor(2 * torch.pi))
            )

        return mean.squeeze(-1), var, log_marginal


class DeepKernelGP(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        kernel: Optional[nn.Module] = None,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_std = noise_std

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        phi_train = self.extract_features(x_train)
        phi_test = self.extract_features(x_test)

        gp = GaussianProcess(kernel=self.kernel, noise_std=self.noise_std)
        mean, var, _ = gp(phi_train, y_train, phi_test)

        return mean, var


class SparseGaussianProcess(nn.Module):
    def __init__(
        self,
        n_inducing: int,
        kernel: Optional[nn.Module] = None,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.n_inducing = n_inducing
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_std = noise_std

        self.inducing_points = None

    def _init_inducing_points(self, x: torch.Tensor):
        indices = torch.randperm(x.shape[0])[: self.n_inducing]
        self.inducing_points = nn.Parameter(x[indices].clone())


class SVGP(nn.Module):
    def __init__(
        self,
        n_inducing: int,
        input_dim: int,
        kernel: Optional[nn.Module] = None,
        noise_std: float = 0.1,
        learnable_inducing: bool = True,
    ):
        super().__init__()
        self.n_inducing = n_inducing
        self.input_dim = input_dim
        self.noise_std = noise_std

        self.kernel = kernel if kernel is not None else RBFKernel()

        if learnable_inducing:
            self.inducing_points = nn.Parameter(
                torch.randn(n_inducing, input_dim) * 0.1
            )
        else:
            self.register_buffer("inducing_points", torch.randn(n_inducing, input_dim))

        self.q_mu = nn.Parameter(torch.randn(n_inducing))
        self.q_sqrt = nn.Parameter(torch.ones(n_inducing))

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        K_uu = self.kernel(self.inducing_points)
        K_uu_noise = K_uu + (self.noise_std**2) * torch.eye(
            self.n_inducing, device=K_uu.device
        )

        K_uf = self.kernel(self.inducing_points, x_train)
        K_ff_diag = self.kernel(x_train).diagonal()

        L = torch.linalg.cholesky(K_uu_noise)
        alpha = torch.cholesky_solve(K_uf @ y_train.unsqueeze(-1), L)

        k_star = self.kernel(x_test, self.inducing_points)
        mean = (k_star @ alpha).squeeze(-1)

        v = torch.cholesky_solve(k_star.t(), L)
        K_ff_star_diag = self.kernel(x_test).diagonal()
        var = K_ff_star_diag - (k_star * v).sum(dim=-1)
        var = torch.clamp(var, min=1e-6)

        return mean, var

    def elbo(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        K_uu = self.kernel(self.inducing_points)
        K_uu_noise = K_uu + (self.noise_std**2) * torch.eye(
            self.n_inducing, device=K_uu.device
        )

        K_uf = self.kernel(self.inducing_points, x)
        K_ff_diag = self.kernel(x).diagonal()

        L = torch.linalg.cholesky(K_uu_noise)
        sigma = torch.diag(self.q_sqrt**2)

        alpha = torch.cholesky_solve(
            K_uf @ y.unsqueeze(-1) + L @ self.q_mu.unsqueeze(-1), L
        )

        log_likelihood = (
            -0.5
            * (K_ff_diag - (K_uf.t() @ torch.cholesky_solve(K_uf, L)).diagonal()).sum()
            / (self.noise_std**2)
        )

        kl_term = 0.5 * (
            (self.q_mu**2 / self.q_sqrt**2).sum()
            + torch.log(sigma).sum()
            - torch.log(K_uu_noise).sum()
            + torch.trace(K_uu_noise @ sigma.inverse())
            - self.n_inducing
        )

        elbo = log_likelihood - kl_term
        return elbo


def inducing_points(
    x: torch.Tensor,
    n_inducing: int,
    method: str = "random",
) -> torch.Tensor:
    if method == "random":
        indices = torch.randperm(x.shape[0])[:n_inducing]
        return x[indices]

    elif method == "kmeans":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_inducing, random_state=42)
        kmeans.fit(x.numpy())
        return torch.from_numpy(kmeans.cluster_centers_)

    elif method == "fps":
        indices = [0]
        distances = torch.full((x.shape[0],), float("inf"))

        for _ in range(n_inducing - 1):
            current_idx = indices[-1]
            dist_to_current = ((x - x[current_idx]) ** 2).sum(dim=-1)
            distances = torch.minimum(distances, dist_to_current)
            new_idx = torch.argmax(distances).item()
            indices.append(new_idx)

        return x[indices]

    else:
        raise ValueError(f"Unknown method: {method}")
