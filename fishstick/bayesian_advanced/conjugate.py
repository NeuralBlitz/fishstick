import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConjugateGradientSolver:
    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        x = x0.clone() if x0 is not None else torch.zeros_like(b)
        r = b - A @ x
        p = r.clone()
        rsold = torch.sum(r * r)

        for i in range(self.max_iter):
            Ap = A @ p
            alpha = rsold / (torch.sum(p * Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r)

            if torch.sqrt(rsnew) < self.tol:
                return x, i + 1

            p = r + (rsnew / (rsold + 1e-10)) * p
            rsold = rsnew

        return x, self.max_iter

    def compute_log_det(self, A: torch.Tensor) -> torch.Tensor:
        return 2 * torch.log(torch.linalg.cholesky(A).diag()).sum()


class NNGPKernel(nn.Module):
    def __init__(
        self,
        depth: int = 2,
        width: int = 128,
        activation: str = "erf",
        scale: float = 1.0,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.activation = activation
        self.scale = scale

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "erf":
            return torch.erf(x / torch.sqrt(torch.tensor(2.0)))
        elif self.activation == "relu":
            return F.relu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        else:
            return torch.tanh(x)

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x2 is None:
            x2 = x1

        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(-1)

        k = torch.matmul(x1, x2.t()) * self.scale

        if self.activation == "erf":
            k = (2 / torch.pi).sqrt() * torch.atan(k * torch.pi / 2).sin()
        elif self.activation in ["relu", "gelu", "tanh"]:
            if self.activation == "relu":
                act = F.relu(k)
            elif self.activation == "gelu":
                act = F.gelu(k)
            else:
                act = torch.tanh(k)
            k = act

        return k

    def nngp_kernel_matrix(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        k = self.forward(x1, x2)
        return k

    def forward_multiply(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        k = self.forward(x)
        return k @ y


class NeuralTangentKernel(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        ntk_type: str = "empirical",
    ):
        super().__init__()
        self.network = network
        self.ntk_type = ntk_type

    def compute_ntk(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        compute_eigenvalues: bool = False,
    ) -> torch.Tensor:
        if x2 is None:
            x2 = x1

        self.network.eval()
        x1.requires_grad_(True)
        x2.requires_grad_(True)

        y1 = self.network(x1)
        y2 = self.network(x2)

        if y1.dim() > 1:
            y1 = y1.mean(dim=-1, keepdim=True)
        if y2.dim() > 1:
            y2 = y2.mean(dim=-1, keepdim=True)

        ntk = torch.autograd.grad(
            outputs=y1,
            inputs=x1,
            grad_outputs=torch.ones_like(y1),
            create_graph=True,
        )[0]

        ntk = torch.autograd.grad(
            outputs=y2,
            inputs=x2,
            grad_outputs=ntk,
            create_graph=False,
        )[0]

        if compute_eigenvalues:
            eigenvalues = torch.linalg.eigvalsh(ntk)
            return ntk, eigenvalues

        return ntk

    def empirical_ntk(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x2 is None:
            x2 = x1

        x1 = x1.detach().requires_grad_(True)
        x2 = x2.detach().requires_grad_(True)

        output1 = self.network(x1)
        output2 = self.network(x2)

        if output1.dim() > 1:
            output1 = output1.squeeze(-1)
        if output2.dim() > 1:
            output2 = output2.squeeze(-1)

        n_samples = output1.shape[0]
        ntk = torch.zeros(n_samples, n_samples, device=x1.device)

        for i in range(n_samples):
            grad_i = torch.autograd.grad(
                output1[i],
                self.network.parameters(),
                retain_graph=True,
            )
            grad_i = torch.cat([g.flatten() for g in grad_i if g is not None])

            for j in range(n_samples):
                grad_j = torch.autograd.grad(
                    output2[j],
                    self.network.parameters(),
                    retain_graph=True,
                )
                grad_j = torch.cat([g.flatten() for g in grad_j if g is not None])

                ntk[i, j] = torch.dot(grad_i, grad_j)

        return ntk

    def infinite_width_limit(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x2 is None:
            x2 = x1

        x1 = x1 / torch.sqrt(torch.tensor(x1.shape[-1]))
        x2 = x2 / torch.sqrt(torch.tensor(x2.shape[-1]))

        k = torch.matmul(x1, x2.t())
        return k


def compute_ntk_matrix(
    network: nn.Module,
    x_train: torch.Tensor,
    x_test: Optional[torch.Tensor] = None,
    compute_train_ntk: bool = True,
    compute_test_train_ntk: bool = True,
) -> dict:
    ntk_solver = NeuralTangentKernel(network)

    results = {}

    if compute_train_ntk:
        train_ntk = ntk_solver.empirical_ntk(x_train)
        results["train_ntk"] = train_ntk

    if compute_test_train_ntk and x_test is not None:
        test_train_ntk = ntk_solver.compute_ntk(x_test, x_train)
        results["test_train_ntk"] = test_train_ntk

    if x_test is not None:
        test_ntk = ntk_solver.compute_ntk(x_test)
        results["test_ntk"] = test_ntk

    return results


class ConjugateGradientBayesianRegression(nn.Module):
    def __init__(
        self,
        input_dim: int,
        prior_precision: float = 1.0,
        noise_precision: float = 100.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision
        self.cg_solver = ConjugateGradientSolver()

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        n = X.shape[0]
        d = self.input_dim

        XtX = X.t() @ X
        Xty = X.t() @ y

        A = XtX + (self.prior_precision / self.noise_precision) * torch.eye(
            d, device=X.device
        )
        b = Xty / self.noise_precision

        w, _ = self.cg_solver.solve(A, b)
        self.weight = w

        return w

    def predict(
        self,
        X: torch.Tensor,
        return_variance: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean = X @ self.weight

        if return_variance:
            XtX = X.t() @ X
            A = XtX + (self.prior_precision / self.noise_precision) * torch.eye(
                self.input_dim, device=X.device
            )
            cov = torch.linalg.inv(A) / self.noise_precision
            variance = (X @ cov * X).sum(dim=-1)
            return mean, variance

        return mean, None
