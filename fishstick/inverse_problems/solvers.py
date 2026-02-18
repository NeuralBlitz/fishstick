"""
Iterative Solvers Module

Implements iterative solvers for inverse problems including:
- Conjugate Gradient method
- ADMM (Alternating Direction Method of Multipliers)
- Primal-Dual algorithms
- Proximal Gradient methods
"""

from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InverseProblemSolver(nn.Module):
    """Base class for inverse problem solvers."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        operator: Callable,
        measurements: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve inverse problem.

        Args:
            operator: Forward operator
            measurements: Measured data
            x_init: Initial solution guess

        Returns:
            Reconstructed solution
        """
        raise NotImplementedError


class ConjugateGradient(InverseProblemSolver):
    """Conjugate Gradient solver for linear systems.

    Solves: Ax = b for symmetric positive definite A

    Example:
        >>> cg = ConjugateGradient(max_iterations=100, tol=1e-6)
        >>> solution = cg(operator, measurements)
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tol: float = 1e-6,
        preconditioner: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.tol = tol
        self.preconditioner = preconditioner

    def forward(
        self,
        operator: Callable,
        b: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Solve using Conjugate Gradient.

        Args:
            operator: Linear operator A (callable or matrix)
            b: Right-hand side
            x_init: Initial guess

        Returns:
            Tuple of (solution, num_iterations)
        """
        if x_init is None:
            x = torch.zeros_like(b)
        else:
            x = x_init.clone()

        if callable(operator):
            Ax = operator(x)
        else:
            Ax = operator @ x

        r = b - Ax
        p = r.clone()

        if self.preconditioner is not None:
            z = self.preconditioner(r)
        else:
            z = r.clone()

        rz = torch.sum(r * z)

        for i in range(self.max_iterations):
            if callable(operator):
                Ap = operator(p)
            else:
                Ap = operator @ p

            pAp = torch.sum(p * Ap)

            if torch.abs(pAp) < 1e-10:
                break

            alpha = rz / pAp

            x = x + alpha * p
            r = r - alpha * Ap

            if self.preconditioner is not None:
                z_new = self.preconditioner(r)
            else:
                z_new = r.clone()

            rz_new = torch.sum(r * z_new)

            if torch.sqrt(torch.sum(r**2)) < self.tol:
                break

            beta = rz_new / rz

            p = r + beta * p
            z = z_new
            rz = rz_new

        return x, i + 1


class ADMM(InverseProblemSolver):
    """Alternating Direction Method of Multipliers.

    Solves: min f(x) + g(z) s.t. Ax + Bz = c

    Example:
        >>> admm = ADMM(num_iterations=100, rho=1.0)
        >>> solution = admm(forward_op, measurements, regularizer)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        rho: float = 1.0,
        alpha: float = 1.0,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.rho = rho
        self.alpha = alpha
        self.tol = tol

    def forward(
        self,
        data_fidelity: Callable,
        regularizer: Callable,
        measurements: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve using ADMM.

        Args:
            data_fidelity: Data fidelity term f(x)
            regularizer: Regularizer g(z)
            measurements: Measurements b
            A: Forward operator matrix
            x_init: Initial guess

        Returns:
            Solution
        """
        if x_init is None:
            x = torch.zeros_like(measurements)
        else:
            x = x_init.clone()

        z = x.clone()
        u = torch.zeros_like(x)

        for _ in range(self.num_iterations):
            x = self._x_update(x, z, u, measurements, data_fidelity, A)
            z = self._z_update(x, z, u, regularizer)
            u = u + self._dual_update(x, z)

        return x

    def _x_update(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        b: torch.Tensor,
        f: Callable,
        A: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """X update step."""
        if A is not None:
            x_new = torch.linalg.solve(
                A.T @ A + self.rho * torch.eye(A.shape[1], device=A.device),
                A.T @ b + self.rho * (z - u),
            )
        else:
            grad = x - b + self.rho * (x - z + u)
            x_new = x - 0.1 * grad

        return x_new

    def _z_update(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        g: Callable,
    ) -> torch.Tensor:
        """Z update step with proximal operator."""
        prox_input = x + u
        z_new = self._prox(g, prox_input, 1.0 / self.rho)
        return z_new

    def _dual_update(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Dual variable update."""
        return x - z

    def _prox(self, g: Callable, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Proximal operator approximation."""
        return x


class PrimalDualHybridGradient(InverseProblemSolver):
    """Primal-Dual Hybrid Gradient (PDHG) method.

    Solves: min_x f(x) + g(Ax) + h(x)

    Example:
        >>> pdhg = PrimalDualHybridGradient(num_iterations=100, theta=1.0)
        >>> solution = pdhg(data_fidelity, regularizer, measurements)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        tau: float = 0.01,
        sigma: float = 0.01,
        theta: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.tau = tau
        self.sigma = sigma
        self.theta = theta

    def forward(
        self,
        data_fidelity: Callable,
        regularizer: Callable,
        measurements: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve using PDHG.

        Args:
            data_fidelity: Data fidelity term
            regularizer: Regularizer
            measurements: Measurements
            A: Forward operator
            x_init: Initial guess

        Returns:
            Solution
        """
        if x_init is None:
            x = torch.zeros_like(measurements)
        else:
            x = x_init.clone()

        x_bar = x.clone()
        y = torch.zeros_like(x)

        for _ in range(self.num_iterations):
            y = self._y_update(y, x_bar, measurements, A)
            x = self._x_update(x, y, measurements, data_fidelity)
            x_bar = x + self.theta * (x - x_bar)

        return x

    def _y_update(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        b: torch.Tensor,
        A: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Dual variable update."""
        if A is not None:
            Ax = A @ x
        else:
            Ax = x

        grad = y + self.sigma * (Ax - b)
        y_new = grad - self.sigma * self._prox_fista(
            grad / self.sigma, 1.0 / self.sigma
        )

        return y_new

    def _x_update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        b: torch.Tensor,
        f: Callable,
    ) -> torch.Tensor:
        """Primal variable update."""
        if callable(f):
            grad = x - b
            x_new = x - self.tau * grad
        else:
            x_new = x - self.tau * (x - b)

        return x_new

    def _prox_fista(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """FISTA proximal step."""
        return torch.sign(x) * torch.relu(torch.abs(x) - lambda_)


class GradientDescent(InverseProblemSolver):
    """Gradient descent solver.

    Basic iterative optimization for inverse problems.

    Example:
        >>> gd = GradientDescent(num_iterations=100, step_size=0.01)
        >>> solution = gd(loss_fn, x_init)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        step_size: float = 0.01,
        momentum: float = 0.0,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.momentum = momentum
        self.tol = tol

    def forward(
        self,
        loss_fn: Callable,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        """Solve using gradient descent.

        Args:
            loss_fn: Loss function to minimize
            x_init: Initial guess

        Returns:
            Solution
        """
        x = x_init.clone()
        v = torch.zeros_like(x)

        for _ in range(self.num_iterations):
            x.requires_grad = True
            loss = loss_fn(x)
            loss.backward()

            with torch.no_grad():
                grad = x.grad
                v = self.momentum * v - self.step_size * grad
                x = x + v

                if torch.norm(grad) < self.tol:
                    break

        return x


class GaussNewton(InverseProblemSolver):
    """Gauss-Newton method for nonlinear least squares.

    Solves: min_x ||f(x) - y||^2

    Example:
        >>> gn = GaussNewton(num_iterations=50)
        >>> solution = gn(residual_fn, x_init)
    """

    def __init__(
        self,
        num_iterations: int = 50,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.tol = tol

    def forward(
        self,
        residual_fn: Callable,
        y: torch.Tensor,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        """Solve using Gauss-Newton.

        Args:
            residual_fn: Residual function r(x) = f(x) - y
            y: Target
            x_init: Initial guess

        Returns:
            Solution
        """
        x = x_init.clone()

        for _ in range(self.num_iterations):
            x.requires_grad = True
            r = residual_fn(x)

            J = []
            for ri in r:
                gi = torch.autograd.grad(ri, x, retain_graph=True)[0]
                J.append(gi)

            J = torch.stack(J)

            r_val = r.detach()

            delta = torch.linalg.lstsq(J, -r_val).solution

            with torch.no_grad():
                x = x + delta

                if torch.norm(delta) < self.tol:
                    break

        return x


class ProximalGradientDescent(InverseProblemSolver):
    """Proximal Gradient Descent.

    Solves: min_x f(x) + g(x) where f is smooth and g is not

    Example:
        >>> pgd = ProximalGradientDescent(num_iterations=100, step_size=0.01)
        >>> solution = pgd(smooth_loss, regularizer, x_init)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        step_size: float = 0.01,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.tol = tol

    def forward(
        self,
        smooth_fn: Callable,
        prox_fn: Callable,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        """Solve using PGD.

        Args:
            smooth_fn: Smooth part f(x)
            prox_fn: Proximal of non-smooth part g(x)
            x_init: Initial guess

        Returns:
            Solution
        """
        x = x_init.clone()

        for _ in range(self.num_iterations):
            x.requires_grad = True
            loss = smooth_fn(x)
            loss.backward()

            with torch.no_grad():
                grad = x.grad
                x_temp = x - self.step_size * grad
                x = prox_fn(x_temp, self.step_size)

                if torch.norm(grad) < self.tol:
                    break

        return x


class FISTA(InverseProblemSolver):
    """Fast Iterative Shrinkage-Thresholding Algorithm.

    Accelerated proximal gradient method for L1 optimization.

    Example:
        >>> fista = FISTA(num_iterations=100, lambda_=0.1)
        >>> solution = fista(measurements, A, lambda_=0.1)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        lambda_: float = 0.1,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.tol = tol

    def forward(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve using FISTA.

        Args:
            A: Forward operator
            b: Measurements
            x_init: Initial guess

        Returns:
            Solution
        """
        if x_init is None:
            x = torch.zeros(A.shape[1], device=A.device)
        else:
            x = x_init.clone()

        y = x.clone()
        t = 1.0

        for _ in range(self.num_iterations):
            x_old = x.clone()

            r = A @ y - b
            grad = A.T @ r

            x = self._soft_threshold(y - 0.1 * grad, self.lambda_)

            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x + ((t - 1) / t_new) * (x - x_old)

            t = t_new

            if torch.norm(x - x_old) < self.tol:
                break

        return x

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding operator."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class SplitBregman(InverseProblemSolver):
    """Split Bregman iteration for L1 problems.

    Solves: min ||Ax - b||_1 + lambda*||x||_1

    Example:
        >>> sb = SplitBregman(num_iterations=50, mu=1.0)
        >>> solution = sb(A, b, lambda_=0.1)
    """

    def __init__(
        self,
        num_iterations: int = 50,
        mu: float = 1.0,
        lambda_: float = 0.1,
        tol: float = 1e-6,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.mu = mu
        self.lambda_ = lambda_
        self.tol = tol

    def forward(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve using Split Bregman.

        Args:
            A: Forward operator
            b: Measurements
            x_init: Initial guess

        Returns:
            Solution
        """
        if x_init is None:
            x = torch.zeros(A.shape[1], device=A.device)
        else:
            x = x_init.clone()

        d = torch.zeros_like(x)
        z = torch.zeros_like(x)

        for _ in range(self.num_iterations):
            x_old = x.clone()

            rhs = A.T @ b + self.mu * (d - z)
            ATA = A.T @ A + self.mu * torch.eye(A.shape[1], device=A.device)
            x = torch.linalg.solve(ATA, rhs)

            d = self._soft_threshold(x + z, self.lambda_ / self.mu)
            z = z + x - d

            if torch.norm(x - x_old) < self.tol:
                break

        return x

    def _soft_threshold(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft thresholding."""
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class StochasticGradientDescent(InverseProblemSolver):
    """Stochastic Gradient Descent for large-scale problems.

    Example:
        >>> sgd = StochasticGradientDescent(num_iterations=1000, lr=0.01)
        >>> solution = sgd(loss_fn, x_init)
    """

    def __init__(
        self,
        num_iterations: int = 1000,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(
        self,
        loss_fn: Callable,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        """Solve using SGD.

        Args:
            loss_fn: Loss function
            x_init: Initial guess

        Returns:
            Solution
        """
        x = x_init.clone()
        v = torch.zeros_like(x)

        for _ in range(self.num_iterations):
            x.requires_grad = True
            loss = loss_fn(x)
            loss.backward()

            with torch.no_grad():
                grad = x.grad

                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * x

                v = self.momentum * v - self.lr * grad
                x = x + v

        return x


class LBFGS(InverseProblemSolver):
    """Limited-memory BFGS optimizer.

    Quasi-Newton method for smooth optimization.

    Example:
        >>> lbfgs = LBFGS(num_iterations=100)
        >>> solution = lbfgs(loss_fn, x_init)
    """

    def __init__(
        self,
        num_iterations: int = 100,
        max_history: int = 10,
        lr: float = 1.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.max_history = max_history
        self.lr = lr

    def forward(
        self,
        loss_fn: Callable,
        x_init: torch.Tensor,
    ) -> torch.Tensor:
        """Solve using L-BFGS.

        Args:
            loss_fn: Loss function
            x_init: Initial guess

        Returns:
            Solution
        """
        x = x_init.clone()

        s_history = []
        y_history = []

        for _ in range(self.num_iterations):
            x.requires_grad = True
            loss = loss_fn(x)
            loss.backward()

            with torch.no_grad():
                grad = x.grad

                if len(s_history) > 0:
                    s = x - x_prev
                    y = grad - grad_prev

                    s_history.append(s)
                    y_history.append(y)

                    if len(s_history) > self.max_history:
                        s_history.pop(0)
                        y_history.pop(0)

                x_prev = x.clone()
                grad_prev = grad.clone()

                direction = self._compute_direction(grad, s_history, y_history)

                x = x - self.lr * direction

        return x

    def _compute_direction(
        self,
        grad: torch.Tensor,
        s_history: list,
        y_history: list,
    ) -> torch.Tensor:
        """Compute L-BFGS direction."""
        if len(s_history) == 0:
            return grad

        alpha = []

        for i in reversed(range(len(s_history))):
            s = s_history[i]
            y = y_history[i]
            rho = 1.0 / (torch.sum(y * s) + 1e-10)
            alpha_i = rho * torch.sum(s * grad)
            alpha.append(alpha_i)
            grad = grad - alpha_i * y

        direction = grad

        for i in range(len(s_history)):
            s = s_history[i]
            y = y_history[i]
            rho = 1.0 / (torch.sum(y * s) + 1e-10)
            beta = rho * torch.sum(y * direction)
            direction = direction + (alpha[i] - beta) * s

        return direction


def create_solver(
    solver_type: str,
    **kwargs,
) -> InverseProblemSolver:
    """Create an iterative solver.

    Args:
        solver_type: Type of solver
        **kwargs: Additional arguments

    Returns:
        Solver instance
    """
    solvers = {
        "cg": ConjugateGradient,
        "admm": ADMM,
        "pdhg": PrimalDualHybridGradient,
        "gd": GradientDescent,
        "gn": GaussNewton,
        "pgd": ProximalGradientDescent,
        "fista": FISTA,
        "split_bregman": SplitBregman,
        "sgd": StochasticGradientDescent,
        "lbfgs": LBFGS,
    }

    if solver_type not in solvers:
        raise ValueError(f"Unknown solver: {solver_type}")

    return solvers[solver_type](**kwargs)
