"""
Portfolio Optimization Module

Mean-variance, risk parity, Black-Litterman and other optimization methods.
"""

from typing import Optional, Tuple
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor


class MeanVarianceOptimization:
    """Markowitz Mean-Variance Portfolio Optimization."""

    def __init__(
        self,
        expected_returns: Tensor,
        covariance_matrix: Tensor,
        risk_free_rate: float = 0.0,
    ):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate

    def optimal_portfolio(
        self,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        allow_short: bool = False,
    ) -> Tensor:
        """Compute optimal portfolio weights."""
        n_assets = len(self.expected_returns)
        cov = self.covariance_matrix.numpy()
        mu = self.expected_returns.numpy()

        if target_return is not None:
            return self._minimize_variance(target_return, cov, mu, allow_short)
        elif target_risk is not None:
            return self._maximize_return(target_risk, cov, mu, allow_short)
        else:
            return self._tangency_portfolio(cov, mu, allow_short)

    def _minimize_variance(
        self,
        target_return: float,
        cov: ndarray,
        mu: ndarray,
        allow_short: bool,
    ) -> Tensor:
        """Minimize variance for target return."""
        n = len(mu)
        ones = np.ones(n)

        A = 2 * cov
        b = np.zeros(n)

        A = np.block(
            [[A, -mu, -ones], [-mu.reshape(1, -1), 0, 0], [-ones.reshape(1, -1), 0, 0]]
        )
        b = np.zeros(n + 2)
        b[n] = -2 * target_return
        b[n + 1] = -1

        if not allow_short:
            bounds = [(0, None) for _ in range(n)]
        else:
            bounds = [(None, None) for _ in range(n)]

        return self._solve_qp(A, b, bounds)

    def _maximize_return(
        self,
        target_risk: float,
        cov: ndarray,
        mu: ndarray,
        allow_short: bool,
    ) -> Tensor:
        """Maximize return for target risk."""
        n = len(mu)

        from scipy.optimize import minimize

        def portfolio_variance(weights):
            return weights @ cov @ weights

        def portfolio_return(weights):
            return weights @ mu

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {
                "type": "eq",
                "fun": lambda w: np.sqrt(portfolio_variance(w)) - target_risk,
            },
        ]
        bounds = [(0, None) if not allow_short else (None, None) for _ in range(n)]

        result = minimize(
            lambda w: -portfolio_return(w),
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return torch.tensor(result.x)

    def _tangency_portfolio(
        self,
        cov: ndarray,
        mu: ndarray,
        allow_short: bool,
    ) -> Tensor:
        """Compute tangency (maximum Sharpe) portfolio."""
        n = len(mu)
        rf = self.risk_free_rate

        excess_returns = mu - rf

        from scipy.optimize import minimize

        def neg_sharpe(weights):
            port_return = weights @ excess_returns
            port_vol = np.sqrt(weights @ cov @ weights)
            return -(port_return / port_vol)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, None) if not allow_short else (None, None) for _ in range(n)]

        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return torch.tensor(result.x)

    def _solve_qp(self, A: ndarray, b: ndarray, bounds) -> Tensor:
        """Solve quadratic program."""
        from scipy.optimize import minimize

        n = len(b) - 2

        def objective(x):
            return 0.5 * x[:n] @ A[:n, :n] @ x[:n] + b[:n] @ x[:n]

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1},
            {"type": "eq", "fun": lambda x: x[n] - x[n + 1]},
        ]

        result = minimize(
            objective,
            x0=np.zeros(n + 2),
            method="SLSQP",
            bounds=bounds * 2 + [(None, None)] * 2,
        )
        return torch.tensor(result.x[:n])

    def efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Compute efficient frontier."""
        returns_range = torch.linspace(
            float(torch.min(self.expected_returns)),
            float(torch.max(self.expected_returns)),
            n_points,
        )

        risks = []
        weights_list = []

        for target_ret in returns_range:
            try:
                weights = self.optimal_portfolio(
                    target_return=target_ret.item(), allow_short=allow_short
                )
                risk = float(torch.sqrt(weights @ self.covariance_matrix @ weights))
                risks.append(risk)
                weights_list.append(weights)
            except:
                continue

        return torch.tensor(risks), torch.stack(weights_list)


class RiskParity:
    """Risk Parity Portfolio Optimization."""

    def __init__(self, covariance_matrix: Tensor):
        self.covariance_matrix = covariance_matrix

    def compute(
        self,
        target_risk: Optional[Tensor] = None,
        allow_short: bool = False,
    ) -> Tensor:
        """Compute risk parity weights."""
        cov = self.covariance_matrix.numpy()
        n = cov.shape[0]

        def risk_contribution(weights):
            port_vol = np.sqrt(weights @ cov @ weights)
            marginal_risk = cov @ weights
            risk_contrib = weights * marginal_risk / port_vol
            return risk_contrib

        def objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.ones(n) / n * np.sum(rc)
            return np.sum((rc - target_rc) ** 2)

        from scipy.optimize import minimize

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, None) if not allow_short else (None, None) for _ in range(n)]

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return torch.tensor(result.x)


class BlackLitterman:
    """Black-Litterman Portfolio Optimization."""

    def __init__(
        self,
        market_returns: Tensor,
        market_cap: Tensor,
        covariance_matrix: Tensor,
        risk_aversion: float = 2.5,
    ):
        self.market_returns = market_returns
        self.market_cap = market_cap
        self.covariance_matrix = covariance_matrix
        self.risk_aversion = risk_aversion

    def compute(
        self,
        views: Optional[Tensor] = None,
        view_confidence: Optional[Tensor] = None,
        allow_short: bool = False,
    ) -> Tensor:
        """Compute Black-Litterman adjusted weights."""
        cov = self.covariance_matrix.numpy()
        n = len(self.market_returns)

        pi = self.risk_aversion * cov @ self.market_cap.numpy()

        if views is not None and view_confidence is not None:
            P = np.eye(len(views))
            Q = views.numpy()
            omega = np.diag(view_confidence.numpy())

            tau = 1.0 / len(self.market_returns)
            M = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(omega) @ P)
            adjusted_returns = M @ (
                np.linalg.inv(tau * cov) @ pi + P.T @ np.linalg.inv(omega) @ Q
            )
        else:
            adjusted_returns = pi

        mvo = MeanVarianceOptimization(
            torch.tensor(adjusted_returns),
            self.covariance_matrix,
        )
        return mvo.optimal_portfolio(allow_short=allow_short)


class MinimumVariance:
    """Minimum Variance Portfolio."""

    def __init__(self, covariance_matrix: Tensor):
        self.covariance_matrix = covariance_matrix

    def compute(self, allow_short: bool = False) -> Tensor:
        """Compute minimum variance weights."""
        cov = self.covariance_matrix.numpy()
        n = cov.shape[0]

        from scipy.optimize import minimize

        def portfolio_variance(weights):
            return weights @ cov @ weights

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, None) if not allow_short else (None, None) for _ in range(n)]

        result = minimize(
            portfolio_variance,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return torch.tensor(result.x)


class MaximumSharpe:
    """Maximum Sharpe Ratio Portfolio."""

    def __init__(
        self,
        expected_returns: Tensor,
        covariance_matrix: Tensor,
        risk_free_rate: float = 0.0,
    ):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate

    def compute(self, allow_short: bool = False) -> Tensor:
        """Compute maximum Sharpe ratio weights."""
        mvo = MeanVarianceOptimization(
            self.expected_returns,
            self.covariance_matrix,
            self.risk_free_rate,
        )
        return mvo.optimal_portfolio(allow_short=allow_short)


class HierarchicalRiskParity:
    """Hierarchical Risk Parity Portfolio."""

    def __init__(self, returns: Tensor):
        self.returns = returns

    def compute(self) -> Tensor:
        """Compute hierarchical risk parity weights."""
        cov = torch.cov(self.returns.T).numpy()
        n = cov.shape[0]

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        distances = np.sqrt(2 * (1 - np.corrcoef(cov)))
        condensed = squareform(distances)
        linkage_matrix = linkage(condensed, method="single")
        clusters = fcluster(linkage_matrix, t=n / 2, criterion="maxclust")

        weights = np.zeros(n)
        for i in range(1, int(max(clusters)) + 1):
            cluster_idx = np.where(clusters == i)[0]
            cluster_cov = cov[np.ix_(cluster_idx, cluster_idx)]
            cluster_var = np.diag(cluster_cov)
            cluster_weights = 1 / (cluster_var + 1e-8)
            cluster_weights = cluster_weights / np.sum(cluster_weights)
            weights[cluster_idx] = cluster_weights

        weights = weights / np.sum(weights)
        return torch.tensor(weights)


class EqualWeightPortfolio:
    """Equal Weight Portfolio."""

    @staticmethod
    def compute(n_assets: int) -> Tensor:
        """Compute equal weight portfolio."""
        return torch.ones(n_assets) / n_assets


class InverseVolatility:
    """Inverse Volatility Weighted Portfolio."""

    def __init__(self, returns: Tensor):
        self.returns = returns

    def compute(self) -> Tensor:
        """Compute inverse volatility weights."""
        volatilities = torch.std(self.returns, dim=0)
        inv_vol = 1 / (volatilities + 1e-8)
        return inv_vol / torch.sum(inv_vol)


class KellyCriterion:
    """Kelly Criterion for position sizing."""

    @staticmethod
    def optimal_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 1.0,
    ) -> float:
        """Compute Kelly fraction."""
        if avg_loss == 0:
            return 0.0
        b = avg_win / avg_loss
        p = win_rate
        kelly = (b * p - (1 - p)) / b
        return max(0, kelly * fraction)


class PortfolioBacktest:
    """Portfolio backtesting and evaluation."""

    def __init__(self, returns: Tensor, weights: Tensor):
        self.returns = returns
        self.weights = weights

    def compute_portfolio_returns(self) -> Tensor:
        """Compute portfolio returns."""
        return (self.returns * self.weights).sum(dim=1)

    def compute_cumulative_returns(self) -> Tensor:
        """Compute cumulative returns."""
        portfolio_returns = self.compute_portfolio_returns()
        return torch.cumprod(1 + portfolio_returns, dim=0) - 1
