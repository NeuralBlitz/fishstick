"""
Risk Assessment Module

Risk metrics, VaR, CVaR, and risk-adjusted performance measures.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor


class ValueAtRisk:
    """Value at Risk (VaR) calculations using multiple methods."""

    @staticmethod
    def historical(
        returns: Tensor,
        confidence_level: float = 0.95,
    ) -> float:
        """Historical VaR."""
        return float(torch.quantile(returns, 1 - confidence_level))

    @staticmethod
    def parametric(
        returns: Tensor,
        confidence_level: float = 0.95,
    ) -> float:
        """Parametric (Gaussian) VaR."""
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - confidence_level))
        return float(mu + z * sigma)

    @staticmethod
    def monte_carlo(
        returns: Tensor,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
    ) -> float:
        """Monte Carlo VaR."""
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        simulated = torch.normal(mu, sigma, (n_simulations,))
        return float(torch.quantile(simulated, 1 - confidence_level))

    @staticmethod
    def cornish_fisher(
        returns: Tensor,
        confidence_level: float = 0.95,
    ) -> float:
        """Cornish-Fisher VaR (accounting for skewness and kurtosis)."""
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - confidence_level))

        skew = torch.mean(((returns - mu) / sigma) ** 3)
        kurtosis = torch.mean(((returns - mu) / sigma) ** 4) - 3

        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * (kurtosis) / 24
            - (2 * z**3 - 5 * z) * (skew**2) / 36
        )

        return float(mu + z_cf * sigma)


class ConditionalVaR:
    """Conditional Value at Risk (CVaR) / Expected Shortfall."""

    @staticmethod
    def historical(
        returns: Tensor,
        confidence_level: float = 0.95,
    ) -> float:
        """Historical CVaR."""
        var = ValueAtRisk.historical(returns, confidence_level)
        return float(torch.mean(returns[returns <= var]))

    @staticmethod
    def parametric(
        returns: Tensor,
        confidence_level: float = 0.95,
    ) -> float:
        """Parametric CVaR (assuming normal distribution)."""
        alpha = 1 - confidence_level
        z_alpha = torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha))
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        cvar = mu - sigma * (
            torch.exp(-(z_alpha**2) / 2) / (np.sqrt(2 * np.pi) * alpha)
        )
        return float(cvar)

    @staticmethod
    def monte_carlo(
        returns: Tensor,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
    ) -> float:
        """Monte Carlo CVaR."""
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        simulated = torch.normal(mu, sigma, (n_simulations,))
        var = ValueAtRisk.monte_carlo(returns, confidence_level, n_simulations)
        return float(torch.mean(simulated[simulated <= var]))


class MaximumDrawdown:
    """Maximum Drawdown calculations."""

    @staticmethod
    def compute(returns: Tensor) -> Tuple[float, int, int]:
        """Compute maximum drawdown and its timing."""
        wealth_index = torch.cumprod(1 + returns, dim=0)
        previous_peaks = torch.cummax(wealth_index, dim=0)[0]
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_dd = torch.min(drawdowns)

        max_idx = torch.argmin(drawdowns).item()
        peak_idx = torch.argmax(wealth_index[: max_idx + 1]).item()

        return float(max_dd), peak_idx, max_idx

    @staticmethod
    def underwater(returns: Tensor) -> Tensor:
        """Compute underwater plot (drawdown series)."""
        wealth_index = torch.cumprod(1 + returns, dim=0)
        previous_peaks = torch.cummax(wealth_index, dim=0)[0]
        return (wealth_index - previous_peaks) / previous_peaks


class PerformanceMetrics:
    """Risk-adjusted performance metrics."""

    @staticmethod
    def sharpe_ratio(
        returns: Tensor,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Sharpe Ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        return float(
            torch.mean(excess_returns)
            / torch.std(excess_returns)
            * np.sqrt(periods_per_year)
        )

    @staticmethod
    def sortino_ratio(
        returns: Tensor,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Sortino Ratio (uses downside deviation)."""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]
        downside_dev = (
            torch.std(downside_returns)
            if len(downside_returns) > 0
            else torch.tensor(1e-8)
        )
        return float(
            torch.mean(excess_returns) / downside_dev * np.sqrt(periods_per_year)
        )

    @staticmethod
    def calmar_ratio(
        returns: Tensor,
        periods_per_year: int = 252,
    ) -> float:
        """Calmar Ratio (return / max drawdown)."""
        annual_return = torch.mean(returns) * periods_per_year
        max_dd, _, _ = MaximumDrawdown.compute(returns)
        return float(annual_return / abs(max_dd))

    @staticmethod
    def information_ratio(
        returns: Tensor,
        benchmark_returns: Tensor,
        periods_per_year: int = 252,
    ) -> float:
        """Information Ratio."""
        active_returns = returns - benchmark_returns
        tracking_error = torch.std(active_returns)
        return float(
            torch.mean(active_returns) / tracking_error * np.sqrt(periods_per_year)
        )

    @staticmethod
    def treynor_ratio(
        returns: Tensor,
        benchmark_returns: Tensor,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """Treynor Ratio."""
        beta = torch.cov(returns, benchmark_returns)[0, 1] / torch.var(
            benchmark_returns
        )
        excess_returns = torch.mean(returns) - risk_free_rate / periods_per_year
        return float(excess_returns / beta * periods_per_year)

    @staticmethod
    def omega_ratio(
        returns: Tensor,
        threshold: float = 0.0,
    ) -> float:
        """Omega Ratio."""
        gains = torch.sum(torch.clamp(returns - threshold, min=0))
        losses = torch.sum(torch.clamp(threshold - returns, min=0))
        return float(gains / (losses + 1e-8))


class RiskMetrics:
    """Comprehensive risk metrics."""

    @staticmethod
    def beta(
        returns: Tensor,
        benchmark_returns: Tensor,
    ) -> float:
        """Market Beta."""
        cov = torch.cov(returns, benchmark_returns)[0, 1]
        benchmark_var = torch.var(benchmark_returns)
        return float(cov / benchmark_var)

    @staticmethod
    def tracking_error(
        returns: Tensor,
        benchmark_returns: Tensor,
        periods_per_year: int = 252,
    ) -> float:
        """Tracking Error."""
        active_returns = returns - benchmark_returns
        return float(torch.std(active_returns) * np.sqrt(periods_per_year))

    @staticmethod
    def tail_ratio(returns: Tensor) -> float:
        """Tail Ratio (95th percentile / 5th percentile)."""
        return float(torch.quantile(returns, 0.95) / abs(torch.quantile(returns, 0.05)))

    @staticmethod
    def skewness(returns: Tensor) -> float:
        """Skewness of returns."""
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        return float(torch.mean(((returns - mu) / sigma) ** 3))

    @staticmethod
    def kurtosis(returns: Tensor) -> float:
        """Excess kurtosis of returns."""
        mu = torch.mean(returns)
        sigma = torch.std(returns)
        return float(torch.mean(((returns - mu) / sigma) ** 4) - 3)


class FactorExposure:
    """Factor exposure and risk decomposition."""

    def __init__(self, factor_returns: Tensor):
        self.factor_returns = factor_returns

    def regression_betas(
        self,
        portfolio_returns: Tensor,
    ) -> Tensor:
        """Compute factor betas via linear regression."""
        X = torch.cat(
            [torch.ones((len(self.factor_returns), 1)), self.factor_returns], dim=1
        )
        y = portfolio_returns

        XtX = X.T @ X
        XtX_inv = torch.inverse(XtX + 1e-6 * torch.eye(XtX.size(0)))
        betas = XtX_inv @ (X.T @ y)
        return betas[1:]

    def r_squared(
        self,
        portfolio_returns: Tensor,
        betas: Tensor,
    ) -> float:
        """R-squared of factor model."""
        X = torch.cat(
            [torch.ones((len(self.factor_returns), 1)), self.factor_returns], dim=1
        )
        X_beta = X[:, 1:] @ betas
        ss_res = torch.sum((portfolio_returns - X_beta) ** 2)
        ss_tot = torch.sum((portfolio_returns - torch.mean(portfolio_returns)) ** 2)
        return float(1 - ss_res / ss_tot)


class StressTest:
    """Stress testing and scenario analysis."""

    @staticmethod
    def historical_scenarios(
        returns: Tensor,
        percentiles: list[float] = [1, 5, 10, 25],
    ) -> dict:
        """Historical stress scenarios."""
        return {
            f"p{int(p)}": float(torch.quantile(returns, p / 100)) for p in percentiles
        }

    @staticmethod
    def factor_shocks(
        factor_betas: Tensor,
        factor_shocks: Tensor,
    ) -> Tensor:
        """Calculate portfolio impact from factor shocks."""
        return factor_betas * factor_shocks

    @staticmethod
    def worst_case(
        returns: Tensor,
        n_days: int = 5,
    ) -> float:
        """Worst case over n days."""
        rolling = torch.nn.functional.avg_pool1d(
            returns.unsqueeze(0).unsqueeze(0),
            kernel_size=n_days,
            stride=1,
        ).squeeze()
        return float(torch.min(rolling))


class RiskReport:
    """Generate comprehensive risk reports."""

    def __init__(self, returns: Tensor, benchmark_returns: Optional[Tensor] = None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns

    def generate(self) -> dict:
        """Generate full risk report."""
        report = {
            "var_95": ValueAtRisk.historical(self.returns, 0.95),
            "var_99": ValueAtRisk.historical(self.returns, 0.99),
            "cvar_95": ConditionalVaR.historical(self.returns, 0.95),
            "cvar_99": ConditionalVaR.historical(self.returns, 0.99),
            "max_drawdown": MaximumDrawdown.compute(self.returns)[0],
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(self.returns),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(self.returns),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(self.returns),
            "skewness": RiskMetrics.skewness(self.returns),
            "kurtosis": RiskMetrics.kurtosis(self.returns),
            "tail_ratio": RiskMetrics.tail_ratio(self.returns),
        }

        if self.benchmark_returns is not None:
            report["beta"] = RiskMetrics.beta(self.returns, self.benchmark_returns)
            report["tracking_error"] = RiskMetrics.tracking_error(
                self.returns, self.benchmark_returns
            )
            report["information_ratio"] = PerformanceMetrics.information_ratio(
                self.returns, self.benchmark_returns
            )

        return report
