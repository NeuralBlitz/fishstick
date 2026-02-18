"""
Volatility Modeling Module

Advanced volatility estimation and forecasting for financial time series.
"""

from typing import Optional, Tuple
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import Module


class GARCH(Module):
    """GARCH(1,1) volatility model."""

    def __init__(self, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.85):
        super().__init__()
        self.omega = torch.tensor(omega, requires_grad=True)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.beta = torch.tensor(beta, requires_grad=True)

    def forward(self, returns: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute conditional volatility."""
        n = len(returns)
        h = torch.ones(n)
        h[0] = torch.var(returns)

        for t in range(1, n):
            h[t] = self.omega + self.alpha * returns[t - 1] ** 2 + self.beta * h[t - 1]

        return torch.sqrt(h), h

    def log_likelihood(self, returns: Tensor) -> Tensor:
        """Compute negative log-likelihood."""
        sigma2, h = self.forward(returns)
        ll = -0.5 * (torch.log(2 * torch.pi) + torch.log(h) + returns**2 / h)
        return -torch.mean(ll)


class EGARCH(Module):
    """Exponential GARCH(1,1) model."""

    def __init__(
        self,
        omega: float = -0.1,
        alpha: float = 0.2,
        beta: float = 0.95,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.omega = torch.tensor(omega, requires_grad=True)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.beta = torch.tensor(beta, requires_grad=True)
        self.gamma = torch.tensor(gamma, requires_grad=True)

    def forward(self, returns: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute conditional volatility."""
        n = len(returns)
        z = returns / (torch.std(returns) + 1e-8)
        log_h = torch.ones(n)
        log_h[0] = torch.log(torch.var(returns))

        for t in range(1, n):
            log_h[t] = (
                self.omega
                + self.alpha * (torch.abs(z[t - 1]) - np.sqrt(2 / np.pi))
                + self.gamma * z[t - 1]
                + self.beta * log_h[t - 1]
            )

        return torch.exp(log_h / 2), torch.exp(log_h)


class GJR_GARCH(Module):
    """GJR-GARCH(1,1) with asymmetric term."""

    def __init__(
        self,
        omega: float = 0.01,
        alpha: float = 0.1,
        beta: float = 0.85,
        gamma: float = 0.1,
    ):
        super().__init__()
        self.omega = torch.tensor(omega, requires_grad=True)
        self.alpha = torch.tensor(alpha, requires_grad=True)
        self.beta = torch.tensor(beta, requires_grad=True)
        self.gamma = torch.tensor(gamma, requires_grad=True)

    def forward(self, returns: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute conditional volatility with asymmetry."""
        n = len(returns)
        h = torch.ones(n)
        h[0] = torch.var(returns)

        for t in range(1, n):
            indicator = 1.0 if returns[t - 1] < 0 else 0.0
            h[t] = (
                self.omega
                + self.alpha * returns[t - 1] ** 2
                + self.gamma * indicator * returns[t - 1] ** 2
                + self.beta * h[t - 1]
            )

        return torch.sqrt(h), h


class EWMAVolatility:
    """Exponentially Weighted Moving Average volatility."""

    def __init__(self, lambda_: float = 0.94):
        self.lambda_ = lambda_

    def compute(self, returns: Tensor, half_life: bool = False) -> Tensor:
        """Compute EWMA volatility."""
        if half_life:
            self.lambda_ = np.exp(-np.log(2) / 14)

        squared_returns = returns**2
        n = len(squared_returns)
        variance = torch.zeros(n)

        variance[0] = squared_returns[0]
        for t in range(1, n):
            variance[t] = (
                self.lambda_ * variance[t - 1] + (1 - self.lambda_) * squared_returns[t]
            )

        return torch.sqrt(variance)


class RealizedVolatility:
    """Realized volatility estimation from high-frequency data."""

    @staticmethod
    def simple(realized_returns: Tensor, window: int = 1) -> Tensor:
        """Simple realized volatility."""
        if window == 1:
            return torch.sqrt(torch.cumsum(realized_returns**2, dim=0))
        else:
            realized = torch.cumsum(realized_returns**2, dim=0)
            return torch.sqrt(realized[window:] - realized[:-window])

    @staticmethod
    def yz_zhang(
        returns: Tensor,
        subsample: int = 5,
    ) -> Tensor:
        """Yao-Zhang realized volatility estimator."""
        n = len(returns)
        rs = returns.view(subsample, n // subsample)
        rv = torch.sum(rs**2, dim=0)
        return torch.sqrt(rv)

    @staticmethod
    def parkinson(high: Tensor, low: Tensor, window: int = 1) -> Tensor:
        """Parkinson volatility estimator using high-low range."""
        log_hl = torch.log(high / low)
        hl_var = (log_hl**2) / (4 * np.log(2))
        return torch.sqrt(torch.cumsum(hl_var, dim=0) / window)

    @staticmethod
    def garman_klass(
        high: Tensor,
        low: Tensor,
        open_: Tensor,
        close: Tensor,
    ) -> Tensor:
        """Garman-Klass volatility estimator."""
        log_hl = torch.log(high / low)
        log_co = torch.log(close / open_)
        hl_term = (0.5 * log_hl**2) - (2 * np.log(2) - 1) * log_co**2
        return torch.sqrt(torch.cumsum(hl_term, dim=0))


class ImpliedVolatility:
    """Implied volatility calculations using Black-Scholes."""

    @staticmethod
    def black_scholes_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        """Black-Scholes call option price."""
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def newton_raphson(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """Calculate implied volatility using Newton-Raphson."""
        sigma = 0.5

        for _ in range(max_iter):
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            from scipy.stats import norm

            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            vega = S * norm.pdf(d1) * np.sqrt(T)

            diff = market_price - price
            if abs(diff) < tol:
                return sigma
            sigma = sigma + diff / vega

        return sigma


class VolatilityForecast:
    """Volatility forecasting combining multiple models."""

    def __init__(self, methods: list[str] = ["garch", "ewma", "historical"]):
        self.methods = methods
        self.garch_model: Optional[GARCH] = None
        self.ewma_model: Optional[EWMAVolatility] = None

    def fit(self, returns: Tensor) -> "VolatilityForecast":
        """Fit volatility models."""
        if "garch" in self.methods:
            self.garch_model = GARCH()
            optimizer = torch.optim.Adam(self.garch_model.parameters(), lr=0.01)
            for _ in range(100):
                loss = self.garch_model.log_likelihood(returns)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if "ewma" in self.methods:
            self.ewma_model = EWMAVolatility()

        return self

    def predict(self, returns: Tensor, horizon: int = 1) -> Tensor:
        """Predict future volatility."""
        predictions = []

        if self.garch_model is not None:
            _, h = self.garch_model(returns)
            garch_pred = h[-1].item() ** 0.5
            predictions.append(torch.tensor([garch_pred] * horizon))

        if self.ewma_model is not None:
            ewma_vol = self.ewma_model.compute(returns)
            predictions.append(ewma_vol[-1:].repeat(horizon))

        if "historical" in self.methods:
            hist_vol = torch.std(returns[-60:])
            predictions.append(hist_vol.repeat(horizon))

        return torch.mean(torch.stack(predictions), dim=0)


class StochasticVolatility(Module):
    """Stochastic volatility model (simplified Heston)."""

    def __init__(
        self, mu: float = 0.0, theta: float = 0.04, kappa: float = 1.0, xi: float = 0.1
    ):
        super().__init__()
        self.mu = torch.tensor(mu)
        self.theta = torch.tensor(theta)
        self.kappa = torch.tensor(kappa)
        self.xi = torch.tensor(xi)

    def simulate(
        self,
        n_steps: int,
        dt: float = 1 / 252,
        S0: float = 100.0,
        v0: float = 0.04,
    ) -> Tuple[Tensor, Tensor]:
        """Simulate paths under stochastic volatility."""
        S = torch.zeros(n_steps)
        v = torch.zeros(n_steps)
        S[0] = S0
        v[0] = v0

        for t in range(1, n_steps):
            z1 = torch.randn(1)
            z2 = torch.randn(1)
            v[t] = torch.clamp(
                v[t - 1]
                + self.kappa * (self.theta - v[t - 1]) * dt
                + self.xi * torch.sqrt(v[t - 1] * dt) * z1,
                min=0.001,
            )
            S[t] = S[t - 1] * torch.exp(
                (self.mu - 0.5 * v[t - 1]) * dt + torch.sqrt(v[t - 1] * dt) * z2
            )

        return S, v


class VolatilitySurface:
    """Volatility surface construction."""

    def __init__(self):
        self.strikes: Optional[Tensor] = None
        self.maturities: Optional[Tensor] = None
        self.volatilities: Optional[Tensor] = None

    def fit(
        self, strikes: ndarray, maturities: ndarray, ivs: ndarray
    ) -> "VolatilitySurface":
        """Fit volatility surface from market data."""
        self.strikes = torch.tensor(strikes)
        self.maturities = torch.tensor(maturities)
        self.volatilities = torch.tensor(ivs)
        return self

    def interpolate(self, strike: float, maturity: float) -> float:
        """Interpolate volatility at given strike and maturity."""
        if self.volatilities is None:
            raise ValueError("Volatility surface not fitted")

        idx_s = torch.argmin(torch.abs(self.strikes - strike))
        idx_t = torch.argmin(torch.abs(self.maturities - maturity))
        return self.volatilities[idx_s, idx_t].item()
