"""
fishstick Finance Module

Comprehensive finance and trading tools for machine learning:
- Financial time series processing
- Volatility modeling (GARCH, EWMA, realized)
- Risk assessment (VaR, CVaR, drawdowns)
- Portfolio optimization (Markowitz, Risk Parity, Black-Litterman)
- Algorithmic trading primitives
"""

from fishstick.finance.timeseries_finance import (
    FinancialTimeSeries,
    TechnicalIndicators,
    OHLCVData,
    FinancialFeatureEngineer,
    FinancialScaler,
)

from fishstick.finance.volatility import (
    GARCH,
    EGARCH,
    GJR_GARCH,
    EWMAVolatility,
    RealizedVolatility,
    ImpliedVolatility,
    VolatilityForecast,
    StochasticVolatility,
    VolatilitySurface,
)

from fishstick.finance.risk import (
    ValueAtRisk,
    ConditionalVaR,
    MaximumDrawdown,
    PerformanceMetrics,
    RiskMetrics,
    FactorExposure,
    StressTest,
    RiskReport,
)

from fishstick.finance.portfolio import (
    MeanVarianceOptimization,
    RiskParity,
    BlackLitterman,
    MinimumVariance,
    MaximumSharpe,
    HierarchicalRiskParity,
    EqualWeightPortfolio,
    InverseVolatility,
    KellyCriterion,
    PortfolioBacktest,
)

from fishstick.finance.trading import (
    OrderType,
    OrderSide,
    OrderStatus,
    Order,
    Position,
    Portfolio,
    SignalGenerator,
    MovingAverageCrossover,
    RSIStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    Backtester,
    TradingStrategy,
    CNNLSTMTrading,
    AttentionTrading,
    ExecutionAlgorithm,
    RiskManager,
)

__all__ = [
    # Time Series
    "FinancialTimeSeries",
    "TechnicalIndicators",
    "OHLCVData",
    "FinancialFeatureEngineer",
    "FinancialScaler",
    # Volatility
    "GARCH",
    "EGARCH",
    "GJR_GARCH",
    "EWMAVolatility",
    "RealizedVolatility",
    "ImpliedVolatility",
    "VolatilityForecast",
    "StochasticVolatility",
    "VolatilitySurface",
    # Risk
    "ValueAtRisk",
    "ConditionalVaR",
    "MaximumDrawdown",
    "PerformanceMetrics",
    "RiskMetrics",
    "FactorExposure",
    "StressTest",
    "RiskReport",
    # Portfolio
    "MeanVarianceOptimization",
    "RiskParity",
    "BlackLitterman",
    "MinimumVariance",
    "MaximumSharpe",
    "HierarchicalRiskParity",
    "EqualWeightPortfolio",
    "InverseVolatility",
    "KellyCriterion",
    "PortfolioBacktest",
    # Trading
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "Order",
    "Position",
    "Portfolio",
    "SignalGenerator",
    "MovingAverageCrossover",
    "RSIStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "Backtester",
    "TradingStrategy",
    "CNNLSTMTrading",
    "AttentionTrading",
    "ExecutionAlgorithm",
    "RiskManager",
]
