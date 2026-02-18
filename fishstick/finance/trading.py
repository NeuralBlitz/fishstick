"""
Algorithmic Trading Module

Trading primitives, backtesting framework, and strategy implementations.
"""

from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.nn import Module


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Order:
    """Trading order representation."""

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ):
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.status = OrderStatus.PENDING


class Position:
    """Position tracking."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0.0
        self.avg_price = 0.0
        self.realized_pnl = 0.0

    def update(self, quantity: float, price: float):
        """Update position after trade."""
        if quantity * self.quantity >= 0:
            total_cost = self.quantity * self.avg_price + quantity * price
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0.0
        else:
            if abs(quantity) >= abs(self.quantity):
                self.realized_pnl += (
                    (self.quantity * price - self.quantity * self.avg_price)
                    if quantity < 0
                    else 0.0
                )
                self.quantity += quantity
                self.avg_price = price if self.quantity != 0 else 0.0
            else:
                self.realized_pnl += quantity * (self.avg_price - price)
                self.quantity += quantity

    def market_value(self, current_price: float) -> float:
        """Current market value."""
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized profit/loss."""
        return self.quantity * (current_price - self.avg_price)


class Portfolio:
    """Portfolio tracking."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_history: List[float] = [initial_capital]
        self.trades: List[Dict] = []

    def update_equity(self, prices: Dict[str, float]):
        """Update portfolio equity."""
        positions_value = sum(
            pos.market_value(prices.get(pos.symbol, 0))
            for pos in self.positions.values()
        )
        self.equity_history.append(self.cash + positions_value)

    def execute_order(self, order: Order, fill_price: float):
        """Execute order."""
        cost = order.quantity * fill_price

        if order.side == OrderSide.BUY:
            if self.cash >= cost:
                self.cash -= cost
                if order.symbol not in self.positions:
                    self.positions[order.symbol] = Position(order.symbol)
                self.positions[order.symbol].update(order.quantity, fill_price)
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.REJECTED
        else:
            if (
                order.symbol in self.positions
                and self.positions[order.symbol].quantity >= order.quantity
            ):
                self.cash += cost
                self.positions[order.symbol].update(-order.quantity, fill_price)
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.REJECTED

        self.trades.append(
            {
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": fill_price,
            }
        )

    def total_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value."""
        positions_value = sum(
            pos.market_value(prices.get(pos.symbol, 0))
            for pos in self.positions.values()
        )
        return self.cash + positions_value


class SignalGenerator:
    """Base class for signal generation."""

    def generate(self, data: Tensor) -> Tensor:
        """Generate trading signals."""
        raise NotImplementedError


class MovingAverageCrossover(SignalGenerator):
    """Moving Average Crossover strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate(self, prices: Tensor) -> Tensor:
        """Generate signals: 1=buy, -1=sell, 0=hold."""
        if len(prices) < self.long_window:
            return torch.zeros(len(prices))

        short_ma = torch.nn.functional.avg_pool1d(
            prices.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=self.short_window,
            stride=1,
        ).squeeze()

        long_ma = torch.nn.functional.avg_pool1d(
            prices.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=self.long_window,
            stride=1,
        ).squeeze()

        signals = torch.zeros(len(prices))
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        return signals


class RSIStrategy(SignalGenerator):
    """RSI-based strategy."""

    def __init__(self, window: int = 14, oversold: float = 30, overbought: float = 70):
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    def generate(self, prices: Tensor) -> Tensor:
        """Generate signals based on RSI."""
        deltas = torch.diff(prices)
        gains = torch.clamp(deltas, min=0)
        losses = torch.clamp(-deltas, min=0)

        avg_gains = torch.nn.functional.avg_pool1d(
            gains.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=self.window,
            stride=1,
        ).squeeze()
        avg_losses = torch.nn.functional.avg_pool1d(
            losses.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=self.window,
            stride=1,
        ).squeeze()

        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = torch.zeros(len(prices))
        signals[rsi < self.oversold] = 1
        signals[rsi > self.overbought] = -1
        return signals


class MomentumStrategy(SignalGenerator):
    """Momentum strategy."""

    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold

    def generate(self, prices: Tensor) -> Tensor:
        """Generate momentum signals."""
        momentum = (prices[self.lookback :] - prices[: -self.lookback]) / (
            prices[: -self.lookback] + 1e-8
        )

        signals = torch.zeros(len(prices))
        signals[self.lookback :][momentum > self.threshold] = 1
        signals[self.lookback :][momentum < -self.threshold] = -1
        return signals


class MeanReversionStrategy(SignalGenerator):
    """Mean reversion strategy."""

    def __init__(self, window: int = 20, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold

    def generate(self, prices: Tensor) -> Tensor:
        """Generate mean reversion signals."""
        if len(prices) < self.window:
            return torch.zeros(len(prices))

        rolling_mean = torch.nn.functional.avg_pool1d(
            prices.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=self.window,
            stride=1,
        ).squeeze()

        rolling_std = torch.std(prices[: self.window])
        z_score = (prices[self.window - 1 :] - rolling_mean) / (rolling_std + 1e-8)

        signals = torch.zeros(len(prices))
        signals[self.window - 1 :][z_score > self.threshold] = -1
        signals[self.window - 1 :][z_score < -self.threshold] = 1
        return signals


class Backtester:
    """Backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(
        self,
        prices: Tensor,
        signals: Tensor,
        position_size: float = 1.0,
    ) -> Dict[str, Any]:
        """Run backtest."""
        capital = self.initial_capital
        position = 0.0
        trades = []
        equity = [capital]

        for i in range(len(prices)):
            current_price = prices[i].item()
            signal = signals[i].item()

            if signal > 0 and position == 0:
                shares = int((capital * position_size) / current_price)
                cost = shares * current_price * (1 + self.commission + self.slippage)
                if cost <= capital:
                    position = shares
                    capital -= cost
                    trades.append(
                        {
                            "type": "buy",
                            "price": current_price,
                            "shares": shares,
                            "idx": i,
                        }
                    )

            elif signal < 0 and position > 0:
                proceeds = (
                    position * current_price * (1 - self.commission - self.slippage)
                )
                capital += proceeds
                trades.append(
                    {
                        "type": "sell",
                        "price": current_price,
                        "shares": position,
                        "idx": i,
                    }
                )
                position = 0

            portfolio_value = capital + position * current_price
            equity.append(portfolio_value)

        returns = torch.diff(torch.tensor(equity)) / torch.tensor(equity[:-1])

        return {
            "equity": equity,
            "returns": returns.numpy(),
            "total_return": (equity[-1] - self.initial_capital) / self.initial_capital,
            "trades": trades,
            "n_trades": len(trades),
        }


class TradingStrategy(Module):
    """Base class for neural trading strategies."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class CNNLSTMTrading(TradingStrategy):
    """CNN-LSTM hybrid trading model."""

    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        num_layers: int = 2,
    ):
        super().__init__(input_dim)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

        self.lstm = torch.nn.LSTM(
            cnn_channels * 2,
            lstm_hidden,
            num_layers,
            batch_first=True,
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out


class AttentionTrading(TradingStrategy):
    """Transformer-based trading strategy."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__(input_dim, d_model)

        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class PositionalEncoding(Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class ExecutionAlgorithm:
    """Order execution algorithms."""

    @staticmethod
    def twap(
        quantity: float,
        n_slices: int,
        start_price: float,
    ) -> List[float]:
        """Time-Weighted Average Price execution."""
        return [start_price] * n_slices

    @staticmethod
    def vwap(
        volumes: Tensor,
        prices: Tensor,
        target_quantity: float,
    ) -> List[float]:
        """Volume-Weighted Average Price execution."""
        total_volume = torch.sum(volumes)
        vwap_prices = torch.cumsum(prices * volumes, dim=0) / torch.cumsum(
            volumes, dim=0
        )
        slice_size = target_quantity / len(prices)
        execution_prices = []

        remaining = target_quantity
        for i in range(len(vwap_prices)):
            execute = min(slice_size, remaining)
            execution_prices.append(vwap_prices[i].item())
            remaining -= execute

        return execution_prices

    @staticmethod
    def implementation_shortfall(
        quantity: float,
        urgency: float,
        start_price: float,
    ) -> Dict[str, float]:
        """Implementation Shortfall (IS) algorithm."""
        return {
            "arrival_price": start_price,
            "estimated_cost": urgency * quantity * start_price * 0.001,
        }


class RiskManager:
    """Risk management for trading."""

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_loss_per_trade: float = 0.02,
        max_daily_loss: float = 0.05,
    ):
        self.max_position_size = max_position_size
        self.max_loss_per_trade = max_loss_per_trade
        self.max_daily_loss = max_daily_loss

    def check_order(
        self,
        order: Order,
        portfolio: Portfolio,
        current_prices: Dict[str, float],
    ) -> bool:
        """Check if order passes risk limits."""
        position_value = order.quantity * current_prices.get(order.symbol, 0)
        portfolio_value = portfolio.total_value(current_prices)

        if position_value / portfolio_value > self.max_position_size:
            return False

        return True

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        risk_per_trade: float,
        portfolio_value: float,
    ) -> int:
        """Calculate position size based on risk."""
        risk_amount = portfolio_value * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        shares = int(risk_amount / risk_per_share)
        return shares
