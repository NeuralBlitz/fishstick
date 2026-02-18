import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm


class BlackScholes:
    @staticmethod
    def call_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    @staticmethod
    def put_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> float:
        if T <= 0:
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    @staticmethod
    def greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
    ) -> dict:
        if T <= 0:
            return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * norm.cdf(d2)
        theta = theta / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }


class DeepHedger(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_assets: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_assets = n_assets

        self.hedging_network = nn.Sequential(
            nn.Linear(n_assets + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
            nn.Tanh(),
        )

    def forward(
        self,
        spot_prices: torch.Tensor,
        time_step: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([spot_prices, time_step.unsqueeze(-1)], dim=-1)
        hedges = self.hedging_network(inputs)
        return hedges

    def compute_pnl(
        self,
        initial_prices: torch.Tensor,
        final_prices: torch.Tensor,
        strikes: torch.Tensor,
        option_type: str = "call",
    ) -> torch.Tensor:
        if option_type == "call":
            payoff = torch.clamp(final_prices - strikes, min=0)
        else:
            payoff = torch.clamp(strikes - final_prices, min=0)

        pnl = payoff.sum(dim=-1)
        return pnl

    def hedge(
        self,
        spot_prices: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return (spot_prices * positions).sum(dim=-1, keepdim=True)


class QLBS:
    def __init__(
        self,
        n_steps: int,
        n_assets: int,
        hidden_dim: int = 32,
        risk_aversion: float = 1.0,
    ):
        self.n_steps = n_steps
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

        self.q_network = nn.Sequential(
            nn.Linear(n_assets + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    def expected_utility(
        self,
        wealth: torch.Tensor,
        wealth_next: torch.Tensor,
    ) -> torch.Tensor:
        return torch.mean(torch.log(wealth_next))

    def q_function(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat([state, action], dim=-1)
        return self.q_network(inputs)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.95,
    ) -> dict:
        current_q = self.q_function(states, actions)

        with torch.no_grad():
            next_actions = torch.softmax(self.q_network(next_states), dim=-1)
            next_q = self.q_function(next_states, next_actions)
            target_q = rewards + gamma * (1 - dones) * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"q_loss": loss.item()}

    def optimal_action(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.q_network(state)
            probs = torch.softmax(q_values / self.risk_aversion, dim=-1)
        return probs

    def compute_portfolio_value(
        self,
        weights: torch.Tensor,
        prices: torch.Tensor,
    ) -> torch.Tensor:
        return (weights * prices).sum(dim=-1)
