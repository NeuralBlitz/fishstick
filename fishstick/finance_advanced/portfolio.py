import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional


class RiskParityOptimizer:
    def __init__(self, risk_target: Optional[float] = None):
        self.risk_target = risk_target

    def compute_weights(
        self, cov_matrix: torch.Tensor, risk_target: Optional[float] = None
    ) -> torch.Tensor:
        n_assets = cov_matrix.shape[0]
        target = risk_target or self.risk_target or 1.0 / n_assets

        weights = torch.ones(n_assets, requires_grad=True, device=cov_matrix.device)

        for _ in range(100):
            portfolio_vol = torch.sqrt(
                torch.matmul(
                    weights.unsqueeze(0), torch.matmul(cov_matrix, weights.unsqueeze(1))
                ).squeeze()
            )
            marginal_contrib = torch.matmul(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol

            target_risk = target * portfolio_vol
            diff = risk_contrib - target_risk
            loss = torch.sum(diff**2)

            loss.backward()
            with torch.no_grad():
                weights -= 0.01 * weights.grad
                weights = torch.clamp(weights, min=0.01)
                weights = weights / weights.sum()
                weights.requires_grad = True

        return weights.detach()


class RLPortfolioAgent(nn.Module):
    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        hidden_dim: int = 128,
        action_std: float = 0.1,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.action_std = action_std

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        action_mean = self.actor(features)
        value = self.critic(features)

        action_mean = torch.softmax(action_mean, dim=-1)

        return action_mean, value

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        action_mean, _ = self.forward(state)
        noise = torch.randn_like(action_mean) * self.action_std
        action = action_mean + noise
        action = torch.softmax(torch.log(action + 1e-8) + noise, dim=-1)
        return action


class PPOAgent:
    def __init__(
        self,
        n_assets: int,
        state_dim: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        hidden_dim: int = 128,
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.agent = RLPortfolioAgent(n_assets, state_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.agent.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.agent.critic.parameters(), lr=lr_critic)

    def compute_returns(
        self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            returns[t] = gae + values[t]

        advantage = returns - values
        return returns, advantage

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        action_means, values = (
            self.agent(states).chunk(2, dim=-1)
            if states.dim() > 1
            else (self.agent(states)[0], self.agent(states)[1])
        )

        _, values = [], []
        for i in range(len(states)):
            _, v = self.agent(states[i])
            values.append(v)
        values = torch.stack(values).squeeze()

        returns, advantages = self.compute_returns(rewards, dones, values.detach())

        action_dist = torch.distributions.Categorical(action_means)
        log_probs = action_dist.log_prob(actions)

        surr1 = log_probs * advantages.detach()
        surr2 = (
            torch.clamp(log_probs, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages.detach()
        )

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values, returns)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
