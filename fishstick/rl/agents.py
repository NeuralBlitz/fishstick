"""
Reinforcement Learning Agents
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Deep Q-Network agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 1e-3,
        target_update_freq: int = 100,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.step_count = 0

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

    def forward(self, x: Tensor) -> Tensor:
        return self.q_network(x)

    def get_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size: int) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


class ActorCritic(nn.Module):
    """Actor-Critic agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.gamma = gamma

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)

    def get_action(self, state: np.ndarray) -> int:
        probs = self.forward(torch.FloatTensor(state))
        return torch.multinomial(probs, 1).item()

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> float:
        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + 1e-8)

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        targets = rewards + (1 - dones) * self.gamma * next_values.detach()

        advantage = targets - values.detach()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, targets)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class PPO(nn.Module):
    """Proximal Policy Optimization agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        learning_rate: float = 3e-4,
    ):
        super().__init__()

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        probs = self.forward(torch.FloatTensor(state))
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action] + 1e-8)
        return action, log_prob

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        probs = self.actor(states)
        values = self.critic(states).squeeze()

        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        return log_probs, values, entropy


class SAC(nn.Module):
    """Soft Actor-Critic agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
    ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.actor(x)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std

    def get_action(self, state: np.ndarray) -> int:
        mean, std = self.forward(torch.FloatTensor(state))
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.argmax().item()

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> float:
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        new_actions = dist.rsample()
        log_probs = dist.log_prob(new_actions).sum(dim=-1)

        q1 = self.critic1(torch.cat([states, new_actions], dim=-1))
        q2 = self.critic2(torch.cat([states, new_actions], dim=-1))
        q = torch.min(q1, q2).squeeze()

        actor_loss = (-q - 0.01 * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        next_mean, next_std = self.forward(next_states)
        next_dist = torch.distributions.Normal(next_mean, next_std)
        next_actions = next_dist.rsample()
        next_log_probs = next_dist.log_prob(next_actions).sum(dim=-1)

        target_q1 = self.target_critic1(
            torch.cat([next_states, next_actions], dim=-1)
        ).squeeze()
        target_q2 = self.target_critic2(
            torch.cat([next_states, next_actions], dim=-1)
        ).squeeze()
        target_q = torch.min(target_q1, target_q2)
        targets = rewards + (1 - dones) * self.gamma * (
            target_q - 0.01 * next_log_probs
        )

        q1 = self.critic1(torch.cat([states, actions], dim=-1)).squeeze()
        q2 = self.critic2(torch.cat([states, actions], dim=-1)).squeeze()

        critic_loss = F.mse_loss(q1, targets.detach()) + F.mse_loss(
            q2, targets.detach()
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._soft_update()

        return critic_loss.item()

    def _soft_update(self):
        for target, source in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target.data.copy_(target.data * (1.0 - self.tau) + source.data * self.tau)

        for target, source in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            target.data.copy_(target.data * (1.0 - self.tau) + source.data * self.tau)


class PolicyGradient(nn.Module):
    """REINFORCE policy gradient agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.gamma = gamma

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.policy(x)

    def get_action(self, state: np.ndarray) -> int:
        probs = self.forward(torch.FloatTensor(state))
        return torch.multinomial(probs, 1).item()

    def update(self, states: Tensor, actions: Tensor, rewards: Tensor) -> float:
        probs = self.policy(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + 1e-8)

        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs.squeeze() * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class TD3(nn.Module):
    """Twin Delayed DDPG (TD3) agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
    ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.target_critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=1e-3,
        )

    def get_action(self, state: np.ndarray) -> int:
        return self.actor(torch.FloatTensor(state)).argmax().item()

    def update(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> float:
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.target_actor(next_states) + noise).clamp(-1, 1)

            target_q1 = self.target_critic1(
                torch.cat([next_states, next_actions], dim=-1)
            )
            target_q2 = self.target_critic2(
                torch.cat([next_states, next_actions], dim=-1)
            )
            target_q = torch.min(target_q1, target_q2).squeeze()
            target = rewards + (1 - dones) * self.gamma * target_q

        q1 = self.critic1(torch.cat([states, actions], dim=-1)).squeeze()
        q2 = self.critic2(torch.cat([states, actions], dim=-1)).squeeze()

        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(
                torch.cat([states, self.actor(states)], dim=-1)
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update()

        self.total_it += 1
        return critic_loss.item()

    def _soft_update(self):
        for target, source in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target.data.copy_(target.data * (1 - self.tau) + source.data * self.tau)

        for target, source in zip(
            self.target_critic1.parameters(), self.critic1.parameters()
        ):
            target.data.copy_(target.data * (1 - self.tau) + source.data * self.tau)
