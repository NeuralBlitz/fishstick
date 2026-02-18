import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
from typing import Optional, Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: list[int] = [256, 256]
    ):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class QLearning:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        target_update_freq: int = 1000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.steps = 0

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

    def get_action(self, state: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax(dim=-1).item()

    def update(self, batch_size: int) -> dict:
        if len(self.replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=-1)[0]
            target_q = rewards + self.gamma * (1 - dones) * next_q

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"q_loss": loss.item(), "epsilon": self.epsilon}


class OfflineRL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.action_dim = action_dim
        self.gamma = gamma

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=-1)[0]
            target_q = rewards + self.gamma * (1 - dones) * next_q

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_network.load_state_dict(self.q_network.state_dict())

        return {"offline_q_loss": loss.item()}

    def get_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax(dim=-1).item()


class ActorNetwork(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: list[int] = [256, 256]
    ):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class BCQ(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        phi: float = 0.03,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.phi = phi

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_critic1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_critic2 = QNetwork(state_dim, action_dim, hidden_dims)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_size = state.shape[0] if state.dim() > 1 else 1
            if state.dim() == 1:
                state = state.unsqueeze(0)

            action = self.actor(state)
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            q = q1 + q2

            perturbation = (torch.rand_like(action) - 0.5) * self.phi
            perturbed_action = (action + perturbation).clamp(-1, 1)

            q_perturbed = self.critic1(state, perturbed_action)
            final_action = torch.where(q_perturbed > q, perturbed_action, action)

            return final_action.squeeze(0) if batch_size == 1 else final_action

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            next_actions = self.actor(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions = self.actor(states)
        q1_new = self.critic1(states, new_actions)
        actor_loss = -q1_new.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "bcq_critic_loss": critic_loss.item(),
            "bcq_actor_loss": actor_loss.item(),
        }


class CQL(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 1.0,
        temp: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.temp = temp

        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims)
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_critic1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_critic2 = QNetwork(state_dim, action_dim, hidden_dims)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate,
        )

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.actor(state).squeeze(0)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            next_actions = self.actor(next_states)
            q1_next = self.target_critic1(next_states, next_actions)
            q2_next = self.target_critic2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        sampled_actions = torch.rand(len(states), self.action_dim) * 2 - 1
        q1_sampled = self.critic1(states, sampled_actions)
        q2_sampled = self.critic2(states, sampled_actions)
        min_q_sampled = torch.min(q1_sampled, q2_sampled)

        cql_loss = self.alpha * (min_q_sampled - q1).mean()

        total_critic_loss = critic_loss + cql_loss

        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        new_actions = self.actor(states)
        q1_new = self.critic1(states, new_actions)
        actor_loss = -q1_new.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "cql_critic_loss": critic_loss.item(),
            "cql_actor_loss": actor_loss.item(),
            "cql_conservative_loss": cql_loss.item(),
        }
