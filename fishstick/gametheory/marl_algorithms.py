"""
Multi-agent RL algorithms.

Implements Q-learning, policy gradient, and other MARL algorithms
for multi-agent environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from fishstick.gametheory.multi_agent_env import (
    MultiAgentEnvironment,
    MultiAgentState,
    MultiAgentStep,
)


@dataclass
class MultiAgentRLAlgorithm(ABC):
    """Abstract base class for MARL algorithms."""
    
    n_agents: int
    obs_dim: int
    act_dim: int
    learning_rate: float = 0.01
    gamma: float = 0.99
    epsilon: float = 0.1
    
    @abstractmethod
    def select_action(
        self, 
        observations: Dict[int, NDArray[np.float64]],
        agent_id: int,
        explore: bool = True
    ) -> int:
        """Select action for an agent."""
        pass
    
    @abstractmethod
    def update(
        self,
        observations: Dict[int, NDArray[np.float64]],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_obs: Dict[int, NDArray[np.float64]],
        done: bool,
    ) -> Dict[str, float]:
        """Update algorithm parameters."""
        pass
    
    def reset(self) -> None:
        """Reset algorithm state."""
        pass


@dataclass
class QLearning(MultiAgentRLAlgorithm):
    """Q-learning for multi-agent settings.
    
    Independent Q-learning where each agent learns its own Q-function.
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        super().__init__(n_agents, obs_dim, act_dim, learning_rate, gamma, epsilon)
        
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_tables: Dict[int, NDArray[np.float64]] = {}
        
        for agent_id in range(n_agents):
            self.q_tables[agent_id] = np.zeros((1000, act_dim))
        
        self._discretize_obs = True
    
    def select_action(
        self,
        observations: Dict[int, NDArray[np.float64]],
        agent_id: int,
        explore: bool = True
    ) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.act_dim)
        
        obs = observations.get(agent_id, np.zeros(self.obs_dim))
        obs_idx = self._obs_to_idx(obs)
        
        q_values = self.q_tables[agent_id][obs_idx]
        return int(np.argmax(q_values))
    
    def _obs_to_idx(self, obs: NDArray[np.float64]) -> int:
        """Convert continuous observation to discrete index."""
        if self._discretize_obs and len(obs) > 0:
            obs_sum = int(np.sum(obs) * 10) % 1000
            return obs_sum
        return 0
    
    def update(
        self,
        observations: Dict[int, NDArray[np.float64]],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_obs: Dict[int, NDArray[np.float64]],
        done: bool,
    ) -> Dict[str, float]:
        """Update Q-tables for all agents."""
        losses = {}
        
        for agent_id in range(self.n_agents):
            obs = observations.get(agent_id, np.zeros(self.obs_dim))
            next_ob = next_obs.get(agent_id, np.zeros(self.obs_dim))
            
            obs_idx = self._obs_to_idx(obs)
            next_idx = self._obs_to_idx(next_ob)
            
            current_q = self.q_tables[agent_id][obs_idx, actions.get(agent_id, 0)]
            
            max_next_q = np.max(self.q_tables[agent_id][next_idx])
            
            target_q = rewards.get(agent_id, 0.0) + \
                      self.gamma * max_next_q * (1 - done)
            
            td_error = target_q - current_q
            self.q_tables[agent_id][obs_idx, actions.get(agent_id, 0)] += \
                self.learning_rate * td_error
            
            losses[f"agent_{agent_id}_loss"] = abs(td_error)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return losses
    
    def reset(self) -> None:
        """Reset epsilon."""
        pass


@dataclass
class DeepQNetwork(nn.Module):
    """Deep Q-Network for MARL."""
    
    obs_dim: int
    act_dim: int
    hidden_dim: int = 64
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.act_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods."""
    
    obs_dim: int
    act_dim: int
    hidden_dim: int = 64
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.act_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


@dataclass
class ValueNetwork(nn.Module):
    """Value network for actor-critic methods."""
    
    obs_dim: int
    hidden_dim: int = 64
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class PolicyGradientMARL(MultiAgentRLAlgorithm):
    """Policy gradient for multi-agent settings.
    
    Uses REINFORCE with baseline for variance reduction.
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        super().__init__(n_agents, obs_dim, act_dim, learning_rate, gamma)
        
        self.entropy_coef = entropy_coef
        
        self.policy_networks: Dict[int, PolicyNetwork] = {}
        self.value_networks: Dict[int, ValueNetwork] = {}
        self.policy_optimizers: Dict[int, torch.optim.Optimizer] = {}
        self.value_optimizers: Dict[int, torch.optim.Optimizer] = {}
        
        for agent_id in range(n_agents):
            self.policy_networks[agent_id] = PolicyNetwork(obs_dim, act_dim)
            self.value_networks[agent_id] = ValueNetwork(obs_dim)
            
            self.policy_optimizers[agent_id] = torch.optim.Adam(
                self.policy_networks[agent_id].parameters(),
                lr=learning_rate
            )
            self.value_optimizers[agent_id] = torch.optim.Adam(
                self.value_networks[agent_id].parameters(),
                lr=learning_rate
            )
        
        self.trajectory_buffer: List[Dict[str, Any]] = []
    
    def select_action(
        self,
        observations: Dict[int, NDArray[np.float64]],
        agent_id: int,
        explore: bool = True
    ) -> int:
        """Select action using current policy."""
        obs = observations.get(agent_id, np.zeros(self.obs_dim))
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy_networks[agent_id](obs_tensor)
        
        if explore:
            action = Categorical(probs).sample().item()
        else:
            action = probs.argmax().item()
        
        return action
    
    def update(
        self,
        observations: Dict[int, NDArray[np.float64]],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_obs: Dict[int, NDArray[np.float64]],
        done: bool,
    ) -> Dict[str, float]:
        """Update policy and value networks."""
        losses = {}
        
        for agent_id in range(self.n_agents):
            obs = observations.get(agent_id, np.zeros(self.obs_dim))
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            probs = self.policy_networks[agent_id](obs_tensor)
            value = self.value_networks[agent_id](obs_tensor)
            
            log_prob = torch.log(probs[0, actions.get(agent_id, 0)] + 1e-8)
            
            advantage = rewards.get(agent_id, 0.0) - value.item()
            
            policy_loss = -log_prob * advantage
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            
            total_policy_loss = policy_loss - self.entropy_coef * entropy
            
            self.policy_optimizers[agent_id].zero_grad()
            total_policy_loss.backward()
            self.policy_optimizers[agent_id].step()
            
            value_loss = F.mse_loss(
                value, 
                torch.tensor([rewards.get(agent_id, 0.0)])
            )
            
            self.value_optimizers[agent_id].zero_grad()
            value_loss.backward()
            self.value_optimizers[agent_id].step()
            
            losses[f"agent_{agent_id}_policy_loss"] = policy_loss.item()
            losses[f"agent_{agent_id}_value_loss"] = value_loss.item()
        
        return losses
    
    def reset(self) -> None:
        """Clear trajectory buffer."""
        self.trajectory_buffer = []


@dataclass
class NashQ-Learning(MultiAgentRLAlgorithm):
    """Nash Q-learning for general-sum games.
    
    Learns a Nash equilibrium strategy in multi-agent settings.
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__(n_agents, obs_dim, act_dim, learning_rate, gamma)
        
        self.q_tables = {
            agent_id: np.zeros((100, act_dim))
            for agent_id in range(n_agents)
        }
    
    def select_action(
        self,
        observations: Dict[int, NDArray[np.float64]],
        agent_id: int,
        explore: bool = True
    ) -> int:
        """Select action using Nash equilibrium strategy."""
        obs = observations.get(agent_id, np.zeros(self.obs_dim))
        obs_idx = int(np.sum(obs) * 10) % 100
        
        q_values = self.q_tables[agent_id][obs_idx]
        
        exp_q = np.exp(q_values - q_values.max())
        probs = exp_q / exp_q.sum()
        
        action = np.random.choice(self.act_dim, p=probs)
        
        return action
    
    def update(
        self,
        observations: Dict[int, NDArray[np.float64]],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_obs: Dict[int, NDArray[np.float64]],
        done: bool,
    ) -> Dict[str, float]:
        """Update using Nash equilibrium as target."""
        losses = {}
        
        for agent_id in range(self.n_agents):
            obs = observations.get(agent_id, np.zeros(self.obs_dim))
            obs_idx = int(np.sum(obs) * 10) % 100
            
            current_q = self.q_tables[agent_id][obs_idx, actions.get(agent_id, 0)]
            
            other_q = np.zeros((self.n_agents, self.act_dim))
            for other_id in range(self.n_agents):
                if other_id != agent_id:
                    other_obs = next_obs.get(other_id, np.zeros(self.obs_dim))
                    other_idx = int(np.sum(other_obs) * 10) % 100
                    other_q[other_id] = self.q_tables[other_id][other_idx]
            
            nash_value = self._compute_nash_value(other_q)
            
            target_q = rewards.get(agent_id, 0.0) + self.gamma * nash_value
            
            td_error = target_q - current_q
            self.q_tables[agent_id][obs_idx, actions.get(agent_id, 0)] += \
                self.learning_rate * td_error
            
            losses[f"agent_{agent_id}_loss"] = abs(td_error)
        
        return losses
    
    def _compute_nash_value(self, q_tables: NDArray[np.float64]) -> float:
        """Compute approximate Nash equilibrium value."""
        return np.mean([np.max(q) for q in q_tables])


@dataclass
class MeanFieldQ(MultiAgentRLAlgorithm):
    """Mean Field Q-Learning.
    
    Approximates interactions between agents using mean field.
    """
    
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__(n_agents, obs_dim, act_dim, learning_rate, gamma)
        
        self.q_tables = {
            agent_id: np.zeros(act_dim)
            for agent_id in range(n_agents)
        }
    
    def select_action(
        self,
        observations: Dict[int, NDArray[np.float64]],
        agent_id: int,
        explore: bool = True
    ) -> int:
        """Select action using greedy policy."""
        q_values = self.q_tables[agent_id]
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.act_dim)
        
        return int(np.argmax(q_values))
    
    def update(
        self,
        observations: Dict[int, NDArray[np.float64]],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_obs: Dict[int, NDArray[np.float64]],
        done: bool,
    ) -> Dict[str, float]:
        """Update using mean field approximation."""
        losses = {}
        
        action_counts = np.zeros(self.act_dim)
        for a in actions.values():
            action_counts[a] += 1
        mean_action_probs = action_counts / self.n_agents
        
        for agent_id in range(self.n_agents):
            current_q = self.q_tables[agent_id][actions.get(agent_id, 0)]
            
            mean_field_q = np.sum(
                mean_action_probs * self.q_tables[agent_id]
            )
            
            target_q = rewards.get(agent_id, 0.0) + \
                      self.gamma * mean_field_q * (1 - done)
            
            td_error = target_q - current_q
            self.q_tables[agent_id][actions.get(agent_id, 0)] += \
                self.learning_rate * td_error
            
            losses[f"agent_{agent_id}_loss"] = abs(td_error)
        
        self.epsilon = max(0.01, self.epsilon * 0.99)
        
        return losses


def train_marl_agent(
    algorithm: MultiAgentRLAlgorithm,
    env: MultiAgentEnvironment,
    num_episodes: int = 1000,
    max_steps: int = 100,
) -> Dict[str, List[float]]:
    """Train a MARL agent in a multi-agent environment.
    
    Args:
        algorithm: The MARL algorithm
        env: The multi-agent environment
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary of training statistics
    """
    episode_rewards: Dict[int, List[float]] = {
        i: [] for i in range(algorithm.n_agents)
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        
        episode_reward = {i: 0.0 for i in range(algorithm.n_agents)}
        
        for step in range(max_steps):
            actions = {}
            
            for agent_id in range(algorithm.n_agents):
                action = algorithm.select_action(
                    state.observation,
                    agent_id,
                    explore=(episode < num_episodes * 0.8)
                )
                actions[agent_id] = action
            
            step_result = env.step(actions)
            
            algorithm.update(
                state.observation,
                actions,
                step_result.rewards,
                step_result.observations,
                step_result.done,
            )
            
            for agent_id in range(algorithm.n_agents):
                episode_reward[agent_id] += step_result.rewards.get(agent_id, 0.0)
            
            if step_result.done:
                break
            
            state.observation = step_result.observations
        
        for agent_id in range(algorithm.n_agents):
            episode_rewards[agent_id].append(episode_reward[agent_id])
    
    return episode_rewards
