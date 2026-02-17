"""
Reinforcement Learning Utilities
"""

from typing import Tuple
import torch
import numpy as np
from collections import deque
import random


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None,
    ):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities.append(priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices: list, priorities: list):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)


class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self, size: int, theta: float = 0.15, mu: float = 0.0, sigma: float = 0.2
    ):
        self.size = size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.x = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.size)
        self.x += dx
        return self.x


class GumbelSoftmax:
    """Gumbel-Softmax for discrete actions."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        return torch.softmax((logits + gumbel) / self.temperature, dim=-1)


class EpsilonScheduler:
    """Epsilon greedy exploration scheduler."""

    def __init__(self, start: float = 1.0, end: float = 0.01, decay_steps: int = 10000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.step_count = 0

    def get_epsilon(self) -> float:
        fraction = min(self.step_count / self.decay_steps, 1.0)
        return self.start + fraction * (self.end - self.start)

    def step(self):
        self.step_count += 1


class MultiStepReward:
    """N-step return computation."""

    def __init__(self, gamma: float = 0.99, n_steps: int = 3):
        self.gamma = gamma
        self.n_steps = n_steps

    def compute(
        self,
        rewards: list,
        dones: list,
        final_value: float = 0.0,
    ) -> Tuple[list, list]:
        returns = []
        advantages = []

        for i in range(len(rewards)):
            ret = 0
            for j in range(self.n_steps):
                if i + j < len(rewards):
                    ret += self.gamma**j * rewards[i + j]
                    if dones[i + j]:
                        break
            else:
                ret += self.gamma**self.n_steps * final_value

            returns.append(ret)

        for i in range(len(returns) - 1):
            advantages.append(returns[i + 1] - returns[i] * self.gamma)
        advantages.append(0)

        return returns, advantages


class RewardScaling:
    """Reward scaling for stability."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.returns = None

    def __call__(self, reward: float, return_: float) -> float:
        if self.returns is None:
            self.returns = return_
        else:
            self.returns = self.gamma * self.returns + reward

        return reward / (abs(self.returns) + 1e-8)


class ActionRepeat:
    """Repeat actions for efficiency."""

    def __init__(self, env, repeat: int = 1):
        self.env = env
        self.repeat = repeat

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, info
