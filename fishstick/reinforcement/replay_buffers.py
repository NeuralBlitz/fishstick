"""Experience replay buffers for reinforcement learning."""

import torch
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
import random


class ReplayBuffer:
    """Basic uniform experience replay buffer."""

    def __init__(
        self, capacity: int, state_dim: int, action_dim: int = 1, device: str = "cpu"
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.states = torch.zeros(capacity, state_dim, device=device)
        self.next_states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)

        self.position = 0
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add a transition to the buffer."""
        idx = self.position

        self.states[idx] = state.to(self.device)
        self.actions[idx] = action.to(self.device)
        self.rewards[idx] = reward.to(self.device)
        self.next_states[idx] = next_state.to(self.device)
        self.dones[idx] = done.to(self.device)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    """Prioritized experience replay with sum tree."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device

        self.states = torch.zeros(capacity, state_dim, device=device)
        self.next_states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)

        self.priorities = torch.zeros(capacity, device=device)
        self.tree = SumTree(capacity)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        td_error: float = 1.0,
    ):
        """Add transition with priority based on TD error."""
        idx = self.position

        self.states[idx] = state.to(self.device)
        self.actions[idx] = action.to(self.device)
        self.rewards[idx] = reward.to(self.device)
        self.next_states[idx] = next_state.to(self.device)
        self.dones[idx] = done.to(self.device)

        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.priorities[idx] = priority
        self.tree.update(idx, priority)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample with prioritized probabilities."""
        self.beta = min(1.0, self.beta + self.beta_increment)

        indices = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            sampled = random.uniform(a, b)
            idx = self.tree.find(sampled)
            indices.append(idx)

        indices = torch.tensor(indices, device=self.device)

        probs = self.priorities[indices] / self.tree.total_priority
        weights = (self.size * probs) ** (-self.beta)
        weights = weights / weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights,
            indices,
        )

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities after learning."""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self.tree.update(idx, priority)

    def __len__(self):
        return self.size


class SumTree:
    """Binary sum tree for prioritized replay."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1)
        self.position = 0

    def update(self, idx: int, priority: float):
        """Update priority at index."""
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] = (
                self.tree[2 * tree_idx + 1] + self.tree[2 * tree_idx + 2]
            )

    def find(self, prefix_sum: float) -> int:
        """Find index for given prefix sum."""
        idx = 0
        while idx < self.capacity - 1:
            if self.tree[2 * idx + 1] >= prefix_sum:
                idx = 2 * idx + 1
            else:
                prefix_sum -= self.tree[2 * idx + 1]
                idx = 2 * idx + 2
        return idx - self.capacity + 1

    @property
    def total_priority(self):
        return self.tree[0]


class EpisodicReplayBuffer:
    """Episodic replay buffer for storing complete episodes."""

    def __init__(
        self, capacity: int, state_dim: int, action_dim: int = 1, device: str = "cpu"
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.episodes: List[Tuple] = []
        self.current_episode = []
        self.position = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add transition to current episode."""
        self.current_episode.append((state, action, reward, next_state, done))

        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            if len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    def sample(
        self, batch_size: int, episode_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, ...]:
        """Sample batch from random episodes."""
        if len(self.episodes) == 0:
            raise ValueError("No episodes in buffer")

        episodes = random.sample(self.episodes, min(batch_size, len(episodes)))

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for episode in episodes:
            if episode_length and len(episode) > episode_length:
                start = random.randint(0, len(episode) - episode_length)
                episode = episode[start : start + episode_length]

            for trans in episode:
                s, a, r, ns, d = trans
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
        )

    def __len__(self):
        return len(self.episodes)


class HindsightReplayBuffer(ReplayBuffer):
    """Hindsight Experience Replay (HER) buffer."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        goal_dim: int = None,
        n_goals: int = 4,
        strategy: str = "future",
        device: str = "cpu",
    ):
        super().__init__(capacity, state_dim, action_dim, device)
        self.goal_dim = goal_dim or state_dim
        self.n_goals = n_goals
        self.strategy = strategy
        self.device = device

        self.goals = torch.zeros(capacity, self.goal_dim, device=device)
        self.episode_goals = []
        self.episode_states = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ):
        """Add transition with goal."""
        self.episode_states.append(state.to(self.device))
        if goal is not None:
            self.episode_goals.append(goal.to(self.device))

        super().add(state, action, reward, next_state, done)

        if goal is not None:
            idx = self.position - 1 if self.position > 0 else self.capacity - 1
            self.goals[idx] = goal.to(self.device)

        if done:
            self._add_her_transitions()

    def _add_her_transitions(self):
        """Add hindsight goals from completed episode."""
        if len(self.episode_states) < 2:
            return

        n = len(self.episode_states)
        achieved_goals = self.episode_states[-1][: self.goal_dim]

        for _ in range(self.n_goals):
            if self.strategy == "future":
                future_idx = random.randint(0, n - 1)
                selected_goal = self.episode_states[future_idx][: self.goal_dim]
            elif self.strategy == "final":
                selected_goal = achieved_goals
            else:
                selected_goal = achieved_goals

            for i in range(n - 1):
                state = self.episode_states[i]
                next_state = self.episode_states[i + 1]

                reward = self._compute_reward(next_state, selected_goal)
                done = torch.tensor([False], device=self.device)

                self.add(state, torch.zeros(1), reward, next_state, done, selected_goal)

        self.episode_states = []
        self.episode_goals = []

    def _compute_reward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute reward based on goal achievement."""
        distance = torch.norm(state[: self.goal_dim] - goal)
        reward = -1.0 if distance > 0.05 else 0.0
        return torch.tensor([reward], device=self.device)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch including goals."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[indices],
            self.goals[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )


def create_replay_buffer(
    buffer_type: str, capacity: int, state_dim: int, action_dim: int = 1, **kwargs
) -> ReplayBuffer:
    """Factory function to create replay buffers."""
    buffers = {
        "uniform": ReplayBuffer,
        "prioritized": PrioritizedReplayBuffer,
        "episodic": EpisodicReplayBuffer,
        "her": HindsightReplayBuffer,
    }

    if buffer_type not in buffers:
        raise ValueError(f"Unknown buffer type: {buffer_type}")

    return buffers[buffer_type](capacity, state_dim, action_dim, **kwargs)
