"""
Multi-Task RL Utilities.

Utilities for multi-task reinforcement learning including:
- Multi-task policy networks
- Task embedding and identification
- Parameter sharing utilities

These tools enable training a single agent across multiple tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from fishstick.reinforcement import (
    GaussianPolicy,
    ActionValueNetwork,
    StateValueNetwork,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task RL.

    Attributes:
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        num_tasks: Number of tasks.
        hidden_dims: Hidden layer dimensions.
        embedding_dim: Dimension of task embedding.
        actor_lr: Learning rate for actor.
        critic_lr: Learning rate for critic.
        gamma: Discount factor.
        tau: Soft update parameter.
        device: Device to use.
    """

    state_dim: int = 0
    action_dim: int = 0
    num_tasks: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    embedding_dim: int = 32
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    device: str = "cuda"


class TaskEmbeddingNetwork(Module):
    """Network for learning task embeddings.

    Learns a latent embedding for each task based on observations.
    """

    def __init__(
        self,
        state_dim: int,
        num_tasks: int,
        embedding_dim: int = 32,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim

        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, embedding_dim))

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], embedding_dim),
        )

    def get_embedding(self, task_id: Optional[int] = None) -> Tensor:
        """Get embedding for a specific task."""
        if task_id is None:
            return self.task_embeddings
        return self.task_embeddings[task_id]

    def forward(self, state: Tensor, task_id: Optional[Tensor] = None) -> Tensor:
        """Encode state with task embedding."""
        state_encoding = self.encoder(state)

        if task_id is not None:
            task_emb = self.task_embeddings[task_id]
        else:
            task_emb = self.task_embeddings.mean(dim=0, keepdim=True).expand(
                state.shape[0], -1
            )

        return state_encoding + task_emb


class TaskIdentifier(Module):
    """Task identification network.

    Identifies which task the agent is currently in based on observations.
    Useful for task inference in multi-task settings.
    """

    def __init__(
        self,
        state_dim: int,
        num_tasks: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_tasks = num_tasks

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_tasks),
        )

    def forward(self, state: Tensor) -> Tensor:
        """Get task logits from state."""
        return self.network(state)

    def predict_task(self, state: Tensor) -> Tensor:
        """Predict task ID from state."""
        logits = self.forward(state)
        return torch.argmax(logits, dim=-1)

    def get_task_probability(self, state: Tensor) -> Tensor:
        """Get probability distribution over tasks."""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


class MultiTaskPolicy(Module):
    """Multi-task policy network.

    A single policy that can handle multiple tasks by conditioning
    on task embeddings.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_tasks: int,
        embedding_dim: int = 32,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim

        self.task_embedding = nn.Parameter(torch.randn(num_tasks, embedding_dim))

        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.policy_heads = nn.ModuleList(
            [
                nn.Linear(hidden_dims[1] + embedding_dim, action_dim * 2)
                for _ in range(num_tasks)
            ]
        )

    def get_embedding(self, task_id: Tensor) -> Tensor:
        """Get embedding for task ID."""
        return self.task_embedding[task_id]

    def forward(
        self,
        state: Tensor,
        task_id: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Get action distribution for a task.

        Args:
            state: State tensor
            task_id: Task ID tensor

        Returns:
            Tuple of (mean, log_std)
        """
        encoded = self.shared_encoder(state)
        task_emb = self.get_embedding(task_id)

        combined = torch.cat([encoded, task_emb], dim=-1)

        mean_std = self.policy_heads[task_id](combined)
        mean, log_std = mean_std.chunk(2, dim=-1)
        log_std = F.softplus(log_std) + 1e-5

        return mean, log_std

    def get_action(
        self,
        state: Tensor,
        task_id: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Get action from policy.

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state, task_id)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            dist = Normal(mean, log_std)
            action = torch.tanh(dist.rsample())
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob


class MultiTaskValueNetwork(Module):
    """Multi-task Q-value network.

    Estimates Q-values for multiple tasks with shared encoder.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_tasks: int,
        embedding_dim: int = 32,
        hidden_dims: List[int] = None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim

        self.task_embedding = nn.Parameter(torch.randn(num_tasks, embedding_dim))

        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.q_heads = nn.ModuleList(
            [
                nn.Linear(hidden_dims[1] + embedding_dim + action_dim, 1)
                for _ in range(num_tasks)
            ]
        )

    def get_embedding(self, task_id: Tensor) -> Tensor:
        """Get embedding for task ID."""
        return self.task_embedding[task_id]

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        task_id: Tensor,
    ) -> Tensor:
        """Get Q-value for state-action pair in a task."""
        encoded = self.shared_encoder(state)
        task_emb = self.get_embedding(task_id)

        combined = torch.cat([encoded, task_emb, action], dim=-1)

        q_value = self.q_heads[task_id](combined)
        return q_value.squeeze(-1)


class MultiTaskAgent(Module):
    """Multi-task RL agent.

    A single agent that can learn to perform multiple tasks simultaneously.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_tasks: int,
        config: Optional[MultiTaskConfig] = None,
    ):
        super().__init__()

        self.config = config or MultiTaskConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks

        self.policy = MultiTaskPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_network1 = MultiTaskValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_network2 = MultiTaskValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_q1 = MultiTaskValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_q2 = MultiTaskValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_tasks=num_tasks,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.target_q1.load_state_dict(self.q_network1.state_dict())
        self.target_q2.load_state_dict(self.q_network2.state_dict())

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.actor_lr,
        )

        self.q_optimizer1 = torch.optim.Adam(
            self.q_network1.parameters(),
            lr=self.config.critic_lr,
        )

        self.q_optimizer2 = torch.optim.Adam(
            self.q_network2.parameters(),
            lr=self.config.critic_lr,
        )

        self.task_identifier = TaskIdentifier(
            state_dim=state_dim,
            num_tasks=num_tasks,
            hidden_dims=self.config.hidden_dims,
        )

        self.gamma = self.config.gamma
        self.tau = self.config.tau

    def get_action(
        self,
        state: Tensor,
        task_id: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Get action for a specific task."""
        action, _ = self.policy.get_action(state, task_id, deterministic)
        return action

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        task_id: Tensor,
    ) -> Dict[str, float]:
        """Update multi-task agent.

        Args:
            state: States
            action: Actions
            reward: Rewards
            next_state: Next states
            done: Done flags
            task_id: Task IDs
        """
        state = state.float()
        action = action.float()
        reward = reward.float()
        next_state = next_state.float()
        done = done.float()

        with torch.no_grad():
            next_action, next_log_pi = self.policy.get_action(
                next_state, task_id, False
            )
            target_q1 = self.target_q1(next_state, next_action, task_id)
            target_q2 = self.target_q2(next_state, next_action, task_id)
            target_q = torch.min(target_q1, target_q2)
            q_target = reward + (1 - done) * self.config.gamma * (
                target_q - next_log_pi
            )

        q1 = self.q_network1(state, action, task_id)
        q2 = self.q_network2(state, action, task_id)

        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        self.q_optimizer1.zero_grad()
        q1_loss.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q2_loss.backward()
        self.q_optimizer2.step()

        new_action, log_pi = self.policy.get_action(state, task_id)
        q1_new = self.q_network1(state.detach(), new_action, task_id)
        q2_new = self.q_network2(state.detach(), new_action, task_id)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = -q_new.mean() + log_pi.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update_target()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
        }

    def _soft_update_target(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_q1.parameters(),
            self.q_network1.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_q2.parameters(),
            self.q_network2.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


def create_multitask_agent(
    state_dim: int,
    action_dim: int,
    num_tasks: int,
    **kwargs,
) -> MultiTaskAgent:
    """Create a multi-task RL agent.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        num_tasks: Number of tasks
        **kwargs: Additional configuration

    Returns:
        MultiTaskAgent instance
    """
    config = MultiTaskConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
        **kwargs,
    )
    return MultiTaskAgent(state_dim, action_dim, num_tasks, config)
