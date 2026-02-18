"""
Reinforcement Learning for Robotics.

Specialized RL agents and algorithms for robot control:
- Joint space control agents
- Task space control agents
- Hybrid position/force agents
- Sim-to-real transfer utilities
"""

from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

from .core import JointState, TaskState, Observation, Action, Trajectory


@dataclass
class StateAction:
    """State-action pair for robotics."""

    observation: Observation
    action: Action
    reward: float
    next_observation: Optional[Observation] = None
    done: bool = False

    @property
    def obs_vector(self) -> Tensor:
        return self.observation.to_vector()

    @property
    def action_vector(self) -> Tensor:
        if self.action.joint_torques is not None:
            return self.action.joint_torques
        elif self.action.target_position is not None:
            return self.action.target_position
        return torch.zeros(1)


class RoboticsAgent(nn.Module):
    """
    Base class for robotics RL agents.

    Provides common infrastructure for:
    - Action masking
    - Safety constraints
    - Domain randomization
    - Sim-to-real transfer
    """

    def __init__(
        self,
        n_joints: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = RoboticsReplayBuffer(capacity=100000)

    def get_action(
        self,
        observation: Observation,
        deterministic: bool = False,
    ) -> Action:
        """Compute action from observation."""
        raise NotImplementedError

    def update(
        self,
        batch_size: int,
    ) -> Dict[str, float]:
        """Update agent from replay buffer."""
        raise NotImplementedError

    def compute_reward(
        self,
        observation: Observation,
        action: Action,
        next_observation: Observation,
    ) -> float:
        """Compute task-specific reward."""
        raise NotImplementedError


class JointSpaceAgent(RoboticsAgent):
    """
    Joint space RL agent for position/velocity control.

    Learns to map joint states to torque commands.
    Suitable for free-space movement tasks.

    Args:
        n_joints: Number of robot joints
        use_velocity: Include velocity in state
        action_scale: Scale factor for action outputs
    """

    def __init__(
        self,
        n_joints: int,
        use_velocity: bool = True,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        action_scale: float = 1.0,
    ):
        state_dim = n_joints * 2 if use_velocity else n_joints
        super().__init__(
            n_joints, state_dim, n_joints, hidden_dim, learning_rate, gamma, tau
        )

        self.use_velocity = use_velocity
        self.action_scale = action_scale

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_joints * 2),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim + n_joints, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_joints * 2),
        )

        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + n_joints, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=learning_rate
        )

    def _get_state(self, obs: Observation) -> Tensor:
        """Extract state vector from observation."""
        if self.use_velocity:
            return obs.proprioceptive
        return obs.joint_state.position

    def get_action(
        self,
        observation: Observation,
        deterministic: bool = False,
    ) -> Action:
        """Compute joint torque action."""
        state = self._get_state(observation)

        with torch.no_grad():
            output = self.actor(state)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = torch.tanh(dist.sample())

            action = action * self.action_scale

        return Action(joint_torques=action)

    def update(
        self,
        batch_size: int,
    ) -> Dict[str, float]:
        """Update actor and critic networks."""
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0}

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.stack([self._get_state(o) for o in states])
        next_states = torch.stack([self._get_state(o) for o in next_states])
        actions = torch.stack([a.joint_torques for a in actions])

        current_q = self.critic(torch.cat([states, actions], dim=-1))

        with torch.no_grad():
            next_output = self.target_actor(next_states)
            next_mean, next_log_std = next_output.chunk(2, dim=-1)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_actions = torch.tanh(next_dist.rsample())

            target_q = self.target_critic(
                torch.cat([next_states, next_actions], dim=-1)
            ).squeeze()

            targets = rewards + (1 - dones) * self.gamma * target_q

        critic_loss = F.mse_loss(current_q.squeeze(), targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        output = self.actor(states)
        mean, log_std = output.chunk(2, dim=-1)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        new_actions = torch.tanh(dist.rsample())

        actor_loss = -self.critic(torch.cat([states, new_actions], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def _soft_update(self):
        """Polyak averaging for target networks."""
        for target, source in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target.data.copy_(target.data * (1 - self.tau) + source.data * self.tau)

        for target, source in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1 - self.tau) + source.data * self.tau)

    def compute_reward(
        self,
        observation: Observation,
        action: Action,
        next_observation: Observation,
    ) -> float:
        """Compute tracking reward."""
        target = next_observation.joint_state.position
        current = observation.joint_state.position

        pos_error = torch.norm(target - current).item()
        vel_error = torch.norm(next_observation.joint_state.velocity).item()

        reward = -pos_error - 0.1 * vel_error

        return reward


class TaskSpaceAgent(RoboticsAgent):
    """
    Task space RL agent for end-effector control.

    Operates in operational space (position/orientation)
    rather than joint space. Better for manipulation tasks.

    Args:
        n_joints: Number of joints
        task_dim: Task space dimension (3 for position, 6 for pose)
    """

    def __init__(
        self,
        n_joints: int,
        task_dim: int = 3,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        state_dim = n_joints * 2 + task_dim * 2
        super().__init__(
            n_joints, state_dim, task_dim, hidden_dim, learning_rate, gamma, tau
        )

        self.task_dim = task_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, task_dim * 2),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim + task_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, task_dim * 2),
        )

        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + task_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=learning_rate
        )

    def _get_state(self, obs: Observation) -> Tensor:
        """Extract full state vector."""
        proprio = obs.proprioceptive

        if obs.task_state is not None:
            task = torch.cat(
                [
                    obs.task_state.position,
                    obs.task_state.linear_velocity,
                ]
            )
        else:
            task = torch.zeros(self.task_dim * 2, device=proprio.device)

        return torch.cat([proprio, task])

    def get_action(
        self,
        observation: Observation,
        deterministic: bool = False,
    ) -> Action:
        """Compute task-space action (velocity command)."""
        state = self._get_state(observation)

        with torch.no_grad():
            output = self.actor(state)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = torch.tanh(dist.sample())

        return Action(target_velocity=action)

    def update(
        self,
        batch_size: int,
    ) -> Dict[str, float]:
        """Update task-space agent."""
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0}

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.stack([self._get_state(o) for o in states])
        next_states = torch.stack([self._get_state(o) for o in next_states])
        actions = torch.stack([a.target_velocity for a in actions])

        current_q = self.critic(torch.cat([states, actions], dim=-1))

        with torch.no_grad():
            next_output = self.target_actor(next_states)
            next_mean, next_log_std = next_output.chunk(2, dim=-1)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_actions = torch.tanh(next_dist.rsample())

            target_q = self.target_critic(
                torch.cat([next_states, next_actions], dim=-1)
            ).squeeze()

            targets = rewards + (1 - dones) * self.gamma * target_q

        critic_loss = F.mse_loss(current_q.squeeze(), targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        output = self.actor(states)
        mean, log_std = output.chunk(2, dim=-1)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)
        new_actions = torch.tanh(dist.rsample())

        actor_loss = -self.critic(torch.cat([states, new_actions], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def _soft_update(self):
        for target, source in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target.data.copy_(target.data * (1 - self.tau) + source.data * self.tau)

        for target, source in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target.data.copy_(target.data * (1 - self.tau) + source.data * self.tau)


class HybridAgent(RoboticsAgent):
    """
    Hybrid position-force RL agent.

    Combines task-space position control with force feedback
    for compliant manipulation tasks.

    Uses impedance control framework where the agent learns
    stiffness and damping parameters.
    """

    def __init__(
        self,
        n_joints: int,
        task_dim: int = 6,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        state_dim = n_joints * 2 + task_dim * 2 + 6
        super().__init__(
            n_joints, state_dim, task_dim * 2, hidden_dim, learning_rate, gamma, tau
        )

        self.task_dim = task_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, task_dim * 2),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim + task_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, task_dim * 2),
        )

        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + task_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.parameters()), lr=learning_rate
        )

    def _get_state(self, obs: Observation) -> Tensor:
        """Extract state with force/torque info."""
        proprio = obs.proprioceptive

        if obs.task_state is not None:
            task = torch.cat(
                [
                    obs.task_state.position,
                    obs.task_state.linear_velocity,
                ]
            )
        else:
            task = torch.zeros(self.task_dim * 2, device=proprio.device)

        wrench = (
            obs.external_wrench
            if obs.external_wrench is not None
            else torch.zeros(6, device=proprio.device)
        )

        return torch.cat([proprio, task, wrench])

    def get_action(
        self,
        observation: Observation,
        deterministic: bool = False,
    ) -> Action:
        """Compute hybrid impedance action."""
        state = self._get_state(observation)

        with torch.no_grad():
            output = self.actor(state)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()

            if deterministic:
                action = torch.tanh(mean)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = torch.tanh(dist.sample())

        target_pos = action[: self.task_dim]
        stiffness = torch.sigmoid(action[self.task_dim :]) * 100

        return Action(
            target_position=target_pos,
            impedance_targets=torch.cat([target_pos, stiffness]),
        )


class RoboticsReplayBuffer:
    """Replay buffer for robotics RL with efficient storage."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.position = 0

    def push(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Optional[Observation],
        done: bool,
    ) -> None:
        """Add transition to buffer."""
        self.buffer.append(
            StateAction(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        )

    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)

        states = [s.observation for s in batch]
        actions = [s.action for s in batch]
        rewards = torch.FloatTensor([s.reward for s in batch])
        next_states = [s.next_observation for s in batch]
        dones = torch.FloatTensor([float(s.done) for s in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DomainRandomizer:
    """
    Domain randomization for sim-to-real transfer.

    Randomizes physics parameters during training to improve
    robustness and enable transfer from simulation to real robot.
    """

    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        randomize_every: int = 100,
    ):
        self.param_ranges = param_ranges
        self.randomize_every = randomize_every
        self.step_count = 0

    def randomize(self) -> Dict[str, float]:
        """Sample new random parameters."""
        params = {}
        for name, (low, high) in self.param_ranges.items():
            params[name] = np.random.uniform(low, high)
        return params

    def should_randomize(self) -> bool:
        """Check if randomization should occur."""
        self.step_count += 1
        return self.step_count % self.randomize_every == 0


class PPORobotics(RoboticsAgent):
    """
    PPO agent adapted for robotics.

    Uses clipped surrogate objective for stable learning
    with continuous control.
    """

    def __init__(
        self,
        n_joints: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        super().__init__(
            n_joints, state_dim, action_dim, hidden_dim, learning_rate, gamma
        )

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.log_probs: List[Tensor] = []
        self.values: List[Tensor] = []
        self.rewards: List[float] = []

    def get_action(
        self,
        observation: Observation,
        deterministic: bool = False,
    ) -> Action:
        """Sample action from policy."""
        state = observation.to_vector()

        output = self.actor(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            action = torch.tanh(dist.sample())
            log_prob = dist.log_prob(action).sum(dim=-1)

        if log_prob is not None:
            self.log_probs.append(log_prob.detach())

        value = self.critic(state)
        self.values.append(value.detach())

        return Action(joint_torques=action)

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """Compute GAE returns."""
        returns = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - 1.0
            else:
                next_non_terminal = 1.0

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * lam * next_non_terminal * gae
            returns.insert(0, gae + self.values[t])
            next_value = self.values[t]

        return torch.tensor(returns)

    def update(self, batch_size: int) -> Dict[str, float]:
        """Update policy with PPO."""
        if len(self.rewards) < batch_size:
            return {"loss": 0.0}

        returns = self.compute_returns(self.gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = 0
        value_loss = 0

        for i in range(len(self.log_probs)):
            advantage = returns[i] - self.values[i]

            ratio = torch.exp(self.log_probs[i])

            surr1 = ratio * advantage
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantage
            )

            policy_loss = policy_loss - torch.min(surr1, surr2).mean()
            value_loss = value_loss + F.mse_loss(self.values[i], returns[i])

        entropy_loss = -self.entropy_coef * sum(self.log_probs).mean()

        loss = policy_loss + self.value_coef * value_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item(),
        }
