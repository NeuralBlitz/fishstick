"""
Hierarchical RL Primitives.

Implements hierarchical reinforcement learning methods:
- Option-Critic: Learning options (temporal abstractions)
- HAC: Hierarchical Actor-Critic

References:
    Bacon et al. "The Option-Critic Architecture" (2017)
    Levy et al. "Multi-Level Hierarchical Reinforcement Learning" (2019)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical, Normal

from fishstick.reinforcement import (
    GaussianPolicy,
    ActionValueNetwork,
    StateValueNetwork,
)

Tensor = torch.Tensor
Module = nn.Module


@dataclass
class OptionCriticConfig:
    """Configuration for Option-Critic."""

    state_dim: int = 0
    action_dim: int = 0
    num_options: int = 4
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    device: str = "cuda"


@dataclass
class HACConfig:
    """Configuration for HAC."""

    state_dim: int = 0
    action_dim: int = 0
    num_levels: int = 2
    goal_dim: int = 2
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    device: str = "cuda"


class Option(Module):
    """An option in hierarchical RL with policy and termination."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim * 2),
        )

        self.termination_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], 1),
            nn.Sigmoid(),
        )

    def get_action(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        output = self.policy_net(state)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = F.softplus(log_std) + 1e-5

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            dist = Normal(mean, log_std)
            action = torch.tanh(dist.rsample())
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def get_termination_prob(self, state: Tensor) -> Tensor:
        return self.termination_net(state).squeeze(-1)


class MetaController(Module):
    """Meta-Controller for selecting options."""

    def __init__(
        self,
        state_dim: int,
        num_options: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_options = num_options

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], num_options),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.network(state)

    def get_option(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        q_values = self.forward(state)
        dist = Categorical(logits=q_values)

        if deterministic:
            option = torch.argmax(q_values, dim=-1)
            log_prob = None
        else:
            option = dist.sample()
            log_prob = dist.log_prob(option)

        return option, log_prob


class SubController(Module):
    """Sub-Controller for primitive actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims or [64, 64],
        )

    def get_action(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        return self.policy.get_action(state, deterministic)


class HierarchicalPolicy(Module):
    """Hierarchical Policy with meta-controller and options."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_options: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options

        self.meta_controller = MetaController(
            state_dim=state_dim,
            num_options=num_options,
            hidden_dims=hidden_dims,
        )

        self.options = nn.ModuleList(
            [Option(state_dim, action_dim, hidden_dims) for _ in range(num_options)]
        )

        self.current_option = None
        self.option_steps = 0
        self.max_option_steps = 10

    def select_option(
        self, state: Tensor, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        option, log_prob = self.meta_controller.get_option(state, deterministic)
        self.current_option = option
        self.option_steps = 0
        return option, log_prob

    def get_action(
        self,
        state: Tensor,
        option: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if option is None:
            option = self.current_option

        option_idx = option.item() if option.dim() == 0 else option
        option_module = self.options[option_idx]

        action, log_prob = option_module.get_action(state, deterministic)
        termination_prob = option_module.get_termination_prob(state)

        self.option_steps += 1

        return action, log_prob, termination_prob

    def should_terminate(self, state: Tensor) -> bool:
        if self.current_option is None:
            return True

        option_idx = (
            self.current_option.item()
            if self.current_option.dim() == 0
            else self.current_option
        )
        option_module = self.options[option_idx]

        termination_prob = option_module.get_termination_prob(state)

        return (
            torch.rand(1).item() < termination_prob.item()
            or self.option_steps >= self.max_option_steps
        )


class OptionCritic(Module):
    """Option-Critic Architecture for hierarchical RL.

    Learns options end-to-end along with the policy over options.

    Reference:
        Bacon et al. "The Option-Critic Architecture" (2017)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[OptionCriticConfig] = None,
    ):
        super().__init__()

        self.config = config or OptionCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = self.config.num_options

        self.hierarchical_policy = HierarchicalPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            num_options=self.num_options,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_option = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=self.num_options,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_option_target = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=self.num_options,
            hidden_dims=self.config.hidden_dims,
        )

        self.q_option_target.load_state_dict(self.q_option.state_dict())

        self.value_network = StateValueNetwork(
            state_dim=state_dim,
            hidden_dims=self.config.hidden_dims,
        )

        self.meta_optimizer = torch.optim.Adam(
            self.hierarchical_policy.parameters(),
            lr=self.config.actor_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.q_option.parameters()) + list(self.value_network.parameters()),
            lr=self.config.critic_lr,
        )

        self.gamma = self.config.gamma
        self.entropy_coef = self.config.entropy_coef

    def get_action(
        self,
        state: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        should_terminate = self.hierarchical_policy.should_terminate(state)

        if should_terminate or self.hierarchical_policy.current_option is None:
            option, option_log_prob = self.hierarchical_policy.select_option(
                state, deterministic
            )
        else:
            option = self.hierarchical_policy.current_option
            option_log_prob = None

        action, log_prob, termination_prob = self.hierarchical_policy.get_action(
            state, option, deterministic
        )

        return action, option, log_prob, termination_prob

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
        option: Tensor,
        next_option: Tensor,
    ) -> Dict[str, float]:
        state = state.float()
        action = action.float()
        reward = reward.float()
        next_state = next_state.float()
        done = done.float()

        q_values = self.q_option(state)
        next_q_values = self.q_option_target(next_state)

        q_option = q_values.gather(1, option.unsqueeze(1)).squeeze(1)

        v_value = self.value_network(state)

        next_v = self.value_network(next_state)

        with torch.no_grad():
            target_q = reward + (1 - done) * self.gamma * next_v

        critic_loss = F.mse_loss(q_option, target_q)

        option_logits = self.hierarchical_policy.meta_controller(state)
        option_dist = torch.distributions.Categorical(logits=option_logits)

        entropy = option_dist.entropy().mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_option_logits = self.hierarchical_policy.meta_controller(state.detach())
        new_option_dist = torch.distributions.Categorical(logits=new_option_logits)
        new_option = new_option_dist.sample()

        new_action, log_prob, _ = self.hierarchical_policy.get_action(
            state.detach(), new_option
        )

        new_q = (
            self.q_option(state.detach()).gather(1, new_option.unsqueeze(1)).squeeze(1)
        )

        meta_loss = -new_q.mean() - self.entropy_coef * entropy

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        for target_param, param in zip(
            self.q_option_target.parameters(),
            self.q_option.parameters(),
        ):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "meta_loss": meta_loss.item(),
            "entropy": entropy.item(),
        }


class HAC(Module):
    """Hierarchical Actor-Critic (HAC).

    Multi-level hierarchy where each level has its own goal and policy.

    Reference:
        Levy et al. "Multi-Level Hierarchical Reinforcement Learning" (2019)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[HACConfig] = None,
    ):
        super().__init__()

        self.config = config or HACConfig(
            state_dim=state_dim,
            action_dim=action_dim,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_levels = self.config.num_levels
        self.goal_dim = self.config.goal_dim

        self.policies = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()

        for level in range(self.num_levels):
            if level == 0:
                input_dim = state_dim + goal_dim
            else:
                input_dim = state_dim + goal_dim + goal_dim

            self.policies.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.config.hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(
                        self.config.hidden_dims[1],
                        action_dim if level == 0 else goal_dim,
                    ),
                )
            )

            critic_input_dim = input_dim + (action_dim if level == 0 else goal_dim)
            self.critics.append(
                nn.Sequential(
                    nn.Linear(critic_input_dim, self.config.hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[1], 1),
                )
            )

            self.target_critics.append(
                nn.Sequential(
                    nn.Linear(critic_input_dim, self.config.hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[1]),
                    nn.ReLU(),
                    nn.Linear(self.config.hidden_dims[1], 1),
                )
            )

            self.target_critics[-1].load_state_dict(self.critics[-1].state_dict())

        self.optimizers = [
            torch.optim.Adam(
                list(policy.parameters()) + list(critic.parameters()),
                lr=self.config.actor_lr,
            )
            for policy, critic in zip(self.policies, self.critics)
        ]

        self.gamma = self.config.gamma

    def get_goal(self, level: int, state: Tensor) -> Tensor:
        if level == 0:
            return torch.zeros(state.shape[0], self.goal_dim, device=state.device)
        return torch.randn(state.shape[0], self.goal_dim, device=state.device)

    def get_action(
        self,
        state: Tensor,
        goal: Tensor,
        level: int,
        deterministic: bool = False,
    ) -> Tensor:
        if level == 0:
            state_goal = torch.cat([state, goal], dim=-1)
        else:
            state_goal = torch.cat([state, goal, goal], dim=-1)

        action = self.policies[level](state_goal)

        if level == 0 and not deterministic:
            action = action + torch.randn_like(action) * 0.1

        return torch.tanh(action)

    def update(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> Dict[str, float]:
        losses = {}

        for level in range(self.num_levels):
            goal = self.get_goal(level, state)

            if level == 0:
                state_input = torch.cat([state, goal], dim=-1)
                next_state_input = torch.cat([next_state, goal], dim=-1)
                action_input = action
            else:
                state_input = torch.cat([state, goal, goal], dim=-1)
                next_state_input = torch.cat([next_state, goal, goal], dim=-1)
                action_input = goal

            q_value = self.critics[level](
                torch.cat([state_input, action_input], dim=-1)
            )

            with torch.no_grad():
                target_action = self.get_action(
                    next_state, goal, level, deterministic=False
                )
                target_q = reward + (1 - done) * self.gamma * self.target_critics[
                    level
                ](torch.cat([next_state_input, target_action], dim=-1))

            critic_loss = F.mse_loss(q_value, target_q)

            self.optimizers[level].zero_grad()
            critic_loss.backward()
            self.optimizers[level].step()

            losses[f"level_{level}_critic_loss"] = critic_loss.item()

            for target_param, param in zip(
                self.target_critics[level].parameters(),
                self.critics[level].parameters(),
            ):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

        return losses
