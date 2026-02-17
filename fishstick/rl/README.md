# Reinforcement Learning

RL algorithms and environments.

## Installation

```bash
pip install fishstick[rl]
```

## Overview

The `rl` module provides reinforcement learning algorithms including DQN, PPO, SAC, and related utilities.

## Usage

```python
from fishstick.rl import DQN, PPO, SAC, ReplayBuffer

# DQN
dqn = DQN(
    state_dim=4,
    action_dim=2,
    hidden_dim=256
)

# PPO
ppo = PPO(
    policy=policy_net,
    value=value_net,
    lr=3e-4
)

# Replay buffer
buffer = ReplayBuffer(capacity=100000)
buffer.push(state, action, reward, next_state, done)
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| `DQN` | Deep Q-Network |
| `PolicyGradient` | Policy gradient |
| `ActorCritic` | Actor-critic |
| `PPO` | Proximal Policy Optimization |
| `SAC` | Soft Actor-Critic |
| `TD3` | Twin Delayed DDPG |

## Utilities

| Utility | Description |
|---------|-------------|
| `ReplayBuffer` | Experience replay buffer |
| `PrioritizedReplayBuffer` | Prioritized replay |
| `GymEnvironment` | OpenAI Gym wrapper |
| `OrnsteinUhlenbeckProcess` | OU noise |

## Examples

See `examples/rl/` for complete examples.
