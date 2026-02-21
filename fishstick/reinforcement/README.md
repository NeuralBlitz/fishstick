# Reinforcement Learning Module

Comprehensive reinforcement learning tools for training agents in various environments.

## Overview

This module provides:

- **Policy Networks**: Categorical, Gaussian, Deterministic
- **Value Networks**: V, Q, Advantage, Dueling
- **Algorithms**: A2C, PPO, SAC
- **Replay Buffers**: Uniform, Prioritized, HER
- **Exploration**: Epsilon-greedy, Entropy, Noisy networks

## Installation

```bash
pip install torch numpy gym
```

## Quick Start

### PPO Agent

```python
import torch
from fishstick.reinforcement import PPO, CategoricalPolicy

# Create policy
policy = CategoricalPolicy(
    state_dim=4,
    action_dim=2,
    hidden_dims=[64, 64]
)

# PPO algorithm
ppo = PPO(
    policy=policy,
    lr=3e-4,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01
)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        ppo.store_transition(state, action, reward, done)
        state = next_state
    
    ppo.update()
```

### SAC Agent

```python
from fishstick.reinforcement import SAC, GaussianPolicy

# SAC for continuous control
sac = SAC(
    state_dim=17,
    action_dim=6,
    hidden_dims=[256, 256],
    gamma=0.99,
    tau=0.005
)

# Training
for step in range(100000):
    # Collect experience
    experience = env.sample()
    sac.store(*experience)
    
    # Update
    if len(sac.replay) > 1000:
        sac.update(batch_size=256)
```

### Replay Buffer

```python
from fishstick.reinforcement import PrioritizedReplayBuffer

buffer = PrioritizedReplayBuffer(
    capacity=100000,
    alpha=0.6,
    beta=0.4
)

# Add experience
buffer.push(state, action, reward, next_state, done)

# Sample
batch = buffer.sample(batch_size=32, beta=0.4)
```

## API Reference

### Policy Networks

| Class | Description |
|-------|-------------|
| `CategoricalPolicy` | Discrete actions |
| `GaussianPolicy` | Continuous actions |
| `DeterministicPolicy` | Deterministic |
| `TanhGaussianPolicy` | Bounded continuous |

### Algorithms

| Class | Description |
|-------|-------------|
| `A2C` | Advantage Actor-Critic |
| `PPO` | Proximal Policy Optimization |
| `SAC` | Soft Actor-Critic |

### Replay Buffers

| Class | Description |
|-------|-------------|
| `UniformReplayBuffer` | Standard buffer |
| `PrioritizedReplayBuffer` | PER |
| `HERReplayBuffer` | Hindsight experience replay |

### Exploration

| Class | Description |
|-------|-------------|
| `EpsilonGreedy` | Exploration |
| `EntropyRegularization` | Entropy bonus |
| `NoisyLinear` | Noisy networks |

## License

MIT License
