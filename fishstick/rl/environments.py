"""
Reinforcement Learning Environments
"""

from typing import Tuple, Any, Dict
import numpy as np


class GymEnvironment:
    """Wrapper for OpenAI Gym environments."""

    def __init__(self, env_name: str, render_mode: str = None):
        self.env_name = env_name
        self.render_mode = render_mode
        self.env = None

    def _get_env(self):
        if self.env is None:
            try:
                import gym

                self.env = gym.make(self.env_name, render_mode=self.render_mode)
            except ImportError:
                raise ImportError(
                    "gymnasium is required. Install with: pip install gymnasium"
                )
        return self.env

    def reset(self) -> np.ndarray:
        env = self._get_env()
        state, _ = env.reset()
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        env = self._get_env()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        return state, reward, done, info

    def render(self):
        if self.env:
            self.env.render()

    def close(self):
        if self.env:
            self.env.close()

    @property
    def action_space(self):
        env = self._get_env()
        return env.action_space

    @property
    def observation_space(self):
        env = self._get_env()
        return env.observation_space


class EnvironmentWrapper:
    """Wrapper for custom RL environments."""

    def __init__(self, env: Any):
        self.env = env

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        return self.env.step(action)

    def render(self):
        if hasattr(self.env, "render"):
            self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class MultiAgentEnvironment:
    """Multi-agent environment wrapper."""

    def __init__(self, env_fn, num_agents: int):
        self.num_agents = num_agents
        self.envs = [env_fn() for _ in range(num_agents)]

    def reset(self) -> list:
        return [env.reset() for env in self.envs]

    def step(self, actions: list) -> Tuple[list, list, list, list]:
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones, infos = zip(*results)
        return list(states), list(rewards), list(dones), list(infos)

    def close(self):
        for env in self.envs:
            env.close()


class VectorEnvironment:
    """Vectorized environment for parallel training."""

    def __init__(self, env_fn, num_envs: int = 4):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]

    def reset(self) -> np.ndarray:
        return np.array([env.reset() for env in self.envs])

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones, infos = zip(*results)
        return np.array(states), np.array(rewards), np.array(dones), list(infos)

    def close(self):
        for env in self.envs:
            env.close()
