"""
Multi-agent RL environments.

Implements multi-agent environment interfaces for MARL research,
including matrix games, Markov games, and cooperative/competitive
settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from fishstick.gametheory.normal_form_game import NormalFormGame, TwoPlayerGame


@dataclass
class MultiAgentState:
    """State in a multi-agent environment.

    Attributes:
        observation: The observation for each agent
        agent_ids: List of agent IDs
        step: Current timestep
    """

    observation: Dict[int, NDArray[np.float64]]
    agent_ids: List[int]
    step: int = 0

    def get_obs(self, agent_id: int) -> NDArray[np.float64]:
        """Get observation for a specific agent."""
        return self.observation.get(agent_id, np.array([]))


@dataclass
class MultiAgentStep:
    """Result of a step in the multi-agent environment.

    Attributes:
        observations: Observations for each agent
        rewards: Rewards for each agent
        done: Whether episode is finished
        info: Additional information
    """

    observations: Dict[int, NDArray[np.float64]]
    rewards: Dict[int, float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class MultiAgentEnvironment(ABC):
    """Abstract base class for multi-agent environments."""

    @abstractmethod
    def reset(self) -> MultiAgentState:
        """Reset the environment."""
        pass

    @abstractmethod
    def step(self, actions: Dict[int, Any]) -> MultiAgentStep:
        """Execute actions for all agents."""
        pass

    @abstractmethod
    def get_agent_ids(self) -> List[int]:
        """Get list of agent IDs."""
        pass

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Number of agents."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """Observation space."""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """Action space."""
        pass


@dataclass
class MatrixGame(MultiAgentEnvironment):
    """A matrix game (simultaneous move) as a multi-agent environment.

    Attributes:
        game: The underlying normal-form game
    """

    game: TwoPlayerGame

    def __post_init__(self) -> None:
        self._current_state: Optional[MultiAgentState] = None
        self._step_count = 0

    def reset(self) -> MultiAgentState:
        """Reset the game."""
        self._step_count = 0
        self._current_state = MultiAgentState(
            observation={i: np.array([i]) for i in range(self.n_agents)},
            agent_ids=self.get_agent_ids(),
            step=0,
        )
        return self._current_state

    def step(self, actions: Dict[int, int]) -> MultiAgentStep:
        """Execute one step of the game."""
        if self._current_state is None:
            self.reset()

        action_profile = tuple(actions.get(i, 0) for i in range(self.n_agents))

        payoffs = self.game.get_payoffs(action_profile)

        rewards = {i: float(payoffs[i]) for i in range(self.n_agents)}

        self._step_count += 1
        done = True

        return MultiAgentStep(
            observations={i: np.array([i]) for i in range(self.n_agents)},
            rewards=rewards,
            done=done,
            info={"action_profile": action_profile, "payoffs": payoffs},
        )

    def get_agent_ids(self) -> List[int]:
        return list(range(self.n_agents))

    @property
    def n_agents(self) -> int:
        return self.game.num_players()

    @property
    def observation_space(self) -> int:
        return 1

    @property
    def action_space(self) -> int:
        return self.game.num_actions(0)


@dataclass
class MarkovGame(MultiAgentEnvironment):
    """A stochastic game (Markov game).

    Extends matrix games to multiple states with transition dynamics.

    Attributes:
        states: Set of game states
        n_agents: Number of agents
        payoff_tensors: Payoff functions for each state and agent
        transition_probs: Transition probabilities
    """

    states: Set[int]
    n_agents: int
    payoff_tensors: Dict[int, NDArray[np.float64]]
    transition_probs: Dict[int, Dict[Tuple[int, ...], Dict[int, float]]]

    def __init__(
        self,
        states: Set[int],
        n_agents: int,
        payoff_tensors: Dict[int, NDArray[np.float64]],
        transition_probs: Dict[int, Dict[Tuple[int, ...], Dict[int, float]]],
        initial_state: int = 0,
    ):
        self.states = states
        self.n_agents = n_agents
        self.payoff_tensors = payoff_tensors
        self.transition_probs = transition_probs
        self.current_state = initial_state
        self._step_count = 0

    def reset(self) -> MultiAgentState:
        """Reset the game."""
        self._step_count = 0
        self.current_state = 0

        return MultiAgentState(
            observation={
                i: np.array([self.current_state]) for i in range(self.n_agents)
            },
            agent_ids=list(range(self.n_agents)),
            step=0,
        )

    def step(self, actions: Dict[int, int]) -> MultiAgentStep:
        """Execute one step."""
        action_profile = tuple(actions.get(i, 0) for i in range(self.n_agents))

        payoff_tensor = self.payoff_tensors[self.current_state]
        payoffs = payoff_tensor[action_profile]

        rewards = {i: float(payoffs[i]) for i in range(self.n_agents)}

        next_state_probs = self.transition_probs.get(self.current_state, {}).get(
            action_profile, {0: 1.0}
        )

        next_state = np.random.choice(
            list(next_state_probs.keys()), p=list(next_state_probs.values())
        )

        self.current_state = next_state
        self._step_count += 1

        done = self._step_count >= 100

        return MultiAgentStep(
            observations={
                i: np.array([self.current_state]) for i in range(self.n_agents)
            },
            rewards=rewards,
            done=done,
            info={
                "state": self.current_state,
                "action_profile": action_profile,
            },
        )

    def get_agent_ids(self) -> List[int]:
        return list(range(self.n_agents))

    @property
    def observation_space(self) -> int:
        return len(self.states)

    @property
    def action_space(self) -> int:
        if self.payoff_tensors:
            first_tensor = next(iter(self.payoff_tensors.values()))
            return first_tensor.shape[0]
        return 2


@dataclass
class CooperativeMatrixGame(MatrixGame):
    """A cooperative matrix game where agents share rewards."""

    def __init__(self, payoff_matrix: NDArray[np.float64]):
        from fishstick.gametheory.normal_form_game import TwoPlayerGame

        n_actions = payoff_matrix.shape[0]

        game = TwoPlayerGame(
            row_payoffs=payoff_matrix,
            col_payoffs=payoff_matrix,
        )

        super().__init__(game)

    def step(self, actions: Dict[int, int]) -> MultiAgentStep:
        """Execute step with shared rewards."""
        result = super().step(actions)

        shared_reward = sum(result.rewards.values()) / self.n_agents
        result.rewards = {i: shared_reward for i in range(self.n_agents)}

        return result


@dataclass
class CompetitiveMatrixGame(MatrixGame):
    """A competitive matrix game (zero-sum)."""

    def __init__(self, payoff_matrix: NDArray[np.float64]):
        from fishstick.gametheory.normal_form_game import ZeroSumGame

        game = ZeroSumGame(payoff_matrix)

        super().__init__(game)


@dataclass
class RepeatedGame(MultiAgentEnvironment):
    """A repeated game where the same matrix game is played multiple times.

    Attributes:
        base_game: The matrix game to repeat
        horizon: Number of repetitions
        discount_factor: Discount factor for future rewards
    """

    base_game: MatrixGame
    horizon: int = 10
    discount_factor: float = 0.95

    def __post_init__(self) -> None:
        self._current_state: Optional[MultiAgentState] = None
        self._step_count = 0
        self._cumulative_rewards: Dict[int, float] = {}

    def reset(self) -> MultiAgentState:
        """Reset the repeated game."""
        self._step_count = 0
        self._cumulative_rewards = {i: 0.0 for i in range(self.n_agents)}

        self._current_state = MultiAgentState(
            observation={i: np.array([0, 0.0]) for i in range(self.n_agents)},
            agent_ids=self.get_agent_ids(),
            step=0,
        )

        return self._current_state

    def step(self, actions: Dict[int, int]) -> MultiAgentStep:
        """Execute one repetition of the base game."""
        step_result = self.base_game.step(actions)

        discount = self.discount_factor**self._step_count

        for agent_id in self._cumulative_rewards:
            self._cumulative_rewards[agent_id] += (
                discount * step_result.rewards[agent_id]
            )

        self._step_count += 1

        obs = {
            i: np.array([self._step_count, self._cumulative_rewards[i]])
            for i in range(self.n_agents)
        }

        return MultiAgentStep(
            observations=obs,
            rewards=step_result.rewards,
            done=self._step_count >= self.horizon,
            info={
                "cumulative_rewards": dict(self._cumulative_rewards),
                "discount": discount,
            },
        )

    def get_agent_ids(self) -> List[int]:
        return self.base_game.get_agent_ids()

    @property
    def n_agents(self) -> int:
        return self.base_game.n_agents

    @property
    def observation_space(self) -> int:
        return 2

    @property
    def action_space(self) -> int:
        return self.base_game.action_space


def create_matrix_game_env(game_type: str, **kwargs) -> MatrixGame:
    """Factory function to create matrix game environments."""

    if game_type == "prisoners_dilemma":
        from fishstick.gametheory.normal_form_game import PrisonersDilemma

        game = PrisonersDilemma(**kwargs)
        return MatrixGame(game)

    elif game_type == "coordination":
        from fishstick.gametheory.normal_form_game import CoordinationGame

        game = CoordinationGame(**kwargs)
        return MatrixGame(game)

    elif game_type == "chicken":
        from fishstick.gametheory.normal_form_game import ChickenGame

        game = ChickenGame(**kwargs)
        return MatrixGame(game)

    elif game_type == "battle_of_sexes":
        from fishstick.gametheory.normal_form_game import BattleOfSexes

        game = BattleOfSexes(**kwargs)
        return MatrixGame(game)

    elif game_type == "rock_paper_scissors":
        from fishstick.gametheory.normal_form_game import create_rock_paper_scissors

        game = create_rock_paper_scissors()
        return MatrixGame(game)

    else:
        raise ValueError(f"Unknown game type: {game_type}")
