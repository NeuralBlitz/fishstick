"""
fishstick Game Theory Module

Comprehensive game theory and multi-agent learning toolkit providing:
- Nash equilibrium solvers for normal-form and zero-sum games
- Cooperative game theory (TU games, Shapley value, Core, Nucleolus)
- Mechanism design primitives (auctions, voting, matching)
- Multi-agent RL interfaces and algorithms
- Evolutionary game theory and dynamics

Usage:
    from fishstick.gametheory import (
        NormalFormGame,
        NashEquilibriumSolver,
        ShapleyValue,
        Auction,
        MultiAgentEnv,
        ReplicatorDynamics,
    )
"""

from fishstick.gametheory.core_types import (
    Strategy,
    MixedStrategy,
    PayoffMatrix,
    GameOutcome,
    Player,
)
from fishstick.gametheory.normal_form_game import (
    NormalFormGame,
    TwoPlayerGame,
    ZeroSumGame,
    CoordinationGame,
    PrisonersDilemma,
    ChickenGame,
    BattleOfSexes,
)
from fishstick.gametheory.nash_solver import (
    NashEquilibriumSolver,
    PureStrategyNE,
    MixedStrategyNE,
    LemkeHowson,
    SupportEnumeration,
)
from fishstick.gametheory.zero_sum_solver import (
    ZeroSumSolver,
    LinearProgrammingSolver,
    FictitiousPlay,
    GradientDescentSolver,
)
from fishstick.gametheory.cooperative_game import (
    TUGame,
    SimpleGame,
    MajorityGame,
    WeightedVotingGame,
    CostGame,
    NetworkGame,
)
from fishstick.gametheory.solution_concepts import (
    SolutionConcept,
    ShapleyValue,
    Nucleolus,
    Core,
    Kernel,
    BanzhafIndex,
    OwenValue,
)
from fishstick.gametheory.mechanism_design import (
    Mechanism,
    MechanismDesign,
    VCGMechanism,
    MyersonMechanism,
)
from fishstick.gametheory.auctions import (
    Auction,
    VickreyAuction,
    DutchAuction,
    EnglishAuction,
    FirstPriceAuction,
    DoubleAuction,
)
from fishstick.gametheory.voting import (
    VotingRule,
    PluralityVoting,
    BordaCount,
    CondorcetWinner,
    ApprovalVoting,
)
from fishstick.gametheory.multi_agent_env import (
    MultiAgentEnvironment,
    MarkovGame,
    MatrixGame,
    CooperativeMatrixGame,
    CompetitiveMatrixGame,
)
from fishstick.gametheory.marl_algorithms import (
    MultiAgentRLAlgorithm,
    QLearning,
    PolicyGradientMARL,
    NashQLearning,
    MeanFieldQ,
)
from fishstick.gametheory.evolutionary_dynamics import (
    EvolutionaryDynamics,
    ReplicatorDynamics,
    BestResponseDynamics,
    LogitDynamics,
    MoranProcess,
)
from fishstick.gametheory.population_games import (
    PopulationGame,
    StablePopulation,
    find_nash_equilibrium_population,
    check_evolutionary_stability,
)

__all__ = [
    # Core types
    "Strategy",
    "MixedStrategy",
    "PayoffMatrix",
    "GameOutcome",
    "Player",
    # Normal form games
    "NormalFormGame",
    "TwoPlayerGame",
    "ZeroSumGame",
    "CoordinationGame",
    "PrisonersDilemma",
    "ChickenGame",
    "BattleOfSexes",
    # Nash solvers
    "NashEquilibriumSolver",
    "PureStrategyNE",
    "MixedStrategyNE",
    "LemkeHowson",
    "SupportEnumeration",
    # Zero-sum solvers
    "ZeroSumSolver",
    "LinearProgrammingSolver",
    "FictitiousPlay",
    "GradientDescentSolver",
    # Cooperative games
    "TUGame",
    "SimpleGame",
    "MajorityGame",
    "WeightedVotingGame",
    "CostGame",
    "NetworkGame",
    # Solution concepts
    "SolutionConcept",
    "ShapleyValue",
    "Nucleolus",
    "Core",
    "Kernel",
    "BanzhafIndex",
    "OwenValue",
    # Mechanism design
    "Mechanism",
    "MechanismDesign",
    "VCGMechanism",
    "MyersonMechanism",
    # Auctions
    "Auction",
    "VickreyAuction",
    "DutchAuction",
    "EnglishAuction",
    "FirstPriceAuction",
    "DoubleAuction",
    # Voting
    "VotingRule",
    "PluralityVoting",
    "BordaCount",
    "CondorcetWinner",
    "ApprovalVoting",
    # Multi-agent environments
    "MultiAgentEnvironment",
    "MarkovGame",
    "MatrixGame",
    "CooperativeMatrixGame",
    "CompetitiveMatrixGame",
    # MARL algorithms
    "MultiAgentRLAlgorithm",
    "QLearning",
    "PolicyGradientMARL",
    "NashQLearning",
    "MeanFieldQ",
    # Evolutionary dynamics
    "EvolutionaryDynamics",
    "ReplicatorDynamics",
    "BestResponseDynamics",
    "LogitDynamics",
    "MoranProcess",
    # Population games
    "PopulationGame",
    "StablePopulation",
    "find_nash_equilibrium_population",
    "check_evolutionary_stability",
]
