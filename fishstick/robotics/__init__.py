"""
fishstick Robotics Module

Robotics and control tools including:
- PID controllers
- Model Predictive Control (MPC)
- Reinforcement learning for robotics
- Trajectory planning
- Robot dynamics simulation
"""

from fishstick.robotics.pid_controller import (
    PIDController,
    CascadePID,
    AdaptivePID,
    PIDTuner,
)
from fishstick.robotics.mpc import (
    MPCController,
    LinearMPC,
    NonlinearMPC,
    QuadraticCost,
)
from fishstick.robotics.robot_rl import (
    RoboticsAgent,
    JointSpaceAgent,
    TaskSpaceAgent,
    HybridAgent,
    StateAction,
)
from fishstick.robotics.motion_policies import (
    MotionPolicy,
    ImitationPolicy,
    GMMPolicy,
    DynamicMovementPrimitive,
)
from fishstick.robotics.trajectory_planning import (
    TrajectoryPlanner,
    RRTStar,
    CHOMP,
    TimeOptimalProfile,
    JacobianInverse,
)
from fishstick.robotics.dynamics import (
    RigidBodyDynamics,
    SerialChainDynamics,
    ForwardKinematics,
    InverseDynamics,
    GravityCompensation,
)

__all__ = [
    # PID Controllers
    "PIDController",
    "CascadePID",
    "AdaptivePID",
    "PIDTuner",
    # MPC
    "MPCController",
    "LinearMPC",
    "NonlinearMPC",
    "QuadraticCost",
    # Robot RL
    "RoboticsAgent",
    "JointSpaceAgent",
    "TaskSpaceAgent",
    "HybridAgent",
    "StateAction",
    # Motion Policies
    "MotionPolicy",
    "ImitationPolicy",
    "GMMPolicy",
    "DynamicMovementPrimitive",
    # Trajectory Planning
    "TrajectoryPlanner",
    "RRTStar",
    "CHOMP",
    "TimeOptimalProfile",
    "JacobianInverse",
    # Dynamics
    "RigidBodyDynamics",
    "SerialChainDynamics",
    "ForwardKinematics",
    "InverseDynamics",
    "GravityCompensation",
]
