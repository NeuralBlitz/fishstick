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
from fishstick.robotics.slam import (
    # Core Types
    SE3Pose,
    Landmark,
    KeyFrame,
    PointCloud,
    IMUData,
    SLAMState,
    SensorType,
    SLAMMode,
    # Visual SLAM
    VisualSLAM,
    ORBSLAM,
    LSDSLAM,
    DirectSparseOdometry,
    SemiDirectVisualOdometry,
    LoopClosureSLAM,
    # LiDAR SLAM
    LiDARSLAM,
    LOAM,
    LeGOLiDARSLAM,
    Cartographer,
    HectorSLAM,
    # Visual-Inertial SLAM
    VisualInertialSLAM,
    VINSMono,
    VINSFusion,
    OKVIS,
    MSCKF,
    # Loop Closure
    LoopClosureDetector,
    DBoW2,
    FABMAP,
    ScanContext,
    NetVLAD,
    # Mapping
    OccupancyGrid,
    OctoMap,
    VoxelGrid,
    PointCloudMap,
    # State Estimation
    StateEstimator,
    EKF_SLAM,
    UKF_SLAM,
    ParticleFilterSLAM,
    # Optimization
    PoseGraphOptimization,
    BundleAdjustment,
    FactorGraph,
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
    # SLAM Core Types
    "SE3Pose",
    "Landmark",
    "KeyFrame",
    "PointCloud",
    "IMUData",
    "SLAMState",
    "SensorType",
    "SLAMMode",
    # Visual SLAM
    "VisualSLAM",
    "ORBSLAM",
    "LSDSLAM",
    "DirectSparseOdometry",
    "SemiDirectVisualOdometry",
    "LoopClosureSLAM",
    # LiDAR SLAM
    "LiDARSLAM",
    "LOAM",
    "LeGOLiDARSLAM",
    "Cartographer",
    "HectorSLAM",
    # Visual-Inertial SLAM
    "VisualInertialSLAM",
    "VINSMono",
    "VINSFusion",
    "OKVIS",
    "MSCKF",
    # Loop Closure
    "LoopClosureDetector",
    "DBoW2",
    "FABMAP",
    "ScanContext",
    "NetVLAD",
    # Mapping
    "OccupancyGrid",
    "OctoMap",
    "VoxelGrid",
    "PointCloudMap",
    # State Estimation
    "StateEstimator",
    "EKF_SLAM",
    "UKF_SLAM",
    "ParticleFilterSLAM",
    # Optimization
    "PoseGraphOptimization",
    "BundleAdjustment",
    "FactorGraph",
]
