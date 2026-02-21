# Robotics Module

Comprehensive robotics and control tools for manipulation, navigation, and autonomous systems.

## Overview

This module provides a complete suite of robotics tools:

- **Control**: PID, Adaptive, Cascade controllers
- **MPC**: Linear and Nonlinear Model Predictive Control
- **Dynamics**: Rigid body, Forward/Inverse kinematics
- **Planning**: RRT*, CHOMP, Trajectory optimization
- **SLAM**: Visual, LiDAR, Visual-Inertial SLAM
- **RL**: Robotics-specific reinforcement learning
- **Motion Policies**: DMPs, Imitation learning, GMM policies

## Installation

```bash
# Core dependencies
pip install torch numpy scipy

# For advanced SLAM
pip install opencv-python
```

## Quick Start

### PID Controller

```python
import torch
from fishstick.robotics import PIDController

# Basic PID
pid = PIDController(
    kp=1.0,  # Proportional gain
    ki=0.1,   # Integral gain
    kd=0.05   # Derivative gain
)

# Setpoint tracking
setpoint = 100.0
current_value = 95.0

for t in range(100):
    error = setpoint - current_value
    control = pid(error)
    current_value += control  # Apply control
    
    # Simulate system response
    current_value += torch.randn(1).item() * 0.1
```

### Model Predictive Control (MPC)

```python
from fishstick.robotics import MPCController, LinearMPC

# Linear MPC for robot arm
mpc = LinearMPC(
    state_dim=6,     # Joint positions + velocities
    action_dim=3,   # Torques
    horizon=20,      # Prediction horizon
    dt=0.01         # Time step
)

# Define quadratic cost
mpc.set_cost(
    Q=torch.eye(6),    # State cost
    R=torch.eye(3) * 0.1  # Control cost
)

# Initial state
state = torch.randn(6)

# Compute optimal control
action = mpc.forward(state)
```

### Forward Kinematics

```python
from fishstick.robotics import ForwardKinematics, SerialChainDynamics

# Define robot (3-DOF planar arm)
dh_params = [
    {'theta': 0.3, 'd': 0.5, 'a': 0.3, 'alpha': 0},  # Joint 1
    {'theta': 0.5, 'd': 0.0, 'a': 0.4, 'alpha': 0},  # Joint 2
    {'theta': -0.2, 'd': 0.0, 'a': 0.3, 'alpha': 0}, # Joint 3
]

fk = ForwardKinematics(dh_params)

# Compute end-effector pose
joint_angles = torch.tensor([0.3, 0.5, -0.2])
pose = fk.forward(joint_angles)

print(f"End-effector position: {pose[:3]}")
print(f"End-effector orientation: {pose[3:]}")
```

### Trajectory Planning with RRT*

```python
from fishstick.robotics import RRTStar, TrajectoryPlanner

# Plan path in joint space
planner = RRTStar(
    dim=3,
    bounds=[-3.14, 3.14],  # Joint limits
    max_iterations=1000,
    goal_sample_prob=0.1
)

# Start and goal
start = torch.tensor([0.0, 0.0, 0.0])
goal = torch.tensor([1.0, 1.0, 1.0])

# Plan
path = planner.plan(start, goal)

print(f"Path found with {len(path)} waypoints")
```

### SLAM - Visual Odometry

```python
from fishstick.robotics import ORBSLAM, SE3Pose

# Initialize ORB-SLAM
slam = ORBSLAM(
    vocab_path="vocabulary.bin",
    settings_path="settings.yaml"
)

# Process frame
frame = cv2.imread("frame.png")  # Load image

pose = slam.process_frame(frame)

if pose is not None:
    print(f"Camera pose: {pose.translation}")
```

### Robot RL Agent

```python
from fishstick.robotics import JointSpaceAgent, StateAction

# Create RL agent for joint control
agent = JointSpaceAgent(
    state_dim=14,    # 7 joint pos + 7 joint vel
    action_dim=7,    # 7 torques
    hidden_dims=[256, 256]
)

# Reset environment
state = agent.reset()

# Collect experience
for step in range(100):
    action = agent.select_action(state)
    next_state, reward, done = env.step(action)
    
    agent.store_transition(state, action, reward, next_state, done)
    state = next_state
    
    if done:
        state = agent.reset()
```

### Dynamic Movement Primitives

```python
from fishstick.robotics import DynamicMovementPrimitive

# Learn trajectory
dmp = DynamicMovementPrimitive(n_bfs=50)

# Demonstration trajectory
demo_trajectory = torch.randn(100, 3)  # 100 timesteps, 3D

# Learn from demonstration
dmp.learn(demo_trajectory)

# Reproduce with different goal
goal = torch.tensor([1.0, 0.5, 0.8])
reproduced = dmp.reproduce(goal, n_steps=100)
```

## API Reference

### Control

| Class | Description |
|-------|-------------|
| `PIDController` | Standard PID controller |
| `CascadePID` | Multi-loop cascade control |
| `AdaptivePID` | Gain-scheduled PID |
| `PIDTuner` | Auto-tuning for PID |

### MPC

| Class | Description |
|-------|-------------|
| `MPCController` | Base MPC controller |
| `LinearMPC` | Linear dynamics MPC |
| `NonlinearMPC` | Nonlinear MPC |
| `QuadraticCost` | QP cost definition |

### Dynamics

| Class | Description |
|-------|-------------|
| `ForwardKinematics` | Compute end-effector pose |
| `InverseDynamics` | Compute required torques |
| `SerialChainDynamics` | Full robot dynamics |
| `GravityCompensation` | Gravity compensation |

### Planning

| Class | Description |
|-------|-------------|
| `RRTStar` | Optimal rapidly-exploring random tree |
| `CHOMP` | Covariant Hamiltonian optimization |
| `TrajectoryPlanner` | General trajectory planning |
| `TimeOptimalProfile` | Time-optimal velocity profiles |

### SLAM

| Class | Description |
|-------|-------------|
| `ORBSLAM` | ORB-SLAM2 implementation |
| `LiDARSLAM` | LiDAR-based SLAM |
| `VINSMono` | Visual-inertial odometry |
| `PoseGraphOptimization` | Graph optimization |
| `OccupancyGrid` | 2D grid mapping |

### RL

| Class | Description |
|-------|-------------|
| `RoboticsAgent` | Base RL agent |
| `JointSpaceAgent` | Joint space control |
| `TaskSpaceAgent` | Task space control |
| `HybridAgent` | Hybrid control |

## Examples

```python
# Full manipulation example
from fishstick.robotics import (
    ForwardKinematics,
    InverseDynamics,
    MPCController,
    TrajectoryPlanner
)
```

## References

- Siciliano et al., "Robotics: Modelling, Planning and Control"
- Khansari-Zadeh & Billard, "Learning Stable Non-Linear Dynamical Systems"
- Mur-Artal & Tardós, "ORB-SLAM2"

## License

MIT License - see project root for details.
