import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Callable


class BehaviorCloning(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256],
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.policy = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy(state)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(state)

    def update(self, states: torch.Tensor, actions: torch.Tensor) -> dict:
        self.optimizer.zero_grad()
        pred_actions = self.forward(states)
        loss = self.criterion(pred_actions, actions)
        loss.backward()
        self.optimizer.step()
        return {"bc_loss": loss.item()}


class BCInteractionDataset(Dataset):
    def __init__(
        self, states: torch.Tensor, actions: torch.Tensor, expert_actions: torch.Tensor
    ):
        self.states = states
        self.actions = actions
        self.expert_actions = expert_actions

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.expert_actions[idx]


class DAgger:
    def __init__(
        self,
        policy: BehaviorCloning,
        expert_policy: Callable,
        state_dim: int,
        action_dim: int,
        iterations: int = 100,
        dagger_ratio: float = 0.5,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        self.policy = policy
        self.expert_policy = expert_policy
        self.iterations = iterations
        self.dagger_ratio = dagger_ratio
        self.batch_size = batch_size

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_expert_actions = []

    def collect_demonstration(
        self,
        env,
        num_episodes: int = 10,
    ) -> Tuple[list, list, list]:
        states, actions, expert_actions = [], [], []

        for _ in range(num_episodes):
            state, done = env.reset(), False
            while not done:
                action = self.policy.get_action(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0)
                expert_action = self.expert_policy(state)

                states.append(state)
                actions.append(action.numpy() if hasattr(action, "numpy") else action)
                expert_actions.append(expert_action)

                state, reward, done, _ = env.step(actions[-1])

        return states, actions, expert_actions

    def train_step(self, states: torch.Tensor, expert_actions: torch.Tensor) -> dict:
        return self.policy.update(states, expert_actions)

    def run(
        self,
        env,
        num_initial_demos: int = 20,
        num_dagger_episodes: int = 5,
    ) -> dict:
        init_states, init_actions, init_expert = self.collect_demonstration(
            env, num_initial_demos
        )
        self.buffer_states.extend(init_states)
        self.buffer_actions.extend(init_actions)
        self.buffer_expert_actions.extend(init_expert)

        logs = {"iterations": [], "losses": []}

        for i in range(self.iterations):
            states_tensor = torch.tensor(self.buffer_states, dtype=torch.float32)
            expert_tensor = torch.tensor(
                self.buffer_expert_actions, dtype=torch.float32
            )

            indices = torch.randperm(len(states_tensor))[: self.batch_size]
            loss_dict = self.train_step(states_tensor[indices], expert_tensor[indices])

            dagger_states, _, dagger_expert = self.collect_demonstration(
                env, num_dagger_episodes
            )
            self.buffer_states.extend(dagger_states)
            self.buffer_expert_actions.extend(dagger_expert)

            logs["iterations"].append(i)
            logs["losses"].append(loss_dict["bc_loss"])

            if i % 10 == 0:
                print(f"DAgger iteration {i}: loss = {loss_dict['bc_loss']:.4f}")

        return logs


class VisualServoingController(nn.Module):
    def __init__(
        self,
        image_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 128],
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.image_dim = image_dim
        self.state_dim = state_dim

        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        combined_dim = hidden_dims[1] + hidden_dims[0]
        self.controller = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim),
            nn.Tanh(),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(
        self,
        image_features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        img_enc = self.image_encoder(image_features)
        state_enc = self.state_encoder(state)
        combined = torch.cat([img_enc, state_enc], dim=-1)
        return self.controller(combined)

    def compute_ibvs_error(
        self,
        current_features: torch.Tensor,
        desired_features: torch.Tensor,
    ) -> torch.Tensor:
        return desired_features - current_features

    def get_action(
        self,
        current_image: torch.Tensor,
        desired_image: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            current_features = self.image_encoder(current_image)
            desired_features = self.image_encoder(desired_image)
            error = self.compute_ibvs_error(current_features, desired_features)
            action = self.forward(error, state)
        return action

    def update(
        self,
        current_images: torch.Tensor,
        desired_images: torch.Tensor,
        states: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> dict:
        self.optimizer.zero_grad()

        current_features = self.image_encoder(current_images)
        desired_features = self.image_encoder(desired_images)
        errors = self.compute_ibvs_error(current_features, desired_features)

        pred_actions = self.forward(errors, states)
        loss = self.criterion(pred_actions, target_actions)

        loss.backward()
        self.optimizer.step()

        return {"vs_loss": loss.item()}


class ImageBasedServoingController(VisualServoingController):
    def __init__(
        self,
        image_channels: int,
        image_size: Tuple[int, int],
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [256, 256, 128],
    ):
        self.image_channels = image_channels
        self.image_size = image_size
        flat_image_dim = image_channels * image_size[0] * image_size[1]

        super().__init__(flat_image_dim, state_dim, action_dim, hidden_dims)

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dims[0]),
            nn.ReLU(),
        )

    def forward(
        self,
        current_image: torch.Tensor,
        desired_image: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        current_features = self.cnn_encoder(current_image)
        desired_features = self.cnn_encoder(desired_image)
        error = self.compute_ibvs_error(current_features, desired_features)

        state_enc = self.state_encoder(state)
        combined = torch.cat([error, state_enc], dim=-1)
        return self.controller(combined)
