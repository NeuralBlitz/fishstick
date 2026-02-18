import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import Any


class AnomalyDetector:
    def __init__(self, model: Any, device: str = "cpu"):
        self.model = model
        self.device = device

    def predict(self, x: torch.Tensor) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray) -> "AnomalyDetector":
        raise NotImplementedError


class IsolationForestDetector(AnomalyDetector):
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        max_samples: int = 256,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
        )
        super().__init__(self.model, device)

    def fit(self, x: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(x)
        return self

    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.model.predict(x)

    def score(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.model.score_samples(x)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.encoder_latent = nn.Linear(hidden_dim, latent_dim)

        self.decoder_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        h = h[-1]
        z = self.encoder_latent(h)
        z = self.decoder_latent(z)
        z = z.unsqueeze(0).repeat(x.size(0), 1)
        out, _ = self.decoder(z)
        out = self.output_layer(out)
        return out


class LSTMAutoencoderDetector(AnomalyDetector):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cpu",
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.model = LSTMAutoencoder(
            input_dim, hidden_dim, latent_dim, num_layers, dropout
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        super().__init__(self.model, device)

    def fit(
        self, x: np.ndarray, epochs: int = 50, batch_size: int = 32
    ) -> "LSTMAutoencoderDetector":
        self.model.train()
        x_tensor = torch.FloatTensor(x).to(self.device)
        dataset = torch.utils.data.TensorDataset(x_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch[0]
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return (errors.cpu().numpy() > errors.cpu().numpy().mean()).astype(int)

    def score(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return -errors.cpu().numpy()


class BeatGAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class BeatGANDetector(AnomalyDetector):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.2,
        device: str = "cpu",
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.model = BeatGAN(input_dim, hidden_dim, latent_dim, dropout).to(device)
        self.g_optimizer = torch.optim.Adam(
            self.model.generator.parameters(), lr=learning_rate
        )
        self.d_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=learning_rate
        )
        self.criterion = nn.BCELoss()
        super().__init__(self.model, device)

    def fit(
        self, x: np.ndarray, epochs: int = 100, batch_size: int = 32
    ) -> "BeatGANDetector":
        self.model.train()
        x_tensor = torch.FloatTensor(x).to(self.device)
        dataset = torch.utils.data.TensorDataset(x_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch[0]
                batch_size_actual = batch.size(0)

                real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)

                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_data = self.model(z)

                self.d_optimizer.zero_grad()
                real_loss = self.criterion(self.model.discriminate(batch), real_labels)
                fake_loss = self.criterion(
                    self.model.discriminate(fake_data.detach()), fake_labels
                )
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.d_optimizer.zero_grad()

                self.g_optimizer.zero_grad()
                z = torch.randn(batch_size_actual, self.latent_dim).to(self.device)
                fake_data = self.model(z)
                g_loss = self.criterion(self.model.discriminate(fake_data), real_labels)
                g_loss.backward()
                self.g_optimizer.step()
        return self

    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            z = torch.randn(x.size(0), self.latent_dim).to(self.device)
            generated = self.model(z)
            errors = torch.mean((x - generated) ** 2, dim=1)
        return (errors.cpu().numpy() > errors.cpu().numpy().mean()).astype(int)

    def score(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            z = torch.randn(x.size(0), self.latent_dim).to(self.device)
            generated = self.model(z)
            errors = torch.mean((x - generated) ** 2, dim=1)
        return -errors.cpu().numpy()


class OneClassSVMDetector(AnomalyDetector):
    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float = "scale",
        nu: float = 0.1,
        device: str = "cpu",
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        super().__init__(self.model, device)

    def fit(self, x: np.ndarray) -> "OneClassSVMDetector":
        self.model.fit(x)
        return self

    def predict(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.model.predict(x)

    def score(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return self.model.score_samples(x)
