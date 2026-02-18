"""
Extreme Event Prediction

Prediction and detection of extreme weather events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ExtremeEventPrediction:
    """Prediction output for extreme events.

    Args:
        event_type: Type of extreme event
        probability: Probability of event occurrence
        intensity: Expected intensity if event occurs
        location: Center location (lat, lon)
        bounding_box: Event bounding box ((lat_min, lon_min), (lat_max, lon_max))
        lead_time: Forecast lead time in hours
    """

    event_type: str
    probability: float
    intensity: float
    location: Tuple[float, float]
    bounding_box: Tuple[Tuple[float, float], Tuple[float, float]]
    lead_time: float


class ExtremeEventClassifier(nn.Module):
    """Classifier for extreme weather events.

    Args:
        input_channels: Number of input climate variables
        num_classes: Number of event classes
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        input_channels: int = 5,
        num_classes: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class HeatWaveDetector:
    """Heat wave detection and prediction.

    Args:
        threshold_temp: Temperature threshold in Kelvin
        duration_min: Minimum duration in days
        spatial_extent: Minimum spatial extent (grid points)
    """

    def __init__(
        self,
        threshold_temp: float = 315.0,
        duration_min: int = 3,
        spatial_extent: int = 10,
    ):
        self.threshold_temp = threshold_temp
        self.duration_min = duration_min
        self.spatial_extent = spatial_extent

    def detect(
        self,
        temperature: Tensor,
    ) -> List[Dict]:
        """Detect heat waves in temperature data.

        Args:
            temperature: Temperature data of shape (T, H, W)

        Returns:
            List of detected heat waves
        """
        above_threshold = temperature > self.threshold_temp

        heat_waves = []
        T, H, W = temperature.shape

        for t in range(T):
            for h in range(H):
                for w in range(W):
                    if above_threshold[t, h, w]:
                        duration = 0
                        extent = 0

                        for dt in range(self.duration_min):
                            if t + dt < T:
                                if above_threshold[t + dt, h, w]:
                                    duration += 1

                        if duration >= self.duration_min:
                            extent = (
                                above_threshold[t : t + self.duration_min].sum().item()
                            )

                            if extent >= self.spatial_extent:
                                heat_waves.append(
                                    {
                                        "start_time": t,
                                        "duration": duration,
                                        "intensity": temperature[t, h, w].item(),
                                        "location": (
                                            h / H * 180 - 90,
                                            w / W * 360 - 180,
                                        ),
                                    }
                                )

        return heat_waves

    def predict(
        self,
        historical_data: Tensor,
        forecast_data: Tensor,
    ) -> List[ExtremeEventPrediction]:
        """Predict heat waves from historical and forecast data.

        Args:
            historical_data: Historical temperature data
            forecast_data: Forecast temperature data

        Returns:
            List of heat wave predictions
        """
        predictions = []

        max_forecast = forecast_data.max(dim=0)[0]
        hot_spots = max_forecast > self.threshold_temp * 0.98

        if hot_spots.any():
            max_temp = max_forecast[hot_spots].max().item()
            probability = min(
                1.0,
                (max_temp - self.threshold_temp * 0.98) / (self.threshold_temp * 0.05),
            )

            predictions.append(
                ExtremeEventPrediction(
                    event_type="heat_wave",
                    probability=probability,
                    intensity=max_temp,
                    location=(0.0, 0.0),
                    bounding_box=((-90, -180), (90, 180)),
                    lead_time=forecast_data.shape[0] * 6,
                )
            )

        return predictions


class TropicalCycloneTracker:
    """Tropical cyclone detection and tracking.

    Args:
        wind_threshold: Wind speed threshold in m/s
        min_pressure: Minimum central pressure in hPa
    """

    def __init__(
        self,
        wind_threshold: float = 17.0,
        min_pressure: float = 980.0,
    ):
        self.wind_threshold = wind_threshold
        self.min_pressure = min_pressure

    def detect_cyclone(
        self,
        wind_speed: Tensor,
        pressure: Optional[Tensor] = None,
    ) -> List[Dict]:
        """Detect tropical cyclones from wind and pressure data.

        Args:
            wind_speed: Wind speed data of shape (T, H, W)
            pressure: Optional surface pressure data

        Returns:
            List of detected cyclones
        """
        low_pressure = pressure < self.min_pressure if pressure is not None else None

        high_wind = wind_speed > self.wind_threshold

        cyclones = []

        return cyclones

    def track(
        self,
        detections: List[Dict],
    ) -> List[Dict]:
        """Track cyclone paths over time.

        Args:
            detections: List of cyclone detections

        Returns:
            List of cyclone tracks
        """
        tracks = []

        return tracks


class ExtremeValueModel(nn.Module):
    """Extreme Value Theory model for tail estimation.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        distribution: Type of extreme value distribution ('gev', 'gpd')
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        distribution: str = "gev",
    ):
        super().__init__()

        self.distribution = distribution

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if distribution == "gev":
            self.output_layer = nn.Linear(hidden_dim, 3)
        elif distribution == "gpd":
            self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        params = self.output_layer(features)
        return params

    def sample(self, params: Tensor, num_samples: int = 100) -> Tensor:
        """Sample from the extreme value distribution.

        Args:
            params: Distribution parameters
            num_samples: Number of samples

        Returns:
            Samples from the distribution
        """
        if self.distribution == "gev":
            shape, loc, scale = params[:, 0], params[:, 1], params[:, 2]
            u = torch.rand(num_samples, *shape.shape, device=params.device)
            samples = shape * torch.log(-torch.log(u)) + loc + scale
        else:
            scale = params[:, 0]
            u = torch.rand(num_samples, *scale.shape, device=params.device)
            samples = -scale * torch.log(u)

        return samples


class SevereStormPredictor(nn.Module):
    """Predictor for severe storms.

    Args:
        input_channels: Number of input variables
        hidden_channels: Hidden channels
    """

    def __init__(
        self,
        input_channels: int = 10,
        hidden_channels: int = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
        )

        self.severity_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.location_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.encoder(x)

        severity = self.severity_head(features)
        location = self.location_head(features)

        return severity, location


class ProbabilityOfPrecipitation(nn.Module):
    """Probability of precipitation (PoP) prediction model.

    Args:
        input_channels: Number of input variables
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def compute_return_period(
    data: Tensor,
    threshold: float,
    distribution: str = "gev",
) -> float:
    """Compute return period for a threshold.

    Args:
        data: Time series data
        threshold: Threshold value
        distribution: Type of distribution

    Returns:
        Return period in time units
    """
    extremes = data[data > threshold]

    if len(extremes) == 0:
        return float("inf")

    n_years = len(data) / 365.0
    return_period = n_years / len(extremes)

    return return_period


def compute_exceedance_probability(
    data: Tensor,
    threshold: float,
    window_size: int = 30,
) -> Tensor:
    """Compute probability of threshold exceedance over rolling window.

    Args:
        data: Time series data
        threshold: Threshold value
        window_size: Rolling window size

    Returns:
        Exceedance probability
    """
    exceeds = (data > threshold).float()

    if len(data) < window_size:
        return exceeds.mean().unsqueeze(0)

    probs = exceeds.unfold(0, window_size, 1).mean(dim=1)

    return probs
