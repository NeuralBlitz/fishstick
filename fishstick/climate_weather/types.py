"""
Climate & Weather Data Types and Utilities

Core types, enums, and utilities for climate and weather modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


class CoordinateSystem(Enum):
    """Coordinate system for climate data."""

    LAT_LON = "lat_lon"
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CUBED_SPHERE = "cubed_sphere"


class VerticalLevelType(Enum):
    """Vertical level types in atmospheric data."""

    SURFACE = "surface"
    PRESSURE = "pressure"
    MODEL = "model"
    HEIGHT = "height"


class VariableType(Enum):
    """Climate/weather variable types."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"
    WIND_U = "wind_u"
    WIND_V = "wind_v"
    WIND_SPEED = "wind_speed"
    PRECIPITATION = "precipitation"
    GEOPOTENTIAL = "geopotential"
    CLOUD = "cloud"
    RADIATION = "radiation"


@dataclass
class ClimateDataSpec:
    """Specification for climate/weather data."""

    variables: List[VariableType]
    spatial_resolution: float
    temporal_resolution: float
    vertical_levels: int
    coordinate_system: CoordinateSystem
    grid_shape: Tuple[int, int, int]

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @property
    def num_grid_points(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]


@dataclass
class WeatherState:
    """Represents a weather state at a given time.

    Args:
        data: Weather data array of shape (B, C, H, W) or (C, H, W)
        timestamp: Timestamp of the weather state
        lead_time: Forecast lead time in hours (0 for analysis)
        variables: List of variable names in the data channels
    """

    data: Union[Tensor, np.ndarray]
    timestamp: datetime
    lead_time: float = 0.0
    variables: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def device(self) -> torch.device:
        return self.data.device

    def to(self, device: Union[str, torch.device]) -> "WeatherState":
        return WeatherState(
            data=self.data.to(device),
            timestamp=self.timestamp,
            lead_time=self.lead_time,
            variables=self.variables.copy(),
        )


@dataclass
class ClimateIndex:
    """Climate index representation.

    Args:
        name: Name of the climate index (e.g., 'ENSO', 'NAO', 'AO')
        values: Index values over time
        timestamps: Corresponding timestamps
        description: Description of the index
    """

    name: str
    values: Union[Tensor, np.ndarray]
    timestamps: List[datetime]
    description: str = ""

    def __post_init__(self):
        if isinstance(self.values, np.ndarray):
            self.values = torch.from_numpy(self.values)

    def to_numpy(self) -> np.ndarray:
        if isinstance(self.values, Tensor):
            return self.values.cpu().numpy()
        return self.values


@dataclass
class ForecastOutput:
    """Output from a weather forecast model.

    Args:
        mean: Mean prediction
        std: Standard deviation (uncertainty)
        quantiles: Optional quantile predictions
        timestamps: Forecast timestamps
        lead_times: Lead times in hours
    """

    mean: Union[Tensor, np.ndarray]
    std: Optional[Union[Tensor, np.ndarray]] = None
    quantiles: Optional[Dict[float, Union[Tensor, np.ndarray]]] = None
    timestamps: Optional[List[datetime]] = None
    lead_times: Optional[Union[Tensor, np.ndarray]] = None

    @property
    def has_uncertainty(self) -> bool:
        return self.std is not None or self.quantiles is not None


@dataclass
class ExtremeEvent:
    """Represents an extreme weather event.

    Args:
        event_type: Type of extreme event
        start_time: Event start time
        end_time: Event end time
        intensity: Event intensity metric
        location: Geographic location (lat, lon)
        probability: Probability of occurrence
    """

    event_type: str
    start_time: datetime
    end_time: datetime
    intensity: float
    location: Tuple[float, float]
    probability: float = 1.0

    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 3600


@dataclass
class AssimilationObservation:
    """Observation for data assimilation.

    Args:
        location: (lat, lon, level) coordinates
        value: Observed value
        error: Observation error standard deviation
        timestamp: Observation time
        variable: Variable type
    """

    location: Tuple[float, float, float]
    value: float
    error: float
    timestamp: datetime
    variable: VariableType


@dataclass
class GridSpec:
    """Specification for a regular latitude-longitude grid.

    Args:
        lats: Latitude values
        lons: Longitude values
        levels: Vertical levels (optional)
        time: Time coordinates (optional)
    """

    lats: Union[Tensor, np.ndarray]
    lons: Union[Tensor, np.ndarray]
    levels: Optional[Union[Tensor, np.ndarray]] = None
    time: Optional[List[datetime]] = None

    def __post_init__(self):
        if isinstance(self.lats, np.ndarray):
            self.lats = torch.from_numpy(self.lats)
        if isinstance(self.lons, np.ndarray):
            self.lons = torch.from_numpy(self.lons)
        if self.levels is not None and isinstance(self.levels, np.ndarray):
            self.levels = torch.from_numpy(self.levels)

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.lats), len(self.lons))

    @property
    def resolution(self) -> Tuple[float, float]:
        lat_res = (self.lats[1] - self.lats[0]).item() if len(self.lats) > 1 else 0
        lon_res = (self.lons[1] - self.lons[0]).item() if len(self.lons) > 1 else 0
        return (lat_res, lon_res)


def create_lat_lon_grid(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution: float,
) -> GridSpec:
    """Create a regular latitude-longitude grid.

    Args:
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        resolution: Grid resolution in degrees

    Returns:
        GridSpec for the specified grid
    """
    lats = torch.arange(lat_min, lat_max + resolution, resolution)
    lons = torch.arange(lon_min, lon_max + resolution, resolution)
    return GridSpec(lats=lats, lons=lons)


def normalize_coordinates(
    lats: Tensor,
    lons: Tensor,
    lat_range: Tuple[float, float] = (-90, 90),
    lon_range: Tuple[float, float] = (-180, 180),
) -> Tuple[Tensor, Tensor]:
    """Normalize latitude and longitude to [0, 1] range.

    Args:
        lats: Latitude tensor
        lons: Longitude tensor
        lat_range: Target latitude range
        lon_range: Target longitude range

    Returns:
        Tuple of (normalized_lats, normalized_lons)
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    norm_lats = (lats - lat_min) / (lat_max - lat_min)
    norm_lons = (lons - lon_min) / (lon_max - lon_min)

    return norm_lats, norm_lons


def geo_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    earth_radius: float = 6371.0,
) -> float:
    """Calculate great circle distance between two points.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2
        earth_radius: Earth radius in km

    Returns:
        Distance in km
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius * c
