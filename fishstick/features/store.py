"""
Feature Store Module for Fishstick
==================================
A comprehensive feature store implementation supporting online (Redis) and offline (Hive/S3) storage,
with feature serving, registry, transformations, monitoring, materialization, and integrations.

Components:
- FeatureStorage: OnlineStore, OfflineStore, FeatureGroup, FeatureEntity
- FeatureServing: FeatureServer, get_features, FeatureVector
- FeatureRegistry: FeatureRegistry, FeatureMetadata
- Transformation: OnDemandTransform, BatchTransform, StreamingTransform
- Monitoring: FeatureDrift, FeatureStats, FeatureQuality
- Materialization: MaterializationEngine with backfill and incremental support
- Integration: Feast, Tecton, SageMaker
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")
FeatureValue = Union[int, float, str, bool, List, np.ndarray, Tensor]


# ============================================================================
# Feature Storage - Core Classes
# ============================================================================


class StorageType(Enum):
    """Types of feature storage."""

    ONLINE = auto()
    OFFLINE = auto()
    HYBRID = auto()


@dataclass
class FeatureEntity:
    """
    Represents an entity for which features are stored.

    An entity is a real-world object (user, item, transaction) that has
    features associated with it.

    Attributes:
        name: Unique entity name
        join_keys: Primary key column(s) for joining features
        description: Human-readable description
        metadata: Additional entity metadata
    """

    name: str
    join_keys: List[str]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureEntity):
            return NotImplemented
        return self.name == other.name


@dataclass
class FeatureGroup:
    """
    A logical grouping of related features.

    Feature groups organize features by domain, source, or use case.
    They can contain features from multiple entities.

    Attributes:
        name: Unique feature group name
        entities: Entities this group contains features for
        features: List of feature names in this group
        description: Human-readable description
        tags: Classification tags
        owner: Team/individual responsible
    """

    name: str
    entities: List[FeatureEntity]
    features: List[str]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_feature(self, feature_name: str) -> None:
        """Add a feature to this group."""
        if feature_name not in self.features:
            self.features.append(feature_name)

    def remove_feature(self, feature_name: str) -> None:
        """Remove a feature from this group."""
        if feature_name in self.features:
            self.features.remove(feature_name)


@dataclass
class FeatureMetadata:
    """
    Metadata describing a feature.

    Attributes:
        name: Feature name
        entity: Associated entity
        dtype: Data type (int, float, string, etc.)
        description: Human-readable description
        owner: Feature owner
        tags: Classification tags
        statistics: Computed feature statistics
        lineage: Data lineage information
        version: Feature version
    """

    name: str
    entity: FeatureEntity
    dtype: str
    description: str = ""
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    lineage: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        result = asdict(self)
        result["entity"] = self.entity.name
        return result


class OnlineStore(ABC):
    """
    Abstract base class for online feature stores.

    Online stores provide low-latency access to feature values
    for real-time inference. Typically backed by Redis, DynamoDB, etc.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the online store."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the online store."""
        pass

    @abstractmethod
    async def get(
        self, entity_key: str, feature_names: Optional[List[str]] = None
    ) -> Dict[str, FeatureValue]:
        """
        Retrieve features for an entity key.

        Args:
            entity_key: Unique entity identifier
            feature_names: Specific features to retrieve (None = all)

        Returns:
            Dictionary mapping feature names to values
        """
        pass

    @abstractmethod
    async def set(
        self,
        entity_key: str,
        features: Dict[str, FeatureValue],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store features for an entity key.

        Args:
            entity_key: Unique entity identifier
            features: Dictionary of feature names and values
            ttl: Time-to-live in seconds (None = no expiration)
        """
        pass

    @abstractmethod
    async def delete(self, entity_key: str) -> None:
        """Delete all features for an entity key."""
        pass

    @abstractmethod
    async def get_batch(
        self, entity_keys: List[str], feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Retrieve features for multiple entity keys.

        Args:
            entity_keys: List of entity identifiers
            feature_names: Specific features to retrieve

        Returns:
            Dictionary mapping entity keys to feature dictionaries
        """
        pass


class RedisOnlineStore(OnlineStore):
    """
    Redis-backed online feature store implementation.

    Provides fast, in-memory storage with optional persistence.
    Supports Redis Cluster for horizontal scaling.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        cluster_mode: bool = False,
        cluster_nodes: Optional[List[Dict[str, Any]]] = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.cluster_mode = cluster_mode
        self.cluster_nodes = cluster_nodes or []
        self._client: Any = None
        self._pipeline: Any = None

    async def connect(self) -> None:
        """Connect to Redis server or cluster."""
        try:
            import redis.asyncio as redis

            if self.cluster_mode:
                startup_nodes = (
                    [
                        {
                            "host": node.get("host", self.host),
                            "port": node.get("port", self.port),
                        }
                        for node in self.cluster_nodes
                    ]
                    if self.cluster_nodes
                    else [{"host": self.host, "port": self.port}]
                )

                self._client = redis.RedisCluster(
                    startup_nodes=startup_nodes,
                    password=self.password,
                    decode_responses=True,
                )
            else:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                )

            # Test connection
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except ImportError:
            logger.error("redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    def _make_key(self, entity_key: str) -> str:
        """Create Redis key for entity."""
        return f"fs:entity:{entity_key}"

    def _serialize_value(self, value: FeatureValue) -> str:
        """Serialize feature value for storage."""
        if isinstance(value, np.ndarray):
            return json.dumps({"__type__": "ndarray", "data": value.tolist()})
        elif isinstance(value, Tensor):
            return json.dumps(
                {"__type__": "tensor", "data": value.cpu().numpy().tolist()}
            )
        elif isinstance(value, (list, dict)):
            return json.dumps(value)
        else:
            return json.dumps(value)

    def _deserialize_value(self, value_str: str) -> FeatureValue:
        """Deserialize feature value from storage."""
        try:
            data = json.loads(value_str)
            if isinstance(data, dict) and "__type__" in data:
                if data["__type__"] == "ndarray":
                    return np.array(data["data"])
                elif data["__type__"] == "tensor":
                    return torch.tensor(data["data"])
            return data
        except json.JSONDecodeError:
            return value_str

    async def get(
        self, entity_key: str, feature_names: Optional[List[str]] = None
    ) -> Dict[str, FeatureValue]:
        """Retrieve features for an entity key from Redis."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        key = self._make_key(entity_key)

        if feature_names:
            # Use HMGET for specific fields
            values = await self._client.hmget(key, feature_names)
            result = {}
            for fname, value in zip(feature_names, values):
                if value is not None:
                    result[fname] = self._deserialize_value(value)
            return result
        else:
            # Use HGETALL for all fields
            data = await self._client.hgetall(key)
            return {k: self._deserialize_value(v) for k, v in data.items()}

    async def set(
        self,
        entity_key: str,
        features: Dict[str, FeatureValue],
        ttl: Optional[int] = None,
    ) -> None:
        """Store features for an entity key in Redis."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        key = self._make_key(entity_key)
        serialized = {k: self._serialize_value(v) for k, v in features.items()}

        await self._client.hset(key, mapping=serialized)

        if ttl:
            await self._client.expire(key, ttl)

    async def delete(self, entity_key: str) -> None:
        """Delete all features for an entity key."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        key = self._make_key(entity_key)
        await self._client.delete(key)

    async def get_batch(
        self, entity_keys: List[str], feature_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """Retrieve features for multiple entity keys."""
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        results = {}
        # Use pipeline for batch operations
        async with self._client.pipeline() as pipe:
            for entity_key in entity_keys:
                key = self._make_key(entity_key)
                if feature_names:
                    pipe.hmget(key, feature_names)
                else:
                    pipe.hgetall(key)

            responses = await pipe.execute()

            for entity_key, response in zip(entity_keys, responses):
                if feature_names:
                    result = {}
                    for fname, value in zip(feature_names, response):
                        if value is not None:
                            result[fname] = self._deserialize_value(value)
                    results[entity_key] = result
                else:
                    results[entity_key] = {
                        k: self._deserialize_value(v) for k, v in response.items()
                    }

        return results


class OfflineStore(ABC):
    """
    Abstract base class for offline feature stores.

    Offline stores provide historical feature data for training,
    batch inference, and analytics. Typically backed by Hive, S3, etc.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the offline store."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the offline store."""
        pass

    @abstractmethod
    def get_historical_features(
        self,
        entity_keys: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
        timestamp_column: str = "event_timestamp",
    ) -> pd.DataFrame:
        """
        Retrieve historical features for entities in a time range.

        Args:
            entity_keys: List of entity identifiers
            feature_names: Features to retrieve
            start_date: Start of time range
            end_date: End of time range
            timestamp_column: Column name for timestamps

        Returns:
            DataFrame with historical feature data
        """
        pass

    @abstractmethod
    def write_features(
        self,
        df: pd.DataFrame,
        feature_group: str,
        partition_columns: Optional[List[str]] = None,
    ) -> None:
        """
        Write feature data to the offline store.

        Args:
            df: DataFrame with feature data
            feature_group: Target feature group
            partition_columns: Columns to partition by
        """
        pass

    @abstractmethod
    def get_feature_statistics(
        self, feature_names: List[str], start_date: datetime, end_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for features over a time range.

        Args:
            feature_names: Features to analyze
            start_date: Start of time range
            end_date: End of time range

        Returns:
            Dictionary mapping feature names to statistics
        """
        pass


class HiveS3OfflineStore(OfflineStore):
    """
    Hive/S3-backed offline feature store implementation.

    Stores historical feature data in S3 with Hive metastore for
    schema management and querying via SQL.
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str = "features",
        hive_host: str = "localhost",
        hive_port: int = 9083,
        database: str = "feature_store",
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.hive_host = hive_host
        self.hive_port = hive_port
        self.database = database
        self._hive_conn: Any = None
        self._spark: Any = None

    def connect(self) -> None:
        """Connect to Hive and initialize S3."""
        try:
            from pyhive import hive
            from pyspark.sql import SparkSession

            # Connect to Hive
            self._hive_conn = hive.connect(
                host=self.hive_host, port=self.hive_port, database=self.database
            )

            # Initialize Spark for S3 operations
            self._spark = (
                SparkSession.builder.appName("FeatureStore")
                .config(
                    "spark.sql.warehouse.dir",
                    f"s3a://{self.s3_bucket}/{self.s3_prefix}",
                )
                .config(
                    "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
                )
                .getOrCreate()
            )

            logger.info(f"Connected to Hive at {self.hive_host}:{self.hive_port}")
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Hive: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Hive."""
        if self._hive_conn:
            self._hive_conn.close()
            self._hive_conn = None
        if self._spark:
            self._spark.stop()
            self._spark = None
        logger.info("Disconnected from Hive")

    def _get_s3_path(self, feature_group: str) -> str:
        """Get S3 path for a feature group."""
        return f"s3a://{self.s3_bucket}/{self.s3_prefix}/{feature_group}"

    def get_historical_features(
        self,
        entity_keys: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
        timestamp_column: str = "event_timestamp",
    ) -> pd.DataFrame:
        """Retrieve historical features from Hive/S3."""
        if not self._spark:
            raise RuntimeError("Not connected to offline store")

        # Build query
        features_str = ", ".join(feature_names)
        entity_keys_str = ", ".join([f"'{k}'" for k in entity_keys])

        query = f"""
        SELECT entity_key, {timestamp_column}, {features_str}
        FROM features
        WHERE entity_key IN ({entity_keys_str})
        AND {timestamp_column} >= '{start_date.isoformat()}'
        AND {timestamp_column} <= '{end_date.isoformat()}'
        ORDER BY {timestamp_column}
        """

        try:
            df = self._spark.sql(query).toPandas()
            return df
        except Exception as e:
            logger.error(f"Failed to query historical features: {e}")
            raise

    def write_features(
        self,
        df: pd.DataFrame,
        feature_group: str,
        partition_columns: Optional[List[str]] = None,
    ) -> None:
        """Write features to S3 via Spark."""
        if not self._spark:
            raise RuntimeError("Not connected to offline store")

        # Convert pandas to Spark DataFrame
        spark_df = self._spark.createDataFrame(df)

        # Write with partitioning
        path = self._get_s3_path(feature_group)
        writer = spark_df.write.mode("append").parquet(path)

        if partition_columns:
            writer = writer.partitionBy(*partition_columns)

        writer.save()
        logger.info(f"Written features to {path}")

    def get_feature_statistics(
        self, feature_names: List[str], start_date: datetime, end_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Compute feature statistics using Spark."""
        if not self._spark:
            raise RuntimeError("Not connected to offline store")

        stats = {}
        for feature in feature_names:
            query = f"""
            SELECT 
                COUNT(*) as count,
                AVG({feature}) as mean,
                STDDEV({feature}) as std,
                MIN({feature}) as min,
                MAX({feature}) as max,
                PERCENTILE_APPROX({feature}, 0.5) as median
            FROM features
            WHERE event_timestamp >= '{start_date.isoformat()}'
            AND event_timestamp <= '{end_date.isoformat()}'
            """

            result = self._spark.sql(query).collect()[0]
            stats[feature] = {
                "count": result["count"],
                "mean": result["mean"],
                "std": result["std"],
                "min": result["min"],
                "max": result["max"],
                "median": result["median"],
            }

        return stats


# ============================================================================
# Feature Store - Main Store Class
# ============================================================================


class FeatureStore:
    """
    Main feature store coordinating online and offline storage.

    The FeatureStore is the central interface for storing, retrieving,
    and managing ML features across online (low-latency) and offline
    (historical) storage backends.

    Attributes:
        name: Store name
        online_store: Online storage backend
        offline_store: Offline storage backend
        registry: Feature registry for metadata
    """

    def __init__(
        self,
        name: str,
        online_store: Optional[OnlineStore] = None,
        offline_store: Optional[OfflineStore] = None,
        registry: Optional[FeatureRegistry] = None,
    ):
        self.name = name
        self.online_store = online_store
        self.offline_store = offline_store
        self.registry = registry or FeatureRegistry()
        self._entities: Dict[str, FeatureEntity] = {}
        self._feature_groups: Dict[str, FeatureGroup] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connections to all storage backends."""
        if self._initialized:
            return

        if self.online_store:
            await self.online_store.connect()

        if self.offline_store:
            self.offline_store.connect()

        self._initialized = True
        logger.info(f"Feature store '{self.name}' initialized")

    async def close(self) -> None:
        """Close all storage connections."""
        if self.online_store:
            await self.online_store.disconnect()

        if self.offline_store:
            self.offline_store.disconnect()

        self._initialized = False
        logger.info(f"Feature store '{self.name}' closed")

    def register_entity(self, entity: FeatureEntity) -> None:
        """Register an entity with the store."""
        self._entities[entity.name] = entity
        self.registry.register_entity(entity)
        logger.info(f"Registered entity: {entity.name}")

    def register_feature_group(self, group: FeatureGroup) -> None:
        """Register a feature group with the store."""
        self._feature_groups[group.name] = group
        self.registry.register_feature_group(group)
        logger.info(f"Registered feature group: {group.name}")

    def register_feature(self, metadata: FeatureMetadata) -> None:
        """Register a feature with the store."""
        self.registry.register_feature(metadata)
        logger.info(f"Registered feature: {metadata.name}")

    async def get_online_features(
        self, entity_keys: List[str], feature_names: List[str]
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Retrieve features from online store.

        Args:
            entity_keys: Entity identifiers
            feature_names: Features to retrieve

        Returns:
            Dictionary mapping entity keys to feature values
        """
        if not self.online_store:
            raise RuntimeError("Online store not configured")

        if not self._initialized:
            raise RuntimeError("Feature store not initialized")

        return await self.online_store.get_batch(entity_keys, feature_names)

    def get_historical_features(
        self,
        entity_keys: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve historical features from offline store.

        Args:
            entity_keys: Entity identifiers
            feature_names: Features to retrieve
            start_date: Start of time range
            end_date: End of time range

        Returns:
            DataFrame with historical feature data
        """
        if not self.offline_store:
            raise RuntimeError("Offline store not configured")

        if not self._initialized:
            raise RuntimeError("Feature store not initialized")

        return self.offline_store.get_historical_features(
            entity_keys, feature_names, start_date, end_date
        )

    async def materialize(
        self,
        feature_group: str,
        entity_keys: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> None:
        """
        Materialize features from offline to online store.

        Args:
            feature_group: Feature group to materialize
            entity_keys: Specific entities (None = all)
            incremental: Only materialize new/changed data
        """
        if not self.online_store or not self.offline_store:
            raise RuntimeError("Both online and offline stores required")

        logger.info(f"Materializing feature group: {feature_group}")

        # Get latest features from offline store
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1) if incremental else datetime.min

        group = self._feature_groups.get(feature_group)
        if not group:
            raise ValueError(f"Feature group not found: {feature_group}")

        # Query all entities if not specified
        if not entity_keys:
            # This would typically query a metadata store
            entity_keys = []

        df = self.offline_store.get_historical_features(
            entity_keys=entity_keys,
            feature_names=group.features,
            start_date=start_date,
            end_date=end_date,
        )

        # Write to online store
        for _, row in df.iterrows():
            entity_key = row["entity_key"]
            features = {col: row[col] for col in group.features if col in row}
            await self.online_store.set(entity_key, features)

        logger.info(f"Materialized {len(df)} records to online store")


# ============================================================================
# Feature Serving
# ============================================================================


@dataclass
class FeatureVector:
    """
    A vector of feature values for an entity.

    Attributes:
        entity_key: Entity identifier
        features: Dictionary of feature names and values
        timestamp: When the features were retrieved
        metadata: Additional context
    """

    entity_key: str
    features: Dict[str, FeatureValue]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_array(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Convert features to numpy array."""
        if feature_order is None:
            feature_order = sorted(self.features.keys())

        values = []
        for name in feature_order:
            val = self.features.get(name)
            if isinstance(val, (np.ndarray, Tensor)):
                values.extend(val.flatten().tolist())
            else:
                values.append(val)

        return np.array(values)

    def to_tensor(self, feature_order: Optional[List[str]] = None) -> Tensor:
        """Convert features to PyTorch tensor."""
        return torch.tensor(self.to_array(feature_order))


class FeatureServer:
    """
    Server for serving features to ML models.

    Provides a unified interface for retrieving features
    from online and offline stores with caching and transformation support.

    Attributes:
        store: Feature store instance
        cache: Optional cache for frequently accessed features
        transformations: Registered feature transformations
    """

    def __init__(
        self, store: FeatureStore, cache: Optional[Any] = None, max_workers: int = 10
    ):
        self.store = store
        self.cache = cache
        self.max_workers = max_workers
        self._transformations: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    def register_transformation(
        self,
        name: str,
        transform: Callable[[Dict[str, FeatureValue]], Dict[str, FeatureValue]],
    ) -> None:
        """Register a feature transformation."""
        self._transformations[name] = transform
        logger.info(f"Registered transformation: {name}")

    async def get_online_features(
        self,
        entity_keys: List[str],
        feature_names: List[str],
        apply_transforms: Optional[List[str]] = None,
    ) -> List[FeatureVector]:
        """
        Get online features for entities.

        Args:
            entity_keys: Entity identifiers
            feature_names: Features to retrieve
            apply_transforms: List of transformations to apply

        Returns:
            List of feature vectors
        """
        # Check cache first
        results: Dict[str, Dict[str, FeatureValue]] = {}
        missing_keys = []

        if self.cache:
            for key in entity_keys:
                cached = self.cache.get(f"features:{key}")
                if cached:
                    results[key] = cached
                else:
                    missing_keys.append(key)
        else:
            missing_keys = entity_keys

        # Fetch missing features from store
        if missing_keys:
            fetched = await self.store.get_online_features(missing_keys, feature_names)

            # Update cache and results
            for key, features in fetched.items():
                if self.cache:
                    self.cache.set(f"features:{key}", features, ttl=300)
                results[key] = features

        # Apply transformations
        if apply_transforms:
            for key, features in results.items():
                for transform_name in apply_transforms:
                    if transform_name in self._transformations:
                        features = self._transformations[transform_name](features)
                results[key] = features

        # Convert to FeatureVector
        vectors = [
            FeatureVector(entity_key=key, features=features)
            for key, features in results.items()
        ]

        return vectors

    def get_historical_features(
        self,
        entity_keys: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
        as_of: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get historical features for training.

        Args:
            entity_keys: Entity identifiers
            feature_names: Features to retrieve
            start_date: Start of time range
            end_date: End of time range
            as_of: Point-in-time for feature values

        Returns:
            DataFrame with historical features
        """
        df = self.store.get_historical_features(
            entity_keys, feature_names, start_date, end_date
        )

        if as_of is not None:
            # Filter to features as of specific timestamp
            df = df[df["event_timestamp"] <= as_of]
            # Get most recent value per entity
            df = df.sort_values("event_timestamp").groupby("entity_key").last()
            df = df.reset_index()

        return df

    async def get_features(
        self, entity_keys: List[str], feature_refs: List[str], source: str = "online"
    ) -> List[FeatureVector]:
        """
        Get features from specified source.

        Args:
            entity_keys: Entity identifiers
            feature_refs: Feature references (e.g., "group:feature_name")
            source: "online" or "offline"

        Returns:
            List of feature vectors
        """
        # Parse feature references
        feature_names = []
        for ref in feature_refs:
            if ":" in ref:
                _, feature_name = ref.split(":", 1)
                feature_names.append(feature_name)
            else:
                feature_names.append(ref)

        if source == "online":
            return await self.get_online_features(entity_keys, feature_names)
        else:
            # Offline source - return most recent features
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)
            df = self.get_historical_features(
                entity_keys, feature_names, start_date, end_date
            )

            vectors = []
            for entity_key in entity_keys:
                entity_df = df[df["entity_key"] == entity_key]
                if not entity_df.empty:
                    features = entity_df.iloc[-1][feature_names].to_dict()
                    vectors.append(
                        FeatureVector(entity_key=entity_key, features=features)
                    )

            return vectors


def get_features(
    store: FeatureStore,
    entity_keys: List[str],
    feature_refs: List[str],
    source: str = "online",
) -> List[FeatureVector]:
    """
    Synchronous wrapper for getting features.

    Args:
        store: Feature store instance
        entity_keys: Entity identifiers
        feature_refs: Feature references
        source: "online" or "offline"

    Returns:
        List of feature vectors
    """
    server = FeatureServer(store)

    if source == "online":
        return asyncio.run(server.get_online_features(entity_keys, feature_refs))
    else:
        return server.get_historical_features(
            entity_keys,
            feature_refs,
            datetime.utcnow() - timedelta(days=365),
            datetime.utcnow(),
        )


def get_online_features(
    store: FeatureStore, entity_keys: List[str], feature_names: List[str]
) -> Dict[str, Dict[str, FeatureValue]]:
    """
    Synchronous wrapper for getting online features.

    Args:
        store: Feature store instance
        entity_keys: Entity identifiers
        feature_names: Features to retrieve

    Returns:
        Dictionary mapping entity keys to feature values
    """
    return asyncio.run(store.get_online_features(entity_keys, feature_names))


def get_historical_features(
    store: FeatureStore,
    entity_keys: List[str],
    feature_names: List[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Synchronous wrapper for getting historical features.

    Args:
        store: Feature store instance
        entity_keys: Entity identifiers
        feature_names: Features to retrieve
        start_date: Start of time range
        end_date: End of time range

    Returns:
        DataFrame with historical feature data
    """
    return store.get_historical_features(
        entity_keys, feature_names, start_date, end_date
    )


# ============================================================================
# Feature Registry
# ============================================================================


class FeatureRegistry:
    """
    Registry for feature metadata and schemas.

    Maintains a catalog of all features, entities, and feature groups
    with their metadata, lineage, and versioning.

    Attributes:
        backend: Storage backend for registry (file, database, etc.)
        features: Dictionary of registered features
        entities: Dictionary of registered entities
        feature_groups: Dictionary of registered feature groups
    """

    def __init__(self, backend: Optional[Any] = None):
        self.backend = backend
        self._features: Dict[str, FeatureMetadata] = {}
        self._entities: Dict[str, FeatureEntity] = {}
        self._feature_groups: Dict[str, FeatureGroup] = {}
        self._lock = asyncio.Lock()

    def register_entity(self, entity: FeatureEntity) -> None:
        """Register an entity in the registry."""
        self._entities[entity.name] = entity
        self._persist()

    def register_feature_group(self, group: FeatureGroup) -> None:
        """Register a feature group in the registry."""
        self._feature_groups[group.name] = group
        self._persist()

    def register_feature(self, metadata: FeatureMetadata) -> None:
        """Register a feature in the registry."""
        self._features[metadata.name] = metadata
        self._persist()

    def get_feature(self, name: str) -> Optional[FeatureMetadata]:
        """Get feature metadata by name."""
        return self._features.get(name)

    def get_entity(self, name: str) -> Optional[FeatureEntity]:
        """Get entity by name."""
        return self._entities.get(name)

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Get feature group by name."""
        return self._feature_groups.get(name)

    def list_features(
        self,
        entity: Optional[str] = None,
        feature_group: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[FeatureMetadata]:
        """
        List features with optional filtering.

        Args:
            entity: Filter by entity name
            feature_group: Filter by feature group
            tags: Filter by tags

        Returns:
            List of feature metadata matching filters
        """
        results = list(self._features.values())

        if entity:
            results = [f for f in results if f.entity.name == entity]

        if feature_group:
            group = self._feature_groups.get(feature_group)
            if group:
                results = [f for f in results if f.name in group.features]

        if tags:
            results = [f for f in results if any(tag in f.tags for tag in tags)]

        return results

    def search_features(self, query: str) -> List[FeatureMetadata]:
        """
        Search features by name or description.

        Args:
            query: Search query string

        Returns:
            List of matching feature metadata
        """
        query_lower = query.lower()
        results = []

        for feature in self._features.values():
            if (
                query_lower in feature.name.lower()
                or query_lower in feature.description.lower()
                or any(query_lower in tag.lower() for tag in feature.tags)
            ):
                results.append(feature)

        return results

    def update_feature_stats(self, feature_name: str, stats: Dict[str, Any]) -> None:
        """Update feature statistics."""
        if feature_name in self._features:
            self._features[feature_name].statistics.update(stats)
            self._features[feature_name].last_updated = datetime.utcnow()
            self._persist()

    def _persist(self) -> None:
        """Persist registry to backend."""
        if self.backend:
            # Would implement persistence to database/file
            pass


def register_feature(
    registry: FeatureRegistry, name: str, entity: FeatureEntity, dtype: str, **kwargs
) -> FeatureMetadata:
    """
    Helper function to register a feature.

    Args:
        registry: Feature registry
        name: Feature name
        entity: Associated entity
        dtype: Data type
        **kwargs: Additional metadata

    Returns:
        Created feature metadata
    """
    metadata = FeatureMetadata(name=name, entity=entity, dtype=dtype, **kwargs)
    registry.register_feature(metadata)
    return metadata


def list_features(registry: FeatureRegistry, **filters) -> List[FeatureMetadata]:
    """
    Helper function to list features.

    Args:
        registry: Feature registry
        **filters: Filter criteria

    Returns:
        List of feature metadata
    """
    return registry.list_features(**filters)


# ============================================================================
# Feature Transformation
# ============================================================================


class FeatureTransformation(ABC):
    """
    Abstract base class for feature transformations.

    Transformations convert raw features into processed features
    suitable for ML models.
    """

    @abstractmethod
    def transform(self, features: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        """
        Transform input features.

        Args:
            features: Input feature dictionary

        Returns:
            Transformed feature dictionary
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Dict[str, str]:
        """Get output feature schema."""
        pass


class OnDemandTransform(FeatureTransformation):
    """
    On-demand feature transformation computed at serving time.

    Used for transformations that must be computed fresh for
    each request (e.g., calculating age from birth date).

    Attributes:
        transform_fn: Transformation function
        input_features: Required input feature names
        output_features: Output feature names and types
    """

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Dict[str, FeatureValue]], Dict[str, FeatureValue]],
        input_features: List[str],
        output_schema: Dict[str, str],
    ):
        self.name = name
        self.transform_fn = transform_fn
        self.input_features = input_features
        self.output_schema = output_schema

    def transform(self, features: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        """Apply on-demand transformation."""
        # Verify all required inputs present
        missing = set(self.input_features) - set(features.keys())
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Apply transformation
        result = self.transform_fn(features)

        # Merge with original features
        features = features.copy()
        features.update(result)

        return features

    def get_output_schema(self) -> Dict[str, str]:
        """Get output schema."""
        return self.output_schema.copy()


class BatchTransform(FeatureTransformation):
    """
    Batch feature transformation for offline processing.

    Applied during materialization or ETL jobs to process
    large volumes of data efficiently.

    Attributes:
        transform_fn: Function operating on DataFrames
        input_columns: Required input columns
        output_schema: Output column specifications
        partition_by: Columns to partition by for parallel processing
    """

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        input_columns: List[str],
        output_schema: Dict[str, str],
        partition_by: Optional[List[str]] = None,
    ):
        self.name = name
        self.transform_fn = transform_fn
        self.input_columns = input_columns
        self.output_schema = output_schema
        self.partition_by = partition_by

    def transform(self, features: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        """Transform single record (for testing)."""
        df = pd.DataFrame([features])
        result_df = self.transform_fn(df)
        return result_df.iloc[0].to_dict()

    def transform_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform batch of records.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        # Verify all required columns present
        missing = set(self.input_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Apply transformation
        return self.transform_fn(df)

    def get_output_schema(self) -> Dict[str, str]:
        """Get output schema."""
        return self.output_schema.copy()


class StreamingTransform(FeatureTransformation):
    """
    Streaming feature transformation for real-time data.

    Processes features from streaming sources (Kafka, Kinesis)
    with support for windowed aggregations.

    Attributes:
        window_size: Processing window duration
        aggregation_fn: Aggregation function
        stateful: Whether transformation maintains state
    """

    def __init__(
        self,
        name: str,
        transform_fn: Callable[[Any], Dict[str, FeatureValue]],
        window_size: timedelta,
        aggregation_fn: Optional[Callable] = None,
        stateful: bool = False,
    ):
        self.name = name
        self.transform_fn = transform_fn
        self.window_size = window_size
        self.aggregation_fn = aggregation_fn
        self.stateful = stateful
        self._state: Dict[str, Any] = {}
        self._buffer: List[Dict[str, Any]] = []

    def transform(self, features: Dict[str, FeatureValue]) -> Dict[str, FeatureValue]:
        """Process single streaming record."""
        if self.stateful:
            # Update state
            self._update_state(features)

            # Apply transformation with state
            return self.transform_fn(features, self._state)
        else:
            return self.transform_fn(features)

    def _update_state(self, features: Dict[str, FeatureValue]) -> None:
        """Update internal state with new record."""
        entity_key = features.get("entity_key")
        if entity_key:
            if entity_key not in self._state:
                self._state[entity_key] = []
            self._state[entity_key].append(features)

    def get_output_schema(self) -> Dict[str, str]:
        """Get output schema."""
        # Would be determined by transform_fn
        return {}


# ============================================================================
# Feature Monitoring
# ============================================================================


@dataclass
class FeatureStats:
    """
    Statistical summary of feature values.

    Attributes:
        count: Number of observations
        mean: Mean value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        median: Median value
        null_count: Number of null values
        distinct_count: Number of distinct values
    """

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    null_count: int = 0
    distinct_count: int = 0

    @classmethod
    def from_series(cls, series: pd.Series) -> FeatureStats:
        """Compute statistics from a pandas Series."""
        return cls(
            count=len(series),
            mean=series.mean(),
            std=series.std(),
            min=series.min(),
            max=series.max(),
            median=series.median(),
            null_count=series.isnull().sum(),
            distinct_count=series.nunique(),
        )


@dataclass
class FeatureDrift:
    """
    Feature drift detection results.

    Attributes:
        feature_name: Name of the feature
        drift_score: Drift metric value
        is_drift: Whether drift is detected
        threshold: Drift detection threshold
        reference_stats: Statistics from reference data
        current_stats: Statistics from current data
        detection_method: Method used for drift detection
    """

    feature_name: str
    drift_score: float
    is_drift: bool
    threshold: float
    reference_stats: FeatureStats
    current_stats: FeatureStats
    detection_method: str = "ks_test"


class FeatureQuality:
    """
    Feature quality assessment.

    Checks feature data for quality issues like:
    - Missing values
    - Outliers
    - Type mismatches
    - Schema violations

    Attributes:
        checks: List of quality checks to perform
    """

    def __init__(self):
        self.checks: List[Callable] = []

    def add_check(self, check: Callable[[pd.Series], Tuple[bool, str]]) -> None:
        """Add a quality check."""
        self.checks.append(check)

    def assess_quality(
        self, df: pd.DataFrame, feature_metadata: Dict[str, FeatureMetadata]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Assess quality of feature data.

        Args:
            df: Feature DataFrame
            feature_metadata: Feature metadata dictionary

        Returns:
            Dictionary mapping feature names to quality issues
        """
        issues = {}

        for feature_name, metadata in feature_metadata.items():
            if feature_name not in df.columns:
                issues[feature_name] = [
                    {
                        "type": "missing_column",
                        "message": f"Feature column '{feature_name}' not found",
                    }
                ]
                continue

            series = df[feature_name]
            feature_issues = []

            # Check null values
            null_ratio = series.isnull().sum() / len(series)
            if null_ratio > 0.1:
                feature_issues.append(
                    {
                        "type": "high_null_ratio",
                        "message": f"{null_ratio:.2%} null values",
                        "severity": "warning" if null_ratio < 0.5 else "error",
                    }
                )

            # Check data type
            expected_dtype = metadata.dtype
            actual_dtype = str(series.dtype)
            if not self._check_dtype_compatibility(expected_dtype, actual_dtype):
                feature_issues.append(
                    {
                        "type": "type_mismatch",
                        "message": f"Expected {expected_dtype}, got {actual_dtype}",
                    }
                )

            # Run custom checks
            for check in self.checks:
                passed, message = check(series)
                if not passed:
                    feature_issues.append(
                        {"type": "custom_check_failed", "message": message}
                    )

            if feature_issues:
                issues[feature_name] = feature_issues

        return issues

    def _check_dtype_compatibility(self, expected: str, actual: str) -> bool:
        """Check if actual dtype matches expected."""
        # Simplified type checking
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        if "int" in expected_lower and "int" in actual_lower:
            return True
        if "float" in expected_lower and (
            "float" in actual_lower or "int" in actual_lower
        ):
            return True
        if "string" in expected_lower and "object" in actual_lower:
            return True

        return expected_lower in actual_lower


class FeatureMonitoring:
    """
    Monitor features for drift, quality, and statistics.

    Continuously tracks feature distributions and alerts on
    significant changes that could affect model performance.

    Attributes:
        reference_data: Baseline data for drift detection
        drift_threshold: Threshold for drift alerts
        quality_checker: Quality assessment instance
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,
        quality_checker: Optional[FeatureQuality] = None,
    ):
        self.drift_threshold = drift_threshold
        self.quality_checker = quality_checker or FeatureQuality()
        self._reference_stats: Dict[str, FeatureStats] = {}
        self._alerts: List[Dict[str, Any]] = []

    def set_reference_data(self, df: pd.DataFrame, feature_names: List[str]) -> None:
        """Set reference data for drift detection."""
        for feature in feature_names:
            if feature in df.columns:
                self._reference_stats[feature] = FeatureStats.from_series(df[feature])

    def detect_drift(
        self, df: pd.DataFrame, feature_names: List[str], method: str = "ks_test"
    ) -> List[FeatureDrift]:
        """
        Detect feature drift compared to reference.

        Args:
            df: Current feature data
            feature_names: Features to check
            method: Drift detection method (ks_test, psi, wasserstein)

        Returns:
            List of drift detection results
        """
        from scipy import stats

        drift_results = []

        for feature in feature_names:
            if feature not in df.columns:
                continue

            current_stats = FeatureStats.from_series(df[feature])
            reference_stats = self._reference_stats.get(feature)

            if reference_stats is None:
                continue

            # Compute drift score
            if method == "ks_test":
                # This would require storing reference distribution
                # Simplified: compare statistics
                drift_score = abs(current_stats.mean - reference_stats.mean) / (
                    reference_stats.std + 1e-10
                )
            elif method == "psi":
                # Population Stability Index
                drift_score = self._compute_psi(df[feature].dropna(), reference_stats)
            else:
                drift_score = 0.0

            is_drift = drift_score > self.drift_threshold

            drift_results.append(
                FeatureDrift(
                    feature_name=feature,
                    drift_score=drift_score,
                    is_drift=is_drift,
                    threshold=self.drift_threshold,
                    reference_stats=reference_stats,
                    current_stats=current_stats,
                    detection_method=method,
                )
            )

            if is_drift:
                self._alerts.append(
                    {
                        "timestamp": datetime.utcnow(),
                        "type": "drift",
                        "feature": feature,
                        "score": drift_score,
                    }
                )

        return drift_results

    def _compute_psi(self, current: pd.Series, reference_stats: FeatureStats) -> float:
        """Compute Population Stability Index."""
        # Simplified PSI calculation
        # Would typically use binned distributions
        return 0.0

    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get monitoring alerts."""
        if since:
            return [a for a in self._alerts if a["timestamp"] > since]
        return self._alerts.copy()

    def compute_statistics(
        self, df: pd.DataFrame, feature_names: List[str]
    ) -> Dict[str, FeatureStats]:
        """Compute statistics for features."""
        stats = {}
        for feature in feature_names:
            if feature in df.columns:
                stats[feature] = FeatureStats.from_series(df[feature])
        return stats


# ============================================================================
# Materialization
# ============================================================================


class MaterializationEngine:
    """
    Engine for materializing features from offline to online store.

    Handles backfills, incremental updates, and scheduled materialization
    to keep online store synchronized with offline source of truth.

    Attributes:
        store: Feature store instance
        batch_size: Number of records to process per batch
        max_workers: Parallel processing workers
    """

    def __init__(
        self, store: FeatureStore, batch_size: int = 10000, max_workers: int = 4
    ):
        self.store = store
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._scheduled_jobs: Dict[str, Any] = {}

    def backfill_features(
        self,
        feature_group: str,
        start_date: datetime,
        end_date: datetime,
        entity_filter: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """
        Backfill historical features to online store.

        Args:
            feature_group: Feature group to backfill
            start_date: Start of backfill period
            end_date: End of backfill period
            entity_filter: Optional filter for entity keys
        """
        logger.info(
            f"Starting backfill for {feature_group}: {start_date} to {end_date}"
        )

        group = self.store._feature_groups.get(feature_group)
        if not group:
            raise ValueError(f"Feature group not found: {feature_group}")

        # Query offline store for all entities in range
        # This is a simplified implementation
        df = self.store.offline_store.get_historical_features(
            entity_keys=[],  # Would query all
            feature_names=group.features,
            start_date=start_date,
            end_date=end_date,
        )

        # Filter entities if needed
        if entity_filter:
            df = df[df["entity_key"].apply(entity_filter)]

        # Process in batches
        total_records = len(df)
        processed = 0

        for i in range(0, total_records, self.batch_size):
            batch = df.iloc[i : i + self.batch_size]

            # Write to online store
            for _, row in batch.iterrows():
                entity_key = row["entity_key"]
                features = {col: row[col] for col in group.features if col in row}
                asyncio.run(self.store.online_store.set(entity_key, features))

            processed += len(batch)
            logger.info(f"Backfill progress: {processed}/{total_records}")

        logger.info(f"Backfill complete: {processed} records materialized")

    def incremental_materialization(
        self, feature_group: str, lookback: timedelta = timedelta(hours=1)
    ) -> None:
        """
        Materialize only new/changed features since last run.

        Args:
            feature_group: Feature group to materialize
            lookback: Time window for incremental changes
        """
        end_date = datetime.utcnow()
        start_date = end_date - lookback

        logger.info(
            f"Incremental materialization for {feature_group}: {start_date} to {end_date}"
        )

        # Use backfill with limited time range
        self.backfill_features(feature_group, start_date, end_date)

    def schedule_materialization(
        self,
        feature_group: str,
        schedule: str,
        incremental: bool = True,
        lookback: Optional[timedelta] = None,
    ) -> str:
        """
        Schedule periodic materialization job.

        Args:
            feature_group: Feature group to materialize
            schedule: Cron expression or interval (e.g., "0 */6 * * *" or "6h")
            incremental: Whether to use incremental mode
            lookback: Time window for incremental mode

        Returns:
            Job ID for the scheduled task
        """
        import schedule

        job_id = str(uuid.uuid4())

        def job():
            if incremental:
                self.incremental_materialization(
                    feature_group, lookback or timedelta(hours=1)
                )
            else:
                # Full materialization
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=7)
                self.backfill_features(feature_group, start_date, end_date)

        # Parse schedule
        if schedule.endswith("h"):
            # Hourly interval
            hours = int(schedule[:-1])
            scheduled_job = schedule.every(hours).hours.do(job)
        elif schedule.endswith("m"):
            # Minute interval
            minutes = int(schedule[:-1])
            scheduled_job = schedule.every(minutes).minutes.do(job)
        else:
            # Assume cron-like (simplified)
            # Would use croniter library for full support
            scheduled_job = schedule.every().hour.do(job)

        self._scheduled_jobs[job_id] = {
            "job": scheduled_job,
            "feature_group": feature_group,
            "schedule": schedule,
        }

        logger.info(f"Scheduled materialization job {job_id} for {feature_group}")
        return job_id

    def cancel_scheduled_job(self, job_id: str) -> bool:
        """Cancel a scheduled materialization job."""
        if job_id in self._scheduled_jobs:
            import schedule

            schedule.cancel_job(self._scheduled_jobs[job_id]["job"])
            del self._scheduled_jobs[job_id]
            logger.info(f"Cancelled job {job_id}")
            return True
        return False


# ============================================================================
# Integrations
# ============================================================================


class FeastIntegration:
    """
    Integration with Feast feature store.

    Allows interoperability with Feast ecosystem for teams
    already using Feast.

    Attributes:
        feast_repo_path: Path to Feast repository
        feast_client: Feast client instance
    """

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._client: Any = None

    def connect(self) -> None:
        """Connect to Feast."""
        try:
            from feast import FeatureStore as FeastFeatureStore

            self._client = FeastFeatureStore(repo_path=self.repo_path)
            logger.info(f"Connected to Feast at {self.repo_path}")
        except ImportError:
            logger.error("feast package not installed")
            raise

    def import_from_feast(
        self, feature_service_name: str
    ) -> Tuple[FeatureGroup, List[FeatureMetadata]]:
        """
        Import feature service from Feast.

        Args:
            feature_service_name: Name of Feast feature service

        Returns:
            Imported feature group and metadata
        """
        if not self._client:
            raise RuntimeError("Not connected to Feast")

        # Get feature service from Feast
        feature_service = self._client.get_feature_service(feature_service_name)

        # Convert to fishstick types
        # This is a simplified conversion
        entity = FeatureEntity(name="imported_entity", join_keys=["entity_key"])

        feature_names = []
        for projection in feature_service.feature_view_projections:
            feature_names.extend([f.name for f in projection.features])

        group = FeatureGroup(
            name=feature_service_name, entities=[entity], features=feature_names
        )

        metadata_list = []
        for name in feature_names:
            metadata_list.append(
                FeatureMetadata(
                    name=name,
                    entity=entity,
                    dtype="float",
                    description=f"Imported from Feast: {feature_service_name}",
                )
            )

        return group, metadata_list

    def export_to_feast(self, feature_group: FeatureGroup, output_path: str) -> None:
        """
        Export feature group to Feast format.

        Args:
            feature_group: Feature group to export
            output_path: Path for Feast configuration
        """
        # Generate Feast feature view definition
        feast_config = f"""
from feast import Entity, Feature, FeatureView, ValueType
from feast.types import Float64, Int64
from datetime import timedelta

# Entity
{feature_group.entities[0].name} = Entity(
    name="{feature_group.entities[0].name}",
    value_type=ValueType.STRING,
    join_keys={feature_group.entities[0].join_keys}
)

# Feature View
{feature_group.name} = FeatureView(
    name="{feature_group.name}",
    entities=["{feature_group.entities[0].name}"],
    ttl=timedelta(days=1),
    features=[
        {", ".join([f'Feature(name="{f}", dtype=Float64)' for f in feature_group.features])}
    ],
    online=True,
    source=None  # Would specify actual source
)
"""

        Path(output_path).write_text(feast_config)
        logger.info(f"Exported feature group to {output_path}")


class TectonIntegration:
    """
    Integration with Tecton feature platform.

    Supports Tecton's real-time and batch feature computation.
    """

    def __init__(self, api_key: Optional[str] = None, workspace: str = "default"):
        self.api_key = api_key
        self.workspace = workspace
        self._client: Any = None

    def connect(self) -> None:
        """Connect to Tecton."""
        try:
            import tecton

            self._client = tecton.init(self.workspace)
            logger.info(f"Connected to Tecton workspace: {self.workspace}")
        except ImportError:
            logger.error("tecton package not installed")
            raise

    def sync_feature_group(self, feature_group: FeatureGroup) -> None:
        """
        Sync feature group to Tecton.

        Args:
            feature_group: Feature group to sync
        """
        # Would implement Tecton SDK calls
        logger.info(f"Syncing feature group {feature_group.name} to Tecton")


class SageMakerIntegration:
    """
    Integration with AWS SageMaker Feature Store.

    Supports SageMaker's online and offline feature store.
    """

    def __init__(self, region: str = "us-east-1", role_arn: Optional[str] = None):
        self.region = region
        self.role_arn = role_arn
        self._sagemaker_client: Any = None
        self._featurestore_runtime: Any = None

    def connect(self) -> None:
        """Connect to AWS SageMaker."""
        try:
            import boto3

            self._sagemaker_client = boto3.client("sagemaker", region_name=self.region)
            self._featurestore_runtime = boto3.client(
                "sagemaker-featurestore-runtime", region_name=self.region
            )
            logger.info(f"Connected to SageMaker in {self.region}")
        except ImportError:
            logger.error("boto3 package not installed")
            raise

    def create_feature_group(
        self,
        name: str,
        entity: FeatureEntity,
        features: List[FeatureMetadata],
        s3_uri: str,
        record_identifier: str = "entity_key",
        event_time: str = "event_timestamp",
    ) -> str:
        """
        Create SageMaker feature group.

        Args:
            name: Feature group name
            entity: Entity definition
            features: List of feature metadata
            s3_uri: S3 location for offline store
            record_identifier: Record ID field name
            event_time: Event timestamp field name

        Returns:
            Feature group ARN
        """
        if not self._sagemaker_client:
            raise RuntimeError("Not connected to SageMaker")

        # Build feature definitions
        feature_definitions = []
        for f in features:
            feature_type = self._map_dtype_to_sagemaker(f.dtype)
            feature_definitions.append(
                {"FeatureName": f.name, "FeatureType": feature_type}
            )

        # Add entity key and timestamp
        feature_definitions.extend(
            [
                {"FeatureName": record_identifier, "FeatureType": "String"},
                {"FeatureName": event_time, "FeatureType": "String"},
            ]
        )

        response = self._sagemaker_client.create_feature_group(
            FeatureGroupName=name,
            RecordIdentifierFeatureName=record_identifier,
            EventTimeFeatureName=event_time,
            FeatureDefinitions=feature_definitions,
            OnlineStoreConfig={"EnableOnlineStore": True},
            OfflineStoreConfig={
                "S3StorageConfig": {"S3Uri": s3_uri, "ResolvedOutputS3Uri": s3_uri}
            },
            RoleArn=self.role_arn,
        )

        feature_group_arn = response["FeatureGroupArn"]
        logger.info(f"Created SageMaker feature group: {feature_group_arn}")

        return feature_group_arn

    def _map_dtype_to_sagemaker(self, dtype: str) -> str:
        """Map fishstick dtype to SageMaker feature type."""
        dtype_lower = dtype.lower()
        if "int" in dtype_lower:
            return "Integral"
        elif "float" in dtype_lower or "double" in dtype_lower:
            return "Fractional"
        else:
            return "String"

    def get_online_features(
        self,
        feature_group_name: str,
        record_identifiers: List[str],
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Get features from SageMaker online store.

        Args:
            feature_group_name: Feature group name
            record_identifiers: Entity identifiers
            feature_names: Features to retrieve

        Returns:
            List of feature records
        """
        if not self._featurestore_runtime:
            raise RuntimeError("Not connected to SageMaker")

        results = []
        for record_id in record_identifiers:
            response = self._featurestore_runtime.get_record(
                FeatureGroupName=feature_group_name,
                RecordIdentifierValueAsString=record_id,
                FeatureNames=feature_names,
            )

            if "Record" in response:
                record = {}
                for feature in response["Record"]:
                    record[feature["FeatureName"]] = list(feature["ValueAsString"])
                results.append(record)

        return results


# ============================================================================
# Utilities
# ============================================================================


class FeatureStoreClient:
    """
    High-level client for interacting with the feature store.

    Provides a simplified API for common operations with
    automatic retries, caching, and error handling.

    Attributes:
        store: Feature store instance
        cache: Optional cache layer
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
    """

    def __init__(
        self,
        store: FeatureStore,
        cache: Optional[Any] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.store = store
        self.cache = cache
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._server = FeatureServer(store, cache)

    async def get_features(
        self, entity_keys: List[str], feature_refs: List[str], use_cache: bool = True
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Get features with automatic retries.

        Args:
            entity_keys: Entity identifiers
            feature_refs: Feature references
            use_cache: Whether to use cache

        Returns:
            Dictionary mapping entity keys to features
        """
        for attempt in range(self.max_retries):
            try:
                vectors = await self._server.get_features(
                    entity_keys, feature_refs, source="online"
                )
                return {v.entity_key: v.features for v in vectors}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{self.max_retries}: {e}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        return {}

    def get_training_data(
        self,
        feature_refs: List[str],
        entity_df: pd.DataFrame,
        timestamp_col: str = "event_timestamp",
    ) -> pd.DataFrame:
        """
        Get point-in-time correct training data.

        Args:
            feature_refs: Feature references
            entity_df: DataFrame with entity keys and timestamps
            timestamp_col: Timestamp column name

        Returns:
            DataFrame with features joined to entities
        """
        # Parse feature references
        feature_names = []
        for ref in feature_refs:
            if ":" in ref:
                _, fname = ref.split(":", 1)
                feature_names.append(fname)
            else:
                feature_names.append(ref)

        # Get unique entities
        entity_keys = entity_df["entity_key"].unique().tolist()

        # Determine time range
        min_ts = entity_df[timestamp_col].min()
        max_ts = entity_df[timestamp_col].max()

        # Fetch historical features
        features_df = self.store.get_historical_features(
            entity_keys=entity_keys,
            feature_names=feature_names,
            start_date=min_ts - timedelta(days=30),
            end_date=max_ts,
        )

        # Point-in-time join
        result = []
        for _, row in entity_df.iterrows():
            entity_key = row["entity_key"]
            target_ts = row[timestamp_col]

            # Get features as of target timestamp
            entity_features = features_df[
                (features_df["entity_key"] == entity_key)
                & (features_df["event_timestamp"] <= target_ts)
            ]

            if not entity_features.empty:
                latest = entity_features.iloc[-1]
                merged = {**row.to_dict(), **latest[feature_names].to_dict()}
                result.append(merged)

        return pd.DataFrame(result)

    async def log_features(
        self,
        entity_key: str,
        features: Dict[str, FeatureValue],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Log features for an entity.

        Args:
            entity_key: Entity identifier
            features: Feature dictionary
            timestamp: Event timestamp
        """
        ts = timestamp or datetime.utcnow()

        # Write to both stores
        if self.store.online_store:
            await self.store.online_store.set(entity_key, features)

        if self.store.offline_store:
            df = pd.DataFrame(
                [{"entity_key": entity_key, "event_timestamp": ts, **features}]
            )
            self.store.offline_store.write_features(df, "logged_features")


class FeatureBuilder:
    """
    Builder pattern for constructing feature pipelines.

    Simplifies creation of complex feature configurations
    with a fluent API.

    Example:
        builder = FeatureBuilder()
        features = (builder
            .with_entity("user", ["user_id"])
            .with_feature("age", "int", description="User age")
            .with_transformation("normalize", normalize_fn)
            .build())
    """

    def __init__(self):
        self._entity: Optional[FeatureEntity] = None
        self._features: List[FeatureMetadata] = []
        self._transformations: List[FeatureTransformation] = []
        self._group_name: Optional[str] = None
        self._tags: List[str] = []

    def with_entity(
        self, name: str, join_keys: List[str], description: str = ""
    ) -> FeatureBuilder:
        """Set the entity for features."""
        self._entity = FeatureEntity(
            name=name, join_keys=join_keys, description=description
        )
        return self

    def with_feature(
        self,
        name: str,
        dtype: str,
        description: str = "",
        owner: str = "",
        tags: Optional[List[str]] = None,
    ) -> FeatureBuilder:
        """Add a feature definition."""
        if not self._entity:
            raise ValueError("Entity must be set before adding features")

        self._features.append(
            FeatureMetadata(
                name=name,
                entity=self._entity,
                dtype=dtype,
                description=description,
                owner=owner,
                tags=tags or [],
            )
        )
        return self

    def with_transformation(
        self, transformation: FeatureTransformation
    ) -> FeatureBuilder:
        """Add a feature transformation."""
        self._transformations.append(transformation)
        return self

    def with_group(self, name: str, tags: Optional[List[str]] = None) -> FeatureBuilder:
        """Set feature group name."""
        self._group_name = name
        self._tags = tags or []
        return self

    def build(self) -> Tuple[FeatureEntity, FeatureGroup, List[FeatureMetadata]]:
        """
        Build the feature configuration.

        Returns:
            Tuple of (entity, feature_group, features)
        """
        if not self._entity:
            raise ValueError("Entity must be set")

        group = FeatureGroup(
            name=self._group_name or f"{self._entity.name}_features",
            entities=[self._entity],
            features=[f.name for f in self._features],
            tags=self._tags,
        )

        return self._entity, group, self._features

    def register(self, store: FeatureStore) -> FeatureBuilder:
        """Register built features to a feature store."""
        entity, group, features = self.build()

        store.register_entity(entity)
        store.register_feature_group(group)

        for feature in features:
            store.register_feature(feature)

        return self


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    # Storage
    "FeatureStore",
    "OnlineStore",
    "RedisOnlineStore",
    "OfflineStore",
    "HiveS3OfflineStore",
    "FeatureGroup",
    "FeatureEntity",
    "FeatureMetadata",
    "StorageType",
    # Serving
    "FeatureServer",
    "FeatureVector",
    "get_features",
    "get_online_features",
    "get_historical_features",
    # Registry
    "FeatureRegistry",
    "register_feature",
    "list_features",
    # Transformation
    "FeatureTransformation",
    "OnDemandTransform",
    "BatchTransform",
    "StreamingTransform",
    # Monitoring
    "FeatureMonitoring",
    "FeatureDrift",
    "FeatureStats",
    "FeatureQuality",
    # Materialization
    "MaterializationEngine",
    # Integration
    "FeastIntegration",
    "TectonIntegration",
    "SageMakerIntegration",
    # Utilities
    "FeatureStoreClient",
    "FeatureBuilder",
    # Types
    "FeatureValue",
]
