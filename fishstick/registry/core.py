"""
Fishstick Model Registry - Comprehensive model management system.

This module provides a complete model registry implementation with:
- Model registration and retrieval
- Semantic versioning
- Multiple storage backends (Local, S3, GCS, Azure)
- Metadata management and model cards
- Lineage tracking
- Deployment lifecycle management
- Search and filtering capabilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import hashlib
import json
import logging
import pickle
import re
import shutil
import tempfile
import uuid


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class ModelNotFoundError(RegistryError):
    """Raised when a model is not found in the registry."""

    pass


class VersionNotFoundError(RegistryError):
    """Raised when a version is not found."""

    pass


class StorageError(RegistryError):
    """Raised when storage operation fails."""

    pass


class ValidationError(RegistryError):
    """Raised when validation fails."""

    pass


# =============================================================================
# Versioning
# =============================================================================


class VersionStage(Enum):
    """Model deployment stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass(frozen=True, order=True)
class SemanticVersion:
    """Semantic versioning implementation (SemVer 2.0.0)."""

    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = field(default=None, compare=False)
    build: Optional[str] = field(default=None, compare=False)

    def __post_init__(self):
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValidationError("Version components must be non-negative")

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse a version string into a SemanticVersion object."""
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValidationError(f"Invalid version string: {version_str}")

        major, minor, patch = map(int, match.groups()[:3])
        prerelease = match.group(4)
        build = match.group(5)

        return cls(major, minor, patch, prerelease, build)

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def bump_major(self) -> "SemanticVersion":
        """Bump major version."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Bump minor version."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        """Bump patch version."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""

    version: SemanticVersion
    model_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    stage: VersionStage = VersionStage.DEVELOPMENT
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    run_id: Optional[str] = None
    user: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": str(self.version),
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "stage": self.stage.value,
            "description": self.description,
            "tags": list(self.tags),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "run_id": self.run_id,
            "user": self.user,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=SemanticVersion.parse(data["version"]),
            model_id=data["model_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            stage=VersionStage(data["stage"]),
            description=data.get("description", ""),
            tags=set(data.get("tags", [])),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            run_id=data.get("run_id"),
            user=data.get("user"),
        )


class SemanticVersioning:
    """Manages semantic versioning for models."""

    def __init__(self):
        self._versions: Dict[str, List[ModelVersion]] = {}

    def version_model(
        self,
        model_id: str,
        version: Optional[SemanticVersion] = None,
        description: str = "",
        tags: Optional[Set[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        run_id: Optional[str] = None,
        user: Optional[str] = None,
    ) -> ModelVersion:
        """Create a new version for a model."""
        if model_id not in self._versions:
            self._versions[model_id] = []

        if version is None:
            version = self._get_next_version(model_id)

        # Check if version already exists
        existing = self.get_version(model_id, version)
        if existing:
            raise ValidationError(
                f"Version {version} already exists for model {model_id}"
            )

        model_version = ModelVersion(
            version=version,
            model_id=model_id,
            description=description,
            tags=tags or set(),
            metrics=metrics or {},
            run_id=run_id,
            user=user,
        )

        self._versions[model_id].append(model_version)
        self._versions[model_id].sort(key=lambda v: v.version)

        logger.info(f"Created version {version} for model {model_id}")
        return model_version

    def _get_next_version(self, model_id: str) -> SemanticVersion:
        """Get the next patch version for a model."""
        if model_id not in self._versions or not self._versions[model_id]:
            return SemanticVersion(0, 0, 1)

        latest = self._versions[model_id][-1].version
        return latest.bump_patch()

    def get_version(
        self,
        model_id: str,
        version: SemanticVersion,
    ) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        if model_id not in self._versions:
            return None

        for v in self._versions[model_id]:
            if v.version == version:
                return v
        return None

    def get_latest_version(
        self,
        model_id: str,
        stage: Optional[VersionStage] = None,
    ) -> Optional[ModelVersion]:
        """Get the latest version of a model, optionally filtered by stage."""
        if model_id not in self._versions:
            return None

        versions = self._versions[model_id]
        if stage:
            versions = [v for v in versions if v.stage == stage]

        return versions[-1] if versions else None

    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions for a model."""
        return list(self._versions.get(model_id, []))

    def update_stage(
        self,
        model_id: str,
        version: SemanticVersion,
        stage: VersionStage,
    ) -> ModelVersion:
        """Update the stage of a version."""
        model_version = self.get_version(model_id, version)
        if not model_version:
            raise VersionNotFoundError(
                f"Version {version} not found for model {model_id}"
            )

        # Remove from list and update
        self._versions[model_id].remove(model_version)
        model_version.stage = stage
        self._versions[model_id].append(model_version)
        self._versions[model_id].sort(key=lambda v: v.version)

        return model_version


# =============================================================================
# Storage Backends
# =============================================================================

T = TypeVar("T")


class StorageBackend(ABC, Generic[T]):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, key: str, data: T, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save data to storage."""
        pass

    @abstractmethod
    def load(self, key: str) -> T:
        """Load data from storage."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data from storage."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a key."""
        pass


class LocalStorage(StorageBackend[bytes]):
    """Local filesystem storage backend."""

    def __init__(self, root_path: Union[str, Path] = ".model_registry"):
        self.root_path = Path(root_path).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _get_path(self, key: str) -> Path:
        """Get the full path for a key."""
        # Sanitize key to prevent directory traversal
        safe_key = key.replace("..", "_").replace("/", "_")
        return self.root_path / safe_key

    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = self.root_path / ".metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = self.root_path / ".metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def save(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save data to local storage."""
        path = self._get_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(data)

        # Compute hash
        file_hash = hashlib.sha256(data).hexdigest()

        # Store metadata
        self._metadata[key] = {
            "path": str(path),
            "size": len(data),
            "hash": file_hash,
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._save_metadata()

        logger.info(f"Saved {key} to local storage ({len(data)} bytes)")
        return str(path)

    def load(self, key: str) -> bytes:
        """Load data from local storage."""
        path = self._get_path(key)
        if not path.exists():
            raise StorageError(f"Key not found: {key}")

        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> bool:
        """Delete data from local storage."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()
            logger.info(f"Deleted {key} from local storage")
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._get_path(key).exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        return [k for k in self._metadata.keys() if k.startswith(prefix)]

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a key."""
        if key not in self._metadata:
            raise StorageError(f"Metadata not found for key: {key}")
        return self._metadata[key].copy()


class S3Storage(StorageBackend[bytes]):
    """AWS S3 storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "models/",
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            try:
                import boto3

                session = boto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region,
                )
                self._client = session.client("s3")
            except ImportError:
                raise StorageError("boto3 is required for S3 storage")
        return self._client

    def _get_key(self, key: str) -> str:
        """Get the full S3 key."""
        return f"{self.prefix}/{key}" if self.prefix else key

    def save(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save data to S3."""
        s3_key = self._get_key(key)
        client = self._get_client()

        file_hash = hashlib.sha256(data).hexdigest()
        extra_args = {
            "Metadata": {
                "hash": file_hash,
                "created_at": datetime.utcnow().isoformat(),
                **{k: str(v) for k, v in (metadata or {}).items()},
            }
        }

        client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=data,
            **extra_args,
        )

        logger.info(f"Saved {key} to S3 ({len(data)} bytes)")
        return f"s3://{self.bucket}/{s3_key}"

    def load(self, key: str) -> bytes:
        """Load data from S3."""
        s3_key = self._get_key(key)
        client = self._get_client()

        response = client.get_object(Bucket=self.bucket, Key=s3_key)
        return response["Body"].read()

    def delete(self, key: str) -> bool:
        """Delete data from S3."""
        s3_key = self._get_key(key)
        client = self._get_client()

        try:
            client.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleted {key} from S3")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {key} from S3: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in S3."""
        s3_key = self._get_key(key)
        client = self._get_client()

        try:
            client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except:
            return False

    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        full_prefix = self._get_key(prefix)
        client = self._get_client()

        paginator = client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        return keys

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a key from S3."""
        s3_key = self._get_key(key)
        client = self._get_client()

        response = client.head_object(Bucket=self.bucket, Key=s3_key)
        return response.get("Metadata", {})


class GCSStorage(StorageBackend[bytes]):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "models/",
        project: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.project = project
        self._client = None

    def _get_client(self):
        """Lazy initialization of GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage

                self._client = storage.Client(project=self.project)
            except ImportError:
                raise StorageError("google-cloud-storage is required for GCS storage")
        return self._client

    def _get_blob_name(self, key: str) -> str:
        """Get the full GCS blob name."""
        return f"{self.prefix}/{key}" if self.prefix else key

    def save(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save data to GCS."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(blob_name)

        file_hash = hashlib.sha256(data).hexdigest()
        blob.metadata = {
            "hash": file_hash,
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        blob.upload_from_string(data)

        logger.info(f"Saved {key} to GCS ({len(data)} bytes)")
        return f"gs://{self.bucket}/{blob_name}"

    def load(self, key: str) -> bytes:
        """Load data from GCS."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise StorageError(f"Key not found: {key}")

        return blob.download_as_bytes()

    def delete(self, key: str) -> bool:
        """Delete data from GCS."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(blob_name)

        if blob.exists():
            blob.delete()
            logger.info(f"Deleted {key} from GCS")
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists in GCS."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(blob_name)
        return blob.exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        full_prefix = self._get_blob_name(prefix)
        client = self._get_client()
        bucket = client.bucket(self.bucket)

        blobs = bucket.list_blobs(prefix=full_prefix)
        keys = []
        for blob in blobs:
            key = blob.name
            if self.prefix:
                key = key[len(self.prefix) + 1 :]
            keys.append(key)
        return keys

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a key from GCS."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        bucket = client.bucket(self.bucket)
        blob = bucket.blob(blob_name)

        blob.reload()
        return blob.metadata or {}


class AzureStorage(StorageBackend[bytes]):
    """Azure Blob Storage backend."""

    def __init__(
        self,
        account_name: str,
        container: str,
        prefix: str = "models/",
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ):
        self.account_name = account_name
        self.container = container
        self.prefix = prefix.strip("/")
        self.account_key = account_key
        self.connection_string = connection_string
        self._client = None

    def _get_client(self):
        """Lazy initialization of Azure Blob client."""
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient

                if self.connection_string:
                    self._client = BlobServiceClient.from_connection_string(
                        self.connection_string
                    )
                else:
                    from azure.storage.blob import BlobClient

                    account_url = f"https://{self.account_name}.blob.core.windows.net"
                    self._client = BlobServiceClient(
                        account_url=account_url,
                        credential=self.account_key,
                    )
            except ImportError:
                raise StorageError("azure-storage-blob is required for Azure storage")
        return self._client

    def _get_blob_name(self, key: str) -> str:
        """Get the full Azure blob name."""
        return f"{self.prefix}/{key}" if self.prefix else key

    def save(
        self,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save data to Azure Blob Storage."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        container_client = client.get_container_client(self.container)
        blob_client = container_client.get_blob_client(blob_name)

        file_hash = hashlib.sha256(data).hexdigest()
        blob_metadata = {
            "hash": file_hash,
            "created_at": datetime.utcnow().isoformat(),
            **{k: str(v) for k, v in (metadata or {}).items()},
        }

        blob_client.upload_blob(data, overwrite=True, metadata=blob_metadata)

        logger.info(f"Saved {key} to Azure ({len(data)} bytes)")
        return f"azure://{self.account_name}/{self.container}/{blob_name}"

    def load(self, key: str) -> bytes:
        """Load data from Azure Blob Storage."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        container_client = client.get_container_client(self.container)
        blob_client = container_client.get_blob_client(blob_name)

        downloader = blob_client.download_blob()
        return downloader.readall()

    def delete(self, key: str) -> bool:
        """Delete data from Azure Blob Storage."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        container_client = client.get_container_client(self.container)
        blob_client = container_client.get_blob_client(blob_name)

        try:
            blob_client.delete_blob()
            logger.info(f"Deleted {key} from Azure")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {key} from Azure: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Azure Blob Storage."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        container_client = client.get_container_client(self.container)
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.exists()

    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix."""
        full_prefix = self._get_blob_name(prefix)
        client = self._get_client()
        container_client = client.get_container_client(self.container)

        blobs = container_client.list_blobs(name_starts_with=full_prefix)
        keys = []
        for blob in blobs:
            key = blob.name
            if self.prefix:
                key = key[len(self.prefix) + 1 :]
            keys.append(key)
        return keys

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a key from Azure Blob Storage."""
        blob_name = self._get_blob_name(key)
        client = self._get_client()
        container_client = client.get_container_client(self.container)
        blob_client = container_client.get_blob_client(blob_name)

        props = blob_client.get_blob_properties()
        return props.metadata or {}


# =============================================================================
# Metadata Management
# =============================================================================


@dataclass
class ModelMetadata:
    """Represents metadata for a registered model."""

    model_id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    owner: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    framework: Optional[str] = None
    task: Optional[str] = None
    license: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner": self.owner,
            "tags": list(self.tags),
            "framework": self.framework,
            "task": self.task,
            "license": self.license,
            "hyperparameters": self.hyperparameters,
            "custom_metadata": self.custom_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            owner=data.get("owner"),
            tags=set(data.get("tags", [])),
            framework=data.get("framework"),
            task=data.get("task"),
            license=data.get("license"),
            hyperparameters=data.get("hyperparameters", {}),
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass
class ModelCard:
    """Represents a model card for documentation."""

    model_id: str
    version: str
    overview: str = ""
    intended_use: str = ""
    factors: str = ""
    metrics: str = ""
    evaluation_data: str = ""
    training_data: str = ""
    ethical_considerations: str = ""
    caveats: str = ""
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "overview": self.overview,
            "intended_use": self.intended_use,
            "factors": self.factors,
            "metrics": self.metrics,
            "evaluation_data": self.evaluation_data,
            "training_data": self.training_data,
            "ethical_considerations": self.ethical_considerations,
            "caveats": self.caveats,
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            overview=data.get("overview", ""),
            intended_use=data.get("intended_use", ""),
            factors=data.get("factors", ""),
            metrics=data.get("metrics", ""),
            evaluation_data=data.get("evaluation_data", ""),
            training_data=data.get("training_data", ""),
            ethical_considerations=data.get("ethical_considerations", ""),
            caveats=data.get("caveats", ""),
            references=data.get("references", []),
        )

    def to_markdown(self) -> str:
        """Convert model card to Markdown format."""
        return f"""# Model Card: {self.model_id} (v{self.version})

## Overview
{self.overview}

## Intended Use
{self.intended_use}

## Factors
{self.factors}

## Metrics
{self.metrics}

## Evaluation Data
{self.evaluation_data}

## Training Data
{self.training_data}

## Ethical Considerations
{self.ethical_considerations}

## Caveats and Recommendations
{self.caveats}

## References
{chr(10).join(f"- {ref}" for ref in self.references)}
"""


class MetadataManager:
    """Manages model metadata and model cards."""

    def __init__(self, storage: Optional[StorageBackend] = None):
        self._metadata: Dict[str, ModelMetadata] = {}
        self._model_cards: Dict[Tuple[str, str], ModelCard] = {}
        self._storage = storage

    def add_metadata(
        self,
        model_id: str,
        name: str,
        description: str = "",
        owner: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        framework: Optional[str] = None,
        task: Optional[str] = None,
        license: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelMetadata:
        """Add metadata for a model."""
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            description=description,
            owner=owner,
            tags=tags or set(),
            framework=framework,
            task=task,
            license=license,
            hyperparameters=hyperparameters or {},
            custom_metadata=custom_metadata or {},
        )

        self._metadata[model_id] = metadata

        if self._storage:
            key = f"metadata/{model_id}.json"
            data = json.dumps(metadata.to_dict()).encode()
            self._storage.save(key, data)

        logger.info(f"Added metadata for model {model_id}")
        return metadata

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self._metadata.get(model_id)

    def update_metadata(
        self,
        model_id: str,
        **kwargs,
    ) -> ModelMetadata:
        """Update metadata for a model."""
        if model_id not in self._metadata:
            raise ModelNotFoundError(f"Model {model_id} not found")

        metadata = self._metadata[model_id]
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.updated_at = datetime.utcnow()

        if self._storage:
            key = f"metadata/{model_id}.json"
            data = json.dumps(metadata.to_dict()).encode()
            self._storage.save(key, data)

        logger.info(f"Updated metadata for model {model_id}")
        return metadata

    def create_model_card(
        self,
        model_id: str,
        version: str,
        **kwargs,
    ) -> ModelCard:
        """Create a model card."""
        card = ModelCard(model_id=model_id, version=version, **kwargs)
        self._model_cards[(model_id, version)] = card

        if self._storage:
            key = f"model_cards/{model_id}_{version}.json"
            data = json.dumps(card.to_dict()).encode()
            self._storage.save(key, data)

        logger.info(f"Created model card for {model_id} v{version}")
        return card

    def get_model_card(self, model_id: str, version: str) -> Optional[ModelCard]:
        """Get a model card."""
        return self._model_cards.get((model_id, version))


# =============================================================================
# Lineage Tracking
# =============================================================================


@dataclass
class Experiment:
    """Represents an experiment run."""

    experiment_id: str
    name: str
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    parent_run_id: Optional[str] = None
    status: str = "running"


@dataclass
class ModelLineageNode:
    """Represents a node in the lineage graph."""

    model_id: str
    version: str
    created_at: datetime
    experiment_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)


class ModelLineage:
    """Tracks model lineage and experiments."""

    def __init__(self):
        self._nodes: Dict[str, ModelLineageNode] = {}
        self._experiments: Dict[str, Experiment] = {}
        self._model_to_node: Dict[Tuple[str, str], str] = {}

    def track_experiment(
        self,
        name: str,
        experiment_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        parent_run_id: Optional[str] = None,
    ) -> Experiment:
        """Track a new experiment."""
        run_id = str(uuid.uuid4())
        experiment = Experiment(
            experiment_id=experiment_id or str(uuid.uuid4()),
            name=name,
            run_id=run_id,
            start_time=datetime.utcnow(),
            parameters=parameters or {},
            parent_run_id=parent_run_id,
        )

        self._experiments[run_id] = experiment
        logger.info(f"Started experiment {name} (run_id: {run_id})")
        return experiment

    def end_experiment(
        self,
        run_id: str,
        status: str = "completed",
        metrics: Optional[Dict[str, float]] = None,
    ) -> Experiment:
        """End an experiment."""
        if run_id not in self._experiments:
            raise RegistryError(f"Experiment run {run_id} not found")

        experiment = self._experiments[run_id]
        experiment.end_time = datetime.utcnow()
        experiment.status = status
        if metrics:
            experiment.metrics.update(metrics)

        logger.info(f"Ended experiment {experiment.name} with status {status}")
        return experiment

    def register_model_lineage(
        self,
        model_id: str,
        version: str,
        experiment_id: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
    ) -> ModelLineageNode:
        """Register a model with lineage tracking."""
        node_id = f"{model_id}:{version}"
        node = ModelLineageNode(
            model_id=model_id,
            version=version,
            created_at=datetime.utcnow(),
            experiment_id=experiment_id,
            parent_ids=parent_ids or [],
        )

        self._nodes[node_id] = node
        self._model_to_node[(model_id, version)] = node_id

        # Update parent nodes with child reference
        for parent_id in parent_ids or []:
            if parent_id in self._nodes:
                self._nodes[parent_id].child_ids.append(node_id)

        logger.info(f"Registered lineage for {model_id} v{version}")
        return node

    def get_ancestors(
        self,
        model_id: str,
        version: str,
        recursive: bool = True,
    ) -> List[ModelLineageNode]:
        """Get ancestor models."""
        node_id = self._model_to_node.get((model_id, version))
        if not node_id or node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        ancestors = []

        for parent_id in node.parent_ids:
            if parent_id in self._nodes:
                parent_node = self._nodes[parent_id]
                ancestors.append(parent_node)
                if recursive:
                    ancestors.extend(self._get_ancestors_recursive(parent_id))

        return ancestors

    def _get_ancestors_recursive(self, node_id: str) -> List[ModelLineageNode]:
        """Recursively get ancestors."""
        if node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        ancestors = []

        for parent_id in node.parent_ids:
            if parent_id in self._nodes:
                parent_node = self._nodes[parent_id]
                ancestors.append(parent_node)
                ancestors.extend(self._get_ancestors_recursive(parent_id))

        return ancestors

    def get_descendants(
        self,
        model_id: str,
        version: str,
        recursive: bool = True,
    ) -> List[ModelLineageNode]:
        """Get descendant models."""
        node_id = self._model_to_node.get((model_id, version))
        if not node_id or node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        descendants = []

        for child_id in node.child_ids:
            if child_id in self._nodes:
                child_node = self._nodes[child_id]
                descendants.append(child_node)
                if recursive:
                    descendants.extend(self._get_descendants_recursive(child_id))

        return descendants

    def _get_descendants_recursive(self, node_id: str) -> List[ModelLineageNode]:
        """Recursively get descendants."""
        if node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        descendants = []

        for child_id in node.child_ids:
            if child_id in self._nodes:
                child_node = self._nodes[child_id]
                descendants.append(child_node)
                descendants.extend(self._get_descendants_recursive(child_id))

        return descendants

    def get_experiment(self, run_id: str) -> Optional[Experiment]:
        """Get an experiment by run ID."""
        return self._experiments.get(run_id)

    def get_lineage_graph(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get the complete lineage graph for a model."""
        node_id = self._model_to_node.get((model_id, version))
        if not node_id or node_id not in self._nodes:
            return {"nodes": [], "edges": []}

        nodes = []
        edges = []
        visited = set()

        def traverse(nid: str):
            if nid in visited:
                return
            visited.add(nid)

            node = self._nodes[nid]
            nodes.append(
                {
                    "id": nid,
                    "model_id": node.model_id,
                    "version": node.version,
                    "experiment_id": node.experiment_id,
                }
            )

            for parent_id in node.parent_ids:
                edges.append({"from": parent_id, "to": nid})
                traverse(parent_id)

            for child_id in node.child_ids:
                edges.append({"from": nid, "to": child_id})
                traverse(child_id)

        traverse(node_id)
        return {"nodes": nodes, "edges": edges}


# =============================================================================
# Deployment
# =============================================================================


@dataclass
class DeploymentTarget:
    """Represents a deployment target."""

    name: str
    environment: str
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Deployment:
    """Represents a model deployment."""

    deployment_id: str
    model_id: str
    version: str
    target: str
    stage: VersionStage
    deployed_at: datetime
    deployed_by: Optional[str] = None
    status: str = "active"
    configuration: Dict[str, Any] = field(default_factory=dict)


class ModelDeployment:
    """Manages model deployments."""

    def __init__(self):
        self._targets: Dict[str, DeploymentTarget] = {}
        self._deployments: Dict[str, Deployment] = {}
        self._model_deployments: Dict[str, List[str]] = {}

    def register_target(
        self,
        name: str,
        environment: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> DeploymentTarget:
        """Register a deployment target."""
        target = DeploymentTarget(
            name=name,
            environment=environment,
            region=region,
            endpoint_url=endpoint_url,
            configuration=configuration or {},
        )
        self._targets[name] = target
        logger.info(f"Registered deployment target: {name}")
        return target

    def stage_model(
        self,
        model_id: str,
        version: str,
        target: str,
        deployed_by: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> Deployment:
        """Stage a model for deployment."""
        if target not in self._targets:
            raise RegistryError(f"Deployment target {target} not found")

        deployment_id = str(uuid.uuid4())
        deployment = Deployment(
            deployment_id=deployment_id,
            model_id=model_id,
            version=version,
            target=target,
            stage=VersionStage.STAGING,
            deployed_at=datetime.utcnow(),
            deployed_by=deployed_by,
            configuration=configuration or {},
        )

        self._deployments[deployment_id] = deployment

        key = f"{model_id}:{version}"
        if key not in self._model_deployments:
            self._model_deployments[key] = []
        self._model_deployments[key].append(deployment_id)

        logger.info(f"Staged model {model_id} v{version} to {target}")
        return deployment

    def promote_model(
        self,
        deployment_id: str,
        new_stage: VersionStage,
        promoted_by: Optional[str] = None,
    ) -> Deployment:
        """Promote a deployment to a new stage."""
        if deployment_id not in self._deployments:
            raise RegistryError(f"Deployment {deployment_id} not found")

        deployment = self._deployments[deployment_id]
        old_stage = deployment.stage
        deployment.stage = new_stage

        logger.info(
            f"Promoted model {deployment.model_id} v{deployment.version} "
            f"from {old_stage.value} to {new_stage.value}"
        )
        return deployment

    def archive_model(
        self,
        model_id: str,
        version: str,
    ) -> List[Deployment]:
        """Archive all deployments for a model version."""
        key = f"{model_id}:{version}"
        deployment_ids = self._model_deployments.get(key, [])
        archived = []

        for deployment_id in deployment_ids:
            deployment = self._deployments[deployment_id]
            deployment.stage = VersionStage.ARCHIVED
            deployment.status = "archived"
            archived.append(deployment)

        logger.info(f"Archived {len(archived)} deployments for {model_id} v{version}")
        return archived

    def get_deployments(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> List[Deployment]:
        """Get deployments for a model."""
        if version:
            key = f"{model_id}:{version}"
            deployment_ids = self._model_deployments.get(key, [])
        else:
            deployment_ids = [
                did
                for key, dids in self._model_deployments.items()
                if key.startswith(f"{model_id}:")
                for did in dids
            ]

        return [self._deployments[did] for did in deployment_ids]

    def list_targets(self) -> List[DeploymentTarget]:
        """List all registered deployment targets."""
        return list(self._targets.values())


# =============================================================================
# Search
# =============================================================================


@dataclass
class SearchResult:
    """Represents a search result."""

    model_id: str
    name: str
    version: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelSearch:
    """Search engine for models."""

    def __init__(self, registry: "ModelRegistry"):
        self.registry = registry
        self._index: Dict[str, Set[str]] = {}  # term -> model_ids

    def index_model(self, model_id: str, text: str):
        """Index a model for search."""
        terms = self._tokenize(text)
        for term in terms:
            if term not in self._index:
                self._index[term] = set()
            self._index[term].add(model_id)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for indexing."""
        # Simple tokenization - lowercase and split on non-alphanumeric
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search for models."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: Dict[str, float] = {}

        for term in query_terms:
            if term in self._index:
                for model_id in self._index[term]:
                    scores[model_id] = scores.get(model_id, 0) + 1

        # Normalize scores
        for model_id in scores:
            scores[model_id] /= len(query_terms)

        # Sort by score
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for model_id, score in sorted_models[:limit]:
            metadata = self.registry.get_metadata(model_id)
            if metadata:
                results.append(
                    SearchResult(
                        model_id=model_id,
                        name=metadata.name,
                        score=score,
                        metadata=metadata.to_dict(),
                    )
                )

        return results

    def filter_by_tag(
        self,
        tags: Union[str, List[str]],
    ) -> List[SearchResult]:
        """Filter models by tags."""
        if isinstance(tags, str):
            tags = [tags]

        results = []
        for model_id in self.registry.list_models():
            metadata = self.registry.get_metadata(model_id)
            if metadata and metadata.tags:
                if all(tag in metadata.tags for tag in tags):
                    results.append(
                        SearchResult(
                            model_id=model_id,
                            name=metadata.name,
                            metadata=metadata.to_dict(),
                        )
                    )

        return results


def search_models(
    registry: "ModelRegistry",
    query: str,
    tags: Optional[List[str]] = None,
    framework: Optional[str] = None,
    task: Optional[str] = None,
    limit: int = 10,
) -> List[SearchResult]:
    """Search for models with filters."""
    search_engine = ModelSearch(registry)

    # Index all models
    for model_id in registry.list_models():
        metadata = registry.get_metadata(model_id)
        if metadata:
            text = f"{metadata.name} {metadata.description} {' '.join(metadata.tags)}"
            search_engine.index_model(model_id, text)

    # Perform search
    results = search_engine.search(query, limit=limit * 2)

    # Apply filters
    filtered = []
    for result in results:
        metadata = result.metadata

        if tags:
            model_tags = set(metadata.get("tags", []))
            if not all(tag in model_tags for tag in tags):
                continue

        if framework and metadata.get("framework") != framework:
            continue

        if task and metadata.get("task") != task:
            continue

        filtered.append(result)

        if len(filtered) >= limit:
            break

    return filtered


def filter_by_tag(registry: "ModelRegistry", tags: Union[str, List[str]]) -> List[str]:
    """Filter model IDs by tags."""
    search_engine = ModelSearch(registry)
    results = search_engine.filter_by_tag(tags)
    return [r.model_id for r in results]


# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class RegisteredModel:
    """Represents a registered model."""

    model_id: str
    metadata: ModelMetadata
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ModelRegistry:
    """Central model registry for managing ML models."""

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        versioning: Optional[SemanticVersioning] = None,
        metadata: Optional[MetadataManager] = None,
        lineage: Optional[ModelLineage] = None,
        deployment: Optional[ModelDeployment] = None,
    ):
        self._models: Dict[str, RegisteredModel] = {}
        self.storage = storage or LocalStorage()
        self.versioning = versioning or SemanticVersioning()
        self.metadata = metadata or MetadataManager(storage)
        self.lineage = lineage or ModelLineage()
        self.deployment = deployment or ModelDeployment()

    def register_model(
        self,
        model: Any,
        name: str,
        description: str = "",
        version: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        owner: Optional[str] = None,
        framework: Optional[str] = None,
        task: Optional[str] = None,
        experiment_id: Optional[str] = None,
        parent_models: Optional[List[Tuple[str, str]]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, ModelVersion]:
        """Register a model in the registry.

        Args:
            model: The model object to register
            name: Human-readable name for the model
            description: Model description
            version: Optional version string (auto-generated if None)
            tags: Set of tags for the model
            owner: Model owner
            framework: ML framework used (e.g., 'pytorch', 'tensorflow')
            task: Task type (e.g., 'classification', 'regression')
            experiment_id: Associated experiment ID
            parent_models: List of (model_id, version) tuples for lineage
            artifacts: Additional artifacts to store
            **kwargs: Additional metadata

        Returns:
            Tuple of (model_id, model_version)
        """
        # Generate model ID
        model_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

        # Create metadata
        self.metadata.add_metadata(
            model_id=model_id,
            name=name,
            description=description,
            owner=owner,
            tags=tags,
            framework=framework,
            task=task,
            custom_metadata=kwargs,
        )

        # Register model
        registered = RegisteredModel(
            model_id=model_id,
            metadata=self.metadata.get_metadata(model_id),
        )
        self._models[model_id] = registered

        # Create version
        if version:
            semver = SemanticVersion.parse(version)
        else:
            semver = SemanticVersion(0, 0, 1)

        model_version = self.versioning.version_model(
            model_id=model_id,
            version=semver,
            description=description,
            tags=tags,
            run_id=experiment_id,
            user=owner,
        )

        # Store model
        self._store_model(model, model_id, str(model_version.version))

        # Register lineage
        parent_ids = None
        if parent_models:
            parent_ids = [f"{mid}:{ver}" for mid, ver in parent_models]

        self.lineage.register_model_lineage(
            model_id=model_id,
            version=str(model_version.version),
            experiment_id=experiment_id,
            parent_ids=parent_ids,
        )

        # Store artifacts
        if artifacts:
            for artifact_name, artifact_data in artifacts.items():
                self._store_artifact(
                    model_id, str(model_version.version), artifact_name, artifact_data
                )

        logger.info(
            f"Registered model {name} with ID {model_id} v{model_version.version}"
        )
        return model_id, model_version

    def _store_model(self, model: Any, model_id: str, version: str):
        """Store model to storage backend."""
        # Serialize model
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        data = buffer.getvalue()

        # Store
        key = f"models/{model_id}/{version}/model.pkl"
        self.storage.save(
            key, data, metadata={"model_id": model_id, "version": version}
        )

    def _store_artifact(
        self,
        model_id: str,
        version: str,
        name: str,
        data: Any,
    ):
        """Store an artifact."""
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        key = f"models/{model_id}/{version}/artifacts/{name}.pkl"
        self.storage.save(key, buffer.getvalue())

    def get_model(
        self,
        model_id: str,
        version: Optional[str] = None,
    ) -> Any:
        """Retrieve a model from the registry.

        Args:
            model_id: The model ID
            version: Optional version string (latest if None)

        Returns:
            The model object
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model {model_id} not found")

        if version is None:
            latest = self.versioning.get_latest_version(model_id)
            if not latest:
                raise VersionNotFoundError(f"No versions found for model {model_id}")
            version = str(latest.version)

        # Load from storage
        key = f"models/{model_id}/{version}/model.pkl"
        try:
            data = self.storage.load(key)
            return pickle.loads(data)
        except Exception as e:
            raise StorageError(f"Failed to load model {model_id} v{version}: {e}")

    def list_models(
        self,
        tags: Optional[List[str]] = None,
        framework: Optional[str] = None,
    ) -> List[str]:
        """List all registered models.

        Args:
            tags: Optional filter by tags
            framework: Optional filter by framework

        Returns:
            List of model IDs
        """
        models = list(self._models.keys())

        if tags or framework:
            filtered = []
            for model_id in models:
                metadata = self.metadata.get_metadata(model_id)
                if metadata:
                    if tags and not all(tag in metadata.tags for tag in tags):
                        continue
                    if framework and metadata.framework != framework:
                        continue
                    filtered.append(model_id)
            return filtered

        return models

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a model."""
        return self.metadata.get_metadata(model_id)

    def add_metadata(
        self,
        model_id: str,
        **kwargs,
    ) -> ModelMetadata:
        """Add or update metadata for a model."""
        return self.metadata.update_metadata(model_id, **kwargs)

    def get_model_version(
        self,
        model_id: str,
        version: str,
    ) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        semver = SemanticVersion.parse(version)
        return self.versioning.get_version(model_id, semver)

    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        return self.versioning.get_latest_version(model_id)


# =============================================================================
# Utilities
# =============================================================================


def compare_models(
    model_a: Any,
    model_b: Any,
    compare_weights: bool = True,
    compare_architecture: bool = True,
) -> Dict[str, Any]:
    """Compare two models and return differences.

    Args:
        model_a: First model
        model_b: Second model
        compare_weights: Whether to compare model weights
        compare_architecture: Whether to compare architecture

    Returns:
        Dictionary with comparison results
    """
    result = {
        "identical": False,
        "type_match": type(model_a) == type(model_b),
        "architecture_match": None,
        "weights_match": None,
        "size_a": 0,
        "size_b": 0,
        "differences": [],
    }

    try:
        import torch
        import torch.nn as nn

        if isinstance(model_a, nn.Module) and isinstance(model_b, nn.Module):
            # Compare architecture
            if compare_architecture:
                arch_a = str(model_a)
                arch_b = str(model_b)
                result["architecture_match"] = arch_a == arch_b
                if not result["architecture_match"]:
                    result["differences"].append("Architecture differs")

            # Compare weights
            if compare_weights:
                weights_match = True
                for (name_a, param_a), (name_b, param_b) in zip(
                    model_a.named_parameters(),
                    model_b.named_parameters(),
                ):
                    if name_a != name_b or not torch.equal(param_a, param_b):
                        weights_match = False
                        result["differences"].append(f"Weight mismatch: {name_a}")

                result["weights_match"] = weights_match

            # Count parameters
            result["size_a"] = sum(p.numel() for p in model_a.parameters())
            result["size_b"] = sum(p.numel() for p in model_b.parameters())

    except ImportError:
        # Fallback for non-PyTorch models
        result["architecture_match"] = str(model_a) == str(model_b)

    # Overall identical check
    result["identical"] = (
        result["type_match"]
        and result.get("architecture_match", True)
        and result.get("weights_match", True)
    )

    return result


def copy_model(
    registry: ModelRegistry,
    source_model_id: str,
    target_name: str,
    source_version: Optional[str] = None,
    new_version: Optional[str] = None,
    **kwargs,
) -> Tuple[str, ModelVersion]:
    """Copy a model to create a new model entry.

    Args:
        registry: The model registry
        source_model_id: Source model ID
        target_name: Name for the new model
        source_version: Source version (latest if None)
        new_version: Version for the new model (auto if None)
        **kwargs: Additional metadata for the new model

    Returns:
        Tuple of (new_model_id, new_version)
    """
    # Get source model
    model = registry.get_model(source_model_id, source_version)
    source_metadata = registry.get_metadata(source_model_id)

    if not source_metadata:
        raise ModelNotFoundError(f"Model {source_model_id} not found")

    # Register as new model
    new_id, new_ver = registry.register_model(
        model=model,
        name=target_name,
        description=kwargs.get("description", f"Copy of {source_metadata.name}"),
        version=new_version,
        tags=kwargs.get("tags", source_metadata.tags.copy()),
        owner=kwargs.get("owner", source_metadata.owner),
        framework=kwargs.get("framework", source_metadata.framework),
        task=kwargs.get("task", source_metadata.task),
        parent_models=[(source_model_id, source_version or str(new_ver.version))],
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["description", "tags", "owner", "framework", "task"]
        },
    )

    logger.info(f"Copied model {source_model_id} to {new_id}")
    return new_id, new_ver


def delete_model(
    registry: ModelRegistry,
    model_id: str,
    version: Optional[str] = None,
    permanent: bool = False,
) -> bool:
    """Delete a model or specific version from the registry.

    Args:
        registry: The model registry
        model_id: Model ID to delete
        version: Optional specific version (all versions if None)
        permanent: If True, permanently delete from storage

    Returns:
        True if deletion was successful
    """
    if model_id not in registry._models:
        raise ModelNotFoundError(f"Model {model_id} not found")

    if version:
        # Delete specific version
        key = f"models/{model_id}/{version}/model.pkl"
        if permanent:
            registry.storage.delete(key)

        # Remove from versions
        semver = SemanticVersion.parse(version)
        versions = registry.versioning.list_versions(model_id)
        for v in versions:
            if v.version == semver:
                registry.versioning._versions[model_id].remove(v)
                break

        logger.info(f"Deleted version {version} of model {model_id}")
    else:
        # Delete all versions
        if permanent:
            keys = registry.storage.list_keys(f"models/{model_id}/")
            for key in keys:
                registry.storage.delete(key)

        # Remove model from registry
        del registry._models[model_id]
        if model_id in registry.versioning._versions:
            del registry.versioning._versions[model_id]

        logger.info(f"Deleted model {model_id}")

    return True


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Registry
    "ModelRegistry",
    "RegisteredModel",
    "register_model",
    "get_model",
    "list_models",
    # Versioning
    "SemanticVersioning",
    "SemanticVersion",
    "ModelVersion",
    "version_model",
    "get_latest_version",
    "VersionStage",
    # Storage
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "GCSStorage",
    "AzureStorage",
    # Metadata
    "ModelMetadata",
    "ModelCard",
    "MetadataManager",
    "add_metadata",
    "get_metadata",
    # Lineage
    "ModelLineage",
    "Experiment",
    "track_experiment",
    "get_ancestors",
    "get_descendants",
    # Deployment
    "ModelDeployment",
    "Deployment",
    "DeploymentTarget",
    "stage_model",
    "promote_model",
    "archive_model",
    # Search
    "ModelSearch",
    "SearchResult",
    "search_models",
    "filter_by_tag",
    # Utilities
    "compare_models",
    "copy_model",
    "delete_model",
    # Exceptions
    "RegistryError",
    "ModelNotFoundError",
    "VersionNotFoundError",
    "StorageError",
    "ValidationError",
]


# Convenience functions
def register_model(
    registry: ModelRegistry,
    model: Any,
    name: str,
    **kwargs,
) -> Tuple[str, ModelVersion]:
    """Convenience function to register a model."""
    return registry.register_model(model, name, **kwargs)


def get_model(
    registry: ModelRegistry, model_id: str, version: Optional[str] = None
) -> Any:
    """Convenience function to get a model."""
    return registry.get_model(model_id, version)


def list_models(registry: ModelRegistry, **kwargs) -> List[str]:
    """Convenience function to list models."""
    return registry.list_models(**kwargs)


def version_model(
    versioning: SemanticVersioning,
    model_id: str,
    **kwargs,
) -> ModelVersion:
    """Convenience function to version a model."""
    return versioning.version_model(model_id, **kwargs)


def get_latest_version(
    versioning: SemanticVersioning,
    model_id: str,
    stage: Optional[VersionStage] = None,
) -> Optional[ModelVersion]:
    """Convenience function to get latest version."""
    return versioning.get_latest_version(model_id, stage)


def add_metadata(
    metadata_manager: MetadataManager,
    model_id: str,
    **kwargs,
) -> ModelMetadata:
    """Convenience function to add metadata."""
    return metadata_manager.add_metadata(model_id, **kwargs)


def get_metadata(
    metadata_manager: MetadataManager, model_id: str
) -> Optional[ModelMetadata]:
    """Convenience function to get metadata."""
    return metadata_manager.get_metadata(model_id)


def track_experiment(
    lineage: ModelLineage,
    name: str,
    **kwargs,
) -> Experiment:
    """Convenience function to track an experiment."""
    return lineage.track_experiment(name, **kwargs)


def get_ancestors(
    lineage: ModelLineage,
    model_id: str,
    version: str,
    **kwargs,
) -> List[ModelLineageNode]:
    """Convenience function to get ancestors."""
    return lineage.get_ancestors(model_id, version, **kwargs)


def get_descendants(
    lineage: ModelLineage,
    model_id: str,
    version: str,
    **kwargs,
) -> List[ModelLineageNode]:
    """Convenience function to get descendants."""
    return lineage.get_descendants(model_id, version, **kwargs)


def stage_model(
    deployment: ModelDeployment,
    model_id: str,
    version: str,
    target: str,
    **kwargs,
) -> Deployment:
    """Convenience function to stage a model."""
    return deployment.stage_model(model_id, version, target, **kwargs)


def promote_model(
    deployment: ModelDeployment,
    deployment_id: str,
    new_stage: VersionStage,
    **kwargs,
) -> Deployment:
    """Convenience function to promote a model."""
    return deployment.promote_model(deployment_id, new_stage, **kwargs)


def archive_model(
    deployment: ModelDeployment,
    model_id: str,
    version: str,
) -> List[Deployment]:
    """Convenience function to archive a model."""
    return deployment.archive_model(model_id, version)


import io
