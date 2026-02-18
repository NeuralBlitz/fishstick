"""
Fishstick - Comprehensive Caching Module
A high-performance, feature-rich caching library for Python.
"""

from __future__ import annotations

import abc
import base64
import functools
import hashlib
import json
import os
import pickle
import random
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    runtime_checkable,
)

# Optional dependencies handling
try:
    import redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import pymemcache

    HAS_MEMCACHED = True
except ImportError:
    HAS_MEMCACHED = False

try:
    import plyvel

    HAS_LEVELDB = True
except ImportError:
    HAS_LEVELDB = False

try:
    import lmdb

    HAS_LMDB = True
except ImportError:
    HAS_LMDB = False

try:
    import msgpack

    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


# ============================================================================
# Base Interfaces
# ============================================================================


@runtime_checkable
class Cache(Protocol[K, V]):
    """Base cache protocol defining common cache operations."""

    def get(self, key: K) -> Optional[V]: ...
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None: ...
    def delete(self, key: K) -> bool: ...
    def exists(self, key: K) -> bool: ...
    def clear(self) -> None: ...
    def keys(self) -> Iterator[K]: ...
    def values(self) -> Iterator[V]: ...
    def items(self) -> Iterator[Tuple[K, V]]: ...
    def size(self) -> int: ...
    def __len__(self) -> int: ...
    def __contains__(self, key: K) -> bool: ...


class BaseCache(abc.ABC, Generic[K, V]):
    """Abstract base class for all cache implementations."""

    @abc.abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Get a value from the cache."""
        pass

    @abc.abstractmethod
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set a value in the cache."""
        pass

    @abc.abstractmethod
    def delete(self, key: K) -> bool:
        """Delete a key from the cache. Returns True if key existed."""
        pass

    @abc.abstractmethod
    def exists(self, key: K) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass

    @abc.abstractmethod
    def keys(self) -> Iterator[K]:
        """Return an iterator over cache keys."""
        pass

    @abc.abstractmethod
    def values(self) -> Iterator[V]:
        """Return an iterator over cache values."""
        pass

    @abc.abstractmethod
    def items(self) -> Iterator[Tuple[K, V]]:
        """Return an iterator over cache items."""
        pass

    @abc.abstractmethod
    def size(self) -> int:
        """Return the number of items in the cache."""
        pass

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, key: K) -> bool:
        return self.exists(key)

    def get_or_set(
        self, key: K, default: Union[V, Callable[[], V]], ttl: Optional[float] = None
    ) -> V:
        """Get a value, or set it if it doesn't exist."""
        value = self.get(key)
        if value is None:
            if callable(default):
                value = default()
            else:
                value = default
            self.set(key, value, ttl)
        return value

    def get_many(self, keys: List[K]) -> Dict[K, V]:
        """Get multiple values at once."""
        return {key: value for key in keys if (value := self.get(key)) is not None}

    def set_many(self, items: Dict[K, V], ttl: Optional[float] = None) -> None:
        """Set multiple values at once."""
        for key, value in items.items():
            self.set(key, value, ttl)

    def delete_many(self, keys: List[K]) -> int:
        """Delete multiple keys at once. Returns count of deleted keys."""
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count


# ============================================================================
# Serialization
# ============================================================================


class Serializer(abc.ABC):
    """Abstract base class for serializers."""

    @abc.abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        pass

    @abc.abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an object."""
        pass


class PickleSerializer(Serializer):
    """Pickle-based serializer."""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class JSONSerializer(Serializer):
    """JSON-based serializer."""

    def __init__(
        self, encoder: Optional[type] = None, decoder: Optional[Callable] = None
    ):
        self.encoder = encoder
        self.decoder = decoder

    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj, cls=self.encoder).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"), object_hook=self.decoder)


class MessagePackSerializer(Serializer):
    """MessagePack-based serializer (requires msgpack package)."""

    def __init__(self, use_bin_type: bool = True, raw: bool = False):
        if not HAS_MSGPACK:
            raise ImportError("msgpack package is required for MessagePackSerializer")
        self.use_bin_type = use_bin_type
        self.raw = raw

    def serialize(self, obj: Any) -> bytes:
        return msgpack.packb(obj, use_bin_type=self.use_bin_type)

    def deserialize(self, data: bytes) -> Any:
        return msgpack.unpackb(data, raw=self.raw)


class CustomSerializer(Serializer):
    """Custom serializer with user-defined encode/decode functions."""

    def __init__(
        self, encode_fn: Callable[[Any], bytes], decode_fn: Callable[[bytes], Any]
    ):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn

    def serialize(self, obj: Any) -> bytes:
        return self.encode_fn(obj)

    def deserialize(self, data: bytes) -> Any:
        return self.decode_fn(data)


class StringSerializer(Serializer):
    """Simple string serializer for text data."""

    def serialize(self, obj: Any) -> bytes:
        return str(obj).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        return data.decode("utf-8")


# ============================================================================
# Eviction Policies
# ============================================================================


class EvictionPolicy(abc.ABC):
    """Abstract base class for eviction policies."""

    @abc.abstractmethod
    def should_evict(self, cache: BaseCache, key: Any, value: Any) -> bool:
        """Determine if an entry should be evicted."""
        pass

    @abc.abstractmethod
    def select_for_eviction(self, cache: BaseCache, keys: List[Any]) -> Any:
        """Select a key for eviction from the given list."""
        pass


class LRUEviction(EvictionPolicy):
    """LRU (Least Recently Used) eviction policy."""

    def __init__(self):
        self.access_times: Dict[Any, float] = {}

    def record_access(self, key: Any) -> None:
        self.access_times[key] = time.time()

    def should_evict(self, cache: BaseCache, key: Any, value: Any) -> bool:
        return False  # LRU doesn't preemptively evict

    def select_for_eviction(self, cache: BaseCache, keys: List[Any]) -> Any:
        if not keys:
            return None
        return min(keys, key=lambda k: self.access_times.get(k, 0))

    def remove(self, key: Any) -> None:
        self.access_times.pop(key, None)


class LFUEviction(EvictionPolicy):
    """LFU (Least Frequently Used) eviction policy."""

    def __init__(self):
        self.access_counts: Dict[Any, int] = defaultdict(int)

    def record_access(self, key: Any) -> None:
        self.access_counts[key] += 1

    def should_evict(self, cache: BaseCache, key: Any, value: Any) -> bool:
        return False

    def select_for_eviction(self, cache: BaseCache, keys: List[Any]) -> Any:
        if not keys:
            return None
        return min(keys, key=lambda k: self.access_counts.get(k, 0))

    def remove(self, key: Any) -> None:
        self.access_counts.pop(key, None)


class RandomEviction(EvictionPolicy):
    """Random eviction policy."""

    def should_evict(self, cache: BaseCache, key: Any, value: Any) -> bool:
        return False

    def select_for_eviction(self, cache: BaseCache, keys: List[Any]) -> Any:
        return random.choice(keys) if keys else None


class TTLExpiration(EvictionPolicy):
    """TTL-based expiration policy."""

    def __init__(self, default_ttl: Optional[float] = None):
        self.default_ttl = default_ttl
        self.expiry_times: Dict[Any, float] = {}

    def set_expiry(self, key: Any, ttl: Optional[float] = None) -> None:
        actual_ttl = ttl if ttl is not None else self.default_ttl
        if actual_ttl is not None:
            self.expiry_times[key] = time.time() + actual_ttl
        else:
            self.expiry_times.pop(key, None)

    def is_expired(self, key: Any) -> bool:
        if key not in self.expiry_times:
            return False
        return time.time() > self.expiry_times[key]

    def should_evict(self, cache: BaseCache, key: Any, value: Any) -> bool:
        return self.is_expired(key)

    def select_for_eviction(self, cache: BaseCache, keys: List[Any]) -> Any:
        expired = [k for k in keys if self.is_expired(k)]
        return expired[0] if expired else None

    def remove(self, key: Any) -> None:
        self.expiry_times.pop(key, None)


class SizeEviction(EvictionPolicy):
    """Size-based eviction policy."""

    def __init__(self, max_size: int, size_fn: Optional[Callable[[Any], int]] = None):
        self.max_size = max_size
        self.size_fn = size_fn or self._default_size_fn
        self.current_size = 0
        self.entry_sizes: Dict[Any, int] = {}

    @staticmethod
    def _default_size_fn(value: Any) -> int:
        try:
            return len(pickle.dumps(value))
        except:
            return 1

    def record_size(self, key: Any, value: Any) -> None:
        size = self.size_fn(value)
        if key in self.entry_sizes:
            self.current_size -= self.entry_sizes[key]
        self.entry_sizes[key] = size
        self.current_size += size

    def remove(self, key: Any) -> None:
        if key in self.entry_sizes:
            self.current_size -= self.entry_sizes[key]
            del self.entry_sizes[key]

    def should_evict(self, cache: BaseCache, key: Any, value: Any) -> bool:
        return self.current_size > self.max_size

    def select_for_eviction(self, cache: BaseCache, keys: List[Any]) -> Any:
        # Remove oldest/largest entries
        if not keys:
            return None
        return max(keys, key=lambda k: self.entry_sizes.get(k, 0))


# ============================================================================
# In-Memory Caches
# ============================================================================


class LRUCache(BaseCache[K, V]):
    """LRU (Least Recently Used) in-memory cache."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._cache.keys()))

    def values(self) -> Iterator[V]:
        with self._lock:
            return iter(list(self._cache.values()))

    def items(self) -> Iterator[Tuple[K, V]]:
        with self._lock:
            return iter(list(self._cache.items()))

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class LFUCache(BaseCache[K, V]):
    """LFU (Least Frequently Used) in-memory cache."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: Dict[K, V] = {}
        self._freq: Dict[K, int] = defaultdict(int)
        self._lock = threading.RLock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._cache:
                return None
            self._freq[key] += 1
            return self._cache[key]

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            if key not in self._cache and len(self._cache) >= self.maxsize:
                # Evict least frequently used
                min_freq = min(self._freq.values())
                for k, freq in list(self._freq.items()):
                    if freq == min_freq:
                        del self._cache[k]
                        del self._freq[k]
                        break
            self._cache[key] = value
            self._freq[key] += 1

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._freq[key]
                return True
            return False

    def exists(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._freq.clear()

    def keys(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._cache.keys()))

    def values(self) -> Iterator[V]:
        with self._lock:
            return iter(list(self._cache.values()))

    def items(self) -> Iterator[Tuple[K, V]]:
        with self._lock:
            return iter(list(self._cache.items()))

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class FIFOCache(BaseCache[K, V]):
    """FIFO (First In, First Out) in-memory cache."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = value

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._cache.keys()))

    def values(self) -> Iterator[V]:
        with self._lock:
            return iter(list(self._cache.values()))

    def items(self) -> Iterator[Tuple[K, V]]:
        with self._lock:
            return iter(list(self._cache.items()))

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class TTLCache(BaseCache[K, V]):
    """TTL (Time To Live) based in-memory cache."""

    def __init__(self, default_ttl: float = 300.0, maxsize: Optional[int] = None):
        self.default_ttl = default_ttl
        self.maxsize = maxsize
        self._cache: Dict[K, V] = {}
        self._expiry: Dict[K, float] = {}
        self._lock = threading.RLock()

    def _is_expired(self, key: K) -> bool:
        if key not in self._expiry:
            return False
        return time.time() > self._expiry[key]

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [k for k, exp in self._expiry.items() if now > exp]
        for key in expired:
            self._cache.pop(key, None)
            self._expiry.pop(key, None)

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._cache or self._is_expired(key):
                self._cache.pop(key, None)
                self._expiry.pop(key, None)
                return None
            return self._cache[key]

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            actual_ttl = ttl if ttl is not None else self.default_ttl

            if (
                self.maxsize
                and len(self._cache) >= self.maxsize
                and key not in self._cache
            ):
                self._cleanup_expired()
                if len(self._cache) >= self.maxsize:
                    # Remove oldest entry
                    oldest = min(self._expiry.items(), key=lambda x: x[1])[0]
                    self._cache.pop(oldest, None)
                    self._expiry.pop(oldest, None)

            self._cache[key] = value
            self._expiry[key] = time.time() + actual_ttl

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._expiry.pop(key, None)
                return True
            return False

    def exists(self, key: K) -> bool:
        with self._lock:
            return key in self._cache and not self._is_expired(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._expiry.clear()

    def keys(self) -> Iterator[K]:
        with self._lock:
            self._cleanup_expired()
            return iter(list(self._cache.keys()))

    def values(self) -> Iterator[V]:
        with self._lock:
            self._cleanup_expired()
            return iter(list(self._cache.values()))

    def items(self) -> Iterator[Tuple[K, V]]:
        with self._lock:
            self._cleanup_expired()
            return iter(list(self._cache.items()))

    def size(self) -> int:
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)


class BoundedCache(BaseCache[K, V]):
    """Bounded cache with configurable eviction policy."""

    def __init__(
        self,
        maxsize: int = 128,
        eviction_policy: Optional[EvictionPolicy] = None,
        default_ttl: Optional[float] = None,
    ):
        self.maxsize = maxsize
        self._cache: Dict[K, V] = {}
        self._eviction = eviction_policy or LRUEviction()
        self._ttl = TTLExpiration(default_ttl) if default_ttl else None
        self._lock = threading.RLock()

        # Track access for LRU/LFU
        if isinstance(self._eviction, LRUEviction):
            self._track_access = self._eviction.record_access
        elif isinstance(self._eviction, LFUEviction):
            self._track_access = self._eviction.record_access
        else:
            self._track_access = lambda x: None

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            if key not in self._cache:
                return None

            if self._ttl and self._ttl.is_expired(key):
                self.delete(key)
                return None

            self._track_access(key)
            return self._cache[key]

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        with self._lock:
            if self._ttl:
                self._ttl.set_expiry(key, ttl)

            if key not in self._cache and len(self._cache) >= self.maxsize:
                keys = list(self._cache.keys())
                to_evict = self._eviction.select_for_eviction(keys)
                if to_evict:
                    self.delete(to_evict)

            self._cache[key] = value
            self._track_access(key)

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self._ttl:
                    self._ttl.remove(key)
                if hasattr(self._eviction, "remove"):
                    self._eviction.remove(key)
                return True
            return False

    def exists(self, key: K) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            if self._ttl and self._ttl.is_expired(key):
                self.delete(key)
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._cache.keys()))

    def values(self) -> Iterator[V]:
        with self._lock:
            return iter(list(self._cache.values()))

    def items(self) -> Iterator[Tuple[K, V]]:
        with self._lock:
            return iter(list(self._cache.items()))

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# ============================================================================
# Disk Caches
# ============================================================================


class DiskCache(BaseCache[str, Any]):
    """SQLite-based disk cache."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
    ):
        self.path = path or Path(tempfile.gettempdir()) / "fishstick_cache.db"
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    expires REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires)")
            conn.commit()

    def _is_expired(self, expires: Optional[float]) -> bool:
        if expires is None:
            return False
        return time.time() > expires

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            with sqlite3.connect(str(self.path)) as conn:
                cursor = conn.execute(
                    "SELECT value, expires FROM cache WHERE key = ?", (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                value_blob, expires = row

                if self._is_expired(expires):
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None

                return self.serializer.deserialize(value_blob)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            actual_ttl = ttl if ttl is not None else self.default_ttl
            expires = time.time() + actual_ttl if actual_ttl else None
            value_blob = self.serializer.serialize(value)

            with sqlite3.connect(str(self.path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, expires)
                    VALUES (?, ?, ?)
                    """,
                    (key, value_blob, expires),
                )
                conn.commit()

    def delete(self, key: str) -> bool:
        with self._lock:
            with sqlite3.connect(str(self.path)) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            with sqlite3.connect(str(self.path)) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()

    def keys(self) -> Iterator[str]:
        with sqlite3.connect(str(self.path)) as conn:
            cursor = conn.execute("SELECT key, expires FROM cache")
            for key, expires in cursor:
                if not self._is_expired(expires):
                    yield key

    def values(self) -> Iterator[Any]:
        with sqlite3.connect(str(self.path)) as conn:
            cursor = conn.execute("SELECT value, expires FROM cache")
            for value_blob, expires in cursor:
                if not self._is_expired(expires):
                    yield self.serializer.deserialize(value_blob)

    def items(self) -> Iterator[Tuple[str, Any]]:
        with sqlite3.connect(str(self.path)) as conn:
            cursor = conn.execute("SELECT key, value, expires FROM cache")
            for key, value_blob, expires in cursor:
                if not self._is_expired(expires):
                    yield key, self.serializer.deserialize(value_blob)

    def size(self) -> int:
        with sqlite3.connect(str(self.path)) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires IS NULL OR expires > ?",
                (time.time(),),
            )
            return cursor.fetchone()[0]


class FileCache(BaseCache[str, Any]):
    """File-based cache storing each entry as a separate file."""

    def __init__(
        self,
        directory: Optional[Union[str, Path]] = None,
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
    ):
        self.directory = (
            Path(directory or tempfile.gettempdir()) / "fishstick_file_cache"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self._lock = threading.RLock()

    def _get_path(self, key: str) -> Path:
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.directory / f"{safe_key}.cache"

    def _is_expired(self, path: Path) -> bool:
        if self.default_ttl is None:
            return False
        return time.time() - path.stat().st_mtime > self.default_ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            path = self._get_path(key)
            if not path.exists():
                return None

            if self._is_expired(path):
                path.unlink(missing_ok=True)
                return None

            try:
                with open(path, "rb") as f:
                    return self.serializer.deserialize(f.read())
            except Exception:
                path.unlink(missing_ok=True)
                return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            path = self._get_path(key)
            data = self.serializer.serialize(value)

            with open(path, "wb") as f:
                f.write(data)

    def delete(self, key: str) -> bool:
        with self._lock:
            path = self._get_path(key)
            if path.exists():
                path.unlink()
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            path = self._get_path(key)
            if not path.exists():
                return False
            if self._is_expired(path):
                path.unlink(missing_ok=True)
                return False
            return True

    def clear(self) -> None:
        with self._lock:
            for path in self.directory.glob("*.cache"):
                path.unlink()

    def keys(self) -> Iterator[str]:
        raise NotImplementedError("FileCache does not support listing keys")

    def values(self) -> Iterator[Any]:
        raise NotImplementedError("FileCache does not support listing values")

    def items(self) -> Iterator[Tuple[str, Any]]:
        raise NotImplementedError("FileCache does not support listing items")

    def size(self) -> int:
        return len(list(self.directory.glob("*.cache")))


class LevelDBCache(BaseCache[str, Any]):
    """LevelDB-based cache (requires plyvel package)."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
    ):
        if not HAS_LEVELDB:
            raise ImportError("plyvel package is required for LevelDBCache")

        self.path = Path(path or tempfile.gettempdir()) / "fishstick_leveldb_cache"
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self._expiry_prefix = b"__exp__"
        self._db = plyvel.DB(str(self.path), create_if_missing=True)
        self._lock = threading.RLock()

    def _expiry_key(self, key: str) -> bytes:
        return self._expiry_prefix + key.encode()

    def _is_expired(self, key: str) -> bool:
        expiry_data = self._db.get(self._expiry_key(key))
        if expiry_data is None:
            return False
        return time.time() > float(expiry_data.decode())

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if self._is_expired(key):
                self.delete(key)
                return None

            data = self._db.get(key.encode())
            if data is None:
                return None

            return self.serializer.deserialize(data)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            actual_ttl = ttl if ttl is not None else self.default_ttl
            data = self.serializer.serialize(value)

            batch = self._db.write_batch()
            batch.put(key.encode(), data)

            if actual_ttl:
                expires = time.time() + actual_ttl
                batch.put(self._expiry_key(key), str(expires).encode())
            else:
                batch.delete(self._expiry_key(key))

            batch.write()

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = self._db.get(key.encode()) is not None
            batch = self._db.write_batch()
            batch.delete(key.encode())
            batch.delete(self._expiry_key(key))
            batch.write()
            return existed

    def exists(self, key: str) -> bool:
        with self._lock:
            if self._is_expired(key):
                self.delete(key)
                return False
            return self._db.get(key.encode()) is not None

    def clear(self) -> None:
        with self._lock:
            # Close and reopen to clear
            self._db.close()
            import shutil

            shutil.rmtree(self.path, ignore_errors=True)
            self._db = plyvel.DB(str(self.path), create_if_missing=True)

    def keys(self) -> Iterator[str]:
        with self._lock:
            for key, _ in self._db:
                if not key.startswith(self._expiry_prefix):
                    key_str = key.decode()
                    if not self._is_expired(key_str):
                        yield key_str

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        return sum(1 for _ in self.keys())


class RocksDBCache(BaseCache[str, Any]):
    """RocksDB-based cache (requires rocksdb package)."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
    ):
        try:
            import rocksdb

            self._rocksdb = rocksdb
        except ImportError:
            raise ImportError("rocksdb package is required for RocksDBCache")

        self.path = Path(path or tempfile.gettempdir()) / "fishstick_rocksdb_cache"
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl

        opts = self._rocksdb.Options()
        opts.create_if_missing = True
        self._db = self._rocksdb.DB(str(self.path), opts)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            data = self._db.get(key.encode())
            if data is None:
                return None
            return self.serializer.deserialize(data)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            data = self.serializer.serialize(value)
            self._db.put(key.encode(), data)

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = self._db.get(key.encode()) is not None
            self._db.delete(key.encode())
            return existed

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._db.get(key.encode()) is not None

    def clear(self) -> None:
        with self._lock:
            import shutil

            self._db.close()
            shutil.rmtree(self.path, ignore_errors=True)
            opts = self._rocksdb.Options()
            opts.create_if_missing = True
            self._db = self._rocksdb.DB(str(self.path), opts)

    def keys(self) -> Iterator[str]:
        it = self._db.itervalues()
        it.seek_to_first()
        for key in it:
            yield key.decode()

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        return sum(1 for _ in self.keys())


class LMDBCache(BaseCache[str, Any]):
    """LMDB-based cache (requires lmdb package)."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
        map_size: int = 104857600,  # 100MB
    ):
        if not HAS_LMDB:
            raise ImportError("lmdb package is required for LMDBCache")

        self.path = Path(path or tempfile.gettempdir()) / "fishstick_lmdb_cache"
        self.path.mkdir(parents=True, exist_ok=True)
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self._env = lmdb.open(str(self.path), map_size=map_size)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            with self._env.begin() as txn:
                data = txn.get(key.encode())
                if data is None:
                    return None
                return self.serializer.deserialize(bytes(data))

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            data = self.serializer.serialize(value)
            with self._env.begin(write=True) as txn:
                txn.put(key.encode(), data)

    def delete(self, key: str) -> bool:
        with self._lock:
            with self._env.begin(write=True) as txn:
                existed = txn.get(key.encode()) is not None
                txn.delete(key.encode())
                return existed

    def exists(self, key: str) -> bool:
        with self._lock:
            with self._env.begin() as txn:
                return txn.get(key.encode()) is not None

    def clear(self) -> None:
        with self._lock:
            with self._env.begin(write=True) as txn:
                cursor = txn.cursor()
                cursor.first()
                for key, _ in cursor:
                    txn.delete(key)

    def keys(self) -> Iterator[str]:
        with self._env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode()

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        with self._env.begin() as txn:
            return txn.stat()["entries"]


# ============================================================================
# Distributed Caches
# ============================================================================


class RedisCache(BaseCache[str, Any]):
    """Redis-based distributed cache (requires redis package)."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
        key_prefix: str = "fishstick:",
    ):
        if not HAS_REDIS:
            raise ImportError("redis package is required for RedisCache")

        self._redis = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=False
        )
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self.key_prefix = key_prefix

    def _make_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        data = self._redis.get(self._make_key(key))
        if data is None:
            return None
        return self.serializer.deserialize(data)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        actual_ttl = ttl if ttl is not None else self.default_ttl
        data = self.serializer.serialize(value)

        if actual_ttl:
            self._redis.setex(self._make_key(key), int(actual_ttl), data)
        else:
            self._redis.set(self._make_key(key), data)

    def delete(self, key: str) -> bool:
        return self._redis.delete(self._make_key(key)) > 0

    def exists(self, key: str) -> bool:
        return self._redis.exists(self._make_key(key)) > 0

    def clear(self) -> None:
        pattern = f"{self.key_prefix}*"
        for key in self._redis.scan_iter(match=pattern):
            self._redis.delete(key)

    def keys(self) -> Iterator[str]:
        pattern = f"{self.key_prefix}*"
        prefix_len = len(self.key_prefix)
        for key in self._redis.scan_iter(match=pattern):
            yield key.decode()[prefix_len:]

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        return sum(1 for _ in self.keys())


class MemcachedCache(BaseCache[str, Any]):
    """Memcached-based distributed cache (requires pymemcache package)."""

    def __init__(
        self,
        servers: Union[str, List[str]] = "localhost:11211",
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
        key_prefix: str = "fishstick:",
    ):
        if not HAS_MEMCACHED:
            raise ImportError("pymemcache package is required for MemcachedCache")

        if isinstance(servers, str):
            servers = [servers]

        self._client = pymemcache.Client(servers[0])
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self.key_prefix = key_prefix
        self._lock = threading.RLock()

    def _make_key(self, key: str) -> str:
        # Memcached has key length limit
        full_key = f"{self.key_prefix}{key}"
        if len(full_key) > 250:
            return hashlib.md5(full_key.encode()).hexdigest()
        return full_key

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            data = self._client.get(self._make_key(key))
            if data is None:
                return None
            return self.serializer.deserialize(data)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            actual_ttl = ttl if ttl is not None else self.default_ttl
            data = self.serializer.serialize(value)

            self._client.set(
                self._make_key(key), data, expire=int(actual_ttl) if actual_ttl else 0
            )

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._client.delete(self._make_key(key))

    def exists(self, key: str) -> bool:
        return self._client.get(self._make_key(key)) is not None

    def clear(self) -> None:
        self._client.flush_all()

    def keys(self) -> Iterator[str]:
        raise NotImplementedError("Memcached does not support listing keys")

    def values(self) -> Iterator[Any]:
        raise NotImplementedError("Memcached does not support listing values")

    def items(self) -> Iterator[Tuple[str, Any]]:
        raise NotImplementedError("Memcached does not support listing items")

    def size(self) -> int:
        raise NotImplementedError("Memcached does not support size operation")


class HazelcastCache(BaseCache[str, Any]):
    """Hazelcast-based distributed cache."""

    def __init__(
        self,
        cluster_members: Optional[List[str]] = None,
        map_name: str = "fishstick_cache",
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
    ):
        try:
            import hazelcast

            self._hz = hazelcast
        except ImportError:
            raise ImportError("hazelcast-python-client package is required")

        config = self._hz.ClientConfig()
        if cluster_members:
            config.network_config.addresses = cluster_members

        self._client = self._hz.HazelcastClient(config)
        self._cache = self._client.get_map(map_name).blocking()
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            data = self._cache.get(key)
            if data is None:
                return None
            return self.serializer.deserialize(data)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            actual_ttl = ttl if ttl is not None else self.default_ttl
            data = self.serializer.serialize(value)

            if actual_ttl:
                self._cache.set(key, data, int(actual_ttl * 1000))
            else:
                self._cache.set(key, data)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._cache.remove(key) is not None

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._cache.contains_key(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> Iterator[str]:
        with self._lock:
            for key in self._cache.key_set():
                yield key

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        with self._lock:
            return self._cache.size()


class IgniteCache(BaseCache[str, Any]):
    """Apache Ignite-based distributed cache."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 10800,
        cache_name: str = "fishstick_cache",
        serializer: Optional[Serializer] = None,
        ttl: Optional[float] = None,
    ):
        try:
            import pyignite
            from pyignite import Client

            self._ignite = pyignite
            self._Client = Client
        except ImportError:
            raise ImportError("pyignite package is required for IgniteCache")

        self._client = self._Client()
        self._client.connect(host, port)
        self._cache = self._client.get_or_create_cache(cache_name)
        self.serializer = serializer or PickleSerializer()
        self.default_ttl = ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            data = self._cache.get(key)
            if data is None:
                return None
            return self.serializer.deserialize(data)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            actual_ttl = ttl if ttl is not None else self.default_ttl
            data = self.serializer.serialize(value)

            if actual_ttl:
                self._cache.put(key, data, ttl=int(actual_ttl * 1000))
            else:
                self._cache.put(key, data)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._cache.remove_key(key)

    def exists(self, key: str) -> bool:
        with self._lock:
            return self._cache.contains_key(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self) -> Iterator[str]:
        with self._lock:
            cursor = self._cache.scan()
            for key, _ in cursor:
                yield key

    def values(self) -> Iterator[Any]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[str, Any]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        with self._lock:
            return self._cache.get_size()


# ============================================================================
# Hierarchical Caches
# ============================================================================


class MultiLevelCache(BaseCache[K, V]):
    """Multi-level cache with L1, L2, and optionally L3 layers."""

    def __init__(
        self,
        l1: BaseCache[K, V],
        l2: BaseCache[K, V],
        l3: Optional[BaseCache[K, V]] = None,
        promote_on_read: bool = True,
        invalidate_on_write: bool = False,
    ):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.promote_on_read = promote_on_read
        self.invalidate_on_write = invalidate_on_write

    def get(self, key: K) -> Optional[V]:
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        value = self.l2.get(key)
        if value is not None:
            if self.promote_on_read:
                self.l1.set(key, value)
            return value

        # Try L3 if exists
        if self.l3:
            value = self.l3.get(key)
            if value is not None:
                if self.promote_on_read:
                    self.l2.set(key, value)
                    self.l1.set(key, value)
                return value

        return None

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        self.l1.set(key, value, ttl)
        self.l2.set(key, value, ttl)
        if self.l3:
            self.l3.set(key, value, ttl)

        if self.invalidate_on_write:
            # Already set above, no invalidation needed
            pass

    def delete(self, key: K) -> bool:
        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key)
        l3_deleted = self.l3.delete(key) if self.l3 else False
        return l1_deleted or l2_deleted or l3_deleted

    def exists(self, key: K) -> bool:
        return (
            self.l1.exists(key)
            or self.l2.exists(key)
            or (self.l3.exists(key) if self.l3 else False)
        )

    def clear(self) -> None:
        self.l1.clear()
        self.l2.clear()
        if self.l3:
            self.l3.clear()

    def keys(self) -> Iterator[K]:
        seen = set()
        for key in self.l1.keys():
            seen.add(key)
            yield key
        for key in self.l2.keys():
            if key not in seen:
                seen.add(key)
                yield key
        if self.l3:
            for key in self.l3.keys():
                if key not in seen:
                    yield key

    def values(self) -> Iterator[V]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[K, V]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        return sum(1 for _ in self.keys())


class L1L2Cache(MultiLevelCache[K, V]):
    """L1/L2 two-level cache with fast L1 and larger L2."""

    def __init__(
        self,
        l1_size: int = 100,
        l2_path: Optional[Union[str, Path]] = None,
        promote_on_read: bool = True,
    ):
        l1 = LRUCache[K, V](maxsize=l1_size)
        l2 = DiskCache(str(l2_path) if l2_path else None)
        super().__init__(l1, l2, promote_on_read=promote_on_read)


class CacheHierarchy(BaseCache[K, V]):
    """Generic cache hierarchy supporting arbitrary levels."""

    def __init__(self, levels: List[BaseCache[K, V]], promote_on_read: bool = True):
        if not levels:
            raise ValueError("At least one cache level required")
        self.levels = levels
        self.promote_on_read = promote_on_read

    def get(self, key: K) -> Optional[V]:
        for i, level in enumerate(self.levels):
            value = level.get(key)
            if value is not None:
                if self.promote_on_read:
                    # Promote to all higher levels
                    for j in range(i):
                        self.levels[j].set(key, value)
                return value
        return None

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        for level in self.levels:
            level.set(key, value, ttl)

    def delete(self, key: K) -> bool:
        deleted = False
        for level in self.levels:
            if level.delete(key):
                deleted = True
        return deleted

    def exists(self, key: K) -> bool:
        return any(level.exists(key) for level in self.levels)

    def clear(self) -> None:
        for level in self.levels:
            level.clear()

    def keys(self) -> Iterator[K]:
        seen = set()
        for level in self.levels:
            for key in level.keys():
                if key not in seen:
                    seen.add(key)
                    yield key

    def values(self) -> Iterator[V]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield value

    def items(self) -> Iterator[Tuple[K, V]]:
        for key in self.keys():
            value = self.get(key)
            if value is not None:
                yield key, value

    def size(self) -> int:
        return sum(1 for _ in self.keys())


# ============================================================================
# Caching Strategies
# ============================================================================


class CacheAside:
    """Cache-aside (lazy loading) pattern implementation."""

    def __init__(self, cache: BaseCache[K, V]):
        self.cache = cache

    def get(self, key: K, loader: Callable[[], V], ttl: Optional[float] = None) -> V:
        """Get from cache or load if not present."""
        value = self.cache.get(key)
        if value is None:
            value = loader()
            self.cache.set(key, value, ttl)
        return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Update cache and data store (caller responsible for data store)."""
        self.cache.set(key, value, ttl)

    def invalidate(self, key: K) -> bool:
        """Invalidate cache entry."""
        return self.cache.delete(key)


class ReadThrough:
    """Read-through pattern implementation."""

    def __init__(self, cache: BaseCache[K, V], data_store: Callable[[K], V]):
        self.cache = cache
        self.data_store = data_store

    def get(self, key: K, ttl: Optional[float] = None) -> V:
        """Get from cache, loading from data store if not present."""
        value = self.cache.get(key)
        if value is None:
            value = self.data_store(key)
            self.cache.set(key, value, ttl)
        return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Update both cache and data store."""
        # Note: data store update is responsibility of caller
        self.cache.set(key, value, ttl)


class WriteThrough:
    """Write-through pattern implementation."""

    def __init__(
        self,
        cache: BaseCache[K, V],
        data_store_write: Callable[[K, V], None],
        data_store_read: Optional[Callable[[K], V]] = None,
    ):
        self.cache = cache
        self.data_store_write = data_store_write
        self.data_store_read = data_store_read

    def get(self, key: K, ttl: Optional[float] = None) -> Optional[V]:
        """Get from cache."""
        return self.cache.get(key)

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Write through to both cache and data store synchronously."""
        self.data_store_write(key, value)
        self.cache.set(key, value, ttl)


class WriteBehind:
    """Write-behind (write-back) pattern implementation."""

    def __init__(
        self,
        cache: BaseCache[K, V],
        data_store_write: Callable[[K, V], None],
        flush_interval: float = 5.0,
        batch_size: int = 100,
    ):
        self.cache = cache
        self.data_store_write = data_store_write
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self._pending: Dict[K, V] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            self._flush()
            self._stop_event.wait(self.flush_interval)

    def _flush(self) -> None:
        with self._lock:
            if not self._pending:
                return

            # Take batch
            batch = dict(list(self._pending.items())[: self.batch_size])
            self._pending = dict(list(self._pending.items())[self.batch_size :])

        # Write to data store
        for key, value in batch.items():
            try:
                self.data_store_write(key, value)
            except Exception:
                # Re-queue on failure
                with self._lock:
                    self._pending[key] = value

    def get(self, key: K, ttl: Optional[float] = None) -> Optional[V]:
        """Get from cache."""
        with self._lock:
            if key in self._pending:
                return self._pending[key]
        return self.cache.get(key)

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Write to cache immediately, queue for data store."""
        self.cache.set(key, value, ttl)
        with self._lock:
            self._pending[key] = value

    def flush(self) -> None:
        """Force flush pending writes."""
        while self._pending:
            self._flush()

    def stop(self) -> None:
        """Stop the flush thread and flush remaining."""
        self._stop_event.set()
        self.flush()
        self._flush_thread.join()


class RefreshAhead:
    """Refresh-ahead pattern implementation."""

    def __init__(
        self,
        cache: BaseCache[K, V],
        data_store: Callable[[K], V],
        refresh_threshold: float = 0.75,
        check_interval: float = 1.0,
    ):
        self.cache = cache
        self.data_store = data_store
        self.refresh_threshold = refresh_threshold
        self.check_interval = check_interval
        self._refresh_times: Dict[K, float] = {}
        self._ttls: Dict[K, float] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            self._check_refresh()
            self._stop_event.wait(self.check_interval)

    def _check_refresh(self) -> None:
        now = time.time()
        with self._lock:
            for key, refresh_time in list(self._refresh_times.items()):
                if now >= refresh_time:
                    ttl = self._ttls.get(key)
                    try:
                        value = self.data_store(key)
                        self.cache.set(key, value, ttl)
                        self._schedule_refresh(key, ttl)
                    except Exception:
                        pass

    def _schedule_refresh(self, key: K, ttl: Optional[float]) -> None:
        if ttl:
            refresh_time = time.time() + (ttl * self.refresh_threshold)
            self._refresh_times[key] = refresh_time
            self._ttls[key] = ttl

    def get(self, key: K, ttl: Optional[float] = None) -> V:
        """Get from cache, schedule refresh if approaching expiry."""
        value = self.cache.get(key)
        if value is None:
            value = self.data_store(key)
            self.cache.set(key, value, ttl)

        self._schedule_refresh(key, ttl)
        return value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set in cache and schedule refresh."""
        self.cache.set(key, value, ttl)
        self._schedule_refresh(key, ttl)

    def stop(self) -> None:
        """Stop the refresh thread."""
        self._stop_event.set()
        self._refresh_thread.join()


# ============================================================================
# Utilities
# ============================================================================


def cached(
    cache: Optional[BaseCache] = None,
    key_fn: Optional[Callable] = None,
    ttl: Optional[float] = None,
):
    """Decorator to cache function results.

    Args:
        cache: Cache instance to use (defaults to LRUCache)
        key_fn: Function to generate cache key from arguments
        ttl: Time-to-live for cached results

    Example:
        @cached(cache=LRUCache(maxsize=100), ttl=300)
        def expensive_function(x, y):
            return x ** y
    """
    if cache is None:
        cache = LRUCache(maxsize=128)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Create deterministic key
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            result = cache.get(cache_key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl)
            return result

        # Attach cache reference for external access
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear

        return wrapper

    return decorator


class CacheManager:
    """Manager for multiple cache instances."""

    def __init__(self):
        self._caches: Dict[str, BaseCache] = {}
        self._lock = threading.RLock()

    def register(self, name: str, cache: BaseCache) -> None:
        """Register a cache with a name."""
        with self._lock:
            self._caches[name] = cache

    def get(self, name: str) -> BaseCache:
        """Get a registered cache by name."""
        with self._lock:
            return self._caches[name]

    def unregister(self, name: str) -> bool:
        """Unregister a cache."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                return True
            return False

    def clear(self, name: Optional[str] = None) -> None:
        """Clear a specific cache or all caches."""
        with self._lock:
            if name:
                if name in self._caches:
                    self._caches[name].clear()
            else:
                for cache in self._caches.values():
                    cache.clear()

    def list_caches(self) -> List[str]:
        """List registered cache names."""
        with self._lock:
            return list(self._caches.keys())

    def stats(self) -> Dict[str, int]:
        """Get size statistics for all caches."""
        with self._lock:
            return {name: cache.size() for name, cache in self._caches.items()}


# Global cache manager instance
_global_manager = CacheManager()


def clear_cache(name: Optional[str] = None) -> None:
    """Clear a cache by name or all caches if no name specified.

    Uses the global cache manager.
    """
    _global_manager.clear(name)


def register_cache(name: str, cache: BaseCache) -> None:
    """Register a cache with the global manager."""
    _global_manager.register(name, cache)


def get_cache(name: str) -> BaseCache:
    """Get a cache from the global manager."""
    return _global_manager.get(name)


# ============================================================================
# Convenience Factory Functions
# ============================================================================


def create_cache(cache_type: str = "lru", **kwargs) -> BaseCache:
    """Factory function to create cache instances.

    Args:
        cache_type: Type of cache (lru, lfu, fifo, ttl, disk, redis, etc.)
        **kwargs: Cache-specific configuration options

    Returns:
        Configured cache instance

    Example:
        cache = create_cache("lru", maxsize=100)
        cache = create_cache("redis", host="localhost", port=6379)
    """
    cache_type = cache_type.lower()

    if cache_type == "lru":
        return LRUCache(**kwargs)
    elif cache_type == "lfu":
        return LFUCache(**kwargs)
    elif cache_type == "fifo":
        return FIFOCache(**kwargs)
    elif cache_type == "ttl":
        return TTLCache(**kwargs)
    elif cache_type == "bounded":
        return BoundedCache(**kwargs)
    elif cache_type == "disk":
        return DiskCache(**kwargs)
    elif cache_type == "file":
        return FileCache(**kwargs)
    elif cache_type == "leveldb":
        return LevelDBCache(**kwargs)
    elif cache_type == "rocksdb":
        return RocksDBCache(**kwargs)
    elif cache_type == "lmdb":
        return LMDBCache(**kwargs)
    elif cache_type == "redis":
        return RedisCache(**kwargs)
    elif cache_type == "memcached":
        return MemcachedCache(**kwargs)
    elif cache_type == "hazelcast":
        return HazelcastCache(**kwargs)
    elif cache_type == "ignite":
        return IgniteCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base classes
    "BaseCache",
    "Cache",
    "Serializer",
    "EvictionPolicy",
    # In-Memory Caches
    "LRUCache",
    "LFUCache",
    "FIFOCache",
    "TTLCache",
    "BoundedCache",
    # Disk Caches
    "DiskCache",
    "FileCache",
    "LevelDBCache",
    "RocksDBCache",
    "LMDBCache",
    # Distributed Caches
    "RedisCache",
    "MemcachedCache",
    "HazelcastCache",
    "IgniteCache",
    # Hierarchical Caches
    "MultiLevelCache",
    "L1L2Cache",
    "CacheHierarchy",
    # Strategies
    "CacheAside",
    "ReadThrough",
    "WriteThrough",
    "WriteBehind",
    "RefreshAhead",
    # Eviction Policies
    "LRUEviction",
    "LFUEviction",
    "RandomEviction",
    "TTLExpiration",
    "SizeEviction",
    # Serializers
    "PickleSerializer",
    "JSONSerializer",
    "MessagePackSerializer",
    "CustomSerializer",
    "StringSerializer",
    # Utilities
    "cached",
    "CacheManager",
    "clear_cache",
    "register_cache",
    "get_cache",
    "create_cache",
]
