"""
Fishstick Database Module - Comprehensive database connectivity and management.

This module provides complete implementations for:
- SQL Databases (PostgreSQL, MySQL, SQLite, SQL Server, Oracle)
- NoSQL Databases (MongoDB, Cassandra, Redis, Elasticsearch, Neo4j)
- Data Lakes (S3, GCS, Azure Blob, HDFS, Delta Lake)
- Vector Databases (Pinecone, Weaviate, Milvus, Chroma, FAISS)
- Feature Stores (Feast, Tecton, SageMaker Feature Store)
- Query builders and execution
- Schema migration tools
- Database utilities
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
    Generator,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import hashlib
import json
import logging
import re
import sqlite3
import uuid
from contextlib import contextmanager
from io import BytesIO, StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class DatabaseError(Exception):
    """Base exception for database errors."""

    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class QueryError(DatabaseError):
    """Raised when query execution fails."""

    pass


class SchemaError(DatabaseError):
    """Raised when schema operation fails."""

    pass


class MigrationError(DatabaseError):
    """Raised when migration operation fails."""

    pass


class VectorDBError(DatabaseError):
    """Raised when vector database operation fails."""

    pass


# =============================================================================
# SQL Databases
# =============================================================================


@dataclass
class ConnectionConfig:
    """Configuration for database connections."""

    host: str = "localhost"
    port: int = 5432
    database: str = ""
    username: str = ""
    password: str = ""
    ssl_mode: str = "prefer"
    timeout: int = 30
    pool_size: int = 5
    max_overflow: int = 10
    extra_params: Dict[str, Any] = field(default_factory=dict)


class SQLConnector(ABC):
    """Abstract base class for SQL database connectors."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connection = None
        self._cursor = None

    @abstractmethod
    def connect(self) -> Any:
        """Establish database connection."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute SQL query."""
        pass

    @abstractmethod
    def fetchall(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """Execute query and fetch all results."""
        pass

    @abstractmethod
    def fetchone(self, query: str, params: Optional[Tuple] = None) -> Optional[Tuple]:
        """Execute query and fetch one result."""
        pass

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connection is not None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class PostgreSQL(SQLConnector):
    """PostgreSQL database connector."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        super().__init__(config or ConnectionConfig(port=5432))
        self._engine = None

    def connect(self) -> Any:
        """Establish PostgreSQL connection."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            self._connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                sslmode=self.config.ssl_mode,
                connect_timeout=self.config.timeout,
                **self.config.extra_params,
            )
            self._cursor = self._connection.cursor(cursor_factory=RealDictCursor)
            logger.info(
                f"Connected to PostgreSQL at {self.config.host}:{self.config.port}"
            )
            return self._connection
        except ImportError:
            raise ConnectionError("psycopg2 is required for PostgreSQL support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def disconnect(self) -> None:
        """Close PostgreSQL connection."""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("PostgreSQL connection closed")

    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute SQL query."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise QueryError(f"Query execution failed: {e}")

    def fetchall(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute query and fetch all results as dictionaries."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            return [dict(row) for row in self._cursor.fetchall()]
        except Exception as e:
            raise QueryError(f"Query fetchall failed: {e}")

    def fetchone(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one result."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            row = self._cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            raise QueryError(f"Query fetchone failed: {e}")

    def copy_from(
        self,
        file_obj: Any,
        table: str,
        columns: Optional[List[str]] = None,
        sep: str = ",",
    ) -> int:
        """Copy data from file-like object to table."""
        if not self.is_connected():
            self.connect()
        try:
            cols = ",".join(columns) if columns else ""
            self._cursor.copy_from(file_obj, table, columns=columns, sep=sep)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise QueryError(f"Copy operation failed: {e}")


class MySQL(SQLConnector):
    """MySQL database connector."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        super().__init__(config or ConnectionConfig(port=3306))

    def connect(self) -> Any:
        """Establish MySQL connection."""
        try:
            import mysql.connector
            from mysql.connector import Error

            self._connection = mysql.connector.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                connection_timeout=self.config.timeout,
                pool_size=self.config.pool_size,
                **self.config.extra_params,
            )
            self._cursor = self._connection.cursor(dictionary=True)
            logger.info(f"Connected to MySQL at {self.config.host}:{self.config.port}")
            return self._connection
        except ImportError:
            raise ConnectionError(
                "mysql-connector-python is required for MySQL support"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MySQL: {e}")

    def disconnect(self) -> None:
        """Close MySQL connection."""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("MySQL connection closed")

    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute SQL query."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise QueryError(f"Query execution failed: {e}")

    def fetchall(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute query and fetch all results."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            return self._cursor.fetchall()
        except Exception as e:
            raise QueryError(f"Query fetchall failed: {e}")

    def fetchone(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one result."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            return self._cursor.fetchone()
        except Exception as e:
            raise QueryError(f"Query fetchone failed: {e}")

    def executemany(self, query: str, params_list: List[Tuple]) -> int:
        """Execute query with multiple parameter sets."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.executemany(query, params_list)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise QueryError(f"Executemany failed: {e}")


class SQLite(SQLConnector):
    """SQLite database connector."""

    def __init__(self, database_path: str = ":memory:"):
        config = ConnectionConfig(database=database_path)
        super().__init__(config)
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        """Establish SQLite connection."""
        try:
            self._connection = sqlite3.connect(
                self.database_path, timeout=self.config.timeout
            )
            self._connection.row_factory = sqlite3.Row
            self._cursor = self._connection.cursor()
            logger.info(f"Connected to SQLite database: {self.database_path}")
            return self._connection
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQLite: {e}")

    def disconnect(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("SQLite connection closed")

    def execute(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute SQL query."""
        if not self.is_connected():
            self.connect()
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            raise QueryError(f"Query execution failed: {e}")

    def fetchall(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute query and fetch all results."""
        if not self.is_connected():
            self.connect()
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            rows = self._cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            raise QueryError(f"Query fetchall failed: {e}")

    def fetchone(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one result."""
        if not self.is_connected():
            self.connect()
        try:
            if params:
                self._cursor.execute(query, params)
            else:
                self._cursor.execute(query)
            row = self._cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            raise QueryError(f"Query fetchone failed: {e}")

    def execute_script(self, script: str) -> None:
        """Execute multiple SQL statements."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.executescript(script)
            self._connection.commit()
        except Exception as e:
            raise QueryError(f"Script execution failed: {e}")


class SQLServer(SQLConnector):
    """SQL Server database connector."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        super().__init__(config or ConnectionConfig(port=1433))

    def connect(self) -> Any:
        """Establish SQL Server connection."""
        try:
            import pyodbc

            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config.host},{self.config.port};"
                f"DATABASE={self.config.database};"
                f"UID={self.config.username};"
                f"PWD={self.config.password};"
                f"Timeout={self.config.timeout}"
            )
            self._connection = pyodbc.connect(conn_str)
            self._cursor = self._connection.cursor()
            logger.info(
                f"Connected to SQL Server at {self.config.host}:{self.config.port}"
            )
            return self._connection
        except ImportError:
            raise ConnectionError("pyodbc is required for SQL Server support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQL Server: {e}")

    def disconnect(self) -> None:
        """Close SQL Server connection."""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("SQL Server connection closed")

    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute SQL query."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise QueryError(f"Query execution failed: {e}")

    def fetchall(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute query and fetch all results."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            columns = [column[0] for column in self._cursor.description]
            return [dict(zip(columns, row)) for row in self._cursor.fetchall()]
        except Exception as e:
            raise QueryError(f"Query fetchall failed: {e}")

    def fetchone(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one result."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            row = self._cursor.fetchone()
            if row:
                columns = [column[0] for column in self._cursor.description]
                return dict(zip(columns, row))
            return None
        except Exception as e:
            raise QueryError(f"Query fetchone failed: {e}")


class Oracle(SQLConnector):
    """Oracle database connector."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        super().__init__(config or ConnectionConfig(port=1521))

    def connect(self) -> Any:
        """Establish Oracle connection."""
        try:
            import cx_Oracle

            dsn = cx_Oracle.makedsn(
                self.config.host, self.config.port, service_name=self.config.database
            )
            self._connection = cx_Oracle.connect(
                user=self.config.username, password=self.config.password, dsn=dsn
            )
            self._cursor = self._connection.cursor()
            logger.info(f"Connected to Oracle at {self.config.host}:{self.config.port}")
            return self._connection
        except ImportError:
            raise ConnectionError("cx_Oracle is required for Oracle support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Oracle: {e}")

    def disconnect(self) -> None:
        """Close Oracle connection."""
        if self._cursor:
            self._cursor.close()
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Oracle connection closed")

    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute SQL query."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            self._connection.commit()
            return self._cursor.rowcount
        except Exception as e:
            self._connection.rollback()
            raise QueryError(f"Query execution failed: {e}")

    def fetchall(
        self, query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute query and fetch all results."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            columns = [col[0] for col in self._cursor.description]
            return [dict(zip(columns, row)) for row in self._cursor.fetchall()]
        except Exception as e:
            raise QueryError(f"Query fetchall failed: {e}")

    def fetchone(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one result."""
        if not self.is_connected():
            self.connect()
        try:
            self._cursor.execute(query, params)
            row = self._cursor.fetchone()
            if row:
                columns = [col[0] for col in self._cursor.description]
                return dict(zip(columns, row))
            return None
        except Exception as e:
            raise QueryError(f"Query fetchone failed: {e}")


# =============================================================================
# NoSQL Databases
# =============================================================================


class MongoDBConnector:
    """MongoDB connector."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_database: str = "admin",
    ):
        self.host = host
        self.port = port
        self.database_name = database
        self.username = username
        self.password = password
        self.auth_database = auth_database
        self._client = None
        self._database = None

    def connect(self) -> Any:
        """Establish MongoDB connection."""
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi

            uri = f"mongodb://{self.host}:{self.port}"
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.auth_database}"

            self._client = MongoClient(uri, server_api=ServerApi("1"))
            self._database = self._client[self.database_name]
            logger.info(f"Connected to MongoDB at {self.host}:{self.port}")
            return self._client
        except ImportError:
            raise ConnectionError("pymongo is required for MongoDB support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {e}")

    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def get_collection(self, collection_name: str) -> Any:
        """Get a collection reference."""
        if not self.is_connected():
            self.connect()
        return self._database[collection_name]

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a single document."""
        coll = self.get_collection(collection)
        result = coll.insert_one(document)
        return str(result.inserted_id)

    def insert_many(
        self, collection: str, documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert multiple documents."""
        coll = self.get_collection(collection)
        result = coll.insert_many(documents)
        return [str(id) for id in result.inserted_ids]

    def find_one(
        self, collection: str, query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find one document matching query."""
        coll = self.get_collection(collection)
        return coll.find_one(query)

    def find(
        self, collection: str, query: Dict[str, Any] = None, limit: int = 0
    ) -> List[Dict[str, Any]]:
        """Find documents matching query."""
        coll = self.get_collection(collection)
        cursor = coll.find(query or {})
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)

    def update_one(
        self, collection: str, query: Dict[str, Any], update: Dict[str, Any]
    ) -> int:
        """Update one document."""
        coll = self.get_collection(collection)
        result = coll.update_one(query, {"$set": update})
        return result.modified_count

    def delete_one(self, collection: str, query: Dict[str, Any]) -> int:
        """Delete one document."""
        coll = self.get_collection(collection)
        result = coll.delete_one(query)
        return result.deleted_count

    def aggregate(
        self, collection: str, pipeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute aggregation pipeline."""
        coll = self.get_collection(collection)
        return list(coll.aggregate(pipeline))


class CassandraConnector:
    """Cassandra connector."""

    def __init__(
        self,
        hosts: List[str],
        keyspace: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 9042,
    ):
        self.hosts = hosts
        self.keyspace = keyspace
        self.username = username
        self.password = password
        self.port = port
        self._cluster = None
        self._session = None

    def connect(self) -> Any:
        """Establish Cassandra connection."""
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider

            auth_provider = None
            if self.username and self.password:
                auth_provider = PlainTextAuthProvider(
                    username=self.username, password=self.password
                )

            self._cluster = Cluster(
                self.hosts, port=self.port, auth_provider=auth_provider
            )
            self._session = self._cluster.connect(self.keyspace)
            logger.info(f"Connected to Cassandra cluster at {', '.join(self.hosts)}")
            return self._session
        except ImportError:
            raise ConnectionError("cassandra-driver is required for Cassandra support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Cassandra: {e}")

    def disconnect(self) -> None:
        """Close Cassandra connection."""
        if self._session:
            self._session.shutdown()
        if self._cluster:
            self._cluster.shutdown()
            self._cluster = None
            logger.info("Cassandra connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._session is not None

    def execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute CQL query."""
        if not self.is_connected():
            self.connect()
        return self._session.execute(query, params)

    def execute_async(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute CQL query asynchronously."""
        if not self.is_connected():
            self.connect()
        return self._session.execute_async(query, params)

    def prepare(self, query: str) -> Any:
        """Prepare a CQL statement."""
        if not self.is_connected():
            self.connect()
        return self._session.prepare(query)


class RedisConnector:
    """Redis connector."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ssl: bool = False,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ssl = ssl
        self._client = None

    def connect(self) -> Any:
        """Establish Redis connection."""
        try:
            import redis

            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                ssl=self.ssl,
                decode_responses=True,
            )
            self._client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return self._client
        except ImportError:
            raise ConnectionError("redis is required for Redis support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        if self._client:
            try:
                self._client.ping()
                return True
            except:
                return False
        return False

    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        if not self.is_connected():
            self.connect()
        return self._client.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value pair."""
        if not self.is_connected():
            self.connect()
        return self._client.set(key, value, ex=ex)

    def delete(self, key: str) -> int:
        """Delete key."""
        if not self.is_connected():
            self.connect()
        return self._client.delete(key)

    def exists(self, key: str) -> int:
        """Check if key exists."""
        if not self.is_connected():
            self.connect()
        return self._client.exists(key)

    def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field."""
        if not self.is_connected():
            self.connect()
        return self._client.hget(key, field)

    def hset(self, key: str, field: str, value: str) -> int:
        """Set hash field."""
        if not self.is_connected():
            self.connect()
        return self._client.hset(key, field, value)

    def lpush(self, key: str, *values) -> int:
        """Push to list."""
        if not self.is_connected():
            self.connect()
        return self._client.lpush(key, *values)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        """Get list range."""
        if not self.is_connected():
            self.connect()
        return self._client.lrange(key, start, end)

    def publish(self, channel: str, message: str) -> int:
        """Publish message to channel."""
        if not self.is_connected():
            self.connect()
        return self._client.publish(channel, message)

    def subscribe(self, *channels: str) -> Any:
        """Subscribe to channels."""
        if not self.is_connected():
            self.connect()
        pubsub = self._client.pubsub()
        pubsub.subscribe(*channels)
        return pubsub


class ElasticsearchConnector:
    """Elasticsearch connector."""

    def __init__(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        cloud_id: Optional[str] = None,
    ):
        self.hosts = hosts
        self.username = username
        self.password = password
        self.api_key = api_key
        self.cloud_id = cloud_id
        self._client = None

    def connect(self) -> Any:
        """Establish Elasticsearch connection."""
        try:
            from elasticsearch import Elasticsearch

            if self.cloud_id:
                self._client = Elasticsearch(
                    cloud_id=self.cloud_id,
                    api_key=self.api_key,
                    basic_auth=(self.username, self.password)
                    if self.username
                    else None,
                )
            else:
                self._client = Elasticsearch(
                    self.hosts,
                    basic_auth=(self.username, self.password)
                    if self.username
                    else None,
                    api_key=self.api_key,
                )
            logger.info(f"Connected to Elasticsearch at {', '.join(self.hosts)}")
            return self._client
        except ImportError:
            raise ConnectionError("elasticsearch is required for Elasticsearch support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Elasticsearch: {e}")

    def disconnect(self) -> None:
        """Close Elasticsearch connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Elasticsearch connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        if self._client:
            return self._client.ping()
        return False

    def index(
        self, index: str, document: Dict[str, Any], doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Index a document."""
        if not self.is_connected():
            self.connect()
        return self._client.index(index=index, id=doc_id, document=document)

    def get(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        if not self.is_connected():
            self.connect()
        try:
            result = self._client.get(index=index, id=doc_id)
            return result["_source"]
        except Exception:
            return None

    def search(
        self, index: str, query: Dict[str, Any], size: int = 10
    ) -> Dict[str, Any]:
        """Search documents."""
        if not self.is_connected():
            self.connect()
        return self._client.search(index=index, query=query, size=size)

    def delete(self, index: str, doc_id: str) -> bool:
        """Delete document."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.delete(index=index, id=doc_id)
            return True
        except Exception:
            return False

    def create_index(
        self, index: str, mappings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create index with optional mappings."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.indices.create(index=index, mappings=mappings)
            return True
        except Exception:
            return False

    def delete_index(self, index: str) -> bool:
        """Delete index."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.indices.delete(index=index)
            return True
        except Exception:
            return False


class Neo4jConnector:
    """Neo4j graph database connector."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    def connect(self) -> Any:
        """Establish Neo4j connection."""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
            return self._driver
        except ImportError:
            raise ConnectionError("neo4j is required for Neo4j support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        if self._driver:
            try:
                self._driver.verify_connectivity()
                return True
            except:
                return False
        return False

    def run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute Cypher query."""
        if not self.is_connected():
            self.connect()
        with self._driver.session(database=self.database) as session:
            return session.run(query, parameters or {})

    def run_transaction(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute Cypher query in transaction."""
        if not self.is_connected():
            self.connect()
        with self._driver.session(database=self.database) as session:
            return session.execute_write(lambda tx: tx.run(query, parameters or {}))

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node."""
        query = f"CREATE (n:{label} $props) RETURN n"
        result = self.run(query, {"props": properties})
        return result.single()

    def create_relationship(
        self,
        from_label: str,
        from_props: Dict[str, Any],
        to_label: str,
        to_props: Dict[str, Any],
        rel_type: str,
        rel_props: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a relationship between nodes."""
        query = f"""
        MATCH (a:{from_label}), (b:{to_label})
        WHERE a.id = $from_id AND b.id = $to_id
        CREATE (a)-[r:{rel_type} $rel_props]->(b)
        RETURN r
        """
        return self.run(
            query,
            {
                "from_id": from_props.get("id"),
                "to_id": to_props.get("id"),
                "rel_props": rel_props or {},
            },
        )


# =============================================================================
# Data Lakes
# =============================================================================


class S3Connector:
    """Amazon S3 connector."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        self.bucket = bucket
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self._client = None

    def connect(self) -> Any:
        """Establish S3 connection."""
        try:
            import boto3
            from botocore.config import Config

            session = boto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
            )

            config = Config(retries={"max_attempts": 3, "mode": "standard"})
            self._client = session.client(
                "s3", config=config, endpoint_url=self.endpoint_url
            )
            logger.info(f"Connected to S3 bucket: {self.bucket}")
            return self._client
        except ImportError:
            raise ConnectionError("boto3 is required for S3 support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to S3: {e}")

    def disconnect(self) -> None:
        """Close S3 connection."""
        self._client = None
        logger.info("S3 connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def upload_file(
        self, local_path: str, key: str, metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload file to S3."""
        if not self.is_connected():
            self.connect()
        try:
            extra_args = {"Metadata": metadata} if metadata else {}
            self._client.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args)
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False

    def download_file(self, key: str, local_path: str) -> bool:
        """Download file from S3."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.download_file(self.bucket, key, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False

    def upload_bytes(
        self, data: bytes, key: str, metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload bytes to S3."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.put_object(
                Bucket=self.bucket, Key=key, Body=data, Metadata=metadata or {}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload bytes to S3: {e}")
            return False

    def download_bytes(self, key: str) -> Optional[bytes]:
        """Download bytes from S3."""
        if not self.is_connected():
            self.connect()
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Failed to download bytes from S3: {e}")
            return None

    def list_objects(
        self, prefix: str = "", max_keys: int = 1000
    ) -> List[Dict[str, Any]]:
        """List objects in S3 bucket."""
        if not self.is_connected():
            self.connect()
        try:
            response = self._client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix, MaxKeys=max_keys
            )
            return response.get("Contents", [])
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []

    def delete_object(self, key: str) -> bool:
        """Delete object from S3."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return False

    def object_exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False


class GCSConnector:
    """Google Cloud Storage connector."""

    def __init__(
        self,
        bucket: str,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
    ):
        self.bucket = bucket
        self.project = project
        self.credentials_path = credentials_path
        self._client = None
        self._bucket = None

    def connect(self) -> Any:
        """Establish GCS connection."""
        try:
            from google.cloud import storage

            if self.credentials_path:
                self._client = storage.Client.from_service_account_json(
                    self.credentials_path, project=self.project
                )
            else:
                self._client = storage.Client(project=self.project)

            self._bucket = self._client.bucket(self.bucket)
            logger.info(f"Connected to GCS bucket: {self.bucket}")
            return self._client
        except ImportError:
            raise ConnectionError("google-cloud-storage is required for GCS support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GCS: {e}")

    def disconnect(self) -> None:
        """Close GCS connection."""
        self._client = None
        self._bucket = None
        logger.info("GCS connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def upload_file(
        self, local_path: str, blob_name: str, metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload file to GCS."""
        if not self.is_connected():
            self.connect()
        try:
            blob = self._bucket.blob(blob_name)
            if metadata:
                blob.metadata = metadata
            blob.upload_from_filename(local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to GCS: {e}")
            return False

    def download_file(self, blob_name: str, local_path: str) -> bool:
        """Download file from GCS."""
        if not self.is_connected():
            self.connect()
        try:
            blob = self._bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download file from GCS: {e}")
            return False

    def upload_bytes(
        self, data: bytes, blob_name: str, metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload bytes to GCS."""
        if not self.is_connected():
            self.connect()
        try:
            blob = self._bucket.blob(blob_name)
            if metadata:
                blob.metadata = metadata
            blob.upload_from_string(data)
            return True
        except Exception as e:
            logger.error(f"Failed to upload bytes to GCS: {e}")
            return False

    def download_bytes(self, blob_name: str) -> Optional[bytes]:
        """Download bytes from GCS."""
        if not self.is_connected():
            self.connect()
        try:
            blob = self._bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            logger.error(f"Failed to download bytes from GCS: {e}")
            return None

    def list_blobs(self, prefix: str = "") -> List[Any]:
        """List blobs in GCS bucket."""
        if not self.is_connected():
            self.connect()
        try:
            return list(self._bucket.list_blobs(prefix=prefix))
        except Exception as e:
            logger.error(f"Failed to list GCS blobs: {e}")
            return []

    def delete_blob(self, blob_name: str) -> bool:
        """Delete blob from GCS."""
        if not self.is_connected():
            self.connect()
        try:
            blob = self._bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Failed to delete GCS blob: {e}")
            return False

    def blob_exists(self, blob_name: str) -> bool:
        """Check if blob exists in GCS."""
        if not self.is_connected():
            self.connect()
        try:
            blob = self._bucket.blob(blob_name)
            return blob.exists()
        except:
            return False


class AzureBlob:
    """Azure Blob Storage connector."""

    def __init__(
        self,
        account_name: str,
        container: str,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
    ):
        self.account_name = account_name
        self.container = container
        self.account_key = account_key
        self.connection_string = connection_string
        self._client = None
        self._container_client = None

    def connect(self) -> Any:
        """Establish Azure Blob connection."""
        try:
            from azure.storage.blob import BlobServiceClient

            if self.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            else:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url, credential=self.account_key
                )

            self._container_client = self._client.get_container_client(self.container)
            logger.info(f"Connected to Azure Blob container: {self.container}")
            return self._client
        except ImportError:
            raise ConnectionError(
                "azure-storage-blob is required for Azure Blob support"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Azure Blob: {e}")

    def disconnect(self) -> None:
        """Close Azure Blob connection."""
        if self._client:
            self._client.close()
        self._client = None
        self._container_client = None
        logger.info("Azure Blob connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def upload_file(
        self, local_path: str, blob_name: str, metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload file to Azure Blob."""
        if not self.is_connected():
            self.connect()
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True, metadata=metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to Azure Blob: {e}")
            return False

    def download_file(self, blob_name: str, local_path: str) -> bool:
        """Download file from Azure Blob."""
        if not self.is_connected():
            self.connect()
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            return True
        except Exception as e:
            logger.error(f"Failed to download file from Azure Blob: {e}")
            return False

    def upload_bytes(
        self, data: bytes, blob_name: str, metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """Upload bytes to Azure Blob."""
        if not self.is_connected():
            self.connect()
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True, metadata=metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to upload bytes to Azure Blob: {e}")
            return False

    def download_bytes(self, blob_name: str) -> Optional[bytes]:
        """Download bytes from Azure Blob."""
        if not self.is_connected():
            self.connect()
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            logger.error(f"Failed to download bytes from Azure Blob: {e}")
            return None

    def list_blobs(self, prefix: str = "") -> List[Any]:
        """List blobs in Azure container."""
        if not self.is_connected():
            self.connect()
        try:
            return list(self._container_client.list_blobs(name_starts_with=prefix))
        except Exception as e:
            logger.error(f"Failed to list Azure blobs: {e}")
            return []

    def delete_blob(self, blob_name: str) -> bool:
        """Delete blob from Azure."""
        if not self.is_connected():
            self.connect()
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            return True
        except Exception as e:
            logger.error(f"Failed to delete Azure blob: {e}")
            return False

    def blob_exists(self, blob_name: str) -> bool:
        """Check if blob exists in Azure."""
        if not self.is_connected():
            self.connect()
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            return blob_client.exists()
        except:
            return False


class HDFSConnector:
    """Hadoop Distributed File System connector."""

    def __init__(
        self, host: str = "localhost", port: int = 9000, user: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.user = user
        self._client = None

    def connect(self) -> Any:
        """Establish HDFS connection."""
        try:
            from hdfs import InsecureClient

            url = f"http://{self.host}:{self.port}"
            self._client = InsecureClient(url, user=self.user)
            self._client.status("/")
            logger.info(f"Connected to HDFS at {self.host}:{self.port}")
            return self._client
        except ImportError:
            raise ConnectionError("hdfs is required for HDFS support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to HDFS: {e}")

    def disconnect(self) -> None:
        """Close HDFS connection."""
        self._client = None
        logger.info("HDFS connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def upload_file(self, local_path: str, hdfs_path: str) -> bool:
        """Upload file to HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.upload(hdfs_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to upload file to HDFS: {e}")
            return False

    def download_file(self, hdfs_path: str, local_path: str) -> bool:
        """Download file from HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.download(hdfs_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download file from HDFS: {e}")
            return False

    def write(self, hdfs_path: str, data: Union[str, bytes]) -> bool:
        """Write data to HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            mode = "wb" if isinstance(data, bytes) else "w"
            with self._client.write(
                hdfs_path, encoding=None if isinstance(data, bytes) else "utf-8"
            ) as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error(f"Failed to write to HDFS: {e}")
            return False

    def read(self, hdfs_path: str) -> Optional[bytes]:
        """Read data from HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            with self._client.read(hdfs_path) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read from HDFS: {e}")
            return None

    def list(self, hdfs_path: str) -> List[Dict[str, Any]]:
        """List directory in HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            return self._client.list(hdfs_path, status=True)
        except Exception as e:
            logger.error(f"Failed to list HDFS directory: {e}")
            return []

    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """Delete file or directory from HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.delete(hdfs_path, recursive=recursive)
            return True
        except Exception as e:
            logger.error(f"Failed to delete from HDFS: {e}")
            return False

    def exists(self, hdfs_path: str) -> bool:
        """Check if path exists in HDFS."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.status(hdfs_path)
            return True
        except:
            return False


class DeltaLake:
    """Delta Lake connector."""

    def __init__(
        self, storage_path: str, storage_options: Optional[Dict[str, str]] = None
    ):
        self.storage_path = storage_path
        self.storage_options = storage_options or {}

    def write_table(
        self,
        data: Any,
        table_path: str,
        mode: str = "overwrite",
        partition_by: Optional[List[str]] = None,
    ) -> bool:
        """Write data to Delta table."""
        try:
            import deltalake
            from deltalake import write_deltalake

            full_path = f"{self.storage_path}/{table_path}"
            write_deltalake(
                full_path,
                data,
                mode=mode,
                partition_by=partition_by,
                storage_options=self.storage_options,
            )
            logger.info(f"Wrote Delta table to {full_path}")
            return True
        except ImportError:
            raise ConnectionError("deltalake is required for Delta Lake support")
        except Exception as e:
            logger.error(f"Failed to write Delta table: {e}")
            return False

    def read_table(self, table_path: str) -> Any:
        """Read Delta table."""
        try:
            from deltalake import DeltaTable

            full_path = f"{self.storage_path}/{table_path}"
            dt = DeltaTable(full_path, storage_options=self.storage_options)
            return dt.to_pandas()
        except ImportError:
            raise ConnectionError("deltalake is required for Delta Lake support")
        except Exception as e:
            logger.error(f"Failed to read Delta table: {e}")
            return None

    def get_version(self, table_path: str) -> int:
        """Get current version of Delta table."""
        try:
            from deltalake import DeltaTable

            full_path = f"{self.storage_path}/{table_path}"
            dt = DeltaTable(full_path, storage_options=self.storage_options)
            return dt.version()
        except Exception as e:
            logger.error(f"Failed to get Delta table version: {e}")
            return -1

    def history(self, table_path: str) -> List[Dict[str, Any]]:
        """Get history of Delta table."""
        try:
            from deltalake import DeltaTable

            full_path = f"{self.storage_path}/{table_path}"
            dt = DeltaTable(full_path, storage_options=self.storage_options)
            return dt.history()
        except Exception as e:
            logger.error(f"Failed to get Delta table history: {e}")
            return []

    def vacuum(self, table_path: str, retention_hours: int = 168) -> List[str]:
        """Vacuum old files from Delta table."""
        try:
            from deltalake import DeltaTable

            full_path = f"{self.storage_path}/{table_path}"
            dt = DeltaTable(full_path, storage_options=self.storage_options)
            return dt.vacuum(retention_hours=retention_hours)
        except Exception as e:
            logger.error(f"Failed to vacuum Delta table: {e}")
            return []


# =============================================================================
# Vector Databases
# =============================================================================


@dataclass
class VectorQueryResult:
    """Result from vector similarity search."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


class PineconeConnector:
    """Pinecone vector database connector."""

    def __init__(self, api_key: str, environment: str = "us-west1-gcp"):
        self.api_key = api_key
        self.environment = environment
        self._index = None
        self._index_name = None

    def connect(self, index_name: str) -> Any:
        """Connect to Pinecone index."""
        try:
            import pinecone

            pinecone.init(api_key=self.api_key, environment=self.environment)
            self._index = pinecone.Index(index_name)
            self._index_name = index_name
            logger.info(f"Connected to Pinecone index: {index_name}")
            return self._index
        except ImportError:
            raise ConnectionError("pinecone-client is required for Pinecone support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")

    def disconnect(self) -> None:
        """Close Pinecone connection."""
        self._index = None
        self._index_name = None
        logger.info("Pinecone connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._index is not None

    def upsert(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        namespace: str = "",
    ) -> bool:
        """Upsert vectors to Pinecone."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Pinecone")
        try:
            self._index.upsert(vectors=vectors, namespace=namespace)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return False

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorQueryResult]:
        """Query Pinecone for similar vectors."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Pinecone")
        try:
            results = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True,
            )
            return [
                VectorQueryResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {}),
                )
                for match in results["matches"]
            ]
        except Exception as e:
            logger.error(f"Failed to query Pinecone: {e}")
            return []

    def delete(self, ids: List[str], namespace: str = "") -> bool:
        """Delete vectors from Pinecone."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Pinecone")
        try:
            self._index.delete(ids=ids, namespace=namespace)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def fetch(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """Fetch vectors by ID."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Pinecone")
        try:
            return self._index.fetch(ids=ids, namespace=namespace)
        except Exception as e:
            logger.error(f"Failed to fetch vectors: {e}")
            return {}


class WeaviateConnector:
    """Weaviate vector database connector."""

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        auth_client_secret: Optional[Any] = None,
    ):
        self.url = url
        self.api_key = api_key
        self.auth_client_secret = auth_client_secret
        self._client = None

    def connect(self) -> Any:
        """Establish Weaviate connection."""
        try:
            import weaviate

            if self.auth_client_secret:
                self._client = weaviate.Client(
                    url=self.url, auth_client_secret=self.auth_client_secret
                )
            elif self.api_key:
                self._client = weaviate.Client(
                    url=self.url, additional_headers={"X-OpenAI-Api-Key": self.api_key}
                )
            else:
                self._client = weaviate.Client(url=self.url)

            logger.info(f"Connected to Weaviate at {self.url}")
            return self._client
        except ImportError:
            raise ConnectionError("weaviate-client is required for Weaviate support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")

    def disconnect(self) -> None:
        """Close Weaviate connection."""
        self._client = None
        logger.info("Weaviate connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        if self._client:
            try:
                self._client.is_live()
                return True
            except:
                return False
        return False

    def create_schema(
        self,
        class_name: str,
        properties: List[Dict[str, Any]],
        vectorizer: str = "none",
    ) -> bool:
        """Create schema in Weaviate."""
        if not self.is_connected():
            self.connect()
        try:
            schema = {
                "class": class_name,
                "vectorizer": vectorizer,
                "properties": properties,
            }
            self._client.schema.create_class(schema)
            return True
        except Exception as e:
            logger.error(f"Failed to create Weaviate schema: {e}")
            return False

    def insert_object(
        self,
        class_name: str,
        properties: Dict[str, Any],
        vector: Optional[List[float]] = None,
        uuid: Optional[str] = None,
    ) -> bool:
        """Insert object into Weaviate."""
        if not self.is_connected():
            self.connect()
        try:
            data_object = {"class": class_name, "properties": properties}
            if vector:
                data_object["vector"] = vector
            if uuid:
                data_object["id"] = uuid

            self._client.data_object.create(**data_object)
            return True
        except Exception as e:
            logger.error(f"Failed to insert Weaviate object: {e}")
            return False

    def query_near_vector(
        self, class_name: str, vector: List[float], limit: int = 10
    ) -> List[VectorQueryResult]:
        """Query Weaviate using nearVector."""
        if not self.is_connected():
            self.connect()
        try:
            result = (
                self._client.query.get(class_name, ["*"])
                .with_near_vector({"vector": vector})
                .with_limit(limit)
                .with_additional(["id", "certainty"])
                .do()
            )

            objects = result["data"]["Get"][class_name]
            return [
                VectorQueryResult(
                    id=obj["_additional"]["id"],
                    score=obj["_additional"]["certainty"],
                    metadata={k: v for k, v in obj.items() if k != "_additional"},
                )
                for obj in objects
            ]
        except Exception as e:
            logger.error(f"Failed to query Weaviate: {e}")
            return []


class MilvusConnector:
    """Milvus vector database connector."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self._connections = None

    def connect(self) -> Any:
        """Establish Milvus connection."""
        try:
            from pymilvus import connections

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
            )
            self._connections = connections
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return connections
        except ImportError:
            raise ConnectionError("pymilvus is required for Milvus support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")

    def disconnect(self) -> None:
        """Close Milvus connection."""
        if self._connections:
            self._connections.disconnect("default")
            self._connections = None
            logger.info("Milvus connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        try:
            from pymilvus import utility

            utility.list_collections()
            return True
        except:
            return False

    def create_collection(
        self, name: str, dim: int, fields: List[Dict[str, Any]], auto_id: bool = True
    ) -> Any:
        """Create collection in Milvus."""
        try:
            from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

            field_schemas = []
            for field in fields:
                field_schemas.append(
                    FieldSchema(
                        name=field["name"],
                        dtype=getattr(DataType, field["dtype"]),
                        is_primary=field.get("is_primary", False),
                        auto_id=field.get("auto_id", False),
                    )
                )

            schema = CollectionSchema(
                fields=field_schemas, description=f"Collection {name}"
            )
            collection = Collection(name=name, schema=schema)

            # Create index on vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(field_name="embedding", index_params=index_params)

            return collection
        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {e}")
            return None

    def insert(self, collection_name: str, entities: List[List[Any]]) -> bool:
        """Insert entities into Milvus collection."""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)
            collection.insert(entities)
            return True
        except Exception as e:
            logger.error(f"Failed to insert into Milvus: {e}")
            return False

    def search(
        self, collection_name: str, vectors: List[List[float]], top_k: int = 10
    ) -> List[List[VectorQueryResult]]:
        """Search Milvus collection."""
        try:
            from pymilvus import Collection

            collection = Collection(collection_name)
            collection.load()

            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            results = collection.search(
                data=vectors, anns_field="embedding", param=search_params, limit=top_k
            )

            all_results = []
            for hits in results:
                query_results = [
                    VectorQueryResult(
                        id=str(hit.id),
                        score=hit.distance,
                    )
                    for hit in hits
                ]
                all_results.append(query_results)

            return all_results
        except Exception as e:
            logger.error(f"Failed to search Milvus: {e}")
            return []


class ChromaConnector:
    """Chroma vector database connector."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        persist_directory: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.persist_directory = persist_directory
        self._client = None

    def connect(self) -> Any:
        """Establish Chroma connection."""
        try:
            import chromadb
            from chromadb.config import Settings

            if self.host and self.port:
                # Client-server mode
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            elif self.persist_directory:
                # Persistent client
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            else:
                # Ephemeral client
                self._client = chromadb.Client()

            logger.info("Connected to Chroma")
            return self._client
        except ImportError:
            raise ConnectionError("chromadb is required for Chroma support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Chroma: {e}")

    def disconnect(self) -> None:
        """Close Chroma connection."""
        self._client = None
        logger.info("Chroma connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
    ) -> Any:
        """Create collection in Chroma."""
        if not self.is_connected():
            self.connect()
        try:
            return self._client.create_collection(
                name=name, metadata=metadata, embedding_function=embedding_function
            )
        except Exception as e:
            logger.error(f"Failed to create Chroma collection: {e}")
            return None

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> bool:
        """Add documents to Chroma collection."""
        if not self.is_connected():
            self.connect()
        try:
            collection = self._client.get_or_create_collection(collection_name)
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            collection.add(
                documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to Chroma: {e}")
            return False

    def query(
        self,
        collection_name: str,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query Chroma collection."""
        if not self.is_connected():
            self.connect()
        try:
            collection = self._client.get_collection(collection_name)
            return collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            logger.error(f"Failed to query Chroma: {e}")
            return {}


class FAISSConnector:
    """FAISS vector database connector (in-memory)."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._index = None
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._id_counter = 0

    def connect(self) -> Any:
        """Initialize FAISS index."""
        try:
            import faiss

            self._index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
            return self._index
        except ImportError:
            raise ConnectionError(
                "faiss-cpu or faiss-gpu is required for FAISS support"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize FAISS: {e}")

    def disconnect(self) -> None:
        """Close FAISS index."""
        self._index = None
        self._metadata = {}
        self._id_counter = 0
        logger.info("FAISS index closed")

    def is_connected(self) -> bool:
        """Check if index is initialized."""
        return self._index is not None

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add vectors to FAISS index."""
        try:
            import numpy as np

            if not self.is_connected():
                self.connect()

            vectors_np = np.array(vectors, dtype=np.float32)
            ids = list(range(self._id_counter, self._id_counter + len(vectors)))
            self._id_counter += len(vectors)

            self._index.add(vectors_np)

            if metadata:
                for i, meta in zip(ids, metadata):
                    self._metadata[i] = meta

            return ids
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS: {e}")
            return []

    def search(
        self, vectors: List[List[float]], k: int = 10
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Search FAISS index."""
        try:
            import numpy as np

            if not self.is_connected():
                self.connect()

            vectors_np = np.array(vectors, dtype=np.float32)
            distances, indices = self._index.search(vectors_np, k)

            return distances.tolist(), indices.tolist()
        except Exception as e:
            logger.error(f"Failed to search FAISS: {e}")
            return [], []

    def save_index(self, filepath: str) -> bool:
        """Save FAISS index to file."""
        try:
            import faiss

            if not self.is_connected():
                return False

            faiss.write_index(self._index, filepath)

            # Save metadata separately
            metadata_path = filepath + ".meta"
            with open(metadata_path, "w") as f:
                json.dump(self._metadata, f)

            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False

    def load_index(self, filepath: str) -> bool:
        """Load FAISS index from file."""
        try:
            import faiss

            self._index = faiss.read_index(filepath)

            # Load metadata
            metadata_path = filepath + ".meta"
            if Path(metadata_path).exists():
                with open(metadata_path, "r") as f:
                    self._metadata = {int(k): v for k, v in json.load(f).items()}

            logger.info(f"Loaded FAISS index from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False


# =============================================================================
# Feature Store
# =============================================================================


class FeastConnector:
    """Feast feature store connector."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self._store = None

    def connect(self) -> Any:
        """Initialize Feast feature store."""
        try:
            from feast import FeatureStore

            self._store = FeatureStore(repo_path=self.repo_path)
            logger.info(f"Connected to Feast feature store at {self.repo_path}")
            return self._store
        except ImportError:
            raise ConnectionError("feast is required for Feast support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Feast: {e}")

    def disconnect(self) -> None:
        """Close Feast connection."""
        self._store = None
        logger.info("Feast connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._store is not None

    def get_features(
        self, entity_rows: List[Dict[str, Any]], feature_refs: List[str]
    ) -> Any:
        """Get features from feature store."""
        if not self.is_connected():
            self.connect()
        try:
            return self._store.get_online_features(
                features=feature_refs, entity_rows=entity_rows
            ).to_dict()
        except Exception as e:
            logger.error(f"Failed to get features from Feast: {e}")
            return None

    def materialize(self, start_date: datetime, end_date: datetime) -> bool:
        """Materialize features to online store."""
        if not self.is_connected():
            self.connect()
        try:
            self._store.materialize(start_date, end_date)
            return True
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")
            return False

    def get_historical_features(self, entity_df: Any, feature_refs: List[str]) -> Any:
        """Get historical features from feature store."""
        if not self.is_connected():
            self.connect()
        try:
            return self._store.get_historical_features(
                entity_df=entity_df, features=feature_refs
            ).to_df()
        except Exception as e:
            logger.error(f"Failed to get historical features: {e}")
            return None


class TectonConnector:
    """Tecton feature store connector."""

    def __init__(self, api_key: str, url: str = "https://tecton.ai"):
        self.api_key = api_key
        self.url = url
        self._workspace = None

    def connect(self, workspace: str = "default") -> Any:
        """Initialize Tecton workspace."""
        try:
            import tecton

            tecton.set_credentials(api_key=self.api_key, url=self.url)
            self._workspace = tecton.get_workspace(workspace)
            logger.info(f"Connected to Tecton workspace: {workspace}")
            return self._workspace
        except ImportError:
            raise ConnectionError("tecton is required for Tecton support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Tecton: {e}")

    def disconnect(self) -> None:
        """Close Tecton connection."""
        self._workspace = None
        logger.info("Tecton connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._workspace is not None

    def get_features(self, feature_service_name: str, join_keys: Dict[str, Any]) -> Any:
        """Get features from feature service."""
        if not self.is_connected():
            raise ConnectionError("Not connected to Tecton")
        try:
            feature_service = self._workspace.get_feature_service(feature_service_name)
            return feature_service.get_features(join_keys).to_pandas()
        except Exception as e:
            logger.error(f"Failed to get features from Tecton: {e}")
            return None


class SageMakerFeatureStore:
    """AWS SageMaker Feature Store connector."""

    def __init__(self, region: str = "us-east-1", role_arn: Optional[str] = None):
        self.region = region
        self.role_arn = role_arn
        self._client = None
        self._runtime_client = None

    def connect(self) -> Any:
        """Establish SageMaker Feature Store connection."""
        try:
            import boto3

            session = boto3.Session(region_name=self.region)
            self._client = session.client("sagemaker-featurestore-runtime")
            self._runtime_client = self._client
            logger.info(f"Connected to SageMaker Feature Store in {self.region}")
            return self._client
        except ImportError:
            raise ConnectionError("boto3 is required for SageMaker support")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SageMaker: {e}")

    def disconnect(self) -> None:
        """Close SageMaker connection."""
        self._client = None
        self._runtime_client = None
        logger.info("SageMaker connection closed")

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._client is not None

    def get_record(
        self,
        feature_group_name: str,
        record_identifier: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get record from feature group."""
        if not self.is_connected():
            self.connect()
        try:
            params = {
                "FeatureGroupName": feature_group_name,
                "RecordIdentifierValueAsString": record_identifier,
            }
            if feature_names:
                params["FeatureNames"] = feature_names

            response = self._client.get_record(**params)
            return {
                f["FeatureName"]: f["ValueAsString"] for f in response.get("Record", [])
            }
        except Exception as e:
            logger.error(f"Failed to get record from SageMaker: {e}")
            return {}

    def put_record(self, feature_group_name: str, record: List[Dict[str, str]]) -> bool:
        """Put record into feature group."""
        if not self.is_connected():
            self.connect()
        try:
            self._client.put_record(FeatureGroupName=feature_group_name, Record=record)
            return True
        except Exception as e:
            logger.error(f"Failed to put record to SageMaker: {e}")
            return False


# =============================================================================
# Query Builders and Execution
# =============================================================================


class SQLQuery:
    """SQL query builder."""

    def __init__(self):
        self._select: List[str] = []
        self._from: Optional[str] = None
        self._where: List[str] = []
        self._params: List[Any] = []
        self._joins: List[str] = []
        self._group_by: List[str] = []
        self._order_by: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None

    def select(self, *columns: str) -> "SQLQuery":
        """Add SELECT clause."""
        self._select.extend(columns)
        return self

    def from_table(self, table: str) -> "SQLQuery":
        """Add FROM clause."""
        self._from = table
        return self

    def where(self, condition: str, *params: Any) -> "SQLQuery":
        """Add WHERE condition."""
        self._where.append(condition)
        self._params.extend(params)
        return self

    def join(self, table: str, on: str, join_type: str = "INNER") -> "SQLQuery":
        """Add JOIN clause."""
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def group_by(self, *columns: str) -> "SQLQuery":
        """Add GROUP BY clause."""
        self._group_by.extend(columns)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> "SQLQuery":
        """Add ORDER BY clause."""
        self._order_by.append(f"{column} {direction}")
        return self

    def limit(self, n: int) -> "SQLQuery":
        """Add LIMIT clause."""
        self._limit = n
        return self

    def offset(self, n: int) -> "SQLQuery":
        """Add OFFSET clause."""
        self._offset = n
        return self

    def build(self) -> Tuple[str, Tuple[Any, ...]]:
        """Build SQL query string."""
        parts = []

        # SELECT
        select_clause = ", ".join(self._select) if self._select else "*"
        parts.append(f"SELECT {select_clause}")

        # FROM
        if self._from:
            parts.append(f"FROM {self._from}")

        # JOINs
        parts.extend(self._joins)

        # WHERE
        if self._where:
            parts.append(f"WHERE {' AND '.join(self._where)}")

        # GROUP BY
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        # ORDER BY
        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")

        # LIMIT
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")

        # OFFSET
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")

        query = " ".join(parts)
        return query, tuple(self._params)


class NoSQLQuery:
    """NoSQL query builder for document databases."""

    def __init__(self):
        self._collection: Optional[str] = None
        self._filter: Dict[str, Any] = {}
        self._projection: Dict[str, Any] = {}
        self._sort: List[Tuple[str, int]] = []
        self._limit: Optional[int] = None
        self._skip: Optional[int] = None

    def collection(self, name: str) -> "NoSQLQuery":
        """Set collection name."""
        self._collection = name
        return self

    def filter(self, **conditions) -> "NoSQLQuery":
        """Add filter conditions."""
        self._filter.update(conditions)
        return self

    def filter_range(
        self, field: str, min_val: Any = None, max_val: Any = None
    ) -> "NoSQLQuery":
        """Add range filter."""
        range_filter = {}
        if min_val is not None:
            range_filter["$gte"] = min_val
        if max_val is not None:
            range_filter["$lte"] = max_val
        if range_filter:
            self._filter[field] = range_filter
        return self

    def project(
        self, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None
    ) -> "NoSQLQuery":
        """Set projection."""
        if include:
            self._projection = {field: 1 for field in include}
        if exclude:
            self._projection.update({field: 0 for field in exclude})
        return self

    def sort(self, field: str, ascending: bool = True) -> "NoSQLQuery":
        """Add sort order."""
        self._sort.append((field, 1 if ascending else -1))
        return self

    def limit(self, n: int) -> "NoSQLQuery":
        """Set limit."""
        self._limit = n
        return self

    def skip(self, n: int) -> "NoSQLQuery":
        """Set skip offset."""
        self._skip = n
        return self

    def build(self) -> Dict[str, Any]:
        """Build query specification."""
        return {
            "collection": self._collection,
            "filter": self._filter,
            "projection": self._projection,
            "sort": self._sort,
            "limit": self._limit,
            "skip": self._skip,
        }


class VectorQuery:
    """Vector similarity query builder."""

    def __init__(self):
        self._vector: Optional[List[float]] = None
        self._top_k: int = 10
        self._filter: Optional[Dict[str, Any]] = None
        self._include_vector: bool = False
        self._include_metadata: bool = True

    def vector(self, vector: List[float]) -> "VectorQuery":
        """Set query vector."""
        self._vector = vector
        return self

    def top_k(self, k: int) -> "VectorQuery":
        """Set number of results."""
        self._top_k = k
        return self

    def filter_by(self, **conditions) -> "VectorQuery":
        """Add metadata filter."""
        self._filter = conditions
        return self

    def include_vector(self, include: bool = True) -> "VectorQuery":
        """Include vectors in results."""
        self._include_vector = include
        return self

    def include_metadata(self, include: bool = True) -> "VectorQuery":
        """Include metadata in results."""
        self._include_metadata = include
        return self

    def build(self) -> Dict[str, Any]:
        """Build query specification."""
        if self._vector is None:
            raise QueryError("Query vector must be set")

        return {
            "vector": self._vector,
            "top_k": self._top_k,
            "filter": self._filter,
            "include_vector": self._include_vector,
            "include_metadata": self._include_metadata,
        }


def execute_query(
    connector: Any, query: Union[str, SQLQuery, NoSQLQuery, VectorQuery], **kwargs
) -> Any:
    """Execute a query on the appropriate connector.

    Args:
        connector: Database connector instance
        query: Query to execute (string or query builder)
        **kwargs: Additional query parameters

    Returns:
        Query results
    """
    if isinstance(query, SQLQuery):
        sql, params = query.build()
        if hasattr(connector, "fetchall"):
            return connector.fetchall(sql, params)
        else:
            raise QueryError(
                f"Connector {type(connector).__name__} does not support SQL"
            )

    elif isinstance(query, NoSQLQuery):
        spec = query.build()
        if hasattr(connector, "find"):
            return connector.find(spec["collection"], spec["filter"])
        else:
            raise QueryError(
                f"Connector {type(connector).__name__} does not support NoSQL"
            )

    elif isinstance(query, VectorQuery):
        spec = query.build()
        if hasattr(connector, "query"):
            return connector.query(
                vector=spec["vector"], top_k=spec["top_k"], filter=spec["filter"]
            )
        else:
            raise QueryError(
                f"Connector {type(connector).__name__} does not support vector queries"
            )

    elif isinstance(query, str):
        # Assume SQL string
        if hasattr(connector, "fetchall"):
            return connector.fetchall(query, kwargs.get("params"))
        else:
            raise QueryError(
                f"Connector {type(connector).__name__} does not support SQL strings"
            )

    else:
        raise QueryError(f"Unknown query type: {type(query)}")


# =============================================================================
# Schema Migration
# =============================================================================


@dataclass
class Migration:
    """Represents a database migration."""

    id: str
    name: str
    version: int
    up_sql: str
    down_sql: str
    applied_at: Optional[datetime] = None


class SchemaMigration:
    """Manages database schema migrations."""

    def __init__(
        self, connector: SQLConnector, migrations_table: str = "schema_migrations"
    ):
        self.connector = connector
        self.migrations_table = migrations_table
        self._migrations: Dict[int, Migration] = {}

    def initialize(self) -> None:
        """Initialize migrations table."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            version INT NOT NULL UNIQUE,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connector.execute(create_table_sql)
        logger.info(f"Initialized migrations table: {self.migrations_table}")

    def register_migration(
        self, name: str, version: int, up_sql: str, down_sql: str
    ) -> Migration:
        """Register a migration."""
        migration = Migration(
            id=str(uuid.uuid4()),
            name=name,
            version=version,
            up_sql=up_sql,
            down_sql=down_sql,
        )
        self._migrations[version] = migration
        return migration

    def get_applied_migrations(self) -> List[int]:
        """Get list of applied migration versions."""
        try:
            result = self.connector.fetchall(
                f"SELECT version FROM {self.migrations_table} ORDER BY version"
            )
            return [row["version"] for row in result]
        except Exception:
            return []

    def migrate_schema(self, target_version: Optional[int] = None) -> List[int]:
        """Apply pending migrations up to target version.

        Args:
            target_version: Target version (None for all)

        Returns:
            List of applied migration versions
        """
        self.initialize()
        applied = set(self.get_applied_migrations())
        pending = sorted([v for v in self._migrations.keys() if v not in applied])

        if target_version:
            pending = [v for v in pending if v <= target_version]

        applied_versions = []
        for version in pending:
            migration = self._migrations[version]
            try:
                # Apply migration
                self.connector.execute(migration.up_sql)

                # Record migration
                self.connector.execute(
                    f"INSERT INTO {self.migrations_table} (id, name, version) VALUES (?, ?, ?)",
                    (migration.id, migration.name, migration.version),
                )

                migration.applied_at = datetime.utcnow()
                applied_versions.append(version)
                logger.info(f"Applied migration {version}: {migration.name}")
            except Exception as e:
                raise MigrationError(f"Failed to apply migration {version}: {e}")

        return applied_versions

    def rollback_migration(self, steps: int = 1) -> List[int]:
        """Rollback migrations.

        Args:
            steps: Number of migrations to rollback

        Returns:
            List of rolled back migration versions
        """
        applied = self.get_applied_migrations()
        to_rollback = sorted(applied, reverse=True)[:steps]

        rolled_back = []
        for version in to_rollback:
            if version not in self._migrations:
                logger.warning(f"Migration {version} not found in registry, skipping")
                continue

            migration = self._migrations[version]
            try:
                # Apply rollback
                self.connector.execute(migration.down_sql)

                # Remove migration record
                self.connector.execute(
                    f"DELETE FROM {self.migrations_table} WHERE version = ?",
                    (migration.version,),
                )

                rolled_back.append(version)
                logger.info(f"Rolled back migration {version}: {migration.name}")
            except Exception as e:
                raise MigrationError(f"Failed to rollback migration {version}: {e}")

        return rolled_back

    def get_status(self) -> Dict[str, Any]:
        """Get migration status."""
        applied = self.get_applied_migrations()
        pending = sorted([v for v in self._migrations.keys() if v not in applied])

        return {
            "current_version": max(applied) if applied else None,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "pending_versions": pending,
            "applied_versions": applied,
        }


def migrate_schema(
    connector: SQLConnector,
    migrations: List[Tuple[str, int, str, str]],
    target_version: Optional[int] = None,
) -> List[int]:
    """Convenience function to run migrations.

    Args:
        connector: SQL database connector
        migrations: List of (name, version, up_sql, down_sql) tuples
        target_version: Target version (None for all)

    Returns:
        List of applied migration versions
    """
    manager = SchemaMigration(connector)

    for name, version, up_sql, down_sql in migrations:
        manager.register_migration(name, version, up_sql, down_sql)

    return manager.migrate_schema(target_version)


def rollback_migration(connector: SQLConnector, steps: int = 1) -> List[int]:
    """Convenience function to rollback migrations.

    Args:
        connector: SQL database connector
        steps: Number of migrations to rollback

    Returns:
        List of rolled back migration versions
    """
    manager = SchemaMigration(connector)
    return manager.rollback_migration(steps)


# =============================================================================
# Utility Functions
# =============================================================================


def connect_database(db_type: str, **kwargs) -> Any:
    """Factory function to create database connector.

    Args:
        db_type: Type of database (postgresql, mysql, sqlite, sqlserver, oracle,
                 mongodb, cassandra, redis, elasticsearch, neo4j,
                 s3, gcs, azure, hdfs, deltalake,
                 pinecone, weaviate, milvus, chroma, faiss,
                 feast, tecton, sagemaker)
        **kwargs: Connection parameters

    Returns:
        Database connector instance
    """
    db_type = db_type.lower()

    # SQL Databases
    if db_type == "postgresql":
        config = ConnectionConfig(**kwargs)
        return PostgreSQL(config)

    elif db_type == "mysql":
        config = ConnectionConfig(**kwargs)
        return MySQL(config)

    elif db_type == "sqlite":
        return SQLite(kwargs.get("database_path", ":memory:"))

    elif db_type == "sqlserver":
        config = ConnectionConfig(**kwargs)
        return SQLServer(config)

    elif db_type == "oracle":
        config = ConnectionConfig(**kwargs)
        return Oracle(config)

    # NoSQL Databases
    elif db_type == "mongodb":
        return MongoDBConnector(**kwargs)

    elif db_type == "cassandra":
        return CassandraConnector(**kwargs)

    elif db_type == "redis":
        return RedisConnector(**kwargs)

    elif db_type == "elasticsearch":
        return ElasticsearchConnector(**kwargs)

    elif db_type == "neo4j":
        return Neo4jConnector(**kwargs)

    # Data Lakes
    elif db_type == "s3":
        return S3Connector(**kwargs)

    elif db_type == "gcs":
        return GCSConnector(**kwargs)

    elif db_type == "azure":
        return AzureBlob(**kwargs)

    elif db_type == "hdfs":
        return HDFSConnector(**kwargs)

    elif db_type == "deltalake":
        return DeltaLake(**kwargs)

    # Vector Databases
    elif db_type == "pinecone":
        return PineconeConnector(**kwargs)

    elif db_type == "weaviate":
        return WeaviateConnector(**kwargs)

    elif db_type == "milvus":
        return MilvusConnector(**kwargs)

    elif db_type == "chroma":
        return ChromaConnector(**kwargs)

    elif db_type == "faiss":
        return FAISSConnector(**kwargs)

    # Feature Stores
    elif db_type == "feast":
        return FeastConnector(**kwargs)

    elif db_type == "tecton":
        return TectonConnector(**kwargs)

    elif db_type == "sagemaker":
        return SageMakerFeatureStore(**kwargs)

    else:
        raise ValueError(f"Unknown database type: {db_type}")


def execute_sql(
    connector: SQLConnector,
    query: str,
    params: Optional[Tuple] = None,
    fetch: bool = True,
) -> Union[List[Dict[str, Any]], int]:
    """Execute SQL query and return results.

    Args:
        connector: SQL database connector
        query: SQL query string
        params: Query parameters
        fetch: Whether to fetch results

    Returns:
        Query results or row count
    """
    if fetch:
        return connector.fetchall(query, params)
    else:
        return connector.execute(query, params)


def fetch_data(
    connector: Any,
    query_or_collection: Union[str, Dict[str, Any]],
    params: Optional[Tuple] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch data from database.

    Args:
        connector: Database connector
        query_or_collection: SQL query or collection/filter dict
        params: Query parameters (for SQL)
        limit: Maximum number of results

    Returns:
        List of records as dictionaries
    """
    if isinstance(connector, SQLConnector):
        # SQL database
        if isinstance(query_or_collection, str):
            query = query_or_collection
            if limit > 0 and "LIMIT" not in query.upper():
                query += f" LIMIT {limit}"
            return connector.fetchall(query, params)
        else:
            raise ValueError("For SQL connectors, query_or_collection must be a string")

    elif isinstance(connector, MongoDBConnector):
        # MongoDB
        if isinstance(query_or_collection, dict):
            collection = query_or_collection.get("collection")
            filter_dict = query_or_collection.get("filter", {})
            return connector.find(collection, filter_dict, limit=limit)
        else:
            raise ValueError("For MongoDB, query_or_collection must be a dict")

    elif isinstance(connector, ElasticsearchConnector):
        # Elasticsearch
        if isinstance(query_or_collection, dict):
            index = query_or_collection.get("index")
            query = query_or_collection.get("query", {"match_all": {}})
            result = connector.search(index, query, size=limit)
            return [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]
        else:
            raise ValueError("For Elasticsearch, query_or_collection must be a dict")

    else:
        raise ValueError(f"Unsupported connector type: {type(connector).__name__}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Exceptions
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "SchemaError",
    "MigrationError",
    "VectorDBError",
    # Configuration
    "ConnectionConfig",
    # SQL Databases
    "SQLConnector",
    "PostgreSQL",
    "MySQL",
    "SQLite",
    "SQLServer",
    "Oracle",
    # NoSQL Databases
    "MongoDBConnector",
    "CassandraConnector",
    "RedisConnector",
    "ElasticsearchConnector",
    "Neo4jConnector",
    # Data Lakes
    "S3Connector",
    "GCSConnector",
    "AzureBlob",
    "HDFSConnector",
    "DeltaLake",
    # Vector Databases
    "PineconeConnector",
    "WeaviateConnector",
    "MilvusConnector",
    "ChromaConnector",
    "FAISSConnector",
    "VectorQueryResult",
    # Feature Store
    "FeastConnector",
    "TectonConnector",
    "SageMakerFeatureStore",
    # Query
    "SQLQuery",
    "NoSQLQuery",
    "VectorQuery",
    "execute_query",
    # Migration
    "Migration",
    "SchemaMigration",
    "migrate_schema",
    "rollback_migration",
    # Utilities
    "connect_database",
    "execute_sql",
    "fetch_data",
]
