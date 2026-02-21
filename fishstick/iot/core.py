"""
Fishstick IoT Module - Comprehensive MQTT/IoT infrastructure

This module provides MQTT client/server implementations, topic management,
message handling, QoS levels, security features, and protocol support.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import queue
import secrets
import socket
import ssl
import struct
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Generator


logger = logging.getLogger(__name__)


class MQTTException(Exception):
    """Base exception for MQTT module."""

    pass


class ConnectionRefused(MQTTException):
    """Connection refused by broker."""

    pass


class ProtocolError(MQTTException):
    """MQTT protocol error."""

    pass


class AuthenticationError(MQTTException):
    """Authentication failed."""

    pass


class AuthorizationError(MQTTException):
    """Authorization failed."""

    pass


# ============================================================================
# QoS Levels
# ============================================================================


class QoSLevel(Enum):
    """MQTT Quality of Service levels."""

    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


AtMostOnce = QoSLevel.AT_MOST_ONCE
AtLeastOnce = QoSLevel.AT_LEAST_ONCE
ExactlyOnce = QoSLevel.EXACTLY_ONCE


# ============================================================================
# Messages
# ============================================================================


@dataclass
class MQTTMessage:
    """Base MQTT message."""

    topic: str
    payload: bytes | str
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE
    retain: bool = False
    packet_id: int | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.payload, str):
            self.payload = self.payload.encode("utf-8")

    @property
    def payload_str(self) -> str:
        """Get payload as string."""
        if isinstance(self.payload, bytes):
            return self.payload.decode("utf-8")
        return self.payload

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        return self.payload

    @classmethod
    def from_bytes(cls, topic: str, payload: bytes, **kwargs) -> MQTTMessage:
        """Create message from bytes."""
        return cls(topic=topic, payload=payload, **kwargs)


@dataclass
class JSONMessage(MQTTMessage):
    """JSON-encoded MQTT message."""

    def __post_init__(self):
        if isinstance(self.payload, bytes):
            self.payload = self.payload.decode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        """Parse JSON payload to dict."""
        if isinstance(self.payload, str):
            return json.loads(self.payload)
        return json.loads(self.payload.decode("utf-8"))

    @classmethod
    def from_dict(
        cls,
        topic: str,
        data: dict[str, Any],
        qos: QoSLevel = QoSLevel.AT_MOST_ONCE,
        retain: bool = False,
        **kwargs,
    ) -> JSONMessage:
        """Create JSON message from dict."""
        payload = json.dumps(data).encode("utf-8")
        return cls(topic=topic, payload=payload, qos=qos, retain=retain, **kwargs)


@dataclass
class BinaryMessage(MQTTMessage):
    """Binary data MQTT message."""

    @classmethod
    def from_base64(cls, topic: str, base64_data: str, **kwargs) -> BinaryMessage:
        """Create binary message from base64 string."""
        payload = base64.b64decode(base64_data)
        return cls(topic=topic, payload=payload, **kwargs)

    def to_base64(self) -> str:
        """Convert payload to base64."""
        if isinstance(self.payload, str):
            payload = self.payload.encode("utf-8")
        else:
            payload = self.payload
        return base64.b64encode(payload).decode("utf-8")


# ============================================================================
# Topics
# ============================================================================


class TopicParser:
    """MQTT topic parser and validator."""

    MAX_TOPIC_LENGTH = 65535
    MAX_TOPIC_LEVELS = 128

    @staticmethod
    def parse(topic: str) -> list[str]:
        """Parse topic string into levels."""
        if not topic:
            raise ValueError("Empty topic")
        if len(topic) > TopicParser.MAX_TOPIC_LENGTH:
            raise ValueError(
                f"Topic too long: {len(topic)} > {TopicParser.MAX_TOPIC_LENGTH}"
            )
        levels = topic.split("/")
        if len(levels) > TopicParser.MAX_TOPIC_LEVELS:
            raise ValueError(f"Too many topic levels: {len(levels)}")
        return levels

    @staticmethod
    def validate(topic: str) -> bool:
        """Validate topic string."""
        try:
            levels = TopicParser.parse(topic)
            for level in levels:
                if level and ("+" in level or "#" in level):
                    if level not in ("+", "#"):
                        return False
                    if level == "#" and level != levels[-1]:
                        return False
            return True
        except ValueError:
            return False

    @staticmethod
    def join(levels: list[str]) -> str:
        """Join topic levels into string."""
        return "/".join(levels)

    @staticmethod
    def extract_levels(topic: str) -> int:
        """Count topic levels."""
        return len(TopicParser.parse(topic))


class WildcardMatching:
    """MQTT wildcard topic matching."""

    @staticmethod
    def match(topic: str, filter: str) -> bool:
        """Match topic against filter with wildcards."""
        if not topic or not filter:
            return False

        topic_levels = TopicParser.parse(topic)
        filter_levels = TopicParser.parse(filter)

        return WildcardMatching._match_levels(topic_levels, filter_levels)

    @staticmethod
    def _match_levels(topic_levels: list[str], filter_levels: list[str]) -> bool:
        """Recursively match topic levels against filter."""
        if not filter_levels:
            return not topic_levels

        if not topic_levels:
            return filter_levels == ["#"]

        filter_level = filter_levels[0]
        topic_level = topic_levels[0]

        if filter_level == "#":
            return True

        if filter_level == "+":
            return WildcardMatching._match_levels(topic_levels[1:], filter_levels[1:])

        if filter_level != topic_level:
            return False

        return WildcardMatching._match_levels(topic_levels[1:], filter_levels[1:])

    @staticmethod
    def matches_any(topic: str, filters: list[str]) -> bool:
        """Check if topic matches any of the filters."""
        return any(WildcardMatching.match(topic, f) for f in filters)


class TopicFilter:
    """Topic filter management."""

    def __init__(self):
        self._filters: dict[str, list[Callable[[MQTTMessage], None]]] = {}

    def add_filter(
        self, topic_filter: str, callback: Callable[[MQTTMessage], None]
    ) -> str:
        """Add a topic filter with callback."""
        filter_id = str(uuid.uuid4())
        if topic_filter not in self._filters:
            self._filters[topic_filter] = []
        self._filters[topic_filter].append(callback)
        return filter_id

    def remove_filter(self, topic_filter: str, filter_id: str) -> bool:
        """Remove a topic filter."""
        if topic_filter in self._filters:
            self._filters[topic_filter] = [
                cb
                for i, cb in enumerate(self._filters[topic_filter])
                if f"{topic_filter}_{i}" != filter_id
            ]
            return True
        return False

    def match(
        self, topic: str, message: MQTTMessage
    ) -> list[Callable[[MQTTMessage], None]]:
        """Find all matching callbacks for a topic."""
        callbacks = []
        for filter_str, cbs in self._filters.items():
            if WildcardMatching.match(topic, filter_str):
                callbacks.extend(cbs)
        return callbacks

    def get_filters(self) -> list[str]:
        """Get all registered filters."""
        return list(self._filters.keys())


# ============================================================================
# Security
# ============================================================================


@dataclass
class TLSConfig:
    """TLS/SSL configuration for MQTT."""

    enabled: bool = False
    ca_file: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    verify_mode: int = ssl.CERT_REQUIRED
    min_version: int = ssl.TLSVersion.TLSv1_2
    ciphers: str | None = None

    def create_context(self) -> ssl.SSLContext | None:
        """Create SSL context from config."""
        if not self.enabled:
            return None

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = self.min_version

        if self.ca_file:
            context.load_verify_locations(self.ca_file)
        if self.cert_file and self.key_file:
            context.load_cert_chain(self.cert_file, self.key_file)
        if self.ciphers:
            context.set_ciphers(self.ciphers)

        context.verify_mode = self.verify_mode
        return context


class Authentication:
    """MQTT authentication mechanisms."""

    def __init__(self):
        self._users: dict[str, tuple[str, str]] = {}
        self._token_cache: dict[str, tuple[str, float]] = {}

    def add_user(self, username: str, password: str) -> None:
        """Add user credentials."""
        self._users[username] = (password, self._hash_password(password))

    def add_user_hashed(self, username: str, password_hash: str) -> None:
        """Add user with pre-hashed password."""
        self._users[username] = ("", password_hash)

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user with password."""
        if username not in self._users:
            return False

        stored_password, stored_hash = self._users[username]
        if stored_password:
            return stored_password == password
        return self._verify_hash(password, stored_hash)

    def authenticate_token(self, token: str) -> str | None:
        """Authenticate using token. Returns username if valid."""
        if token in self._token_cache:
            username, expiry = self._token_cache[token]
            if expiry > time.time():
                return username
            del self._token_cache[token]
        return None

    def generate_token(self, username: str, ttl: int = 3600) -> str:
        """Generate authentication token for user."""
        if username not in self._users:
            raise AuthenticationError(f"User not found: {username}")

        token = secrets.token_urlsafe(32)
        self._token_cache[token] = (username, time.time() + ttl)
        return token

    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def _verify_hash(password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return Authentication._hash_password(password) == password_hash


class Authorization:
    """MQTT authorization for topics."""

    def __init__(self):
        self._permissions: dict[str, dict[str, set[str]]] = {}

    def add_permission(self, username: str, topic: str, actions: set[str]) -> None:
        """Add permission for user on topic."""
        if username not in self._permissions:
            self._permissions[username] = {"read": set(), "write": set()}

        for action in actions:
            if action in ("read", "subscribe", "publish"):
                if action == "read":
                    self._permissions[username]["read"].add(topic)
                else:
                    self._permissions[username]["write"].add(topic)

    def authorize(self, username: str, topic: str, action: str) -> bool:
        """Check if user is authorized for action on topic."""
        if username not in self._permissions:
            return False

        perms = self._permissions[username]
        action_key = "read" if action in ("read", "subscribe") else "write"

        allowed_topics = perms.get(action_key, set())
        for allowed_topic in allowed_topics:
            if WildcardMatching.match(topic, allowed_topic):
                return True
        return False

    def revoke_permission(self, username: str, topic: str) -> bool:
        """Revoke permission for user on topic."""
        if username in self._permissions:
            for action in ("read", "write"):
                self._permissions[username][action].discard(topic)
            return True
        return False


# ============================================================================
# Protocols
# ============================================================================


class MQTTProtocol(ABC):
    """Abstract base class for MQTT protocols."""

    PROTOCOL_NAME: str = "MQTT"
    PROTOCOL_VERSION: int = 0

    @abstractmethod
    def connect(self, client_id: str, clean_session: bool, keepalive: int) -> bytes:
        """Create CONNECT packet."""
        pass

    @abstractmethod
    def connack(self, return_code: int) -> bytes:
        """Create CONNACK packet."""
        pass

    @abstractmethod
    def publish(
        self, topic: str, payload: bytes, qos: QoSLevel, retain: bool, packet_id: int
    ) -> bytes:
        """Create PUBLISH packet."""
        pass

    @abstractmethod
    def subscribe(self, packet_id: int, topics: list[tuple[str, QoSLevel]]) -> bytes:
        """Create SUBSCRIBE packet."""
        pass

    @abstractmethod
    def unsubscribe(self, packet_id: int, topics: list[str]) -> bytes:
        """Create UNSUBSCRIBE packet."""
        pass

    @abstractmethod
    def pingreq(self) -> bytes:
        """Create PINGREQ packet."""
        pass

    @abstractmethod
    def pingresp(self) -> bytes:
        """Create PINGRESP packet."""
        pass

    @abstractmethod
    def disconnect(self) -> bytes:
        """Create DISCONNECT packet."""
        pass


class MQTT311(MQTTProtocol):
    """MQTT 3.1.1 protocol implementation."""

    PROTOCOL_NAME = "MQIsdp"
    PROTOCOL_VERSION = 3

    CONNECT = 1
    CONNACK = 2
    PUBLISH = 3
    PUBACK = 4
    PUBREC = 5
    PUBREL = 6
    PUBCOMP = 7
    SUBSCRIBE = 8
    SUBACK = 9
    UNSUBSCRIBE = 10
    UNSUBACK = 11
    PINGREQ = 12
    PINGRESP = 13
    DISCONNECT = 14

    @staticmethod
    def _encode_length(length: int) -> bytes:
        """Encode remaining length."""
        result = bytearray()
        while True:
            byte = length % 128
            length //= 128
            if length > 0:
                byte |= 0x80
            result.append(byte)
            if length == 0:
                break
        return bytes(result)

    @staticmethod
    def _decode_length(data: bytes) -> tuple[int, int]:
        """Decode remaining length."""
        length = 0
        multiplier = 1
        pos = 0
        for byte in data:
            length += (byte & 0x7F) * multiplier
            multiplier *= 128
            pos += 1
            if byte & 0x80 == 0:
                break
        return length, pos

    def connect(self, client_id: str, clean_session: bool, keepalive: int) -> bytes:
        """Create CONNECT packet."""
        payload = bytearray()

        client_id_bytes = client_id.encode("utf-8")
        payload.extend(struct.pack("!H", len(client_id_bytes)))
        payload.extend(client_id_bytes)

        flags = 0
        if clean_session:
            flags |= 0x02

        variable = bytearray()
        variable.extend(self.PROTOCOL_NAME.encode("utf-8"))
        variable.append(0)
        variable.append(self.PROTOCOL_VERSION)
        variable.append(flags)
        variable.extend(struct.pack("!H", keepalive))

        return self._build_packet(self.CONNECT, bytes(variable) + bytes(payload))

    def connack(self, return_code: int) -> bytes:
        """Create CONNACK packet."""
        variable = bytes([0, return_code])
        return self._build_packet(self.CONNACK, variable)

    def publish(
        self,
        topic: str,
        payload: bytes,
        qos: QoSLevel,
        retain: bool,
        packet_id: int = 0,
    ) -> bytes:
        """Create PUBLISH packet."""
        topic_bytes = topic.encode("utf-8")
        variable = bytearray()
        variable.extend(struct.pack("!H", len(topic_bytes)))
        variable.extend(topic_bytes)

        if qos != QoSLevel.AT_MOST_ONCE:
            variable.extend(struct.pack("!H", packet_id))

        flags = qos.value << 1
        if retain:
            flags |= 0x01

        return self._build_packet(self.PUBLISH, bytes(variable) + payload, flags)

    def puback(self, packet_id: int) -> bytes:
        """Create PUBACK packet."""
        variable = struct.pack("!H", packet_id)
        return self._build_packet(self.PUBACK, variable)

    def subscribe(self, packet_id: int, topics: list[tuple[str, QoSLevel]]) -> bytes:
        """Create SUBSCRIBE packet."""
        payload = bytearray()
        payload.extend(struct.pack("!H", packet_id))

        for topic, qos in topics:
            topic_bytes = topic.encode("utf-8")
            payload.extend(struct.pack("!H", len(topic_bytes)))
            payload.extend(topic_bytes)
            payload.append(qos.value)

        return self._build_packet(self.SUBSCRIBE, bytes(payload))

    def suback(self, packet_id: int, return_codes: list[int]) -> bytes:
        """Create SUBACK packet."""
        payload = bytearray()
        payload.extend(struct.pack("!H", packet_id))
        payload.extend(return_codes)
        return self._build_packet(self.SUBACK, bytes(payload))

    def unsubscribe(self, packet_id: int, topics: list[str]) -> bytes:
        """Create UNSUBSCRIBE packet."""
        payload = bytearray()
        payload.extend(struct.pack("!H", packet_id))

        for topic in topics:
            topic_bytes = topic.encode("utf-8")
            payload.extend(struct.pack("!H", len(topic_bytes)))
            payload.extend(topic_bytes)

        return self._build_packet(self.UNSUBSCRIBE, bytes(payload))

    def unsuback(self, packet_id: int) -> bytes:
        """Create UNSUBACK packet."""
        variable = struct.pack("!H", packet_id)
        return self._build_packet(self.UNSUBACK, variable)

    def pingreq(self) -> bytes:
        """Create PINGREQ packet."""
        return self._build_packet(self.PINGREQ, b"")

    def pingresp(self) -> bytes:
        """Create PINGRESP packet."""
        return self._build_packet(self.PINGRESP, b"")

    def disconnect(self) -> bytes:
        """Create DISCONNECT packet."""
        return self._build_packet(self.DISCONNECT, b"")

    def _build_packet(self, packet_type: int, payload: bytes, flags: int = 0) -> bytes:
        """Build complete MQTT packet."""
        header = (packet_type << 4) | flags
        length = self._encode_length(len(payload))
        return bytes([header]) + length + payload


class MQTT500(MQTT311):
    """MQTT 5.0 protocol implementation."""

    PROTOCOL_NAME = "MQTT"
    PROTOCOL_VERSION = 5

    CONNECT = 1
    CONNACK = 2
    PUBLISH = 3
    PUBACK = 4
    PUBREC = 5
    PUBREL = 6
    PUBCOMP = 7
    SUBSCRIBE = 8
    SUBACK = 9
    UNSUBSCRIBE = 10
    UNSUBACK = 11
    PINGREQ = 12
    PINGRESP = 13
    DISCONNECT = 14
    AUTH = 15

    PROPERTIES = {
        "payload_format": 0x01,
        "message_expiry": 0x02,
        "topic_alias": 0x23,
        "response_topic": 0x08,
        "correlation_data": 0x09,
        "subscription_id": 0x0B,
        "session_expiry": 0x11,
        "assigned_client_id": 0x12,
        "server_keepalive": 0x13,
        "auth_method": 0x15,
        "auth_data": 0x16,
        "user_property": 0x26,
    }

    def connect(
        self,
        client_id: str,
        clean_session: bool,
        keepalive: int,
        properties: dict[str, Any] | None = None,
    ) -> bytes:
        """Create CONNECT packet with MQTT 5.0 properties."""
        properties = properties or {}
        props_bytes = self._encode_properties(properties)

        variable = bytearray()
        variable.extend(self.PROTOCOL_NAME.encode("utf-8"))
        variable.append(0)
        variable.append(self.PROTOCOL_VERSION)
        variable.append(0xC0 if clean_session else 0x80)
        variable.extend(struct.pack("!H", keepalive))
        variable.extend(self._encode_length(len(props_bytes)))
        variable.extend(props_bytes)

        client_id_bytes = client_id.encode("utf-8")
        payload = bytearray()
        payload.extend(struct.pack("!H", len(client_id_bytes)))
        payload.extend(client_id_bytes)

        return self._build_packet(self.CONNECT, bytes(variable) + bytes(payload))

    def connack(
        self, return_code: int, properties: dict[str, Any] | None = None
    ) -> bytes:
        """Create CONNACK packet with properties."""
        properties = properties or {}
        props_bytes = self._encode_properties(properties)

        flags = 0x00
        variable = bytearray()
        variable.append(flags)
        variable.append(return_code)
        variable.extend(props_bytes)

        return self._build_packet(self.CONNACK, bytes(variable))

    def _encode_properties(self, properties: dict[str, Any]) -> bytes:
        """Encode properties for MQTT 5.0."""
        result = bytearray()
        for key, value in properties.items():
            prop_id = self.PROPERTIES.get(key)
            if prop_id is None:
                continue

            if key == "user_property":
                for k, v in value:
                    result.append(prop_id)
                    result.extend(struct.pack("!H", len(k)))
                    result.extend(k.encode("utf-8"))
                    result.extend(struct.pack("!H", len(v)))
                    result.extend(v.encode("utf-8"))
            elif isinstance(value, str):
                result.append(prop_id)
                result.extend(struct.pack("!H", len(value)))
                result.extend(value.encode("utf-8"))
            elif isinstance(value, int):
                result.append(prop_id)
                result.extend(struct.pack("!I", value))
            elif isinstance(value, bytes):
                result.append(prop_id)
                result.extend(struct.pack("!H", len(value)))
                result.extend(value)

        return bytes(result)


# ============================================================================
# MQTT Client
# ============================================================================


class MQTTClient:
    """MQTT client implementation."""

    def __init__(
        self,
        client_id: str | None = None,
        broker: str = "localhost",
        port: int = 1883,
        keepalive: int = 60,
        clean_session: bool = True,
        qos: QoSLevel = QoSLevel.AT_MOST_ONCE,
        protocol: MQTTProtocol | None = None,
        tls_config: TLSConfig | None = None,
        auth: Authentication | None = None,
    ):
        self.client_id = client_id or str(uuid.uuid4())
        self.broker = broker
        self.port = port
        self.keepalive = keepalive
        self.clean_session = clean_session
        self.default_qos = qos
        self.protocol = protocol or MQTT311()
        self.tls_config = tls_config or TLSConfig()
        self.auth = auth

        self._socket: socket.socket | None = None
        self._connected = False
        self._packet_id = 0
        self._subscriptions: dict[str, tuple[Callable, QoSLevel]] = {}
        self._pending_acks: dict[int, asyncio.Future] = {}
        self._message_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def connect(self, username: str | None = None, password: str | None = None) -> bool:
        """Connect to MQTT broker."""
        try:
            if self.tls_config.enabled:
                context = self.tls_config.create_context()
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket = context.wrap_socket(
                    self._socket, server_hostname=self.broker
                )
            else:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            self._socket.settimeout(10)
            self._socket.connect((self.broker, self.port))

            connect_packet = self.protocol.connect(
                self.client_id, self.clean_session, self.keepalive
            )
            self._socket.send(connect_packet)

            response = self._recv_packet()
            if response and response[0] >> 4 == self.protocol.CONNACK:
                return_code = response[-1]
                if return_code == 0:
                    self._connected = True
                    self._running = True
                    self._thread = threading.Thread(target=self._loop, daemon=True)
                    self._thread.start()
                    logger.info(
                        f"Connected to MQTT broker at {self.broker}:{self.port}"
                    )
                    return True
                else:
                    raise ConnectionRefused(
                        f"Connection refused with code: {return_code}"
                    )

            raise ProtocolError("Invalid CONNACK response")

        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            self._connected = False
            raise

    def _recv_packet(self) -> bytes | None:
        """Receive a packet from the broker."""
        try:
            header = self._socket.recv(1)
            if not header:
                return None

            packet_type = header[0] >> 4

            length_byte = self._socket.recv(1)
            length = 0
            multiplier = 1
            while length_byte[0] & 0x80:
                length += (length_byte[0] & 0x7F) * multiplier
                multiplier *= 128
                length_byte = self._socket.recv(1)
            length += length_byte[0] * multiplier

            payload = b""
            if length > 0:
                payload = self._socket.recv(length)

            return bytes([header[0]]) + length_byte + payload

        except Exception as e:
            logger.error(f"Error receiving packet: {e}")
            return None

    def _loop(self):
        """Main client loop."""
        while self._running:
            try:
                self._socket.settimeout(1)
                packet = self._recv_packet()
                if packet:
                    self._handle_packet(packet)
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in client loop: {e}")
                break

        self._connected = False

    def _handle_packet(self, packet: bytes) -> None:
        """Handle incoming packet."""
        packet_type = packet[0] >> 4

        if packet_type == self.protocol.PUBLISH:
            self._handle_publish(packet)
        elif packet_type == self.protocol.PUBACK:
            self._handle_puback(packet)
        elif packet_type == self.protocol.SUBACK:
            self._handle_suback(packet)
        elif packet_type == self.protocol.PINGRESP:
            pass

    def _handle_publish(self, packet: bytes) -> None:
        """Handle PUBLISH packet."""
        header = packet[0]
        qos_val = (header >> 1) & 0x03
        retain = bool(header & 0x01)

        pos = 1
        while packet[pos] & 0x80:
            pos += 1
        pos += 1

        topic_len = struct.unpack("!H", packet[pos : pos + 2])[0]
        pos += 2
        topic = packet[pos : pos + topic_len].decode("utf-8")
        pos += topic_len

        packet_id = None
        if qos_val > 0:
            packet_id = struct.unpack("!H", packet[pos : pos + 2])[0]
            pos += 2

        payload = packet[pos:]

        message = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=QoSLevel(qos_val),
            retain=retain,
            packet_id=packet_id,
        )

        for sub_topic, (callback, _) in self._subscriptions.items():
            if WildcardMatching.match(topic, sub_topic):
                callback(message)

    def _handle_puback(self, packet: bytes) -> None:
        """Handle PUBACK packet."""
        if len(packet) >= 4:
            packet_id = struct.unpack("!H", packet[2:4])[0]
            if packet_id in self._pending_acks:
                self._pending_acks[packet_id].set_result(True)
                del self._pending_acks[packet_id]

    def _handle_suback(self, packet: bytes) -> None:
        """Handle SUBACK packet."""
        if len(packet) >= 4:
            packet_id = struct.unpack("!H", packet[2:4])[0]
            if packet_id in self._pending_acks:
                self._pending_acks[packet_id].set_result(True)
                del self._pending_acks[packet_id]

    def publish(
        self,
        topic: str,
        payload: bytes | str,
        qos: QoSLevel | None = None,
        retain: bool = False,
    ) -> bool:
        """Publish message to topic."""
        if not self._connected:
            raise MQTTException("Not connected to broker")

        qos = qos or self.default_qos
        packet_id = None
        if qos != QoSLevel.AT_MOST_ONCE:
            self._packet_id = (self._packet_id % 65535) + 1
            packet_id = self._packet_id

        packet = self.protocol.publish(topic, payload, qos, retain, packet_id)
        self._socket.send(packet)

        if qos == QoSLevel.AT_MOST_ONCE:
            return True

        if packet_id in self._pending_acks:
            try:
                self._pending_acks[packet_id].result(timeout=30)
                return True
            except Exception:
                return False

        return False

    def subscribe(
        self,
        topic: str,
        callback: Callable[[MQTTMessage], None],
        qos: QoSLevel | None = None,
    ) -> bool:
        """Subscribe to topic."""
        if not self._connected:
            raise MQTTException("Not connected to broker")

        qos = qos or self.default_qos

        self._packet_id = (self._packet_id % 65535) + 1
        packet_id = self._packet_id

        packet = self.protocol.subscribe(packet_id, [(topic, qos)])
        self._socket.send(packet)

        self._subscriptions[topic] = (callback, qos)

        future = asyncio.Future()
        self._pending_acks[packet_id] = future

        try:
            future.result(timeout=30)
            return True
        except Exception:
            return False

    def unsubscribe(self, topic: str) -> bool:
        """Unsubscribe from topic."""
        if not self._connected:
            raise MQTTException("Not connected to broker")

        self._packet_id = (self._packet_id % 65535) + 1
        packet_id = self._packet_id

        packet = self.protocol.unsubscribe(packet_id, [topic])
        self._socket.send(packet)

        if topic in self._subscriptions:
            del self._subscriptions[topic]

        return True

    def disconnect(self) -> None:
        """Disconnect from broker."""
        if self._connected and self._socket:
            try:
                packet = self.protocol.disconnect()
                self._socket.send(packet)
            except Exception:
                pass

        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        self._connected = False
        logger.info("Disconnected from MQTT broker")

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def __enter__(self) -> "MQTTClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


# ============================================================================
# MQTT Server
# ============================================================================


class MQTTServer:
    """MQTT broker/server implementation."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        protocol: MQTTProtocol | None = None,
        tls_config: TLSConfig | None = None,
        auth: Authentication | None = None,
        authorization: Authorization | None = None,
    ):
        self.host = host
        self.port = port
        self.protocol = protocol or MQTT311()
        self.tls_config = tls_config or TLSConfig()
        self.auth = auth
        self.authorization = authorization

        self._socket: socket.socket | None = None
        self._running = False
        self._clients: dict[str, dict[str, Any]] = {}
        self._subscriptions: dict[str, set[str]] = {}
        self._topic_filters = TopicFilter()
        self._retained_messages: dict[str, MQTTMessage] = {}
        self._thread: threading.Thread | None = None

    def create_server(self) -> socket.socket:
        """Create server socket."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(100)
        logger.info(f"MQTT server created on {self.host}:{self.port}")
        return self._socket

    def start_server(self) -> None:
        """Start MQTT server."""
        if not self._socket:
            self.create_server()

        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info(f"MQTT server started on {self.host}:{self.port}")

    def _accept_loop(self) -> None:
        """Accept incoming connections."""
        while self._running:
            try:
                self._socket.settimeout(1)
                client_socket, address = self._socket.accept()
                logger.info(f"Client connected from {address}")
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True,
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                break

    def _handle_client(self, client_socket: socket.socket, address: tuple) -> None:
        """Handle client connection."""
        client_id = None

        try:
            while self._running:
                client_socket.settimeout(30)
                packet = self._recv_packet(client_socket)
                if not packet:
                    break

                packet_type = packet[0] >> 4

                if packet_type == self.protocol.CONNECT:
                    client_id = self._handle_connect(client_socket, packet)
                    if client_id:
                        self._clients[client_id] = {"socket": client_socket}
                elif packet_type == self.protocol.PUBLISH:
                    self.handle_message(client_socket, packet, client_id)
                elif packet_type == self.protocol.SUBSCRIBE:
                    self._handle_subscribe(client_socket, packet, client_id)
                elif packet_type == self.protocol.UNSUBSCRIBE:
                    self._handle_unsubscribe(client_socket, packet, client_id)
                elif packet_type == self.protocol.PINGREQ:
                    self._handle_pingreq(client_socket)
                elif packet_type == self.protocol.DISCONNECT:
                    break

        except Exception as e:
            logger.error(f"Error handling client {address}: {e}")
        finally:
            if client_id and client_id in self._clients:
                self._cleanup_client(client_id)
            try:
                client_socket.close()
            except Exception:
                pass

    def _recv_packet(self, sock: socket.socket) -> bytes | None:
        """Receive packet from client."""
        try:
            header = sock.recv(1)
            if not header:
                return None

            length_byte = sock.recv(1)
            length = 0
            multiplier = 1
            while length_byte[0] & 0x80:
                length += (length_byte[0] & 0x7F) * multiplier
                multiplier *= 128
                length_byte = sock.recv(1)
            length += length_byte[0] * multiplier

            payload = b""
            if length > 0:
                payload = sock.recv(length)

            return bytes([header[0]]) + length_byte + payload

        except Exception:
            return None

    def _handle_connect(self, sock: socket.socket, packet: bytes) -> str | None:
        """Handle CONNECT packet."""
        pos = 2

        proto_name_len = struct.unpack("!H", packet[pos : pos + 2])[0]
        pos += 2
        proto_name = packet[pos : pos + proto_name_len].decode("utf-8")
        pos += proto_name_len

        pos += 1
        flags = packet[pos]
        pos += 1

        keepalive = struct.unpack("!H", packet[pos : pos + 2])[0]
        pos += 2

        client_id_len = struct.unpack("!H", packet[pos : pos + 2])[0]
        pos += 2
        client_id = packet[pos : pos + client_id_len].decode("utf-8")

        clean_session = bool(flags & 0x02)

        connack_packet = self.protocol.connack(0)
        sock.send(connack_packet)

        logger.info(f"Client {client_id} connected")
        return client_id

    def handle_message(
        self, sock: socket.socket, packet: bytes, client_id: str | None
    ) -> None:
        """Handle incoming PUBLISH message."""
        header = packet[0]
        qos_val = (header >> 1) & 0x03

        pos = 1
        while packet[pos] & 0x80:
            pos += 1
        pos += 1

        topic_len = struct.unpack("!H", packet[pos : pos + 2])[0]
        pos += 2
        topic = packet[pos : pos + topic_len].decode("utf-8")
        pos += topic_len

        packet_id = None
        if qos_val > 0:
            packet_id = struct.unpack("!H", packet[pos : pos + 2])[0]
            pos += 2

        payload = packet[pos:]

        if self.auth and client_id:
            if not self.authorization.authorize(client_id, topic, "publish"):
                logger.warning(f"Unauthorized publish from {client_id} to {topic}")
                return

        message = MQTTMessage(
            topic=topic,
            payload=payload,
            qos=QoSLevel(qos_val),
            packet_id=packet_id,
        )

        if header & 0x01:
            self._retained_messages[topic] = message

        callbacks = self._topic_filters.match(topic, message)
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")

        for sub_topic, clients in self._subscriptions.items():
            if WildcardMatching.match(topic, sub_topic):
                for sub_client_id in clients:
                    if sub_client_id in self._clients:
                        client_sock = self._clients[sub_client_id]["socket"]
                        try:
                            client_sock.send(packet)
                        except Exception:
                            pass

        if qos_val == 1:
            puback_packet = self.protocol.puback(packet_id)
            sock.send(puback_packet)

    def _handle_subscribe(
        self, sock: socket.socket, packet: bytes, client_id: str | None
    ) -> None:
        """Handle SUBSCRIBE packet."""
        if len(packet) < 4:
            return

        packet_id = struct.unpack("!H", packet[2:4])[0]
        pos = 4

        return_codes = []

        while pos < len(packet):
            topic_len = struct.unpack("!H", packet[pos : pos + 2])[0]
            pos += 2
            topic = packet[pos : pos + topic_len].decode("utf-8")
            pos += topic_len

            if pos < len(packet):
                qos = packet[pos]
                pos += 1
            else:
                qos = 0

            if self.auth and client_id:
                if not self.authorization.authorize(client_id, topic, "subscribe"):
                    return_codes.append(0x80)
                    continue

            if topic not in self._subscriptions:
                self._subscriptions[topic] = set()
            if client_id:
                self._subscriptions[topic].add(client_id)

            return_codes.append(qos)

        suback_packet = self.protocol.suback(packet_id, return_codes)
        sock.send(suback_packet)

    def _handle_unsubscribe(
        self, sock: socket.socket, packet: bytes, client_id: str | None
    ) -> None:
        """Handle UNSUBSCRIBE packet."""
        if len(packet) < 4:
            return

        packet_id = struct.unpack("!H", packet[2:4])[0]
        pos = 4

        while pos < len(packet):
            topic_len = struct.unpack("!H", packet[pos : pos + 2])[0]
            pos += 2
            topic = packet[pos : pos + topic_len].decode("utf-8")
            pos += topic_len

            if topic in self._subscriptions and client_id:
                self._subscriptions[topic].discard(client_id)

        unsuback_packet = self.protocol.unsuback(packet_id)
        sock.send(unsuback_packet)

    def _handle_pingreq(self, sock: socket.socket) -> None:
        """Handle PINGREQ packet."""
        pingresp_packet = self.protocol.pingresp()
        sock.send(pingresp_packet)

    def _cleanup_client(self, client_id: str) -> None:
        """Clean up client on disconnect."""
        if client_id in self._clients:
            del self._clients[client_id]

        for topic in self._subscriptions:
            self._subscriptions[topic].discard(client_id)

        logger.info(f"Client {client_id} disconnected")

    def stop_server(self) -> None:
        """Stop MQTT server."""
        self._running = False

        for client_info in self._clients.values():
            try:
                client_info["socket"].close()
            except Exception:
                pass

        self._clients.clear()

        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("MQTT server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    def __enter__(self) -> "MQQTTServer":
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_server()


# ============================================================================
# Utilities
# ============================================================================


def mqtt_client(
    broker: str = "localhost",
    port: int = 1883,
    client_id: str | None = None,
    username: str | None = None,
    password: str | None = None,
    tls_config: TLSConfig | None = None,
    keepalive: int = 60,
    clean_session: bool = True,
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE,
    protocol_version: int = 311,
) -> MQTTClient:
    """Create and configure MQTT client."""
    if protocol_version == 500:
        protocol = MQTT500()
    else:
        protocol = MQTT311()

    auth = None
    if username and password:
        auth = Authentication()
        auth.add_user(username, password)

    client = MQTTClient(
        client_id=client_id,
        broker=broker,
        port=port,
        keepalive=keepalive,
        clean_session=clean_session,
        qos=qos,
        protocol=protocol,
        tls_config=tls_config,
        auth=auth,
    )

    return client


def mqtt_server(
    host: str = "localhost",
    port: int = 1883,
    tls_config: TLSConfig | None = None,
    username: str | None = None,
    password: str | None = None,
) -> MQTTServer:
    """Create and configure MQTT server."""
    auth = None
    authorization = Authorization()

    if username and password:
        auth = Authentication()
        auth.add_user(username, password)
        authorization.add_permission(username, "#", {"read", "write"})

    server = MQTTServer(
        host=host,
        port=port,
        tls_config=tls_config,
        auth=auth,
        authorization=authorization,
    )

    return server
