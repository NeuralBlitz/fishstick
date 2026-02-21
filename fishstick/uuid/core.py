import hashlib
import os
import re
import struct
import time
import uuid as _uuid_lib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union


UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)

UUID_REGEX_NO_DASH = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


@dataclass
class Namespace:
    DNS: str = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
    URL: str = "6ba7b811-9dad-11d1-80b4-00c04fd430c8"
    OID: str = "6ba7b812-9dad-11d1-80b4-00c04fd430c8"


NAMESPACE_DNS = Namespace.DNS
NAMESPACE_URL = Namespace.URL
NAMESPACE_OID = Namespace.OID


class UUIDValidator:
    @staticmethod
    def validate(value: str) -> bool:
        return is_valid_uuid(value)

    def __call__(self, value: str) -> bool:
        return self.validate(value)


def is_valid_uuid(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if UUID_REGEX.match(value):
        return True
    if UUID_REGEX_NO_DASH.match(value):
        return True
    try:
        _uuid_lib.UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def validate_uuid(value: str) -> bool:
    return is_valid_uuid(value)


def generate_uuid(version: int = 4) -> str:
    if version == 1:
        return generate_uuid1()
    elif version == 4:
        return generate_uuid4()
    elif version == 5:
        raise ValueError("UUIDv5 requires namespace and name arguments")
    elif version == 6:
        return generate_uuid6()
    else:
        raise ValueError(f"Unsupported UUID version: {version}")


def generate_uuid1() -> str:
    return str(_uuid_lib.uuid1())


def generate_uuid4() -> str:
    return str(_uuid_lib.uuid4())


def generate_uuid5(namespace: str, name: str) -> str:
    if not is_valid_uuid(namespace):
        raise ValueError(f"Invalid namespace UUID: {namespace}")
    ns_uuid = _uuid_lib.UUID(namespace)
    return str(_uuid_lib.uuid5(ns_uuid, name))


def generate_uuid6() -> str:
    timestamp = int(time.time() * 10000000) + 0x01B21DD213814000
    time_low = timestamp & 0xFFFFFFFF
    time_mid = (timestamp >> 32) & 0xFFFF
    time_hi = (timestamp >> 48) & 0x0FFF
    time_hi |= 0x6000
    clock_seq = (timestamp >> 28) & 0x3FFF
    clock_seq |= 0x8000
    node = (int.from_bytes(os.urandom(6), "big") & 0xFFFFFFFFFFFF) | 0x800000000000
    uuid_bytes = struct.pack(
        "<IHHHBBBBBB",
        time_low,
        time_mid,
        time_hi,
        clock_seq,
        (node >> 40) & 0xFF,
        (node >> 32) & 0xFF,
        (node >> 24) & 0xFF,
        (node >> 16) & 0xFF,
        (node >> 8) & 0xFF,
        node & 0xFF,
    )
    return str(_uuid_lib.UUID(bytes=uuid_bytes))


def to_string(uuid_obj: Union[str, _uuid_lib.UUID]) -> str:
    if isinstance(uuid_obj, _uuid_lib.UUID):
        return str(uuid_obj)
    if isinstance(uuid_obj, str):
        if is_valid_uuid(uuid_obj):
            return str(_uuid_lib.UUID(uuid_obj))
    raise ValueError(f"Invalid UUID: {uuid_obj}")


def from_string(value: str) -> str:
    if not is_valid_uuid(value):
        raise ValueError(f"Invalid UUID string: {value}")
    return str(_uuid_lib.UUID(value))


def to_bytes(uuid_value: Union[str, _uuid_lib.UUID]) -> bytes:
    if isinstance(uuid_value, str):
        uuid_value = _uuid_lib.UUID(uuid_value)
    return uuid_value.bytes


def from_bytes(data: bytes) -> str:
    if len(data) != 16:
        raise ValueError(f"Expected 16 bytes, got {len(data)}")
    return str(_uuid_lib.UUID(bytes=data))


def compare_uuid(uuid1: str, uuid2: str) -> int:
    if not is_valid_uuid(uuid1):
        raise ValueError(f"Invalid UUID: {uuid1}")
    if not is_valid_uuid(uuid2):
        raise ValueError(f"Invalid UUID: {uuid2}")
    u1 = _uuid_lib.UUID(uuid1)
    u2 = _uuid_lib.UUID(uuid2)
    if u1 < u2:
        return -1
    elif u1 > u2:
        return 1
    return 0


def is_equal(uuid1: str, uuid2: str) -> bool:
    if not is_valid_uuid(uuid1) or not is_valid_uuid(uuid2):
        return False
    return _uuid_lib.UUID(uuid1) == _uuid_lib.UUID(uuid2)


def parse_uuid(value: str) -> _uuid_lib.UUID:
    if not is_valid_uuid(value):
        raise ValueError(f"Invalid UUID: {value}")
    return _uuid_lib.UUID(value)


def parse_string(value: str) -> str:
    return from_string(value)


def uuid1() -> str:
    return generate_uuid1()


def uuid4() -> str:
    return generate_uuid4()


def uuid5(namespace: str, name: str) -> str:
    return generate_uuid5(namespace, name)


def uuid6() -> str:
    return generate_uuid6()
