"""
Fishstick - Comprehensive Serialization Module
A high-performance, feature-rich serialization library for Python.
"""

from __future__ import annotations

import abc
import base64
import bz2
import dataclasses
import gzip
import hashlib
import json
import lzma
import pickle
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

try:
    import msgpack

    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    from google.protobuf import descriptor_pb2
    from google.protobuf import descriptor_pool
    from google.protobuf import message_factory
    from google.protobuf import reflection
    from google.protobuf import symbol_database

    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False


T = TypeVar("T")
SerializerType = TypeVar("SerializerType")


# ============================================================================
# Base Interfaces
# ============================================================================


@runtime_checkable
class Serializer(Protocol):
    """Base serializer protocol defining common serialization operations."""

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        ...

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an object."""
        ...


class SerializerBase(abc.ABC):
    """Abstract base class for all serializer implementations."""

    @abc.abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        pass

    @abc.abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an object."""
        pass


# ============================================================================
# JSON Serialization
# ============================================================================


class JSONSerializer(SerializerBase):
    """JSON-based serializer with configurable encoding/decoding."""

    def __init__(
        self,
        indent: Optional[int] = None,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        allow_nan: bool = True,
    ):
        self.indent = indent
        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii
        self.allow_nan = allow_nan

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to JSON bytes."""
        return json.dumps(
            obj,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii,
            allow_nan=self.allow_nan,
        ).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to an object."""
        return json.loads(data.decode("utf-8"))

    def serialize_to_str(self, obj: Any) -> str:
        """Serialize an object to a JSON string."""
        return json.dumps(
            obj,
            indent=self.indent,
            sort_keys=self.sort_keys,
            ensure_ascii=self.ensure_ascii,
            allow_nan=self.allow_nan,
        )

    def deserialize_from_str(self, data: str) -> Any:
        """Deserialize a JSON string to an object."""
        return json.loads(data)


def serialize_json(obj: Any, **kwargs) -> bytes:
    """Serialize an object to JSON bytes.

    Args:
        obj: The object to serialize.
        **kwargs: Additional arguments passed to JSONSerializer.

    Returns:
        JSON-encoded bytes.
    """
    return JSONSerializer(**kwargs).serialize(obj)


def deserialize_json(data: bytes) -> Any:
    """Deserialize JSON bytes to an object.

    Args:
        data: JSON-encoded bytes.

    Returns:
        The deserialized object.
    """
    return JSONSerializer().deserialize(data)


# ============================================================================
# JSON Schema Validation
# ============================================================================


@dataclasses.dataclass
class SchemaValidationError:
    """Schema validation error details."""

    path: str
    message: str
    value: Any


class SchemaValidator:
    """JSON Schema validator for serialized data."""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, data: Any) -> List[SchemaValidationError]:
        """Validate data against the schema.

        Args:
            data: The data to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[SchemaValidationError] = []
        self._validate_recursive(data, self.schema, "", errors)
        return errors

    def is_valid(self, data: Any) -> bool:
        """Check if data is valid against the schema.

        Args:
            data: The data to validate.

        Returns:
            True if valid, False otherwise.
        """
        return len(self.validate(data)) == 0

    def _validate_recursive(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
        errors: List[SchemaValidationError],
    ) -> None:
        """Recursively validate data against schema."""
        if "type" in schema:
            expected_type = schema["type"]
            if not self._check_type(data, expected_type, schema):
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"Expected type {expected_type}, got {type(data).__name__}",
                        value=data,
                    )
                )
                return

        if "enum" in schema and data not in schema["enum"]:
            errors.append(
                SchemaValidationError(
                    path=path,
                    message=f"Value must be one of {schema['enum']}, got {data}",
                    value=data,
                )
            )

        if "const" in schema and data != schema["const"]:
            errors.append(
                SchemaValidationError(
                    path=path,
                    message=f"Value must be {schema['const']}, got {data}",
                    value=data,
                )
            )

        if "minimum" in schema and isinstance(data, (int, float)):
            if data < schema["minimum"]:
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"Value {data} is less than minimum {schema['minimum']}",
                        value=data,
                    )
                )

        if "maximum" in schema and isinstance(data, (int, float)):
            if data > schema["maximum"]:
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"Value {data} is greater than maximum {schema['maximum']}",
                        value=data,
                    )
                )

        if "minLength" in schema and isinstance(data, str):
            if len(data) < schema["minLength"]:
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"String length {len(data)} is less than minLength {schema['minLength']}",
                        value=data,
                    )
                )

        if "maxLength" in schema and isinstance(data, str):
            if len(data) > schema["maxLength"]:
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"String length {len(data)} is greater than maxLength {schema['maxLength']}",
                        value=data,
                    )
                )

        if "pattern" in schema and isinstance(data, str):
            import re

            if not re.match(schema["pattern"], data):
                errors.append(
                    SchemaValidationError(
                        path=path,
                        message=f"String does not match pattern {schema['pattern']}",
                        value=data,
                    )
                )

        if "items" in schema and isinstance(data, list):
            for i, item in enumerate(data):
                self._validate_recursive(item, schema["items"], f"{path}[{i}]", errors)

        if "properties" in schema and isinstance(data, dict):
            for key, value in data.items():
                if key in schema["properties"]:
                    self._validate_recursive(
                        value, schema["properties"][key], f"{path}.{key}", errors
                    )

    def _check_type(
        self, data: Any, expected_type: str, schema: Dict[str, Any]
    ) -> bool:
        """Check if data matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        if expected_type not in type_map:
            return True

        expected = type_map[expected_type]
        return isinstance(data, expected)


# ============================================================================
# Pickle Serialization
# ============================================================================


class PickleSerializer(SerializerBase):
    """Pickle-based serializer with configurable protocol."""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to pickle bytes."""
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes to an object."""
        return pickle.loads(data)


def serialize_pickle(obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
    """Serialize an object to pickle bytes.

    Args:
        obj: The object to serialize.
        protocol: Pickle protocol version (default: highest available).

    Returns:
        Pickle-encoded bytes.
    """
    return PickleSerializer(protocol=protocol).serialize(obj)


def deserialize_pickle(data: bytes) -> Any:
    """Deserialize pickle bytes to an object.

    Args:
        data: Pickle-encoded bytes.

    Returns:
        The deserialized object.
    """
    return PickleSerializer().deserialize(data)


class SecurePickle(SerializerBase):
    """Secure pickle deserializer that restricts allowed classes.

    This prevents arbitrary code execution vulnerabilities in pickle.
    """

    def __init__(
        self,
        allowed_classes: Optional[List[Type]] = None,
        allowed_modules: Optional[List[str]] = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ):
        self.allowed_classes = set(allowed_classes or [])
        self.allowed_modules = set(allowed_modules or [])
        self.protocol = protocol

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to pickle bytes."""
        return pickle.dumps(obj, protocol=self.protocol)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes with class restrictions."""
        if self.allowed_classes or self.allowed_modules:
            restricted_loader = pickle.RestrictedUnpickler(
                self.allowed_classes, self.allowed_modules
            )
            return restricted_loader.loads(data)
        return pickle.loads(data)

    def add_allowed_class(self, cls: Type) -> None:
        """Add a class to the allowed list."""
        self.allowed_classes.add(cls)

    def add_allowed_module(self, module: str) -> None:
        """Add a module to the allowed list."""
        self.allowed_modules.add(module)


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows specific classes/modules."""

    def __init__(self, allowed_classes: set, allowed_modules: set, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed_classes = allowed_classes
        self.allowed_modules = allowed_modules

    def find_class(self, module: str, name: str) -> Any:
        """Restrict class lookup to allowed classes/modules."""
        if module in self.allowed_modules:
            return super().find_class(module, name)

        for allowed_cls in self.allowed_classes:
            if allowed_cls.__module__ == module and allowed_cls.__name__ == name:
                return allowed_cls

        raise pickle.UnpicklingError(
            f"Class {module}.{name} is not allowed. "
            f"Allowed modules: {self.allowed_modules}, "
            f"Allowed classes: {[c.__name__ for c in self.allowed_classes]}"
        )


# ============================================================================
# MessagePack Serialization
# ============================================================================


class MessagePackSerializer(SerializerBase):
    """MessagePack-based serializer for efficient binary encoding."""

    def __init__(
        self, use_bin_type: bool = True, raw: bool = False, strict_types: bool = True
    ):
        if not HAS_MSGPACK:
            raise ImportError("msgpack package is required for MessagePackSerializer")
        self.use_bin_type = use_bin_type
        self.raw = raw
        self.strict_types = strict_types

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to MessagePack bytes."""
        return msgpack.packb(
            obj,
            use_bin_type=self.use_bin_type,
            strict_types=self.strict_types,
        )

    def deserialize(self, data: bytes) -> Any:
        """Deserialize MessagePack bytes to an object."""
        return msgpack.unpackb(data, raw=self.raw)

    def serialize_to_file(self, obj: Any, filepath: Union[str, Path]) -> None:
        """Serialize an object to a MessagePack file."""
        with open(filepath, "wb") as f:
            msgpack.packb(obj, f, use_bin_type=self.use_bin_type)

    def deserialize_from_file(self, filepath: Union[str, Path]) -> Any:
        """Deserialize a MessagePack file to an object."""
        with open(filepath, "rb") as f:
            return msgpack.unpackb(f.read(), raw=self.raw)


def serialize_msgpack(obj: Any) -> bytes:
    """Serialize an object to MessagePack bytes.

    Args:
        obj: The object to serialize.

    Returns:
        MessagePack-encoded bytes.
    """
    return MessagePackSerializer().serialize(obj)


def deserialize_msgpack(data: bytes) -> Any:
    """Deserialize MessagePack bytes to an object.

    Args:
        data: MessagePack-encoded bytes.

    Returns:
        The deserialized object.
    """
    return MessagePackSerializer().deserialize(data)


# ============================================================================
# Protobuf Serialization
# ============================================================================


class ProtobufSerializer(SerializerBase):
    """Protobuf-based serializer for efficient binary encoding."""

    def __init__(self, message_class: Optional[Type] = None):
        if not HAS_PROTOBUF:
            raise ImportError("protobuf package is required for ProtobufSerializer")
        self.message_class = message_class

    def serialize(self, obj: Any) -> bytes:
        """Serialize a protobuf message to bytes."""
        if self.message_class is None:
            raise ValueError("message_class must be set for serialization")
        if hasattr(obj, "SerializeToString"):
            return obj.SerializeToString()
        return bytes(obj)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to a protobuf message."""
        if self.message_class is None:
            raise ValueError("message_class must be set for deserialization")
        msg = self.message_class()
        msg.ParseFromString(data)
        return msg

    def serialize_dict(self, obj: Dict[str, Any]) -> bytes:
        """Serialize a dictionary to protobuf bytes."""
        if self.message_class is None:
            raise ValueError("message_class must be set")
        msg = self.message_class(**obj)
        return msg.SerializeToString()

    def deserialize_to_dict(self, data: bytes) -> Dict[str, Any]:
        """Deserialize protobuf bytes to a dictionary."""
        if self.message_class is None:
            raise ValueError("message_class must be set")
        msg = self.message_class()
        msg.ParseFromString(data)
        return dict(msg)


def generate_proto(
    schema: Dict[str, Any],
    message_name: str = "GeneratedMessage",
    package_name: str = "generated",
) -> str:
    """Generate a protobuf schema from a dictionary definition.

    Args:
        schema: Dictionary defining the protobuf message structure.
        message_name: Name for the generated message.
        package_name: Protobuf package name.

    Returns:
        Protobuf schema as a string.
    """
    lines = [f'syntax = "proto3";', "", f"package {package_name};", ""]

    def get_proto_type(field_type: str) -> str:
        type_map = {
            "string": "string",
            "int": "int32",
            "int32": "int32",
            "int64": "int64",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "bytes": "bytes",
        }
        return type_map.get(field_type, field_type)

    def generate_message(
        schema: Dict[str, Any], name: str, indent: int = 0
    ) -> List[str]:
        msg_lines = [f"{'  ' * indent}message {name} {{"]
        field_number = 1

        for field_name, field_def in schema.items():
            if isinstance(field_def, dict):
                if "type" in field_def:
                    proto_type = get_proto_type(field_def["type"])
                    repeated = "repeated " if field_def.get("repeated", False) else ""
                    optional = "optional " if field_def.get("optional", False) else ""
                    msg_lines.append(
                        f"{'  ' * (indent + 1)}{repeated}{optional}{proto_type} {field_name} = {field_number};"
                    )
                    field_number += 1
                elif "message" in field_def:
                    nested = generate_message(
                        field_def["message"], field_name, indent + 1
                    )
                    msg_lines.extend(nested)
                    msg_lines.append(f"{'  ' * (indent + 1)}message {field_name} {{")
                    field_number += 1

        msg_lines.append(f"{'  ' * indent}}}")
        return msg_lines

    lines.extend(generate_message(schema, message_name))
    return "\n".join(lines)


def compile_proto(
    proto_content: str,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Compile a protobuf schema string to a message class.

    Args:
        proto_content: Protobuf schema as a string.
        output_dir: Optional directory for generated files.

    Returns:
        Dictionary with compiled message classes.
    """
    if not HAS_PROTOBUF:
        raise ImportError("protobuf package is required for compile_proto")

    from google.protobuf import descriptor_pb2
    from google.protobuf import descriptor_pool
    from google.protobuf import message_factory
    from google.protobuf import symbol_database

    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "generated.proto"
    file_descriptor.proto = proto_content

    return {"file_descriptor": file_descriptor, "message_classes": {}}


# ============================================================================
# YAML Serialization
# ============================================================================


class YAMLSerializer(SerializerBase):
    """YAML-based serializer with configurable options."""

    def __init__(
        self,
        default_flow_style: bool = False,
        sort_keys: bool = False,
        indent: int = 2,
        allow_unicode: bool = True,
    ):
        if not HAS_YAML:
            raise ImportError("yaml package is required for YAMLSerializer")
        self.default_flow_style = default_flow_style
        self.sort_keys = sort_keys
        self.indent = indent
        self.allow_unicode = allow_unicode

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to YAML bytes."""
        return yaml.dump(
            obj,
            default_flow_style=self.default_flow_style,
            sort_keys=self.sort_keys,
            indent=self.indent,
            allow_unicode=self.allow_unicode,
        ).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize YAML bytes to an object."""
        return yaml.safe_load(data.decode("utf-8"))

    def serialize_to_str(self, obj: Any) -> str:
        """Serialize an object to a YAML string."""
        return yaml.dump(
            obj,
            default_flow_style=self.default_flow_style,
            sort_keys=self.sort_keys,
            indent=self.indent,
            allow_unicode=self.allow_unicode,
        )

    def deserialize_from_str(self, data: str) -> Any:
        """Deserialize a YAML string to an object."""
        return yaml.safe_load(data)

    def serialize_all(self, obj: List[Any]) -> bytes:
        """Serialize multiple documents to YAML bytes."""
        return yaml.dump_all(
            obj,
            default_flow_style=self.default_flow_style,
            sort_keys=self.sort_keys,
            indent=self.indent,
            allow_unicode=self.allow_unicode,
        ).encode("utf-8")

    def deserialize_all(self, data: bytes) -> List[Any]:
        """Deserialize multiple YAML documents."""
        return list(yaml.safe_load_all(data.decode("utf-8")))


def serialize_yaml(obj: Any, **kwargs) -> bytes:
    """Serialize an object to YAML bytes.

    Args:
        obj: The object to serialize.
        **kwargs: Additional arguments passed to YAMLSerializer.

    Returns:
        YAML-encoded bytes.
    """
    return YAMLSerializer(**kwargs).serialize(obj)


def deserialize_yaml(data: bytes) -> Any:
    """Deserialize YAML bytes to an object.

    Args:
        data: YAML-encoded bytes.

    Returns:
        The deserialized object.
    """
    return YAMLSerializer().deserialize(data)


# ============================================================================
# Compression
# ============================================================================


class CompressionBase(abc.ABC):
    """Abstract base class for compression implementations."""

    @abc.abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abc.abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass


class GzipCompress(CompressionBase):
    """GZIP compression implementation."""

    def __init__(self, compresslevel: int = 9):
        self.compresslevel = compresslevel

    def compress(self, data: bytes) -> bytes:
        """Compress data using GZIP."""
        return gzip.compress(data, compresslevel=self.compresslevel)

    def decompress(self, data: bytes) -> bytes:
        """Decompress GZIP data."""
        return gzip.decompress(data)


class BZ2Compress(CompressionBase):
    """BZ2 compression implementation."""

    def __init__(self, compresslevel: int = 9):
        self.compresslevel = compresslevel

    def compress(self, data: bytes) -> bytes:
        """Compress data using BZ2."""
        return bz2.compress(data, compresslevel=self.compresslevel)

    def decompress(self, data: bytes) -> bytes:
        """Decompress BZ2 data."""
        return bz2.decompress(data)


class LZMACompress(CompressionBase):
    """LZMA compression implementation."""

    def __init__(self, preset: int = 6):
        self.preset = preset

    def compress(self, data: bytes) -> bytes:
        """Compress data using LZMA."""
        return lzma.compress(data, preset=self.preset)

    def decompress(self, data: bytes) -> bytes:
        """Decompress LZMA data."""
        return lzma.decompress(data)


class ZstandardCompress(CompressionBase):
    """Zstandard compression implementation."""

    def __init__(self, level: int = 3):
        if not HAS_ZSTD:
            raise ImportError("zstandard package is required for ZstandardCompress")
        self.level = level
        self._compressor = zstd.ZstdCompressor(level=self.level)
        self._decompressor = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        """Compress data using Zstandard."""
        return self._compressor.compress(data)

    def decompress(self, data: bytes) -> bytes:
        """Decompress Zstandard data."""
        return self._decompressor.decompress(data)

    def compress_dict(self, data: bytes, dictionary: bytes) -> bytes:
        """Compress data using a dictionary."""
        compressor = zstd.ZstdCompressor(level=self.level)
        compressor.load_dict(dictionary)
        return compressor.compress(data)

    def decompress_dict(self, data: bytes, dictionary: bytes) -> bytes:
        """Decompress data using a dictionary."""
        decompressor = zstd.ZstdDecompressor()
        decompressor.load_dict(dictionary)
        return decompressor.decompress(data)


# ============================================================================
# Encoding
# ============================================================================


class Base64Encode:
    """Base64 encoding and decoding utilities."""

    @staticmethod
    def encode(data: Union[bytes, str]) -> str:
        """Encode data to Base64 string."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def decode(data: str) -> bytes:
        """Decode Base64 string to bytes."""
        return base64.b64decode(data)

    @staticmethod
    def encode_url(data: Union[bytes, str]) -> str:
        """Encode data to URL-safe Base64 string."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    @staticmethod
    def decode_url(data: str) -> bytes:
        """Decode URL-safe Base64 string to bytes."""
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)


class URLEncode:
    """URL encoding and decoding utilities."""

    @staticmethod
    def encode(data: str, safe: str = "") -> str:
        """URL encode a string."""
        from urllib.parse import quote

        return quote(data, safe=safe)

    @staticmethod
    def decode(data: str) -> str:
        """URL decode a string."""
        from urllib.parse import unquote

        return unquote(data)

    @staticmethod
    def encode_dict(data: Dict[str, Any]) -> str:
        """URL encode a dictionary."""
        from urllib.parse import urlencode

        return urlencode(data)

    @staticmethod
    def decode_dict(data: str) -> Dict[str, str]:
        """URL decode to a dictionary."""
        from urllib.parse import parse_qs

        result = parse_qs(data)
        return {k: v[0] if len(v) == 1 else v for k, v in result.items()}


class HTMLEncode:
    """HTML encoding and decoding utilities."""

    ESCAPE_TABLE = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "/": "&#x2F;",
    }

    UNESCAPE_TABLE = {v: k for k, v in ESCAPE_TABLE.items()}

    @staticmethod
    def encode(data: str, quote: bool = True) -> str:
        """HTML encode a string."""
        if quote:
            import html

            return html.escape(data, quote=quote)
        for char, escape in HTMLEncode.ESCAPE_TABLE.items():
            data = data.replace(char, escape)
        return data

    @staticmethod
    def decode(data: str) -> str:
        """HTML decode a string."""
        import html

        return html.unescape(data)

    @staticmethod
    def encode_dict(data: Dict[str, Any]) -> Dict[str, str]:
        """HTML encode all values in a dictionary."""
        return {k: HTMLEncode.encode(str(v)) for k, v in data.items()}

    @staticmethod
    def decode_dict(data: Dict[str, str]) -> Dict[str, str]:
        """HTML decode all values in a dictionary."""
        return {k: HTMLEncode.decode(v) for k, v in data.items()}


# ============================================================================
# Unified Serialization API
# ============================================================================


class SerializationFormat:
    """Enumeration of supported serialization formats."""

    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    YAML = "yaml"
    PROTOBUF = "protobuf"


class CompressFormat:
    """Enumeration of supported compression formats."""

    GZIP = "gzip"
    BZ2 = "bz2"
    LZMA = "lzma"
    ZSTD = "zstd"


class EncodeFormat:
    """Enumeration of supported encoding formats."""

    BASE64 = "base64"
    URL = "url"
    HTML = "html"


def serialize(
    obj: Any,
    format: str = SerializationFormat.JSON,
    compress: Optional[str] = None,
    encode: Optional[str] = None,
    **kwargs,
) -> Union[bytes, str]:
    """Unified serialization function.

    Args:
        obj: The object to serialize.
        format: Serialization format (json, pickle, msgpack, yaml, protobuf).
        compress: Optional compression (gzip, bz2, lzma, zstd).
        encode: Optional encoding (base64, url, html).
        **kwargs: Additional format-specific arguments.

    Returns:
        Serialized data as bytes or string depending on encoding.
    """
    serializer: SerializerBase

    if format == SerializationFormat.JSON:
        serializer = JSONSerializer(**kwargs)
    elif format == SerializationFormat.PICKLE:
        serializer = PickleSerializer(**kwargs)
    elif format == SerializationFormat.MSGPACK:
        serializer = MessagePackSerializer(**kwargs)
    elif format == SerializationFormat.YAML:
        serializer = YAMLSerializer(**kwargs)
    elif format == SerializationFormat.PROTOBUF:
        serializer = ProtobufSerializer(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    data = serializer.serialize(obj)

    if compress == CompressFormat.GZIP:
        data = GzipCompress().compress(data)
    elif compress == CompressFormat.BZ2:
        data = BZ2Compress().compress(data)
    elif compress == CompressFormat.LZMA:
        data = LZMACompress().compress(data)
    elif compress == CompressFormat.ZSTD:
        data = ZstandardCompress().compress(data)

    if encode == EncodeFormat.BASE64:
        return Base64Encode.encode(data)
    elif encode == EncodeFormat.URL:
        return URLEncode.encode(data.decode("latin-1"))
    elif encode == EncodeFormat.HTML:
        return HTMLEncode.encode(data.decode("latin-1"))

    return data


def deserialize(
    data: Union[bytes, str],
    format: str = SerializationFormat.JSON,
    compress: Optional[str] = None,
    encode: Optional[str] = None,
    **kwargs,
) -> Any:
    """Unified deserialization function.

    Args:
        data: The serialized data.
        format: Serialization format (json, pickle, msgpack, yaml, protobuf).
        compress: Optional compression (gzip, bz2, lzma, zstd).
        encode: Optional encoding (base64, url, html).
        **kwargs: Additional format-specific arguments.

    Returns:
        The deserialized object.
    """
    if isinstance(data, str):
        if encode == EncodeFormat.BASE64:
            data = Base64Encode.decode(data)
        elif encode == EncodeFormat.URL:
            data = URLEncode.decode(data).encode("latin-1")
        elif encode == EncodeFormat.HTML:
            data = HTMLEncode.decode(data).encode("latin-1")
        else:
            data = data.encode("utf-8")

    if isinstance(data, bytes):
        if compress == CompressFormat.GZIP:
            data = GzipCompress().decompress(data)
        elif compress == CompressFormat.BZ2:
            data = BZ2Compress().decompress(data)
        elif compress == CompressFormat.LZMA:
            data = LZMACompress().decompress(data)
        elif compress == CompressFormat.ZSTD:
            data = ZstandardCompress().decompress(data)

    serializer: SerializerBase

    if format == SerializationFormat.JSON:
        serializer = JSONSerializer(**kwargs)
    elif format == SerializationFormat.PICKLE:
        serializer = PickleSerializer(**kwargs)
    elif format == SerializationFormat.MSGPACK:
        serializer = MessagePackSerializer(**kwargs)
    elif format == SerializationFormat.YAML:
        serializer = YAMLSerializer(**kwargs)
    elif format == SerializationFormat.PROTOBUF:
        serializer = ProtobufSerializer(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return serializer.deserialize(data)


# ============================================================================
# Serialization Utilities
# ============================================================================


def get_serializer(
    format: str,
    **kwargs,
) -> SerializerBase:
    """Get a serializer instance for the specified format.

    Args:
        format: Serialization format.
        **kwargs: Format-specific arguments.

    Returns:
        Serializer instance.
    """
    if format == SerializationFormat.JSON:
        return JSONSerializer(**kwargs)
    elif format == SerializationFormat.PICKLE:
        return PickleSerializer(**kwargs)
    elif format == SerializationFormat.MSGPACK:
        return MessagePackSerializer(**kwargs)
    elif format == SerializationFormat.YAML:
        return YAMLSerializer(**kwargs)
    elif format == SerializationFormat.PROTOBUF:
        return ProtobufSerializer(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_compressor(format: str, **kwargs) -> CompressionBase:
    """Get a compressor instance for the specified format.

    Args:
        format: Compression format.
        **kwargs: Format-specific arguments.

    Returns:
        Compressor instance.
    """
    if format == CompressFormat.GZIP:
        return GzipCompress(**kwargs)
    elif format == CompressFormat.BZ2:
        return BZ2Compress(**kwargs)
    elif format == CompressFormat.LZMA:
        return LZMACompress(**kwargs)
    elif format == CompressFormat.ZSTD:
        return ZstandardCompress(**kwargs)
    else:
        raise ValueError(f"Unsupported compression format: {format}")


def hash_data(data: bytes, algorithm: str = "sha256") -> str:
    """Calculate hash of serialized data.

    Args:
        data: Data to hash.
        algorithm: Hash algorithm (md5, sha1, sha256, sha512).

    Returns:
        Hex digest of the hash.
    """
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def verify_hash(data: bytes, hash_value: str, algorithm: str = "sha256") -> bool:
    """Verify data hash.

    Args:
        data: Data to verify.
        hash_value: Expected hash value.
        algorithm: Hash algorithm used.

    Returns:
        True if hash matches, False otherwise.
    """
    return hash_data(data, algorithm) == hash_value


def get_size(data: bytes) -> Dict[str, int]:
    """Get size information for serialized data.

    Args:
        data: Serialized data.

    Returns:
        Dictionary with size information in bytes.
    """
    return {
        "raw_size": len(data),
        "size_kb": len(data) / 1024,
        "size_mb": len(data) / (1024 * 1024),
    }


# ============================================================================
# Exportable API
# ============================================================================

__all__ = [
    # Base classes
    "Serializer",
    "SerializerBase",
    # JSON
    "JSONSerializer",
    "serialize_json",
    "deserialize_json",
    "SchemaValidator",
    # Pickle
    "PickleSerializer",
    "serialize_pickle",
    "deserialize_pickle",
    "SecurePickle",
    # MessagePack
    "MessagePackSerializer",
    "serialize_msgpack",
    # Protobuf
    "ProtobufSerializer",
    "generate_proto",
    "compile_proto",
    # YAML
    "YAMLSerializer",
    "serialize_yaml",
    "deserialize_yaml",
    # Compression
    "GzipCompress",
    "BZ2Compress",
    "LZMACompress",
    "ZstandardCompress",
    # Encoding
    "Base64Encode",
    "URLEncode",
    "HTMLEncode",
    # Unified API
    "serialize",
    "deserialize",
    "SerializationFormat",
    "CompressFormat",
    "EncodeFormat",
    # Utilities
    "get_serializer",
    "get_compressor",
    "hash_data",
    "verify_hash",
    "get_size",
]
