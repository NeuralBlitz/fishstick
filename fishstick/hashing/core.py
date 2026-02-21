"""
Fishstick Hashing Module - Comprehensive hashing utilities.

This module provides MD5, SHA, BLAKE, CRC, HMAC, password hashing,
MurmerHash, and utility functions for file and string hashing.
"""

import hashlib
import hmac
import os
import struct
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
from dataclasses import dataclass


class HashingError(Exception):
    """Base exception for hashing operations."""

    pass


class HashVerificationError(HashingError):
    """Exception raised when hash verification fails."""

    pass


@dataclass
class HashResult:
    """Container for hash computation results."""

    algorithm: str
    hash: str
    hex: str
    digest: bytes

    def __str__(self) -> str:
        return self.hex


class HashAlgorithm(ABC):
    """Abstract base class for hash algorithms."""

    @abstractmethod
    def compute(self, data: Union[bytes, str]) -> HashResult:
        """Compute hash of data."""
        pass

    @abstractmethod
    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        """Verify data against expected hash."""
        pass


class MD5Hash(HashAlgorithm):
    """MD5 hash implementation."""

    def __init__(self):
        self.name = "MD5"
        self.digest_size = 16

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.md5(data).digest()
        hex_hash = hashlib.md5(data).hexdigest()
        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class SHA1Hash(HashAlgorithm):
    """SHA1 hash implementation."""

    def __init__(self):
        self.name = "SHA1"
        self.digest_size = 20

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.sha1(data).digest()
        hex_hash = hashlib.sha1(data).hexdigest()
        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class SHA256Hash(HashAlgorithm):
    """SHA256 hash implementation."""

    def __init__(self):
        self.name = "SHA256"
        self.digest_size = 32

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.sha256(data).digest()
        hex_hash = hashlib.sha256(data).hexdigest()
        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class SHA512Hash(HashAlgorithm):
    """SHA512 hash implementation."""

    def __init__(self):
        self.name = "SHA512"
        self.digest_size = 64

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.sha512(data).digest()
        hex_hash = hashlib.sha512(data).hexdigest()
        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class BLAKE2bHash(HashAlgorithm):
    """BLAKE2b hash implementation."""

    def __init__(self, digest_size: int = 64):
        self.name = "BLAKE2b"
        self.digest_size = min(digest_size, 64)

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.blake2b(data, digest_size=self.digest_size).digest()
        hex_hash = hashlib.blake2b(data, digest_size=self.digest_size).hexdigest()
        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class BLAKE2sHash(HashAlgorithm):
    """BLAKE2s hash implementation."""

    def __init__(self, digest_size: int = 32):
        self.name = "BLAKE2s"
        self.digest_size = min(digest_size, 32)

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")
        digest = hashlib.blake2s(data, digest_size=self.digest_size).digest()
        hex_hash = hashlib.blake2s(data, digest_size=self.digest_size).hexdigest()
        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class BLAKE3Hash(HashAlgorithm):
    """BLAKE3 hash implementation (requires blake3 package)."""

    def __init__(self):
        self.name = "BLAKE3"
        self.digest_size = 32
        self._blake3 = None
        try:
            import blake3

            self._blake3 = blake3
        except ImportError:
            pass

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")

        if self._blake3 is not None:
            digest = self._blake3.blake3(data).digest()
            hex_hash = self._blake3.blake3(data).hexdigest()
        else:
            raise HashingError("BLAKE3 requires blake3 package: pip install blake3")

        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class CRC32Hash:
    """CRC32 hash implementation."""

    def __init__(self):
        self.name = "CRC32"
        self._table = self._build_table()

    def _build_table(self):
        table = [0] * 256
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc >>= 1
            table[i] = crc
        return table

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")

        crc = 0xFFFFFFFF
        for byte in data:
            crc = self._table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
        crc ^= 0xFFFFFFFF

        digest = struct.pack(">I", crc)
        hex_hash = format(crc, "08x")

        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class CRC64Hash:
    """CRC64 hash implementation."""

    def __init__(self):
        self.name = "CRC64"
        self._table = self._build_table()
        self.poly = 0xD58247BF

    def _build_table(self):
        table = [0] * 256
        for i in range(256):
            crc = i << 48
            for _ in range(8):
                if crc & 0x800000000000:
                    crc = (crc << 1) ^ (self.poly << 40)
                else:
                    crc <<= 1
                crc &= 0xFFFFFFFFFFFF
            table[i] = crc
        return table

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")

        crc = 0
        for byte in data:
            crc = self._table[(crc >> 56) ^ (byte & 0xFF)] ^ (crc << 8)
            crc &= 0xFFFFFFFFFFFF

        digest = struct.pack(">Q", crc)
        hex_hash = format(crc, "016x")

        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class CRC32:
    """CRC32 functional interface."""

    @staticmethod
    def hash(data: Union[bytes, str]) -> str:
        hasher = CRC32Hash()
        return hasher.compute(data).hex


class HMAC:
    """HMAC implementation."""

    def __init__(self, key: Union[bytes, str], algorithm: str = "sha256"):
        if isinstance(key, str):
            key = key.encode("utf-8")
        self.key = key
        self.algorithm = algorithm

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")

        if self.algorithm == "sha256":
            h = hmac.new(self.key, data, hashlib.sha256)
            algo_name = "HMAC-SHA256"
        elif self.algorithm == "sha512":
            h = hmac.new(self.key, data, hashlib.sha512)
            algo_name = "HMAC-SHA512"
        elif self.algorithm == "sha1":
            h = hmac.new(self.key, data, hashlib.sha1)
            algo_name = "HMAC-SHA1"
        elif self.algorithm == "md5":
            h = hmac.new(self.key, data, hashlib.md5)
            algo_name = "HMAC-MD5"
        else:
            raise HashingError(f"Unsupported HMAC algorithm: {self.algorithm}")

        digest = h.digest()
        hex_hash = h.hexdigest()

        return HashResult(
            algorithm=algo_name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class BCryptHash:
    """BCrypt password hashing (requires bcrypt package)."""

    def __init__(self, rounds: int = 12):
        self.name = "BCrypt"
        self.rounds = rounds
        self._bcrypt = None
        try:
            import bcrypt

            self._bcrypt = bcrypt
        except ImportError:
            pass

    def hash(self, password: Union[bytes, str]) -> str:
        if self._bcrypt is None:
            raise HashingError("BCrypt requires bcrypt package: pip install bcrypt")

        if isinstance(password, str):
            password = password.encode("utf-8")

        salt = self._bcrypt.gensalt(rounds=self.rounds)
        hashed = self._bcrypt.hashpw(password, salt)
        return hashed.decode("utf-8")

    def verify(self, password: Union[bytes, str], hashed: str) -> bool:
        if self._bcrypt is None:
            raise HashingError("BCrypt requires bcrypt package: pip install bcrypt")

        if isinstance(password, str):
            password = password.encode("utf-8")

        return self._bcrypt.checkpw(password, hashed.encode("utf-8"))


class Argon2Hash:
    """Argon2 password hashing (requires argon2-cffi package)."""

    def __init__(
        self, time_cost: int = 2, memory_cost: int = 65536, parallelism: int = 2
    ):
        self.name = "Argon2"
        self.time_cost = time_cost
        self.memory_cost = memory_cost
        self.parallelism = parallelism
        self._argon2 = None
        try:
            from argon2 import PasswordHasher

            self._ph = PasswordHasher(
                time_cost=time_cost, memory_cost=memory_cost, parallelism=parallelism
            )
        except ImportError:
            pass

    def hash(self, password: Union[bytes, str]) -> str:
        if self._ph is None:
            raise HashingError(
                "Argon2 requires argon2-cffi package: pip install argon2-cffi"
            )

        if isinstance(password, str):
            password = password.encode("utf-8")

        return self._ph.hash(
            password.decode("utf-8") if isinstance(password, bytes) else password
        )

    def verify(self, password: Union[bytes, str], hashed: str) -> bool:
        if self._ph is None:
            raise HashingError(
                "Argon2 requires argon2-cffi package: pip install argon2-cffi"
            )

        try:
            self._ph.verify(
                hashed,
                password if isinstance(password, str) else password.decode("utf-8"),
            )
            return True
        except Exception:
            return False


class PBKDF2Hash:
    """PBKDF2 password hashing."""

    def __init__(
        self, iterations: int = 100000, key_length: int = 32, salt_length: int = 16
    ):
        self.name = "PBKDF2"
        self.iterations = iterations
        self.key_length = key_length
        self.salt_length = salt_length

    def hash(
        self, password: Union[bytes, str], salt: Optional[bytes] = None
    ) -> Tuple[str, str]:
        if isinstance(password, str):
            password = password.encode("utf-8")

        if salt is None:
            salt = os.urandom(self.salt_length)

        derived_key = hashlib.pbkdf2_hmac(
            "sha256", password, salt, self.iterations, dklen=self.key_length
        )

        return salt.hex(), derived_key.hex()

    def verify(self, password: Union[bytes, str], salt: str, hashed: str) -> bool:
        if isinstance(password, str):
            password = password.encode("utf-8")

        salt_bytes = bytes.fromhex(salt)
        derived_key = hashlib.pbkdf2_hmac(
            "sha256", password, salt_bytes, self.iterations, dklen=self.key_length
        )

        return derived_key.hex() == hashed


class MurmurHash2:
    """MurmurHash2 implementation."""

    def __init__(self, seed: int = 0):
        self.name = "MurmurHash2"
        self.seed = seed

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")

        m = 0x5BD1E995
        r = 24
        length = len(data)
        h = self.seed ^ length

        for i in range(0, length - length % 4, 4):
            k = struct.unpack("<I", data[i : i + 4])[0]
            k = (k * m) & 0xFFFFFFFF
            k ^= k >> r
            k = (k * m) & 0xFFFFFFFF
            h = (h * m) & 0xFFFFFFFF
            h ^= k
            h = (h * m) & 0xFFFFFFFF

        remaining = length % 4
        if remaining >= 3:
            h ^= data[i + 2] << 16
        if remaining >= 2:
            h ^= data[i + 1] << 8
        if remaining >= 1:
            h ^= data[i]
            h = (h * m) & 0xFFFFFFFF

        h ^= h >> 13
        h = (h * m) & 0xFFFFFFFF
        h ^= h >> 15

        digest = struct.pack("<I", h)
        hex_hash = format(h, "08x")

        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


class MurmurHash3:
    """MurmurHash3 implementation."""

    def __init__(self, seed: int = 0):
        self.name = "MurmurHash3"
        self.seed = seed

    def compute(self, data: Union[bytes, str]) -> HashResult:
        if isinstance(data, str):
            data = data.encode("utf-8")

        nblocks = len(data) // 4
        h1 = self.seed
        c1 = 0xCC9E2D51
        c2 = 0x1B873593

        for i in range(nblocks):
            k1 = struct.unpack("<I", data[i * 4 : (i + 1) * 4])[0]
            k1 = (k1 * c1) & 0xFFFFFFFF
            k1 = (k1 << 15) | (k1 >> 17)
            k1 = (k1 * c2) & 0xFFFFFFFF

            h1 ^= k1
            h1 = (h1 << 13) | (h1 >> 19)
            h1 = ((h1 * 5) + 0xE6546B64) & 0xFFFFFFFF

        tail = data[nblocks * 4 :]
        k1 = 0
        for i, byte in enumerate(tail):
            k1 ^= byte << (i * 8)

        if len(tail) > 0:
            k1 = (k1 * c1) & 0xFFFFFFFF
            k1 = (k1 << 15) | (k1 >> 17)
            k1 = (k1 * c2) & 0xFFFFFFFF
            h1 ^= k1

        h1 ^= len(data)
        h1 ^= h1 >> 16
        h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
        h1 ^= h1 >> 13
        h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
        h1 ^= h1 >> 16

        digest = struct.pack("<I", h1)
        hex_hash = format(h1, "08x")

        return HashResult(
            algorithm=self.name, hash=hex_hash, hex=hex_hash, digest=digest
        )

    def verify(self, data: Union[bytes, str], expected_hash: str) -> bool:
        result = self.compute(data)
        return result.hex.lower() == expected_hash.lower()


def murmur_hash(data: Union[bytes, str], seed: int = 0, version: int = 3) -> str:
    """Compute MurmurHash of data.

    Args:
        data: Data to hash
        seed: Seed value
        version: MurmurHash version (2 or 3)

    Returns:
        Hex string of hash
    """
    if version == 2:
        hasher = MurmurHash2(seed)
    else:
        hasher = MurmurHash3(seed)

    return hasher.compute(data).hex


def md5(data: Union[bytes, str]) -> str:
    """Compute MD5 hash of data.

    Args:
        data: Data to hash

    Returns:
        Hex string of MD5 hash
    """
    return MD5Hash().compute(data).hex


def md5_file(filepath: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of file.

    Args:
        filepath: Path to file
        chunk_size: Size of chunks to read

    Returns:
        Hex string of MD5 hash
    """
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_md5(data: Union[bytes, str], expected_hash: str) -> bool:
    """Verify MD5 hash of data.

    Args:
        data: Data to verify
        expected_hash: Expected hash value

    Returns:
        True if hash matches
    """
    return MD5Hash().verify(data, expected_hash)


def sha256(data: Union[bytes, str]) -> str:
    """Compute SHA256 hash of data.

    Args:
        data: Data to hash

    Returns:
        Hex string of SHA256 hash
    """
    return SHA256Hash().compute(data).hex


def sha512(data: Union[bytes, str]) -> str:
    """Compute SHA512 hash of data.

    Args:
        data: Data to hash

    Returns:
        Hex string of SHA512 hash
    """
    return SHA512Hash().compute(data).hex


def hmac_sha256(key: Union[bytes, str], data: Union[bytes, str]) -> str:
    """Compute HMAC-SHA256 of data.

    Args:
        key: HMAC key
        data: Data to hash

    Returns:
        Hex string of HMAC-SHA256
    """
    return HMAC(key, "sha256").compute(data).hex


def hmac_sha512(key: Union[bytes, str], data: Union[bytes, str]) -> str:
    """Compute HMAC-SHA512 of data.

    Args:
        key: HMAC key
        data: Data to hash

    Returns:
        Hex string of HMAC-SHA512
    """
    return HMAC(key, "sha512").compute(data).hex


def hash_file(filepath: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """Compute hash of file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, sha512, blake2b, blake2s)
        chunk_size: Size of chunks to read

    Returns:
        Hex string of hash
    """
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "blake2b":
        hasher = hashlib.blake2b()
    elif algorithm == "blake2s":
        hasher = hashlib.blake2s()
    else:
        raise HashingError(f"Unsupported algorithm: {algorithm}")

    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_string(data: str, algorithm: str = "sha256") -> str:
    """Compute hash of string.

    Args:
        data: String to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512, blake2b, blake2s, crc32)

    Returns:
        Hex string of hash
    """
    if algorithm == "md5":
        return md5(data)
    elif algorithm == "sha1":
        hasher = SHA1Hash()
    elif algorithm == "sha256":
        return sha256(data)
    elif algorithm == "sha512":
        return sha512(data)
    elif algorithm == "blake2b":
        return BLAKE2bHash().compute(data).hex
    elif algorithm == "blake2s":
        return BLAKE2sHash().compute(data).hex
    elif algorithm == "crc32":
        return CRC32.hash(data)
    else:
        raise HashingError(f"Unsupported algorithm: {algorithm}")

    return hasher.compute(data).hex


def verify_hash(
    data: Union[bytes, str], expected_hash: str, algorithm: str = "sha256"
) -> bool:
    """Verify hash of data.

    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm

    Returns:
        True if hash matches
    """
    if algorithm == "md5":
        return verify_md5(data, expected_hash)
    elif algorithm == "sha256":
        return SHA256Hash().verify(data, expected_hash)
    elif algorithm == "sha512":
        return SHA512Hash().verify(data, expected_hash)
    elif algorithm == "sha1":
        return SHA1Hash().verify(data, expected_hash)
    elif algorithm == "blake2b":
        return BLAKE2bHash().verify(data, expected_hash)
    elif algorithm == "blake2s":
        return BLAKE2sHash().verify(data, expected_hash)
    elif algorithm == "crc32":
        return CRC32Hash().verify(data, expected_hash)
    else:
        raise HashingError(f"Unsupported algorithm: {algorithm}")
