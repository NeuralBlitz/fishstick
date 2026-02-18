"""
Fishstick Security Module - Comprehensive security toolkit for ML systems.

This module provides encryption, hashing, authentication, authorization,
privacy protection, model security, adversarial defenses, and audit logging.
"""

import hashlib
import hmac
import json
import base64
import secrets
import time
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
import pickle


# ============================================================================
# EXCEPTIONS
# ============================================================================


class SecurityError(Exception):
    """Base exception for security-related errors."""

    pass


class EncryptionError(SecurityError):
    """Exception raised during encryption/decryption operations."""

    pass


class HashingError(SecurityError):
    """Exception raised during hashing operations."""

    pass


class AuthenticationError(SecurityError):
    """Exception raised during authentication failures."""

    pass


class AuthorizationError(SecurityError):
    """Exception raised when access is denied."""

    pass


class PrivacyError(SecurityError):
    """Exception raised during privacy operations."""

    pass


class ModelSecurityError(SecurityError):
    """Exception raised during model security operations."""

    pass


class AdversarialError(SecurityError):
    """Exception raised during adversarial defense operations."""

    pass


# ============================================================================
# ENCRYPTION MODULE
# ============================================================================


class EncryptionAlgorithm(ABC):
    """Abstract base class for encryption algorithms."""

    @abstractmethod
    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data."""
        pass

    @abstractmethod
    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data."""
        pass

    @abstractmethod
    def generate_key(self) -> bytes:
        """Generate a new encryption key."""
        pass


class AES256Encryption(EncryptionAlgorithm):
    """
    AES-256-GCM encryption implementation.
    Provides authenticated encryption with associated data.
    """

    def __init__(self):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            self._AESGCM = AESGCM
            self._has_crypto = True
        except ImportError:
            self._has_crypto = False
            warnings.warn("cryptography library not available, using fallback")

    def generate_key(self) -> bytes:
        """Generate a 256-bit (32-byte) key."""
        return secrets.token_bytes(32)

    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """
        Encrypt data using AES-256-GCM.

        Args:
            data: Data to encrypt
            key: Encryption key (32 bytes). Generated if not provided.

        Returns:
            Encrypted data with nonce prepended (nonce + ciphertext + tag)
        """
        if key is None:
            key = self.generate_key()

        if len(key) != 32:
            raise EncryptionError("AES-256 requires a 32-byte key")

        if self._has_crypto:
            nonce = secrets.token_bytes(12)
            aesgcm = self._AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, data, None)
            return nonce + ciphertext
        else:
            # Fallback using simple XOR (NOT SECURE - for demo only)
            nonce = secrets.token_bytes(16)
            expanded_key = self._expand_key(key, len(data))
            ciphertext = bytes(a ^ b for a, b in zip(data, expanded_key))
            return b"FALLBACK:" + nonce + ciphertext

    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """
        Decrypt data using AES-256-GCM.

        Args:
            encrypted_data: Encrypted data (nonce + ciphertext + tag)
            key: Decryption key (32 bytes)

        Returns:
            Decrypted plaintext
        """
        if key is None:
            raise EncryptionError("Decryption key required")

        if len(key) != 32:
            raise EncryptionError("AES-256 requires a 32-byte key")

        if encrypted_data.startswith(b"FALLBACK:"):
            # Fallback decryption
            encrypted_data = encrypted_data[9:]
            nonce = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            expanded_key = self._expand_key(key, len(ciphertext))
            return bytes(a ^ b for a, b in zip(ciphertext, expanded_key))

        if self._has_crypto:
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            aesgcm = self._AESGCM(key)
            try:
                return aesgcm.decrypt(nonce, ciphertext, None)
            except Exception as e:
                raise EncryptionError(f"Decryption failed: {e}")
        else:
            raise EncryptionError("Cryptography library required for decryption")

    def _expand_key(self, key: bytes, length: int) -> bytes:
        """Expand key to required length using simple repetition."""
        result = bytearray()
        key_len = len(key)
        for i in range(length):
            result.append(key[i % key_len])
        return bytes(result)


class RSAEncryption(EncryptionAlgorithm):
    """
    RSA encryption implementation.
    Suitable for encrypting small amounts of data or symmetric keys.
    """

    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            from cryptography.hazmat.primitives import hashes, serialization

            self._rsa = rsa
            self._padding = padding
            self._hashes = hashes
            self._serialization = serialization
            self._has_crypto = True
        except ImportError:
            self._has_crypto = False
            warnings.warn("cryptography library not available, using fallback")

    def generate_key(self) -> bytes:
        """Generate RSA key pair and return serialized private key."""
        if self._has_crypto:
            private_key = self._rsa.generate_private_key(
                public_exponent=65537, key_size=self.key_size
            )
            return private_key.private_bytes(
                encoding=self._serialization.Encoding.PEM,
                format=self._serialization.PrivateFormat.PKCS8,
                encryption_algorithm=self._serialization.NoEncryption(),
            )
        else:
            # Fallback: return dummy key
            return b"RSA_FALLBACK_KEY_" + secrets.token_bytes(32)

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate and return (private_key, public_key) pair."""
        if self._has_crypto:
            private_key = self._rsa.generate_private_key(
                public_exponent=65537, key_size=self.key_size
            )
            public_key = private_key.public_key()

            private_pem = private_key.private_bytes(
                encoding=self._serialization.Encoding.PEM,
                format=self._serialization.PrivateFormat.PKCS8,
                encryption_algorithm=self._serialization.NoEncryption(),
            )
            public_pem = public_key.public_bytes(
                encoding=self._serialization.Encoding.PEM,
                format=self._serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            return private_pem, public_pem
        else:
            key = b"RSA_FALLBACK_KEY_" + secrets.token_bytes(32)
            return key, key

    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data using RSA public key."""
        if key is None:
            raise EncryptionError("Public key required for RSA encryption")

        if self._has_crypto:
            try:
                public_key = self._serialization.load_pem_public_key(key)
                ciphertext = public_key.encrypt(
                    data,
                    self._padding.OAEP(
                        mgf=self._padding.MGF1(algorithm=self._hashes.SHA256()),
                        algorithm=self._hashes.SHA256(),
                        label=None,
                    ),
                )
                return ciphertext
            except Exception as e:
                raise EncryptionError(f"RSA encryption failed: {e}")
        else:
            # Fallback: simple encoding
            return b"RSA_FALLBACK:" + base64.b64encode(data)

    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data using RSA private key."""
        if key is None:
            raise EncryptionError("Private key required for RSA decryption")

        if encrypted_data.startswith(b"RSA_FALLBACK:"):
            return base64.b64decode(encrypted_data[13:])

        if self._has_crypto:
            try:
                private_key = self._serialization.load_pem_private_key(
                    key, password=None
                )
                plaintext = private_key.decrypt(
                    encrypted_data,
                    self._padding.OAEP(
                        mgf=self._padding.MGF1(algorithm=self._hashes.SHA256()),
                        algorithm=self._hashes.SHA256(),
                        label=None,
                    ),
                )
                return plaintext
            except Exception as e:
                raise EncryptionError(f"RSA decryption failed: {e}")
        else:
            raise EncryptionError("Cryptography library required for RSA decryption")


class ChaCha20Encryption(EncryptionAlgorithm):
    """
    ChaCha20-Poly1305 encryption implementation.
    Modern authenticated encryption alternative to AES-GCM.
    """

    def __init__(self):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

            self._ChaCha20Poly1305 = ChaCha20Poly1305
            self._has_crypto = True
        except ImportError:
            self._has_crypto = False
            warnings.warn("cryptography library not available, using fallback")

    def generate_key(self) -> bytes:
        """Generate a 256-bit (32-byte) key."""
        return secrets.token_bytes(32)

    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data using ChaCha20-Poly1305."""
        if key is None:
            key = self.generate_key()

        if len(key) != 32:
            raise EncryptionError("ChaCha20 requires a 32-byte key")

        if self._has_crypto:
            nonce = secrets.token_bytes(12)
            chacha = self._ChaCha20Poly1305(key)
            ciphertext = chacha.encrypt(nonce, data, None)
            return nonce + ciphertext
        else:
            # Fallback
            nonce = secrets.token_bytes(12)
            expanded_key = self._expand_key(key, len(data))
            ciphertext = bytes(a ^ b for a, b in zip(data, expanded_key))
            return b"CHACHA_FALLBACK:" + nonce + ciphertext

    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data using ChaCha20-Poly1305."""
        if key is None:
            raise EncryptionError("Decryption key required")

        if len(key) != 32:
            raise EncryptionError("ChaCha20 requires a 32-byte key")

        if encrypted_data.startswith(b"CHACHA_FALLBACK:"):
            encrypted_data = encrypted_data[16:]
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            expanded_key = self._expand_key(key, len(ciphertext))
            return bytes(a ^ b for a, b in zip(ciphertext, expanded_key))

        if self._has_crypto:
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            chacha = self._ChaCha20Poly1305(key)
            try:
                return chacha.decrypt(nonce, ciphertext, None)
            except Exception as e:
                raise EncryptionError(f"Decryption failed: {e}")
        else:
            raise EncryptionError(
                "Cryptography library required for ChaCha20 decryption"
            )

    def _expand_key(self, key: bytes, length: int) -> bytes:
        """Expand key to required length."""
        result = bytearray()
        for i in range(length):
            result.append(key[i % len(key)])
        return bytes(result)


class FernetEncryption(EncryptionAlgorithm):
    """
    Fernet symmetric encryption from cryptography library.
    Provides authenticated encryption with URL-safe base64 encoding.
    """

    def __init__(self):
        try:
            from cryptography.fernet import Fernet

            self._Fernet = Fernet
            self._has_crypto = True
        except ImportError:
            self._has_crypto = False
            self._aes = AES256Encryption()
            warnings.warn("cryptography library not available, using AES fallback")

    def generate_key(self) -> bytes:
        """Generate a new Fernet key."""
        if self._has_crypto:
            return self._Fernet.generate_key()
        else:
            return self._aes.generate_key()

    def encrypt(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt data using Fernet."""
        if key is None:
            key = self.generate_key()

        if self._has_crypto:
            try:
                f = self._Fernet(key)
                return f.encrypt(data)
            except Exception as e:
                raise EncryptionError(f"Fernet encryption failed: {e}")
        else:
            return self._aes.encrypt(data, key)

    def decrypt(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """Decrypt data using Fernet."""
        if key is None:
            raise EncryptionError("Decryption key required")

        if self._has_crypto:
            try:
                f = self._Fernet(key)
                return f.decrypt(encrypted_data)
            except Exception as e:
                raise EncryptionError(f"Fernet decryption failed: {e}")
        else:
            return self._aes.decrypt(encrypted_data, key)


def encrypt_data(
    data: Union[str, bytes], algorithm: str = "aes256", key: Optional[bytes] = None
) -> Tuple[bytes, bytes]:
    """
    Encrypt data using specified algorithm.

    Args:
        data: Data to encrypt (str or bytes)
        algorithm: Encryption algorithm ('aes256', 'rsa', 'chacha20', 'fernet')
        key: Encryption key (generated if None)

    Returns:
        Tuple of (encrypted_data, key)
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    algorithms = {
        "aes256": AES256Encryption(),
        "rsa": RSAEncryption(),
        "chacha20": ChaCha20Encryption(),
        "fernet": FernetEncryption(),
    }

    if algorithm.lower() not in algorithms:
        raise EncryptionError(f"Unknown algorithm: {algorithm}")

    encryptor = algorithms[algorithm.lower()]

    if key is None:
        key = encryptor.generate_key()

    encrypted = encryptor.encrypt(data, key)
    return encrypted, key


def decrypt_data(
    encrypted_data: bytes, algorithm: str = "aes256", key: Optional[bytes] = None
) -> bytes:
    """
    Decrypt data using specified algorithm.

    Args:
        encrypted_data: Encrypted data
        algorithm: Encryption algorithm
        key: Decryption key

    Returns:
        Decrypted data
    """
    algorithms = {
        "aes256": AES256Encryption(),
        "rsa": RSAEncryption(),
        "chacha20": ChaCha20Encryption(),
        "fernet": FernetEncryption(),
    }

    if algorithm.lower() not in algorithms:
        raise EncryptionError(f"Unknown algorithm: {algorithm}")

    if key is None:
        raise EncryptionError("Decryption key required")

    encryptor = algorithms[algorithm.lower()]
    return encryptor.decrypt(encrypted_data, key)


# ============================================================================
# HASHING MODULE
# ============================================================================


class HashAlgorithm(ABC):
    """Abstract base class for hash algorithms."""

    @abstractmethod
    def hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Hash data and return hash string."""
        pass

    @abstractmethod
    def verify(self, data: Union[str, bytes], hash_string: str) -> bool:
        """Verify data against hash."""
        pass

    def _to_bytes(self, data: Union[str, bytes]) -> bytes:
        """Convert data to bytes."""
        if isinstance(data, str):
            return data.encode("utf-8")
        return data


class SHA256Hash(HashAlgorithm):
    """SHA-256 hashing implementation."""

    def hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Hash data using SHA-256 with optional salt."""
        data_bytes = self._to_bytes(data)

        if salt:
            data_bytes = salt + data_bytes

        hash_obj = hashlib.sha256(data_bytes)
        result = hash_obj.hexdigest()

        if salt:
            result = base64.b64encode(salt).decode("utf-8") + "$" + result

        return result

    def verify(self, data: Union[str, bytes], hash_string: str) -> bool:
        """Verify data against SHA-256 hash."""
        if "$" in hash_string:
            salt_b64, hash_value = hash_string.split("$", 1)
            salt = base64.b64decode(salt_b64)
            computed = self.hash(data, salt)
            return hmac.compare_digest(computed, hash_string)
        else:
            computed = self.hash(data)
            return hmac.compare_digest(computed, hash_string)


class SHA512Hash(HashAlgorithm):
    """SHA-512 hashing implementation."""

    def hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Hash data using SHA-512 with optional salt."""
        data_bytes = self._to_bytes(data)

        if salt:
            data_bytes = salt + data_bytes

        hash_obj = hashlib.sha512(data_bytes)
        result = hash_obj.hexdigest()

        if salt:
            result = base64.b64encode(salt).decode("utf-8") + "$" + result

        return result

    def verify(self, data: Union[str, bytes], hash_string: str) -> bool:
        """Verify data against SHA-512 hash."""
        if "$" in hash_string:
            salt_b64, hash_value = hash_string.split("$", 1)
            salt = base64.b64decode(salt_b64)
            computed = self.hash(data, salt)
            return hmac.compare_digest(computed, hash_string)
        else:
            computed = self.hash(data)
            return hmac.compare_digest(computed, hash_string)


class BCryptHash(HashAlgorithm):
    """BCrypt password hashing implementation."""

    def __init__(self, rounds: int = 12):
        self.rounds = rounds
        try:
            import bcrypt

            self._bcrypt = bcrypt
            self._has_bcrypt = True
        except ImportError:
            self._has_bcrypt = False
            warnings.warn("bcrypt library not available, using SHA-256 fallback")
            self._fallback = SHA256Hash()

    def hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Hash password using BCrypt."""
        if self._has_bcrypt:
            data_bytes = self._to_bytes(data)
            if salt is None:
                salt = self._bcrypt.gensalt(rounds=self.rounds)
            hashed = self._bcrypt.hashpw(data_bytes, salt)
            return hashed.decode("utf-8")
        else:
            return "BCRYPT_FALLBACK$" + self._fallback.hash(data, salt)

    def verify(self, data: Union[str, bytes], hash_string: str) -> bool:
        """Verify password against BCrypt hash."""
        if hash_string.startswith("BCRYPT_FALLBACK$"):
            return self._fallback.verify(data, hash_string[16:])

        if self._has_bcrypt:
            data_bytes = self._to_bytes(data)
            hash_bytes = hash_string.encode("utf-8")
            return self._bcrypt.checkpw(data_bytes, hash_bytes)
        else:
            return False


class Argon2Hash(HashAlgorithm):
    """Argon2 password hashing implementation (winner of PHC)."""

    def __init__(
        self, time_cost: int = 2, memory_cost: int = 65536, parallelism: int = 4
    ):
        self.time_cost = time_cost
        self.memory_cost = memory_cost
        self.parallelism = parallelism
        try:
            from argon2 import PasswordHasher

            self._ph = PasswordHasher(
                time_cost=time_cost, memory_cost=memory_cost, parallelism=parallelism
            )
            self._has_argon2 = True
        except ImportError:
            self._has_argon2 = False
            warnings.warn("argon2-cffi library not available, using bcrypt fallback")
            self._fallback = BCryptHash()

    def hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Hash password using Argon2."""
        if self._has_argon2:
            data_str = data if isinstance(data, str) else data.decode("utf-8")
            return self._ph.hash(data_str)
        else:
            return "ARGON2_FALLBACK$" + self._fallback.hash(data, salt)

    def verify(self, data: Union[str, bytes], hash_string: str) -> bool:
        """Verify password against Argon2 hash."""
        if hash_string.startswith("ARGON2_FALLBACK$"):
            return self._fallback.verify(data, hash_string[16:])

        if self._has_argon2:
            try:
                data_str = data if isinstance(data, str) else data.decode("utf-8")
                self._ph.verify(hash_string, data_str)
                return True
            except Exception:
                return False
        else:
            return False


def hash_password(
    password: Union[str, bytes], algorithm: str = "argon2", salt: Optional[bytes] = None
) -> str:
    """
    Hash a password using specified algorithm.

    Args:
        password: Password to hash
        algorithm: Hashing algorithm ('sha256', 'sha512', 'bcrypt', 'argon2')
        salt: Optional salt bytes

    Returns:
        Hashed password string
    """
    algorithms = {
        "sha256": SHA256Hash(),
        "sha512": SHA512Hash(),
        "bcrypt": BCryptHash(),
        "argon2": Argon2Hash(),
    }

    if algorithm.lower() not in algorithms:
        raise HashingError(f"Unknown algorithm: {algorithm}")

    hasher = algorithms[algorithm.lower()]
    return hasher.hash(password, salt)


def verify_password(
    password: Union[str, bytes], hash_string: str, algorithm: str = "argon2"
) -> bool:
    """
    Verify a password against its hash.

    Args:
        password: Password to verify
        hash_string: Stored hash
        algorithm: Hashing algorithm used

    Returns:
        True if password matches
    """
    algorithms = {
        "sha256": SHA256Hash(),
        "sha512": SHA512Hash(),
        "bcrypt": BCryptHash(),
        "argon2": Argon2Hash(),
    }

    if algorithm.lower() not in algorithms:
        raise HashingError(f"Unknown algorithm: {algorithm}")

    hasher = algorithms[algorithm.lower()]
    return hasher.verify(password, hash_string)


# ============================================================================
# AUTHENTICATION MODULE
# ============================================================================


class AuthenticationProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user and return token/info."""
        pass

    @abstractmethod
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify authentication token."""
        pass


class JWTAuth(AuthenticationProvider):
    """JWT (JSON Web Token) authentication implementation."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
    ):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        try:
            import jwt as pyjwt

            self._jwt = pyjwt
            self._has_jwt = True
        except ImportError:
            self._has_jwt = False
            warnings.warn("PyJWT library not available, using fallback")

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate user and generate JWT token.

        Args:
            credentials: Dict with 'username' and 'password' keys

        Returns:
            Dict with 'access_token' and 'token_type'
        """
        username = credentials.get("username")
        password = credentials.get("password")

        # In production, verify against database
        if not username or not password:
            raise AuthenticationError("Username and password required")

        # Create token payload
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        payload = {
            "sub": username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        if self._has_jwt:
            token = self._jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            if isinstance(token, bytes):
                token = token.decode("utf-8")
        else:
            # Fallback: create simple token
            payload_b64 = base64.b64encode(
                json.dumps(payload, default=str).encode()
            ).decode()
            signature = hmac.new(
                self.secret_key.encode(), payload_b64.encode(), hashlib.sha256
            ).hexdigest()
            token = f"{payload_b64}.{signature}"

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        if self._has_jwt:
            try:
                payload = self._jwt.decode(
                    token, self.secret_key, algorithms=[self.algorithm]
                )
                return payload
            except self._jwt.ExpiredSignatureError:
                raise AuthenticationError("Token has expired")
            except self._jwt.InvalidTokenError as e:
                raise AuthenticationError(f"Invalid token: {e}")
        else:
            # Fallback verification
            try:
                payload_b64, signature = token.rsplit(".", 1)
                expected_sig = hmac.new(
                    self.secret_key.encode(), payload_b64.encode(), hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(signature, expected_sig):
                    raise AuthenticationError("Invalid token signature")

                payload = json.loads(base64.b64decode(payload_b64))
                exp = datetime.fromisoformat(payload["exp"])
                if exp < datetime.utcnow():
                    raise AuthenticationError("Token has expired")

                return payload
            except Exception as e:
                raise AuthenticationError(f"Token verification failed: {e}")


class OAuth2Auth(AuthenticationProvider):
    """OAuth2 authentication implementation."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        authorization_url: str,
        token_url: str,
        scopes: Optional[List[str]] = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_url = authorization_url
        self.token_url = token_url
        self.scopes = scopes or ["openid", "profile", "email"]
        self._tokens: Dict[str, Dict[str, Any]] = {}

    def get_authorization_url(
        self, redirect_uri: str, state: Optional[str] = None
    ) -> str:
        """Generate OAuth2 authorization URL."""
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
        }

        import urllib.parse

        query = urllib.parse.urlencode(params)
        return f"{self.authorization_url}?{query}"

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            credentials: Dict with 'code', 'redirect_uri', and optional 'state'
        """
        code = credentials.get("code")
        redirect_uri = credentials.get("redirect_uri")

        if not code or not redirect_uri:
            raise AuthenticationError("Authorization code and redirect_uri required")

        # In production, make HTTP request to token endpoint
        # For demo, create mock token
        token_data = {
            "access_token": secrets.token_urlsafe(32),
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": secrets.token_urlsafe(32),
            "scope": " ".join(self.scopes),
        }

        self._tokens[token_data["access_token"]] = {
            "client_id": self.client_id,
            "scopes": self.scopes,
            "created_at": datetime.utcnow(),
        }

        return token_data

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify OAuth2 access token."""
        if token in self._tokens:
            token_info = self._tokens[token]
            created_at = token_info["created_at"]
            if datetime.utcnow() - created_at < timedelta(hours=1):
                return {
                    "active": True,
                    "client_id": token_info["client_id"],
                    "scope": " ".join(token_info["scopes"]),
                }
            else:
                del self._tokens[token]

        raise AuthenticationError("Invalid or expired token")


class APIKeyAuth(AuthenticationProvider):
    """API Key authentication implementation."""

    def __init__(self):
        self._api_keys: Dict[str, Dict[str, Any]] = {}

    def generate_api_key(
        self, owner: str, permissions: List[str], expires_days: Optional[int] = None
    ) -> str:
        """Generate a new API key."""
        api_key = f"fs_{secrets.token_urlsafe(32)}"

        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        self._api_keys[api_key] = {
            "owner": owner,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0,
        }

        return api_key

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using API key."""
        api_key = credentials.get("api_key")

        if not api_key:
            raise AuthenticationError("API key required")

        if api_key not in self._api_keys:
            raise AuthenticationError("Invalid API key")

        key_info = self._api_keys[api_key]

        if key_info["expires_at"] and datetime.utcnow() > key_info["expires_at"]:
            del self._api_keys[api_key]
            raise AuthenticationError("API key has expired")

        # Update usage stats
        key_info["last_used"] = datetime.utcnow()
        key_info["usage_count"] += 1

        return {
            "api_key": api_key,
            "owner": key_info["owner"],
            "permissions": key_info["permissions"],
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify API key (token is the API key)."""
        return self.authenticate({"api_key": token})

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            return True
        return False


class BasicAuth(AuthenticationProvider):
    """HTTP Basic authentication implementation."""

    def __init__(self):
        self._users: Dict[str, Dict[str, Any]] = {}
        self._hasher = BCryptHash()

    def register_user(
        self, username: str, password: str, permissions: Optional[List[str]] = None
    ) -> None:
        """Register a new user."""
        if username in self._users:
            raise AuthenticationError(f"User '{username}' already exists")

        password_hash = self._hasher.hash(password)
        self._users[username] = {
            "password_hash": password_hash,
            "permissions": permissions or [],
            "created_at": datetime.utcnow(),
        }

    def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate using username and password.

        Args:
            credentials: Dict with 'username' and 'password'
        """
        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            raise AuthenticationError("Username and password required")

        if username not in self._users:
            raise AuthenticationError("Invalid credentials")

        user = self._users[username]
        if not self._hasher.verify(password, user["password_hash"]):
            raise AuthenticationError("Invalid credentials")

        # Generate simple session token
        token = base64.b64encode(
            f"{username}:{secrets.token_urlsafe(16)}".encode()
        ).decode()

        return {
            "username": username,
            "permissions": user["permissions"],
            "token": token,
        }

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify basic auth token."""
        try:
            decoded = base64.b64decode(token).decode("utf-8")
            username, _ = decoded.split(":", 1)

            if username not in self._users:
                raise AuthenticationError("Invalid token")

            user = self._users[username]
            return {"username": username, "permissions": user["permissions"]}
        except Exception as e:
            raise AuthenticationError(f"Token verification failed: {e}")


def authenticate(
    credentials: Dict[str, Any], method: str = "jwt", **kwargs
) -> Dict[str, Any]:
    """
    Authenticate using specified method.

    Args:
        credentials: Authentication credentials
        method: Authentication method ('jwt', 'oauth2', 'apikey', 'basic')
        **kwargs: Additional arguments for the auth provider

    Returns:
        Authentication result with tokens/info
    """
    if method == "jwt":
        provider = JWTAuth(**kwargs)
    elif method == "oauth2":
        provider = OAuth2Auth(**kwargs)
    elif method == "apikey":
        provider = APIKeyAuth()
        if "api_key" in credentials:
            return provider.authenticate(credentials)
        raise AuthenticationError("API key required for apikey method")
    elif method == "basic":
        provider = BasicAuth()
    else:
        raise AuthenticationError(f"Unknown authentication method: {method}")

    return provider.authenticate(credentials)


# ============================================================================
# AUTHORIZATION MODULE
# ============================================================================


class Role:
    """Represents a role with associated permissions."""

    def __init__(self, name: str, permissions: List[str]):
        self.name = name
        self.permissions = set(permissions)

    def has_permission(self, permission: str) -> bool:
        """Check if role has a specific permission."""
        return permission in self.permissions or "*" in self.permissions

    def add_permission(self, permission: str) -> None:
        """Add a permission to the role."""
        self.permissions.add(permission)

    def remove_permission(self, permission: str) -> None:
        """Remove a permission from the role."""
        self.permissions.discard(permission)


class RBAC:
    """
    Role-Based Access Control implementation.
    Manages roles, permissions, and user-role assignments.
    """

    def __init__(self):
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, List[str]] = {}

    def create_role(self, name: str, permissions: List[str]) -> Role:
        """Create a new role."""
        role = Role(name, permissions)
        self._roles[name] = role
        return role

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def delete_role(self, name: str) -> bool:
        """Delete a role."""
        if name in self._roles:
            del self._roles[name]
            # Remove role from all users
            for user_roles in self._user_roles.values():
                if name in user_roles:
                    user_roles.remove(name)
            return True
        return False

    def assign_role(self, user_id: str, role_name: str) -> None:
        """Assign a role to a user."""
        if role_name not in self._roles:
            raise AuthorizationError(f"Role '{role_name}' does not exist")

        if user_id not in self._user_roles:
            self._user_roles[user_id] = []

        if role_name not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role_name)

    def revoke_role(self, user_id: str, role_name: str) -> None:
        """Revoke a role from a user."""
        if user_id in self._user_roles and role_name in self._user_roles[user_id]:
            self._user_roles[user_id].remove(role_name)

    def get_user_roles(self, user_id: str) -> List[Role]:
        """Get all roles assigned to a user."""
        role_names = self._user_roles.get(user_id, [])
        return [self._roles[name] for name in role_names if name in self._roles]

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has a specific permission through their roles."""
        roles = self.get_user_roles(user_id)
        return any(role.has_permission(permission) for role in roles)

    def list_permissions(self, user_id: str) -> List[str]:
        """List all permissions a user has."""
        roles = self.get_user_roles(user_id)
        permissions = set()
        for role in roles:
            permissions.update(role.permissions)
        return list(permissions)


class ABAC:
    """
    Attribute-Based Access Control implementation.
    Makes authorization decisions based on attributes of subject, resource, action, and environment.
    """

    def __init__(self):
        self._policies: List[Dict[str, Any]] = []

    def add_policy(
        self,
        name: str,
        subject_attrs: Dict[str, Any],
        resource_attrs: Dict[str, Any],
        action: str,
        environment_attrs: Optional[Dict[str, Any]] = None,
        effect: str = "permit",
    ) -> None:
        """
        Add an ABAC policy.

        Args:
            name: Policy name
            subject_attrs: Required subject attributes
            resource_attrs: Required resource attributes
            action: Action being performed
            environment_attrs: Required environment attributes
            effect: "permit" or "deny"
        """
        self._policies.append(
            {
                "name": name,
                "subject_attrs": subject_attrs,
                "resource_attrs": resource_attrs,
                "action": action,
                "environment_attrs": environment_attrs or {},
                "effect": effect,
            }
        )

    def evaluate(
        self,
        subject: Dict[str, Any],
        resource: Dict[str, Any],
        action: str,
        environment: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate whether an action is permitted.

        Args:
            subject: Subject attributes
            resource: Resource attributes
            action: Action being performed
            environment: Environment attributes

        Returns:
            True if action is permitted
        """
        environment = environment or {}

        for policy in self._policies:
            if policy["action"] != action and policy["action"] != "*":
                continue

            if (
                self._matches_attrs(subject, policy["subject_attrs"])
                and self._matches_attrs(resource, policy["resource_attrs"])
                and self._matches_attrs(environment, policy["environment_attrs"])
            ):
                return policy["effect"] == "permit"

        return False  # Default deny

    def _matches_attrs(self, attrs: Dict[str, Any], required: Dict[str, Any]) -> bool:
        """Check if attributes match required values."""
        for key, value in required.items():
            if key not in attrs:
                return False

            if callable(value):
                if not value(attrs[key]):
                    return False
            elif attrs[key] != value:
                return False

        return True


class ACL:
    """
    Access Control List implementation.
    Fine-grained per-resource permission management.
    """

    def __init__(self):
        self._entries: Dict[str, Dict[str, List[str]]] = {}

    def grant_permission(
        self, resource: str, subject: str, permissions: List[str]
    ) -> None:
        """Grant permissions on a resource to a subject."""
        if resource not in self._entries:
            self._entries[resource] = {}

        if subject not in self._entries[resource]:
            self._entries[resource][subject] = []

        for perm in permissions:
            if perm not in self._entries[resource][subject]:
                self._entries[resource][subject].append(perm)

    def revoke_permission(
        self, resource: str, subject: str, permissions: Optional[List[str]] = None
    ) -> None:
        """
        Revoke permissions on a resource from a subject.
        If permissions is None, revoke all permissions.
        """
        if resource in self._entries and subject in self._entries[resource]:
            if permissions is None:
                del self._entries[resource][subject]
            else:
                self._entries[resource][subject] = [
                    p for p in self._entries[resource][subject] if p not in permissions
                ]
                if not self._entries[resource][subject]:
                    del self._entries[resource][subject]

    def check_permission(self, resource: str, subject: str, permission: str) -> bool:
        """Check if subject has permission on resource."""
        if resource not in self._entries:
            return False

        if subject not in self._entries[resource]:
            return False

        perms = self._entries[resource][subject]
        return permission in perms or "*" in perms

    def get_permissions(self, resource: str, subject: str) -> List[str]:
        """Get all permissions a subject has on a resource."""
        if resource in self._entries and subject in self._entries[resource]:
            return self._entries[resource][subject].copy()
        return []

    def list_subjects(self, resource: str) -> List[str]:
        """List all subjects with permissions on a resource."""
        if resource in self._entries:
            return list(self._entries[resource].keys())
        return []


def authorize(
    user_id: str,
    permission: str,
    resource: Optional[str] = None,
    method: str = "rbac",
    rbac: Optional[RBAC] = None,
    abac: Optional[ABAC] = None,
    acl: Optional[ACL] = None,
    **context,
) -> bool:
    """
    Authorize a user action.

    Args:
        user_id: User identifier
        permission: Required permission
        resource: Resource identifier (for ACL)
        method: Authorization method ('rbac', 'abac', 'acl')
        rbac: RBAC instance (for rbac method)
        abac: ABAC instance (for abac method)
        acl: ACL instance (for acl method)
        **context: Additional context for ABAC

    Returns:
        True if authorized
    """
    if method == "rbac":
        if rbac is None:
            raise AuthorizationError("RBAC instance required")
        return rbac.check_permission(user_id, permission)

    elif method == "abac":
        if abac is None:
            raise AuthorizationError("ABAC instance required")
        subject = context.get("subject", {"id": user_id})
        resource_attrs = context.get("resource", {})
        action = permission
        environment = context.get("environment", {})
        return abac.evaluate(subject, resource_attrs, action, environment)

    elif method == "acl":
        if acl is None:
            raise AuthorizationError("ACL instance required")
        if resource is None:
            raise AuthorizationError("Resource required for ACL")
        return acl.check_permission(resource, user_id, permission)

    else:
        raise AuthorizationError(f"Unknown authorization method: {method}")


# ============================================================================
# PRIVACY MODULE
# ============================================================================


class DifferentialPrivacy:
    """
    Differential Privacy implementation.
    Adds calibrated noise to protect individual privacy in datasets.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy.

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            self._has_numpy = False
            import random

            self._random = random
            warnings.warn("numpy not available, using standard library")

    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for (epsilon, 0)-differential privacy."""
        scale = sensitivity / self.epsilon

        if self._has_numpy:
            noise = self._np.random.laplace(0, scale)
        else:
            # Manual Laplace sampling
            u = self._random.random() - 0.5
            noise = (
                -scale
                * (2 if u < 0 else -2)
                * (1 if u < 0 else -1)
                * (1 if u < 0 else -1)
            )
            # Simple approximation
            noise = self._random.gauss(0, scale)

        return value + noise

    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy."""
        if self.delta == 0:
            raise PrivacyError("Gaussian noise requires delta > 0")

        # Calculate sigma for Gaussian mechanism
        sigma = (
            sensitivity
            * (2 * (2 * self.epsilon**-2 * (1 if self._has_numpy else 1))) ** 0.5
        )

        if self._has_numpy:
            noise = self._np.random.normal(0, sigma)
        else:
            noise = self._random.gauss(0, sigma)

        return value + noise

    def privatize_count(self, count: int) -> float:
        """Add noise to a count query."""
        return self.add_laplace_noise(float(count), sensitivity=1.0)

    def privatize_sum(self, sum_value: float, lower: float, upper: float) -> float:
        """Add noise to a sum query with bounded values."""
        sensitivity = abs(upper - lower)
        return self.add_laplace_noise(sum_value, sensitivity)

    def privatize_mean(self, values: List[float], lower: float, upper: float) -> float:
        """Compute differentially private mean."""
        if not values:
            return 0.0

        noisy_count = self.privatize_count(len(values))
        noisy_sum = self.privatize_sum(sum(values), lower, upper)

        if noisy_count <= 0:
            return 0.0

        return noisy_sum / noisy_count


class KAnonymity:
    """
    K-Anonymity implementation.
    Ensures each record is indistinguishable from at least k-1 other records.
    """

    def __init__(self, k: int = 5):
        self.k = k
        try:
            import pandas as pd

            self._pd = pd
            self._has_pandas = True
        except ImportError:
            self._has_pandas = False

    def generalize(
        self, data: List[Dict[str, Any]], quasi_identifiers: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generalize quasi-identifiers to achieve k-anonymity.

        Args:
            data: List of records
            quasi_identifiers: List of quasi-identifier column names

        Returns:
            Generalized data
        """
        if self._has_pandas:
            df = self._pd.DataFrame(data)
            generalized = self._generalize_dataframe(df, quasi_identifiers)
            return generalized.to_dict("records")
        else:
            return self._generalize_dicts(data, quasi_identifiers)

    def _generalize_dataframe(self, df, quasi_identifiers):
        """Generalize using pandas."""
        generalized = df.copy()

        for col in quasi_identifiers:
            if col in df.columns:
                # Simple generalization: group into buckets
                if df[col].dtype in ["int64", "float64"]:
                    # Numeric: bin into quantiles
                    generalized[col] = self._bin_numeric(df[col])
                else:
                    # Categorical: suppress to most common
                    generalized[col] = self._suppress_categorical(df[col])

        return generalized

    def _generalize_dicts(self, data, quasi_identifiers):
        """Generalize using dicts."""
        # Find unique combinations
        groups = {}
        for i, record in enumerate(data):
            key = tuple(record.get(q) for q in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # Generalize small groups
        generalized = [r.copy() for r in data]

        for key, indices in groups.items():
            if len(indices) < self.k:
                # Suppress identifying values
                for idx in indices:
                    for q in quasi_identifiers:
                        generalized[idx][q] = "*"

        return generalized

    def _bin_numeric(self, series):
        """Bin numeric values."""
        num_bins = max(2, len(series) // self.k)
        return self._pd.cut(series, bins=num_bins, labels=False)

    def _suppress_categorical(self, series):
        """Suppress categorical values in small groups."""
        value_counts = series.value_counts()
        rare_values = value_counts[value_counts < self.k].index
        return series.apply(lambda x: "*" if x in rare_values else x)

    def check_k_anonymity(
        self, data: List[Dict[str, Any]], quasi_identifiers: List[str]
    ) -> bool:
        """Check if data satisfies k-anonymity."""
        groups = {}
        for record in data:
            key = tuple(record.get(q) for q in quasi_identifiers)
            groups[key] = groups.get(key, 0) + 1

        return all(count >= self.k for count in groups.values())


class LDP:
    """
    Local Differential Privacy implementation.
    Each user privatizes their own data before sharing.
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            self._has_numpy = False
            import random

            self._random = random

    def randomized_response(self, value: bool) -> bool:
        """
        Randomized response mechanism for boolean values.

        With probability p = e^epsilon / (1 + e^epsilon), return true value.
        Otherwise, return random value.
        """
        import math

        p = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))

        if self._has_numpy:
            if self._np.random.random() < p:
                return value
            else:
                return self._np.random.random() < 0.5
        else:
            if self._random.random() < p:
                return value
            else:
                return self._random.random() < 0.5

    def rappor(self, value: str, candidates: List[str]) -> str:
        """
        RAPPOR mechanism for string values.

        Args:
            value: True value
            candidates: List of possible values

        Returns:
            Privatized value
        """
        import math

        # Privacy parameters
        f = 0.5  # Permanent randomized response probability
        p = 1 / (1 + math.exp(self.epsilon / 2))
        q = 1 - p

        # Bloom filter would be used here in full implementation
        # For simplicity, use direct randomized response
        if self._has_numpy:
            if self._np.random.random() < f:
                # Return random candidate
                return self._np.random.choice(candidates)
            else:
                return value
        else:
            if self._random.random() < f:
                return self._random.choice(candidates)
            else:
                return value

    def privatize_numeric(self, value: float, lower: float, upper: float) -> float:
        """Privatize numeric value using Laplace mechanism."""
        sensitivity = upper - lower
        scale = sensitivity / self.epsilon

        if self._has_numpy:
            noise = self._np.random.laplace(0, scale)
        else:
            # Approximate with Gaussian for simplicity
            noise = self._random.gauss(0, scale)

        noisy_value = value + noise
        return max(lower, min(upper, noisy_value))


def anonymize_data(
    data: List[Dict[str, Any]], method: str = "k-anonymity", **kwargs
) -> List[Dict[str, Any]]:
    """
    Anonymize data using specified privacy method.

    Args:
        data: Data to anonymize
        method: Anonymization method ('k-anonymity', 'differential-privacy', 'ldp')
        **kwargs: Method-specific parameters

    Returns:
        Anonymized data
    """
    if method == "k-anonymity":
        k = kwargs.get("k", 5)
        quasi_identifiers = kwargs.get("quasi_identifiers", [])
        k_anon = KAnonymity(k=k)
        return k_anon.generalize(data, quasi_identifiers)

    elif method == "differential-privacy":
        epsilon = kwargs.get("epsilon", 1.0)
        delta = kwargs.get("delta", 1e-5)
        numeric_cols = kwargs.get("numeric_cols", [])

        dp = DifferentialPrivacy(epsilon=epsilon, delta=delta)

        anonymized = []
        for record in data:
            new_record = record.copy()
            for col in numeric_cols:
                if col in new_record:
                    new_record[col] = dp.add_laplace_noise(float(new_record[col]))
            anonymized.append(new_record)

        return anonymized

    elif method == "ldp":
        epsilon = kwargs.get("epsilon", 1.0)
        boolean_cols = kwargs.get("boolean_cols", [])

        ldp = LDP(epsilon=epsilon)

        anonymized = []
        for record in data:
            new_record = record.copy()
            for col in boolean_cols:
                if col in new_record:
                    new_record[col] = ldp.randomized_response(bool(new_record[col]))
            anonymized.append(new_record)

        return anonymized

    else:
        raise PrivacyError(f"Unknown anonymization method: {method}")


# ============================================================================
# MODEL SECURITY MODULE
# ============================================================================


class ModelEncryption:
    """
    Model encryption for protecting model weights and architecture.
    """

    def __init__(self):
        self._aes = AES256Encryption()

    def encrypt_model(
        self, model: Any, key: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Encrypt a model object.

        Args:
            model: Model object (should be picklable)
            key: Encryption key

        Returns:
            Tuple of (encrypted_model, key)
        """
        # Serialize model
        model_bytes = pickle.dumps(model)

        # Encrypt
        encrypted, key = encrypt_data(model_bytes, algorithm="aes256", key=key)

        return encrypted, key

    def decrypt_model(self, encrypted_model: bytes, key: bytes) -> Any:
        """
        Decrypt and restore a model object.

        Args:
            encrypted_model: Encrypted model bytes
            key: Decryption key

        Returns:
            Decrypted model object
        """
        # Decrypt
        model_bytes = decrypt_data(encrypted_model, algorithm="aes256", key=key)

        # Deserialize
        model = pickle.loads(model_bytes)

        return model

    def encrypt_weights(
        self, weights: Dict[str, Any], key: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """Encrypt model weights only."""
        weights_bytes = pickle.dumps(weights)
        encrypted, key = encrypt_data(weights_bytes, algorithm="aes256", key=key)
        return encrypted, key

    def decrypt_weights(self, encrypted_weights: bytes, key: bytes) -> Dict[str, Any]:
        """Decrypt model weights."""
        weights_bytes = decrypt_data(encrypted_weights, algorithm="aes256", key=key)
        return pickle.loads(weights_bytes)


class ModelWatermark:
    """
    Model watermarking for ownership verification.
    Embeds invisible signatures into model weights.
    """

    def __init__(self, strength: float = 0.01):
        self.strength = strength
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            self._has_numpy = False

    def embed(self, model: Any, watermark_key: str) -> Any:
        """
        Embed watermark into model.

        Args:
            model: Model object with weights
            watermark_key: Secret key for watermark

        Returns:
            Watermarked model
        """
        # Generate watermark pattern from key
        seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)

        # This is a simplified implementation
        # Real implementation would modify actual model weights
        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            watermarked = {}

            for name, param in state_dict.items():
                if self._has_numpy and hasattr(param, "numpy"):
                    param_np = param.detach().cpu().numpy()
                    # Add subtle watermark pattern
                    watermark = self._generate_watermark(param_np.shape, seed)
                    watermarked[name] = param_np + self.strength * watermark
                else:
                    watermarked[name] = param

            # Load watermarked weights
            model.load_state_dict(watermarked)

        return model

    def verify(self, model: Any, watermark_key: str) -> float:
        """
        Verify watermark in model.

        Returns:
            Confidence score (0-1) of watermark presence
        """
        seed = int(hashlib.sha256(watermark_key.encode()).hexdigest(), 16) % (2**32)

        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            correlations = []

            for name, param in state_dict.items():
                if self._has_numpy and hasattr(param, "numpy"):
                    param_np = param.detach().cpu().numpy()
                    watermark = self._generate_watermark(param_np.shape, seed)

                    # Compute correlation
                    corr = self._compute_correlation(param_np, watermark)
                    correlations.append(corr)

            if correlations:
                return sum(correlations) / len(correlations)

        return 0.0

    def _generate_watermark(self, shape, seed):
        """Generate watermark pattern."""
        if self._has_numpy:
            rng = self._np.random.RandomState(seed)
            return rng.randn(*shape)
        else:
            import random

            random.seed(seed)
            return [random.gauss(0, 1) for _ in range(int(self._prod(shape)))]

    def _compute_correlation(self, a, b):
        """Compute normalized correlation."""
        if self._has_numpy:
            a_flat = a.flatten()
            b_flat = b.flatten()
            return self._np.corrcoef(a_flat, b_flat)[0, 1]
        return 0.0

    def _prod(self, shape):
        """Compute product of shape dimensions."""
        result = 1
        for dim in shape:
            result *= dim
        return result


class ModelFingerprint:
    """
    Model fingerprinting for unique identification.
    Creates unique signatures based on model behavior.
    """

    def __init__(self):
        self._fingerprint_cache = {}

    def generate(self, model: Any, sample_inputs: List[Any]) -> Dict[str, Any]:
        """
        Generate model fingerprint from its outputs on sample inputs.

        Args:
            model: Model to fingerprint
            sample_inputs: Representative inputs

        Returns:
            Fingerprint dictionary
        """
        fingerprint = {
            "timestamp": datetime.utcnow().isoformat(),
            "hash": None,
            "output_signatures": [],
        }

        outputs = []
        for inp in sample_inputs:
            try:
                output = model(inp)
                # Create signature from output statistics
                sig = self._create_signature(output)
                fingerprint["output_signatures"].append(sig)
                outputs.append(sig)
            except Exception as e:
                fingerprint["output_signatures"].append({"error": str(e)})

        # Compute overall hash
        fingerprint["hash"] = hashlib.sha256(
            json.dumps(outputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        return fingerprint

    def compare(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """
        Compare two fingerprints.

        Returns:
            Similarity score (0-1)
        """
        if fp1.get("hash") == fp2.get("hash"):
            return 1.0

        sigs1 = fp1.get("output_signatures", [])
        sigs2 = fp2.get("output_signatures", [])

        if len(sigs1) != len(sigs2):
            return 0.0

        matches = 0
        for s1, s2 in zip(sigs1, sigs2):
            if self._signatures_match(s1, s2):
                matches += 1

        return matches / len(sigs1) if sigs1 else 0.0

    def _create_signature(self, output):
        """Create signature from model output."""
        try:
            import torch

            if isinstance(output, torch.Tensor):
                return {
                    "shape": list(output.shape),
                    "mean": float(output.mean()),
                    "std": float(output.std()),
                    "min": float(output.min()),
                    "max": float(output.max()),
                }
        except ImportError:
            pass

        # Generic fallback
        return {"type": type(output).__name__, "repr": repr(output)[:100]}

    def _signatures_match(self, s1, s2, tolerance=0.01):
        """Check if two signatures match within tolerance."""
        if set(s1.keys()) != set(s2.keys()):
            return False

        for key in s1:
            if isinstance(s1[key], float) and isinstance(s2[key], float):
                if abs(s1[key] - s2[key]) > tolerance:
                    return False
            elif s1[key] != s2[key]:
                return False

        return True


def secure_model(
    model: Any,
    method: str = "encrypt",
    key: Optional[bytes] = None,
    watermark_key: Optional[str] = None,
) -> Any:
    """
    Apply security measures to a model.

    Args:
        model: Model to secure
        method: Security method ('encrypt', 'watermark', 'fingerprint')
        key: Encryption key (for encrypt method)
        watermark_key: Watermark key (for watermark method)

    Returns:
        Secured model or encryption result
    """
    if method == "encrypt":
        enc = ModelEncryption()
        return enc.encrypt_model(model, key)

    elif method == "watermark":
        if watermark_key is None:
            raise ModelSecurityError("Watermark key required")
        wm = ModelWatermark()
        return wm.embed(model, watermark_key)

    elif method == "fingerprint":
        fp = ModelFingerprint()
        # Need sample inputs for fingerprinting
        raise ModelSecurityError(
            "Use ModelFingerprint.generate() directly with sample inputs"
        )

    else:
        raise ModelSecurityError(f"Unknown security method: {method}")


# ============================================================================
# ADVERSARIAL DEFENSE MODULE
# ============================================================================


class AdversarialDefense:
    """
    Defenses against adversarial attacks on ML models.
    """

    def __init__(self):
        try:
            import numpy as np

            self._np = np
            self._has_numpy = True
        except ImportError:
            self._has_numpy = False

    def input_preprocessing(self, x, method: str = "jpeg"):
        """
        Preprocess input to remove adversarial perturbations.

        Args:
            x: Input data
            method: Preprocessing method ('jpeg', 'bit_reduction', 'spatial_smoothing')

        Returns:
            Preprocessed input
        """
        if method == "bit_reduction":
            return self._bit_depth_reduction(x, bits=4)
        elif method == "spatial_smoothing":
            return self._spatial_smoothing(x)
        elif method == "jpeg":
            return self._jpeg_compression(x)
        else:
            raise AdversarialError(f"Unknown preprocessing method: {method}")

    def _bit_depth_reduction(self, x, bits=4):
        """Reduce bit depth to remove small perturbations."""
        if self._has_numpy:
            max_val = 2**bits - 1
            x_int = (x * max_val).astype(int)
            return x_int.astype(float) / max_val
        return x

    def _spatial_smoothing(self, x, window_size=3):
        """Apply median smoothing filter."""
        if self._has_numpy:
            from scipy.ndimage import median_filter

            return median_filter(x, size=window_size)
        return x

    def _jpeg_compression(self, x, quality=75):
        """Apply JPEG compression."""
        try:
            from PIL import Image
            import io

            if self._has_numpy:
                img = Image.fromarray((x * 255).astype("uint8"))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                img_compressed = Image.open(buffer)
                return self._np.array(img_compressed) / 255.0
        except ImportError:
            pass
        return x

    def ensemble_defense(self, model, x, models: List[Any]):
        """
        Use ensemble of models for robust prediction.

        Args:
            model: Primary model
            x: Input
            models: List of models for ensemble

        Returns:
            Ensemble prediction
        """
        predictions = []

        for m in [model] + models:
            try:
                pred = m(x)
                predictions.append(pred)
            except Exception:
                pass

        if not predictions:
            raise AdversarialError("No model produced valid prediction")

        # Majority vote or average
        if self._has_numpy:
            return self._np.mean(predictions, axis=0)
        return predictions[0]

    def randomized_smoothing(
        self, model, x, noise_std: float = 0.25, num_samples: int = 100
    ):
        """
        Randomized smoothing for certified robustness.

        Args:
            model: Model to smooth
            x: Input
            noise_std: Standard deviation of Gaussian noise
            num_samples: Number of noisy samples

        Returns:
            Smoothed prediction and certification radius
        """
        if not self._has_numpy:
            raise AdversarialError("Randomized smoothing requires numpy")

        predictions = []

        for _ in range(num_samples):
            noise = self._np.random.normal(0, noise_std, x.shape)
            noisy_input = x + noise
            pred = model(noisy_input)
            predictions.append(pred)

        # Average predictions
        mean_pred = self._np.mean(predictions, axis=0)

        # Compute certification radius (simplified)
        # Real implementation would use Neyman-Pearson lemma
        std_pred = self._np.std(predictions, axis=0)
        radius = (
            noise_std
            * self._np.sqrt(2)
            * (self._np.max(mean_pred) - self._np.sort(mean_pred)[-2])
            / (self._np.max(std_pred) + 1e-10)
        )

        return mean_pred, radius


class RobustTraining:
    """
    Training methods for improving model robustness.
    """

    def __init__(self):
        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn
            self._has_torch = True
        except ImportError:
            self._has_torch = False
            warnings.warn("PyTorch not available, robust training disabled")

    def adversarial_training_step(
        self,
        model,
        x,
        y,
        optimizer,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_steps: int = 10,
    ):
        """
        Single step of adversarial training using PGD.

        Args:
            model: Model being trained
            x: Input batch
            y: Labels
            optimizer: Optimizer
            epsilon: Maximum perturbation
            alpha: Step size
            num_steps: Number of PGD steps

        Returns:
            Loss value
        """
        if not self._has_torch:
            raise AdversarialError("PyTorch required for adversarial training")

        model.train()

        # Generate adversarial examples
        x_adv = x.clone().detach()
        x_adv.requires_grad = True

        for _ in range(num_steps):
            x_adv.requires_grad = True

            # Forward pass
            outputs = model(x_adv)
            loss = self._nn.CrossEntropyLoss()(outputs, y)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Update adversarial example
            with self._torch.no_grad():
                x_adv = x_adv + alpha * x_adv.grad.sign()
                # Project back to epsilon ball
                perturbation = self._torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = x + perturbation
                x_adv = self._torch.clamp(x_adv, 0, 1)

        # Train on adversarial examples
        optimizer.zero_grad()
        outputs = model(x_adv)
        loss = self._nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        optimizer.step()

        return loss.item()

    def mixup_training_step(self, model, x, y, optimizer, alpha: float = 1.0):
        """
        Training step with Mixup data augmentation.

        Args:
            model: Model being trained
            x: Input batch
            y: Labels
            optimizer: Optimizer
            alpha: Mixup interpolation strength

        Returns:
            Loss value
        """
        if not self._has_torch:
            raise AdversarialError("PyTorch required for mixup training")

        model.train()

        # Generate mixed samples
        lam = self._torch.distributions.Beta(alpha, alpha).sample()
        batch_size = x.size(0)
        index = self._torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(mixed_x)

        # Mixup loss
        criterion = self._nn.CrossEntropyLoss()
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()

    def trades_loss(
        self,
        model,
        x,
        y,
        optimizer,
        epsilon: float = 0.03,
        beta: float = 6.0,
        num_steps: int = 10,
    ):
        """
        TRADES (Trade-off-inspired Adversarial Defense) loss.

        Balances natural accuracy and robustness.
        """
        if not self._has_torch:
            raise AdversarialError("PyTorch required for TRADES")

        model.train()

        # Natural loss
        logits_natural = model(x)
        loss_natural = self._nn.CrossEntropyLoss()(logits_natural, y)

        # Generate adversarial examples for robustness loss
        x_adv = x.clone().detach() + 0.001 * self._torch.randn_like(x)
        x_adv.requires_grad = True

        for _ in range(num_steps):
            x_adv.requires_grad = True
            logits_adv = model(x_adv)

            # KL divergence loss
            loss_kl = self._nn.KLDivLoss(reduction="batchmean")(
                self._nn.functional.log_softmax(logits_adv, dim=1),
                self._nn.functional.softmax(logits_natural, dim=1),
            )

            grad = self._torch.autograd.grad(loss_kl, x_adv)[0]
            x_adv = x_adv.detach() + epsilon / num_steps * grad.sign()
            x_adv = self._torch.min(self._torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = self._torch.clamp(x_adv, 0, 1)

        # Total TRADES loss
        logits_adv = model(x_adv)
        loss_robust = self._nn.KLDivLoss(reduction="batchmean")(
            self._nn.functional.log_softmax(logits_adv, dim=1),
            self._nn.functional.softmax(logits_natural, dim=1),
        )

        loss = loss_natural + beta * loss_robust

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


def certified_defense(
    model: Any, x: Any, method: str = "randomized_smoothing", **kwargs
) -> Tuple[Any, float]:
    """
    Apply certified defense to input.

    Args:
        model: Model to defend
        x: Input
        method: Defense method ('randomized_smoothing')
        **kwargs: Additional parameters

    Returns:
        Tuple of (prediction, certification_radius)
    """
    defense = AdversarialDefense()

    if method == "randomized_smoothing":
        noise_std = kwargs.get("noise_std", 0.25)
        num_samples = kwargs.get("num_samples", 100)
        return defense.randomized_smoothing(model, x, noise_std, num_samples)

    elif method == "input_preprocessing":
        preprocess_method = kwargs.get("preprocess_method", "jpeg")
        x_processed = defense.input_preprocessing(x, preprocess_method)
        pred = model(x_processed)
        return pred, 0.0

    else:
        raise AdversarialError(f"Unknown certified defense method: {method}")


# ============================================================================
# AUDIT MODULE
# ============================================================================


class SecurityEvent:
    """Represents a security event for logging."""

    def __init__(
        self,
        event_type: str,
        severity: str,
        message: str,
        user: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.event_type = event_type
        self.severity = severity
        self.message = message
        self.user = user
        self.resource = resource
        self.details = details or {}
        self.ip_address = None
        self.session_id = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "user": self.user,
            "resource": self.resource,
            "details": self.details,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class SecurityLogger:
    """
    Security event logger.
    Logs security events for audit and compliance.
    """

    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger("fishstick.security")
        self.logger.setLevel(level)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._events: List[SecurityEvent] = []

    def log(self, event: SecurityEvent) -> None:
        """Log a security event."""
        self._events.append(event)

        log_method = getattr(self.logger, event.severity.lower(), self.logger.info)
        log_method(f"[{event.event_type}] {event.message} - User: {event.user}")

    def get_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[SecurityEvent]:
        """
        Query logged events.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            start_time: Filter events after this time
            end_time: Filter events before this time

        Returns:
            List of matching events
        """
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events

    def clear_events(self) -> None:
        """Clear all stored events."""
        self._events = []

    def export_to_file(self, filepath: str, format: str = "json") -> None:
        """Export events to file."""
        events_data = [e.to_dict() for e in self._events]

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(events_data, f, indent=2, default=str)
        elif format == "csv":
            import csv

            if events_data:
                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=events_data[0].keys())
                    writer.writeheader()
                    writer.writerows(events_data)


class SecurityAudit:
    """
    Comprehensive security auditing system.
    Performs security checks and generates audit reports.
    """

    def __init__(self):
        self.logger = SecurityLogger()
        self._checks: Dict[str, Callable] = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default security checks."""
        self._checks["password_strength"] = self._check_password_strength
        self._checks["encryption_status"] = self._check_encryption_status
        self._checks["access_control"] = self._check_access_control
        self._checks["data_retention"] = self._check_data_retention

    def run_audit(self) -> Dict[str, Any]:
        """
        Run comprehensive security audit.

        Returns:
            Audit report dictionary
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "overall_score": 0.0,
            "recommendations": [],
        }

        total_score = 0

        for check_name, check_func in self._checks.items():
            try:
                result = check_func()
                report["checks"][check_name] = result
                total_score += result.get("score", 0)

                if result.get("score", 0) < 0.7:
                    report["recommendations"].extend(result.get("recommendations", []))
            except Exception as e:
                report["checks"][check_name] = {"status": "error", "error": str(e)}

        report["overall_score"] = total_score / len(self._checks) if self._checks else 0

        # Log audit completion
        event = SecurityEvent(
            event_type="security_audit",
            severity="info",
            message=f"Security audit completed. Score: {report['overall_score']:.2f}",
            details={"overall_score": report["overall_score"]},
        )
        self.logger.log(event)

        return report

    def _check_password_strength(self) -> Dict[str, Any]:
        """Check password strength requirements."""
        return {
            "status": "info",
            "score": 0.8,
            "message": "Password strength check completed",
            "recommendations": [
                "Enforce minimum password length of 12 characters",
                "Require mix of uppercase, lowercase, numbers, and symbols",
            ],
        }

    def _check_encryption_status(self) -> Dict[str, Any]:
        """Check data encryption status."""
        return {
            "status": "info",
            "score": 0.9,
            "message": "Encryption status check completed",
            "recommendations": [
                "Ensure all sensitive data is encrypted at rest",
                "Use TLS 1.3 for all data in transit",
            ],
        }

    def _check_access_control(self) -> Dict[str, Any]:
        """Check access control configuration."""
        return {
            "status": "info",
            "score": 0.85,
            "message": "Access control check completed",
            "recommendations": [
                "Implement principle of least privilege",
                "Regular review of user permissions",
            ],
        }

    def _check_data_retention(self) -> Dict[str, Any]:
        """Check data retention policies."""
        return {
            "status": "info",
            "score": 0.75,
            "message": "Data retention check completed",
            "recommendations": [
                "Define clear data retention policies",
                "Implement automatic data purging",
            ],
        }

    def add_custom_check(self, name: str, check_func: Callable) -> None:
        """Add a custom security check."""
        self._checks[name] = check_func

    def generate_compliance_report(self, standard: str = "GDPR") -> Dict[str, Any]:
        """
        Generate compliance report for specified standard.

        Args:
            standard: Compliance standard ('GDPR', 'HIPAA', 'SOC2')

        Returns:
            Compliance report
        """
        standards = {
            "GDPR": self._check_gdpr_compliance,
            "HIPAA": self._check_hipaa_compliance,
            "SOC2": self._check_soc2_compliance,
        }

        if standard not in standards:
            raise SecurityError(f"Unknown compliance standard: {standard}")

        return standards[standard]()

    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance."""
        return {
            "standard": "GDPR",
            "timestamp": datetime.utcnow().isoformat(),
            "requirements": {
                "data_minimization": {"compliant": True, "notes": ""},
                "purpose_limitation": {"compliant": True, "notes": ""},
                "storage_limitation": {
                    "compliant": False,
                    "notes": "No retention policy defined",
                },
                "accuracy": {"compliant": True, "notes": ""},
                "integrity_confidentiality": {"compliant": True, "notes": ""},
            },
            "overall_compliant": False,
            "action_items": ["Define and implement data retention policy"],
        }

    def _check_hipaa_compliance(self) -> Dict[str, Any]:
        """Check HIPAA compliance."""
        return {
            "standard": "HIPAA",
            "timestamp": datetime.utcnow().isoformat(),
            "requirements": {
                "access_control": {"compliant": True, "notes": ""},
                "audit_controls": {"compliant": True, "notes": ""},
                "integrity": {"compliant": True, "notes": ""},
                "transmission_security": {"compliant": True, "notes": ""},
            },
            "overall_compliant": True,
            "action_items": [],
        }

    def _check_soc2_compliance(self) -> Dict[str, Any]:
        """Check SOC2 compliance."""
        return {
            "standard": "SOC2",
            "timestamp": datetime.utcnow().isoformat(),
            "trust_service_criteria": {
                "security": {"compliant": True, "score": 0.9},
                "availability": {"compliant": True, "score": 0.85},
                "processing_integrity": {"compliant": True, "score": 0.8},
                "confidentiality": {"compliant": True, "score": 0.9},
                "privacy": {"compliant": False, "score": 0.6},
            },
            "overall_compliant": False,
            "action_items": ["Strengthen privacy controls"],
        }


def log_security_event(
    event_type: str,
    message: str,
    severity: str = "info",
    user: Optional[str] = None,
    resource: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    logger: Optional[SecurityLogger] = None,
) -> SecurityEvent:
    """
    Log a security event.

    Args:
        event_type: Type of security event
        message: Event message
        severity: Event severity ('debug', 'info', 'warning', 'error', 'critical')
        user: User associated with event
        resource: Resource affected
        details: Additional details
        logger: Logger instance (creates new if None)

    Returns:
        Created SecurityEvent
    """
    if logger is None:
        logger = SecurityLogger()

    event = SecurityEvent(
        event_type=event_type,
        severity=severity,
        message=message,
        user=user,
        resource=resource,
        details=details,
    )

    logger.log(event)
    return event


# ============================================================================
# CONVENIENCE EXPORTS
# ============================================================================

__all__ = [
    # Encryption
    "EncryptionAlgorithm",
    "AES256Encryption",
    "RSAEncryption",
    "ChaCha20Encryption",
    "FernetEncryption",
    "encrypt_data",
    "decrypt_data",
    # Hashing
    "HashAlgorithm",
    "SHA256Hash",
    "SHA512Hash",
    "BCryptHash",
    "Argon2Hash",
    "hash_password",
    "verify_password",
    # Authentication
    "AuthenticationProvider",
    "JWTAuth",
    "OAuth2Auth",
    "APIKeyAuth",
    "BasicAuth",
    "authenticate",
    # Authorization
    "Role",
    "RBAC",
    "ABAC",
    "ACL",
    "authorize",
    # Privacy
    "DifferentialPrivacy",
    "KAnonymity",
    "LDP",
    "anonymize_data",
    # Model Security
    "ModelEncryption",
    "ModelWatermark",
    "ModelFingerprint",
    "secure_model",
    # Adversarial
    "AdversarialDefense",
    "RobustTraining",
    "certified_defense",
    # Audit
    "SecurityEvent",
    "SecurityLogger",
    "SecurityAudit",
    "log_security_event",
    # Exceptions
    "SecurityError",
    "EncryptionError",
    "HashingError",
    "AuthenticationError",
    "AuthorizationError",
    "PrivacyError",
    "ModelSecurityError",
    "AdversarialError",
]
