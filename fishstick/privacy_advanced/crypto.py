import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import secrets


class SecretShare:
    def __init__(self, threshold: int, num_shares: int, prime: Optional[int] = None):
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime or self._generate_prime()

    def _generate_prime(self, bits: int = 64) -> int:
        import random

        while True:
            p = random.getrandbits(bits)
            if self._is_prime(p):
                return p

    def _is_prime(self, n: int, k: int = 40) -> bool:
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False

        import random

        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def split_secret(self, secret: int) -> List[Tuple[int, int]]:
        if secret >= self.prime:
            raise ValueError("Secret must be less than prime")

        coefficients = [secret] + [
            secrets.randbelow(self.prime) for _ in range(self.threshold - 1)
        ]

        shares = []
        for i in range(1, self.num_shares + 1):
            x = i
            y = sum(c * (x**j) for j, c in enumerate(coefficients)) % self.prime
            shares.append((x, y))

        return shares

    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        secret = 0
        for i, (x_i, y_i) in enumerate(shares[: self.threshold]):
            numerator = 1
            denominator = 1
            for j, (x_j, _) in enumerate(shares[: self.threshold]):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime

            lagrange_coeff = numerator * pow(denominator, -1, self.prime)
            secret = (secret + y_i * lagrange_coeff) % self.prime

        return secret

    def share_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        flat = tensor.flatten().numpy()
        shares = []

        for i in range(self.num_shares):
            share_values = []
            for val in flat:
                val_int = int(val * (2**32))
                share_i = self.split_secret(val_int)[i]
                share_values.append(share_i[1])
            shares.append(torch.tensor(share_values, dtype=torch.float32))

        return shares

    def reconstruct_tensor(
        self, shares: List[torch.Tensor], shape: Tuple[int, ...]
    ) -> torch.Tensor:
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")

        reconstructed_values = []
        num_elements = shares[0].shape[0]

        for idx in range(num_elements):
            share_list = [
                (i + 1, int(shares[i][idx].item())) for i in range(len(shares))
            ]
            val_int = self.reconstruct_secret(share_list)
            reconstructed_values.append(val_int / (2**32))

        return torch.tensor(reconstructed_values).reshape(shape)


def reconstruct_secret(shares: List[Tuple[int, int]], prime: int) -> int:
    threshold = len(shares)
    secret = 0

    for i, (x_i, y_i) in enumerate(shares[:threshold]):
        numerator = 1
        denominator = 1
        for j, (x_j, _) in enumerate(shares[:threshold]):
            if i != j:
                numerator = (numerator * (-x_j)) % prime
                denominator = (denominator * (x_i - x_j)) % prime

        lagrange_coeff = numerator * pow(denominator, -1, prime)
        secret = (secret + y_i * lagrange_coeff) % prime

    return secret


class SecureAggregator:
    def __init__(
        self, num_clients: int, threshold: Optional[int] = None, secure_rng: bool = True
    ):
        self.num_clients = num_clients
        self.threshold = threshold or (num_clients // 2 + 1)
        self.secure_rng = secure_rng
        self.secret_sharing = SecretShare(self.threshold, num_clients)

        self.client_public_keys: Dict[int, bytes] = {}
        self.client_shares: Dict[int, Dict[int, int]] = {}
        self.accumulated_shares: Dict[int, int] = {}

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        private_key = secrets.token_bytes(32)
        public_key = hashlib.sha256(private_key).digest()
        return private_key, public_key

    def register_client(self, client_id: int, public_key: bytes) -> None:
        self.client_public_keys[client_id] = public_key

    def add_client_update(
        self, client_id: int, model_update: Dict[str, torch.Tensor]
    ) -> None:
        for name, param in model_update.items():
            shares = self.secret_sharing.share_tensor(param.data)

            if name not in self.client_shares:
                self.client_shares[name] = {}

            self.client_shares[name][client_id] = shares

    def aggregate(self) -> Dict[str, torch.Tensor]:
        if len(self.client_shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} clients to aggregate")

        aggregated = {}

        for name, client_shares in self.client_shares.items():
            sample_shares = list(client_shares.values())[0]
            shape = None

            for client_id, shares in client_shares.items():
                for i, share in enumerate(shares):
                    if i not in self.accumulated_shares:
                        self.accumulated_shares[i] = 0
                    self.accumulated_shares[i] += share.item()

            if shape is None:
                sample_client = list(client_shares.values())[0]
                sample_share = sample_client[0]
                shape = (len(sample_share),)

        for name, client_shares in self.client_shares.items():
            sample_client = list(client_shares.values())[0]
            aggregated[name] = torch.zeros(len(sample_client[0]))

            selected_clients = list(client_shares.keys())[: self.threshold]

            for i in range(len(sample_client)):
                share_sum = sum(
                    client_shares[cid][i].item() for cid in selected_clients
                )
                aggregated[name][i] = share_sum / len(selected_clients)

        return aggregated

    def compute_blinded_updates(
        self, model_update: Dict[str, torch.Tensor], client_id: int
    ) -> Dict[str, torch.Tensor]:
        seed = hashlib.sha256(str(client_id).encode()).digest()
        rng = torch.Generator()
        rng.manual_seed(int.from_bytes(seed[:4], "big"))

        blinded = {}
        for name, param in model_update.items():
            mask = torch.randn_like(param, generator=rng) * 0.1
            blinded[name] = param + mask

        return blinded

    def verify_share(self, client_id: int, share: int, commitment: bytes) -> bool:
        share_bytes = share.to_bytes(32, "big")
        expected = hashlib.sha256(share_bytes).digest()
        return expected == commitment


class HomomorphicEncryption:
    def __init__(self, key_size: int = 2048, encryption_depth: int = 1):
        self.key_size = key_size
        self.encryption_depth = encryption_depth
        self.public_key: Optional[bytes] = None
        self.private_key: Optional[bytes] = None
        self.is_encrypted = False

    def keygen(self) -> Tuple[bytes, bytes]:
        import random

        p = secrets.randbits(self.key_size // 2)
        q = secrets.randbits(self.key_size // 2)

        n = p * q
        g = n + 1

        self.private_key = p.to_bytes(128, "big")
        self.public_key = g.to_bytes(128, "big") + n.to_bytes(128, "big")

        return self.public_key, self.private_key

    def encrypt(self, plaintext: int) -> int:
        if not self.public_key:
            raise ValueError("Keys not generated")

        g = int.from_bytes(self.public_key[:128], "big")
        n = int.from_bytes(self.public_key[128:], "big")

        r = secrets.randbelow(n)
        noise = secrets.randbits(32)

        ciphertext = (g * plaintext + r * noise) % (n * n)
        self.is_encrypted = True

        return ciphertext

    def decrypt(self, ciphertext: int) -> int:
        if not self.private_key:
            raise ValueError("Private key not available")

        p = int.from_bytes(self.private_key, "big")
        n = p * (p - 1)

        mu = pow(p, -1, p - 1)

        x = pow(ciphertext, p - 1, n)
        plaintext = (x - 1) // p * mu % p

        return plaintext

    def add(self, c1: int, c2: int) -> int:
        if not self.public_key:
            raise ValueError("Keys not generated")

        n = int.from_bytes(self.public_key[128:], "big")
        return (c1 * c2) % (n * n)

    def multiply(self, c1: int, c2: int) -> int:
        if not self.public_key:
            raise ValueError("Keys not generated")

        n = int.from_bytes(self.public_key[128:], "big")
        return pow(c1, c2, n * n)

    def encrypt_tensor(self, tensor: torch.Tensor) -> List[int]:
        flat = tensor.flatten().numpy()
        return [self.encrypt(int(val * (2**32))) for val in flat]

    def decrypt_tensor(
        self, ciphertexts: List[int], shape: Tuple[int, ...]
    ) -> torch.Tensor:
        decrypted = [self.decrypt(c) / (2**32) for c in ciphertexts]
        return torch.tensor(decrypted).reshape(shape)


class PaillierEncryption(HomomorphicEncryption):
    def __init__(self, key_size: int = 2048):
        super().__init__(key_size)

    def keygen(self) -> Tuple[Tuple[int, int], int]:
        import random

        def generate_prime(bits):
            while True:
                p = random.getrandbits(bits)
                if self._is_prime(p):
                    return p

        p = generate_prime(self.key_size // 2)
        q = generate_prime(self.key_size // 2)

        n = p * q
        lam = (p - 1) * (q - 1)
        mu = pow(lam, -1, n)

        self.private_key = (lam, mu)
        self.public_key = (n, lam)

        return self.public_key, self.private_key

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
            if n % p == 0:
                return n == p
        return True

    def encrypt(self, plaintext: int) -> int:
        n, _ = self.public_key
        r = secrets.randbelow(n)
        return (pow(n + 1, plaintext, n * n) * pow(r, n, n * n)) % (n * n)

    def decrypt(self, ciphertext: int) -> int:
        n, _ = self.public_key
        lam, mu = self.private_key

        x = pow(ciphertext, lam, n * n)
        l = (x - 1) // n
        return (l * mu) % n

    def add(self, c1: int, c2: int) -> int:
        n, _ = self.public_key
        return (c1 * c2) % (n * n)

    def scalar_multiply(self, ciphertext: int, scalar: int) -> int:
        n, _ = self.public_key
        return pow(ciphertext, scalar, n * n)


def federated_secure_average(
    client_updates: List[Dict[str, torch.Tensor]], threshold: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    if not client_updates:
        return {}

    if threshold and len(client_updates) < threshold:
        raise ValueError(f"Need at least {threshold} clients")

    keys = client_updates[0].keys()
    aggregated = {}

    for key in keys:
        stacked = torch.stack([update[key] for update in client_updates])
        aggregated[key] = stacked.mean(dim=0)

    return aggregated


def secure_dot_product(
    vec1: torch.Tensor, vec2: torch.Tensor, he: HomomorphicEncryption
) -> int:
    encrypted1 = he.encrypt_tensor(vec1)
    encrypted2 = he.encrypt_tensor(vec2)

    result = 0
    for e1, e2 in zip(encrypted1, encrypted2):
        product = he.multiply(e1, e2)
        result = he.add(result, product)

    return he.decrypt(result)
