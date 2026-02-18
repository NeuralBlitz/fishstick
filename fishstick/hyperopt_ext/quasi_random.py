from __future__ import annotations

import math
from typing import Any

import numpy as np


class QuasiRandomSequence:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int, dim: int) -> np.ndarray:
        raise NotImplementedError


class SobolSequence(QuasiRandomSequence):
    def __init__(self, seed: int | None = None, max_dim: int = 1000):
        super().__init__(seed)
        self.max_dim = max_dim
        self._direction_vectors = self._initialize_direction_vectors()

    def _initialize_direction_vectors(self) -> dict[int, np.ndarray]:
        vectors = {}
        for dim in range(1, self.max_dim + 1):
            vectors[dim] = self._generate_direction_vector(dim)
        return vectors

    def _generate_direction_vector(self, dim: int) -> np.ndarray:
        poly = [
            1,
            3,
            7,
            11,
            13,
            19,
            25,
            37,
            59,
            47,
            61,
            109,
            103,
            151,
            193,
            203,
            211,
            239,
            281,
            307,
            401,
            419,
            457,
            503,
        ]
        if dim > len(poly):
            poly = poly * (dim // len(poly) + 1)

        v = np.ones(dim, dtype=np.int64)
        for i in range(2, dim + 1):
            binary = bin(poly[i - 2])[2:]
            v[i - 1] = 1
            for j, bit in enumerate(reversed(binary)):
                if bit == "1":
                    v[i - 1] ^= v[i - 1 - j - 1] << 1

        return v

    def generate(self, n: int, dim: int) -> np.ndarray:
        if dim > self.max_dim:
            raise ValueError(f"Dimension {dim} exceeds maximum {self.max_dim}")

        samples = np.zeros((n, dim), dtype=np.float64)
        x = np.zeros(dim, dtype=np.int64)

        for i in range(n):
            x = x ^ (1 << int(math.log2(i + 1)))
            for d in range(dim):
                samples[i, d] = x[d] / (2**32)

        return samples


class HaltonSequence(QuasiRandomSequence):
    def __init__(self, seed: int | None = None, base: int = 2):
        super().__init__(seed)
        self.base = base

    def _van_der_corput(self, n: int) -> np.ndarray:
        samples = np.zeros(n)
        for i in range(n):
            base = self.base
            value = 0
            denom = 1
            x = i + 1
            while x > 0:
                digit = x % base
                value += digit / denom
                denom *= base
                x //= base
            samples[i] = value
        return samples

    def generate(self, n: int, dim: int) -> np.ndarray:
        samples = np.zeros((n, dim))
        for d in range(dim):
            prime = self._get_prime(d)
            original_base = self.base
            self.base = prime
            samples[:, d] = self._van_der_corput(n)
            self.base = original_base
        return samples

    def _get_prime(self, n: int) -> int:
        primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
            73,
            79,
            83,
            89,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            163,
            167,
            173,
            179,
            181,
            191,
            193,
            197,
            199,
        ]
        return primes[n] if n < len(primes) else primes[-1]


class LatinHypercube(QuasiRandomSequence):
    def __init__(self, seed: int | None = None, criterion: str = "center"):
        super().__init__(seed)
        self.criterion = criterion

    def generate(self, n: int, dim: int) -> np.ndarray:
        samples = np.zeros((n, dim))

        for d in range(dim):
            samples[:, d] = np.linspace(0, 1, n + 1)[:n]
            samples[:, d] += self.rng.uniform(-0.5 / n, 0.5 / n, n)

        if self.criterion == "center":
            samples = self._center(samples)
        elif self.criterion == "maximin":
            samples = self._maximin(samples)

        return samples

    def _center(self, samples: np.ndarray) -> np.ndarray:
        for d in range(samples.shape[1]):
            samples[:, d] = np.sort(samples[:, d])
        return samples

    def _maximin(self, samples: np.ndarray, max_iter: int = 100) -> np.ndarray:
        best_samples = samples.copy()
        best_min_dist = self._min_distance(samples)

        for _ in range(max_iter):
            for d in range(samples.shape[1]):
                perm = self.rng.permutation(samples.shape[0])
                samples[:, d] = samples[perm, d]

            min_dist = self._min_distance(samples)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_samples = samples.copy()

        return best_samples

    def _min_distance(self, samples: np.ndarray) -> float:
        min_dist = float("inf")
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = np.linalg.norm(samples[i] - samples[j])
                min_dist = min(min_dist, dist)
        return min_dist


class HammersleySequence(QuasiRandomSequence):
    def __init__(self, seed: int | None = None):
        super().__init__(seed)

    def _van_der_corput(self, n: int, base: int) -> float:
        value = 0
        denom = 1
        x = n
        while x > 0:
            digit = x % base
            value += digit / denom
            denom *= base
            x //= base
        return value

    def generate(self, n: int, dim: int) -> np.ndarray:
        samples = np.zeros((n, dim))

        for i in range(n):
            samples[i, 0] = (i + 0.5) / n
            for d in range(1, dim):
                prime = self._get_prime(d - 1)
                samples[i, d] = self._van_der_corput(i + 1, prime)

        return samples

    def _get_prime(self, n: int) -> int:
        primes = [
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            47,
            53,
            59,
            61,
            67,
            71,
        ]
        return primes[n % len(primes)]


def get_quasi_random_sequence(name: str, **kwargs: Any) -> QuasiRandomSequence:
    name = name.lower()
    if name == "sobol":
        return SobolSequence(**kwargs)
    elif name == "halton":
        return HaltonSequence(**kwargs)
    elif name == "lhs" or name == "latin_hypercube":
        return LatinHypercube(**kwargs)
    elif name == "hammersley":
        return HammersleySequence(**kwargs)
    else:
        raise ValueError(f"Unknown quasi-random sequence: {name}")
