"""Mock embedding provider for testing.

Produces deterministic vectors from text hashing — no model needed.
Same text always produces the same vector. Different texts produce different vectors.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from semhex.embeddings.base import EmbeddingProvider

_DEFAULT_DIMS = 64


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embeddings from text hashing.

    Useful for tests — no model download, no API key, fully reproducible.
    Vectors are normalized to unit length so cosine similarity works.
    """

    def __init__(self, dimensions: int = _DEFAULT_DIMS):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return "mock"

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate a deterministic vector from the text's hash."""
        # Use SHA-256 to get enough bytes, then expand to fill dimensions
        h = hashlib.sha256(text.encode("utf-8")).digest()

        # Use the hash as a seed for reproducible random vector
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dimensions).astype(np.float32)

        # Normalize to unit length (so cosine similarity is meaningful)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec
