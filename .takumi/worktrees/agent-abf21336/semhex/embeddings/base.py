"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(ABC):
    """Interface for text embedding models."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vectors."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the provider."""
        ...

    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string into a vector.

        Args:
            text: Input text to embed.

        Returns:
            1-D numpy array of shape (dimensions,).
        """
        ...

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed multiple texts into vectors.

        Default implementation calls embed() in a loop.
        Subclasses should override for batched API calls.

        Args:
            texts: List of input texts.

        Returns:
            2-D numpy array of shape (len(texts), dimensions).
        """
        return np.array([self.embed(t) for t in texts], dtype=np.float32)
