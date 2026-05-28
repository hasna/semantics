"""Local embedding provider using sentence-transformers.

Runs entirely locally — no API key needed. Downloads model on first use (~90MB).
Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from semhex.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers based local embedding provider."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install semhex[local]"
            )
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    def embed(self, text: str) -> NDArray[np.float32]:
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
        return vecs.astype(np.float32)
