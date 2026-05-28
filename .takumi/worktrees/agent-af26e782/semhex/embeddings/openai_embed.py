"""OpenAI embedding provider using text-embedding-3-small.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from semhex.core.auth import load_api_key as _load_api_key
from semhex.embeddings.base import EmbeddingProvider

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMS = 1536


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API embedding provider."""

    def __init__(self, model: str = _DEFAULT_MODEL, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install semhex[openai]"
            )
        self._model = model
        resolved_api_key = api_key or _load_api_key("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self._client = OpenAI(api_key=resolved_api_key)

    @property
    def dimensions(self) -> int:
        return _DEFAULT_DIMS

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed(self, text: str) -> NDArray[np.float32]:
        response = self._client.embeddings.create(input=[text], model=self._model)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        # OpenAI API supports batch embedding natively
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
        # Normalize each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return vecs / norms
