"""Embedding providers for SemHex."""

from __future__ import annotations

from semhex.core.auth import load_api_key as _load_api_key
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings.mock import MockEmbeddingProvider


def get_provider(name: str = "auto") -> EmbeddingProvider:
    """Get an embedding provider by name.

    Args:
        name: Provider name. Options:
            - "auto": Try local first, fall back to mock
            - "mock": Deterministic mock (no model, for tests)
            - "local": sentence-transformers (requires pip install semhex[local])
            - "openai": OpenAI API (requires pip install semhex[openai] + OPENAI_API_KEY)

    Returns:
        An EmbeddingProvider instance.
    """
    if name == "mock":
        return MockEmbeddingProvider()

    if name == "local":
        from semhex.embeddings.local_embed import LocalEmbeddingProvider
        return LocalEmbeddingProvider()

    if name == "openai":
        from semhex.embeddings.openai_embed import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider()

    if name == "auto":
        # Try local first (free, no API key)
        try:
            from semhex.embeddings.local_embed import LocalEmbeddingProvider
            return LocalEmbeddingProvider()
        except ImportError:
            pass

        # Try OpenAI if a key is available via env or ~/.secrets
        api_key = _load_api_key("OPENAI_API_KEY")
        if api_key:
            try:
                from semhex.embeddings.openai_embed import OpenAIEmbeddingProvider
                return OpenAIEmbeddingProvider(api_key=api_key)
            except ImportError:
                pass

        # Fall back to mock
        return MockEmbeddingProvider()

    raise ValueError(f"Unknown embedding provider: {name!r}")


__all__ = ["EmbeddingProvider", "MockEmbeddingProvider", "get_provider"]
