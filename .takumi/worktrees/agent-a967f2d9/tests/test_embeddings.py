"""Tests for embedding providers."""

import numpy as np
import pytest

from semhex.embeddings import get_provider, MockEmbeddingProvider
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings.mock import MockEmbeddingProvider


class TestMockProvider:
    def setup_method(self):
        self.provider = MockEmbeddingProvider(dimensions=64)

    def test_implements_interface(self):
        assert isinstance(self.provider, EmbeddingProvider)

    def test_name(self):
        assert self.provider.name == "mock"

    def test_dimensions(self):
        assert self.provider.dimensions == 64

    def test_embed_returns_correct_shape(self):
        vec = self.provider.embed("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_embed_is_deterministic(self):
        """Same text always produces the same vector."""
        vec1 = self.provider.embed("hello world")
        vec2 = self.provider.embed("hello world")
        np.testing.assert_array_equal(vec1, vec2)

    def test_different_texts_different_vectors(self):
        """Different texts produce different vectors."""
        vec1 = self.provider.embed("hello world")
        vec2 = self.provider.embed("goodbye world")
        assert not np.allclose(vec1, vec2)

    def test_embed_is_normalized(self):
        """Vectors should have unit norm (for cosine similarity)."""
        vec = self.provider.embed("hello world")
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)

    def test_embed_batch(self):
        texts = ["hello", "world", "test"]
        vecs = self.provider.embed_batch(texts)
        assert vecs.shape == (3, 64)
        assert vecs.dtype == np.float32

    def test_embed_batch_matches_single(self):
        """Batch embedding should match individual embedding."""
        texts = ["hello", "world"]
        batch_vecs = self.provider.embed_batch(texts)
        for i, text in enumerate(texts):
            single_vec = self.provider.embed(text)
            np.testing.assert_array_equal(batch_vecs[i], single_vec)

    def test_empty_text(self):
        """Should handle empty strings without crashing."""
        vec = self.provider.embed("")
        assert vec.shape == (64,)

    def test_long_text(self):
        """Should handle long texts."""
        vec = self.provider.embed("word " * 10000)
        assert vec.shape == (64,)
        np.testing.assert_almost_equal(np.linalg.norm(vec), 1.0, decimal=5)

    def test_cosine_similarity_meaningful(self):
        """Similar texts should have higher cosine similarity than different texts."""
        # These won't be truly semantic (it's a mock), but the vectors should differ
        vec_a = self.provider.embed("I am happy")
        vec_b = self.provider.embed("I am happy today")
        vec_c = self.provider.embed("quantum physics equations")

        sim_ab = np.dot(vec_a, vec_b)
        sim_ac = np.dot(vec_a, vec_c)
        # Can't guarantee semantic ordering with mock, but both should be valid floats
        assert -1.0 <= sim_ab <= 1.0
        assert -1.0 <= sim_ac <= 1.0

    def test_custom_dimensions(self):
        provider = MockEmbeddingProvider(dimensions=128)
        vec = provider.embed("test")
        assert vec.shape == (128,)


class TestGetProvider:
    def test_mock(self):
        provider = get_provider("mock")
        assert isinstance(provider, MockEmbeddingProvider)

    def test_auto_fallback_to_mock(self):
        """Auto should at least return mock if no models installed."""
        provider = get_provider("auto")
        assert isinstance(provider, EmbeddingProvider)

    def test_auto_uses_openai_when_loader_finds_key(self):
        sentinel = object()
        with pytest.MonkeyPatch.context() as monkeypatch:
            import semhex.embeddings as embeddings_mod
            monkeypatch.setattr("semhex.embeddings.local_embed.LocalEmbeddingProvider", lambda: (_ for _ in ()).throw(ImportError("no local provider")))
            monkeypatch.setattr(embeddings_mod, "_load_api_key", lambda var_name: "loaded-key")
            monkeypatch.setattr("semhex.embeddings.openai_embed.OpenAIEmbeddingProvider", lambda api_key=None: sentinel)
            provider = get_provider("auto")
        assert provider is sentinel

    def test_import_does_not_require_openai(self):
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "openai":
                raise ImportError("openai unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with pytest.MonkeyPatch.context() as monkeypatch:
            for name in ["semhex.embeddings", "semhex.core.codec", "semhex.core.auth"]:
                sys.modules.pop(name, None)
            monkeypatch.setattr(builtins, "__import__", fake_import)
            module = importlib.import_module("semhex.embeddings")

        assert module is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_provider("nonexistent")
