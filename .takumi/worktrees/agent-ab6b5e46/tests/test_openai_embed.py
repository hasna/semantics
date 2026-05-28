"""Tests for OpenAI embedding provider — mocked, no real API calls."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestOpenAIEmbeddingProvider:
    @pytest.fixture
    def provider(self):
        """OpenAIEmbeddingProvider with a mocked OpenAI client."""
        with patch("openai.OpenAI") as MockOpenAI:
            client = MagicMock()
            MockOpenAI.return_value = client
            from semhex.embeddings.openai_embed import OpenAIEmbeddingProvider
            p = OpenAIEmbeddingProvider(api_key="test-key")
            p._client = client
            yield p, client

    def test_name(self, provider):
        p, _ = provider
        assert p.name == "openai:text-embedding-3-small"

    def test_dimensions(self, provider):
        p, _ = provider
        assert p.dimensions == 1536

    def test_embed_returns_array(self, provider):
        p, client = provider
        fake_vec = list(np.random.randn(1536))
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vec)]
        )
        result = p.embed("hello world")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1536,)
        assert result.dtype == np.float32

    def test_embed_normalized(self, provider):
        p, client = provider
        fake_vec = list(np.ones(1536))
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vec)]
        )
        result = p.embed("hello")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_embed_calls_api_once(self, provider):
        p, client = provider
        fake_vec = list(np.random.randn(1536))
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vec)]
        )
        p.embed("test text")
        assert client.embeddings.create.call_count == 1

    def test_embed_passes_text(self, provider):
        p, client = provider
        fake_vec = list(np.random.randn(1536))
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vec)]
        )
        p.embed("specific query")
        call_args = client.embeddings.create.call_args
        assert "specific query" in str(call_args)

    def test_embed_batch_returns_matrix(self, provider):
        p, client = provider
        vecs = [list(np.random.randn(1536)) for _ in range(3)]
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=v) for v in vecs]
        )
        result = p.embed_batch(["a", "b", "c"])
        assert result.shape == (3, 1536)
        assert result.dtype == np.float32

    def test_embed_batch_normalized(self, provider):
        p, client = provider
        vecs = [list(np.ones(1536)) for _ in range(2)]
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=v) for v in vecs]
        )
        result = p.embed_batch(["a", "b"])
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embed_batch_single_api_call(self, provider):
        """Batch embed should use one API call, not N calls."""
        p, client = provider
        vecs = [list(np.random.randn(1536)) for _ in range(5)]
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=v) for v in vecs]
        )
        p.embed_batch(["a", "b", "c", "d", "e"])
        assert client.embeddings.create.call_count == 1

    @patch("semhex.embeddings.openai_embed._load_api_key", return_value="loaded-key")
    @patch("openai.OpenAI")
    def test_uses_loaded_key_when_api_key_not_passed(self, mock_openai_cls, _mock_load_key):
        from semhex.embeddings.openai_embed import OpenAIEmbeddingProvider

        OpenAIEmbeddingProvider()

        mock_openai_cls.assert_called_once_with(api_key="loaded-key")

    @patch("semhex.embeddings.openai_embed._load_api_key", return_value=None)
    @patch("openai.OpenAI")
    def test_raises_when_key_missing(self, mock_openai_cls, _mock_load_key):
        from semhex.embeddings.openai_embed import OpenAIEmbeddingProvider

        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            OpenAIEmbeddingProvider()

        mock_openai_cls.assert_not_called()

    def test_zero_vector_not_normalized_to_nan(self, provider):
        """Zero vector should not produce NaN — norm guard prevents division by zero."""
        p, client = provider
        fake_vec = list(np.zeros(1536))
        client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=fake_vec)]
        )
        result = p.embed("zero vector")
        assert not np.any(np.isnan(result))
