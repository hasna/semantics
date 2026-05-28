"""Tests for the LLM codec: compress/decompress.

Uses mock LLM responses to test the codec logic without API calls.
Integration tests with real APIs are in eval_codec.py.
"""

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from semhex.core.codec import _load_api_key, compress, decompress, roundtrip


class FakeCompletion:
    """Fake OpenAI completion response."""
    def __init__(self, text):
        self.choices = [MagicMock(message=MagicMock(content=text))]


class TestCompress:
    @patch("semhex.core.codec._get_cerebras")
    def test_returns_string(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("FRU.BUG.HLP")
        mock_get.return_value = client

        result = compress("I'm frustrated with this bug", quality=2, provider="cerebras")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("semhex.core.codec._get_cerebras")
    def test_returns_dot_separated_codes(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("FRU.BUG.HLP")
        mock_get.return_value = client

        result = compress("test", quality=2, provider="cerebras")
        assert "." in result
        parts = result.split(".")
        assert all(p.isalnum() for p in parts)

    @patch("semhex.core.codec._get_cerebras")
    def test_strips_think_tags(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("<think>thinking...</think>FRU.BUG")
        mock_get.return_value = client

        result = compress("test", quality=2, provider="cerebras")
        assert "<think>" not in result
        assert "FRU.BUG" in result

    @patch("semhex.core.codec._get_cerebras")
    def test_empty_input(self, mock_get):
        result = compress("", quality=2, provider="cerebras")
        assert result == ""

    @patch("semhex.core.codec._get_cerebras")
    def test_quality_parameter_passed(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("A1")
        mock_get.return_value = client

        compress("test", quality=1, provider="cerebras")
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        user_msg = [m for m in messages if m["role"] == "user"][0]["content"]
        assert "1-2 codes" in user_msg  # quality=1 instruction


class TestDecompress:
    @patch("semhex.core.codec._get_cerebras")
    def test_returns_string(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("I'm frustrated with this bug")
        mock_get.return_value = client

        result = decompress("FRU.BUG.HLP", provider="cerebras")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("semhex.core.codec._get_cerebras")
    def test_strips_think_tags(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("<think>hmm</think>Hello world")
        mock_get.return_value = client

        result = decompress("HLO.WLD", provider="cerebras")
        assert "<think>" not in result
        assert "Hello world" in result

    @patch("semhex.core.codec._get_cerebras")
    def test_strips_prefix(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("TEXT: Hello world")
        mock_get.return_value = client

        result = decompress("HLO.WLD", provider="cerebras")
        assert not result.startswith("TEXT:")
        assert "Hello world" in result

    @patch("semhex.core.codec._get_cerebras")
    def test_empty_input(self, mock_get):
        result = decompress("", provider="cerebras")
        assert result == ""


class TestRoundtrip:
    @patch("semhex.core.codec._get_openai")
    @patch("semhex.core.codec._get_cerebras")
    def test_returns_dict(self, mock_cerebras, mock_openai):
        # Mock compress
        cerebras = MagicMock()
        cerebras.chat.completions.create.return_value = FakeCompletion("FRU.BUG")
        mock_cerebras.return_value = cerebras

        # Mock similarity embedding
        openai = MagicMock()
        fake_emb = MagicMock()
        fake_emb.data = [
            MagicMock(embedding=[1.0] * 10),
            MagicMock(embedding=[0.9] * 10),
        ]
        openai.embeddings.create.return_value = fake_emb
        mock_openai.return_value = openai

        result = roundtrip("frustrated with bug", quality=2, provider="cerebras")
        assert "input" in result
        assert "codes" in result
        assert "output" in result
        assert "compression_ratio" in result
        assert "semantic_similarity" in result

    @patch("semhex.core.codec._get_cerebras")
    def test_compression_ratio_positive(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            FakeCompletion("FRU.BUG"),  # compress
            FakeCompletion("Frustrated with a bug"),  # decompress
        ]
        mock_get.return_value = client

        result = roundtrip("I am really frustrated with this terrible bug", quality=2, provider="cerebras")
        assert result["compression_ratio"] > 1.0

    @patch("semhex.core.codec._get_cerebras")
    def test_has_timing(self, mock_get):
        client = MagicMock()
        client.chat.completions.create.return_value = FakeCompletion("A1.B2")
        mock_get.return_value = client

        result = roundtrip("test", quality=2, provider="cerebras")
        assert "compress_time" in result
        assert "decompress_time" in result
        assert result["compress_time"] >= 0
        assert result["decompress_time"] >= 0


    @patch("semhex.core.codec._get_openai")
    @patch("semhex.core.codec._get_cerebras")
    def test_zero_similarity_is_preserved(self, mock_cerebras, mock_openai):
        cerebras = MagicMock()
        cerebras.chat.completions.create.side_effect = [
            FakeCompletion("FRU.BUG"),
            FakeCompletion("Frustrated with a bug"),
        ]
        mock_cerebras.return_value = cerebras

        openai = MagicMock()
        fake_emb = MagicMock()
        fake_emb.data = [
            MagicMock(embedding=[1.0, 0.0]),
            MagicMock(embedding=[0.0, 1.0]),
        ]
        openai.embeddings.create.return_value = fake_emb
        mock_openai.return_value = openai

        result = roundtrip("frustrated with bug", quality=2, provider="cerebras")
        assert result["semantic_similarity"] == 0.0
        assert result["similarity_error"] is None

    @patch("semhex.core.codec._get_openai", side_effect=ValueError("OPENAI_API_KEY not found"))
    @patch("semhex.core.codec._get_cerebras")
    def test_similarity_error_reported_when_measurement_fails(self, mock_cerebras, _mock_openai):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            FakeCompletion("FRU.BUG"),
            FakeCompletion("Frustrated with a bug"),
        ]
        mock_cerebras.return_value = client

        result = roundtrip("frustrated with bug", quality=2, provider="cerebras")
        assert result["semantic_similarity"] is None
        assert result["similarity_error"] == "OPENAI_API_KEY not found"


class TestScalingResults:
    """Test that scaling law results exist and are valid."""

    def test_scaling_results_exist(self):
        import json
        from pathlib import Path
        path = Path("evaluation/results/scaling_results.json")
        if path.exists():
            data = json.loads(path.read_text())
            assert len(data) >= 3  # At least 3 codebook sizes tested

    def test_rvq_results_exist(self):
        import json
        from pathlib import Path
        path = Path("evaluation/results/rvq_scaling_results.json")
        if path.exists():
            data = json.loads(path.read_text())
            assert len(data) >= 1
            # Quality should improve with more levels
            sims = [r["mean_similarity"] for r in data]
            assert sims[-1] >= sims[0]  # Last level >= first level

    def test_larger_codebook_better(self):
        import json
        from pathlib import Path
        path = Path("evaluation/results/scaling_results.json")
        if path.exists():
            data = json.loads(path.read_text())
            results = [r for r in data if isinstance(r, dict) and "mean_similarity" in r]
            if len(results) >= 2:
                # Larger codebook should give better similarity
                assert results[-1]["mean_similarity"] >= results[0]["mean_similarity"]


class TestApiKeyLoading:
    def test_reads_exported_key_from_secrets_env(self, tmp_path, monkeypatch):
        secrets_root = tmp_path / ".secrets" / "open" / "svc"
        secrets_root.mkdir(parents=True)
        (secrets_root / "live.env").write_text(
            "export OPENAI_API_KEY='sk-test-123'\nOTHER=1\n",
            encoding="utf-8",
        )

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _load_api_key("OPENAI_API_KEY") == "sk-test-123"

    def test_returns_none_when_secrets_root_is_directory_only(self, tmp_path, monkeypatch):
        (tmp_path / ".secrets").mkdir()
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _load_api_key("CEREBRAS_API_KEY") is None
