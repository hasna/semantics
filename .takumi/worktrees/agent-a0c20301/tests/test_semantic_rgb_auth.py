"""Auth and secret-loading regression tests for semantic_rgb."""

from __future__ import annotations

from pathlib import Path

import pytest

import semhex.core.semantic_rgb as semantic_rgb


@pytest.fixture(autouse=True)
def reset_clients(monkeypatch):
    monkeypatch.setattr(semantic_rgb, "_cerebras_client", None)
    monkeypatch.setattr(semantic_rgb, "_openai_client", None)


def test_load_api_key_reads_env_files(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    secrets_file = tmp_path / ".secrets" / "hasna" / "shared" / "live.env"
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text('export OPENAI_API_KEY="sk-test-openai"\n')

    monkeypatch.setattr(semantic_rgb.Path, "home", classmethod(lambda cls: tmp_path))

    assert semantic_rgb._load_api_key("OPENAI_API_KEY") == "sk-test-openai"


def test_get_openai_raises_when_key_missing(monkeypatch):
    monkeypatch.setattr(semantic_rgb, "_load_api_key", lambda var_name: None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
        semantic_rgb._get_openai()


def test_get_cerebras_uses_loaded_key(monkeypatch):
    captured = {}

    def fake_openai(*, base_url=None, api_key=None):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr(semantic_rgb, "_load_api_key", lambda var_name: "cb-test-key")
    monkeypatch.setattr(semantic_rgb, "OpenAI", fake_openai)

    client = semantic_rgb._get_cerebras()
    assert client is semantic_rgb._cerebras_client
    assert captured == {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key": "cb-test-key",
    }


def test_get_openai_raises_helpful_import_error_when_openai_missing(monkeypatch):
    monkeypatch.setattr(semantic_rgb, "_load_api_key", lambda var_name: "sk-test-openai")
    monkeypatch.setattr(semantic_rgb, "OpenAI", None)

    with pytest.raises(ImportError, match=r"pip install semhex\[openai\]"):
        semantic_rgb._get_openai()


def test_get_client_auto_does_not_swallow_unexpected_errors(monkeypatch):
    monkeypatch.setattr(semantic_rgb, "_get_cerebras", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        semantic_rgb._get_client("auto")


def test_get_client_auto_falls_back_to_openai(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(semantic_rgb, "_get_cerebras", lambda: (_ for _ in ()).throw(ValueError("missing cerebras")))
    monkeypatch.setattr(semantic_rgb, "_get_openai", lambda: sentinel)

    client, model = semantic_rgb._get_client("auto")
    assert client is sentinel
    assert model == "gpt-4o-mini"
