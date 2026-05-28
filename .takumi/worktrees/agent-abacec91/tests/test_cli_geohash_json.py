"""Targeted CLI geohash JSON-output regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from click.testing import CliRunner

import semhex.cli as cli


RUNNER = CliRunner()


def _invoke(*args: str):
    return RUNNER.invoke(cli.main, list(args))


def test_hash_json_output(monkeypatch):
    import openai
    import semhex.core.auth as auth_mod
    import semhex.core.geohash_v2 as geohash_mod

    class DummyClient:
        class embeddings:
            @staticmethod
            def create(input, model, dimensions):
                return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            self.hex_length = 16
            self.total_bits = 64

        def load(self, state_name: str):
            return None

        def encode(self, vec):
            return "ABCD1234"

    monkeypatch.setattr(auth_mod, "load_api_key", lambda var_name: "sk-test-openai")
    monkeypatch.setattr(openai, "OpenAI", lambda api_key=None: DummyClient())
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)

    result = _invoke("hash", "compress this", "-j")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data == {
        "input": "compress this",
        "code": "ABCD1234",
        "bits": 64,
        "bits_per_dimension": 4,
        "hex_chars": 16,
        "state": "matryoshka_64d_4b",
    }


def test_hash_json_output_uses_canonical_2bit_state(monkeypatch):
    import openai
    import semhex.core.auth as auth_mod
    import semhex.core.geohash_v2 as geohash_mod

    class DummyClient:
        class embeddings:
            @staticmethod
            def create(input, model, dimensions):
                return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            self.hex_length = 8
            self.total_bits = 32

        def load(self, state_name: str):
            return None

        def encode(self, vec):
            return "BEEF"

    monkeypatch.setattr(auth_mod, "load_api_key", lambda var_name: "sk-test-openai")
    monkeypatch.setattr(openai, "OpenAI", lambda api_key=None: DummyClient())
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)

    result = _invoke("hash", "compress this", "--bits", "2", "-j")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"] == "matryoshka_64d_2b"
    assert data["bits_per_dimension"] == 2


def test_hash_json_error_on_missing_state(monkeypatch):
    import openai
    import semhex.core.geohash_v2 as geohash_mod

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            pass

        def load(self, state_name: str):
            raise FileNotFoundError(state_name)

    monkeypatch.setattr(openai, "OpenAI", lambda: object())
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)

    result = _invoke("hash", "compress this", "-j")
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data == {
        "state": "matryoshka_64d_4b",
        "bits_per_dimension": 4,
        "error": "Trained state 'matryoshka_64d_4b' not found. Run training first.",
    }


def test_hash_json_error_on_missing_openai_key(monkeypatch):
    import openai
    import semhex.core.auth as auth_mod
    import semhex.core.geohash_v2 as geohash_mod

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            pass

        def load(self, state_name: str):
            return None

    monkeypatch.setattr(auth_mod, "load_api_key", lambda var_name: None)
    monkeypatch.setattr(openai, "OpenAI", lambda api_key=None: (_ for _ in ()).throw(AssertionError("client should not be created")))
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)

    result = _invoke("hash", "compress this", "-j")
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data == {
        "state": "matryoshka_64d_4b",
        "bits_per_dimension": 4,
        "error": "OPENAI_API_KEY not found",
    }


def test_unhash_json_output_uses_canonical_2bit_state(monkeypatch):
    import semhex.core.geohash_v2 as geohash_mod

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            pass

        def load(self, state_name: str):
            return None

        def decode(self, code: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    labels = {
        "0": {"hex_code": "AAA111", "examples": ["first region"]},
    }

    original_read_text = Path.read_text
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)
    monkeypatch.setattr(np, "load", lambda path: np.array([[0.9, 0.1]], dtype=np.float32))
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps(labels) if self.name == "labels.json" else original_read_text(self, *args, **kwargs),
    )

    result = _invoke("unhash", "DEADBEEF", "--bits", "2", "-j")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"] == "matryoshka_64d_2b"
    assert data["bits_per_dimension"] == 2


def test_unhash_json_output(monkeypatch):
    import semhex.core.geohash_v2 as geohash_mod

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            pass

        def load(self, state_name: str):
            return None

        def decode(self, code: str):
            return np.array([1.0, 0.0], dtype=np.float32)

    labels = {
        "0": {"hex_code": "AAA111", "examples": ["first region"]},
        "1": {"hex_code": "BBB222", "examples": ["second region"]},
        "2": {"hex_code": "CCC333", "examples": ["third region"]},
    }

    original_read_text = Path.read_text
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)
    monkeypatch.setattr(
        np,
        "load",
        lambda path: np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *args, **kwargs: json.dumps(labels) if self.name == "labels.json" else original_read_text(self, *args, **kwargs),
    )

    result = _invoke("unhash", "DEADBEEF", "--neighbors", "2", "-j")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["code"] == "DEADBEEF"
    assert data["bits_per_dimension"] == 4
    assert data["neighbors"] == 2
    assert data["state"] == "matryoshka_64d_4b"
    assert data["nearest_regions"] == [
        {
            "index": 0,
            "hex_code": "AAA111",
            "similarity": 0.9,
            "examples": ["first region"],
        },
        {
            "index": 1,
            "hex_code": "BBB222",
            "similarity": 0.8,
            "examples": ["second region"],
        },
    ]
