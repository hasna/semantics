"""Targeted CLI validation and failure-mode regression tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import semhex.cli as cli


RUNNER = CliRunner()


def _invoke(*args: str):
    return RUNNER.invoke(cli.main, list(args))


@pytest.mark.parametrize(
    ("args", "expected_fragment"),
    [
        (("encode", "hello", "--depth", "0"), "Invalid value for '--depth' / '-d'"),
        (("roundtrip", "hello", "--depth", "3"), "Invalid value for '--depth' / '-d'"),
        (("blend", "$00.0001", "$00.0002", "--weight", "1.5"), "Invalid value for '--weight' / '-w'"),
        (("compress", "hello", "--quality", "5"), "Invalid value for '--quality' / '-q'"),
        (("decompress", "BUG.HLP", "--provider", "bogus"), "Invalid value for '--provider' / '-p'"),
        (("hash", "hello", "--bits", "3"), "Invalid value for '--bits' / '-b'"),
        (("eval", "composition", "--n-pairs", "0"), "Invalid value for '--n-pairs'"),
    ],
)
def test_invalid_option_values_exit_non_zero(args, expected_fragment):
    result = _invoke(*args)
    assert result.exit_code == 2
    assert expected_fragment in result.output


def test_inspect_missing_code_returns_json_error(monkeypatch):
    import semhex.core.decoder as decoder_mod
    import semhex.core.format as format_mod

    monkeypatch.setattr(cli, "_get_codebook", lambda version="v0.1": object())
    monkeypatch.setattr(format_mod, "parse_code", lambda code: code)
    monkeypatch.setattr(
        decoder_mod,
        "decode",
        lambda codes, codebook=None, k_neighbors=None: SimpleNamespace(decoded=[], summary=""),
    )

    result = _invoke("inspect", "$00.0001", "-j")
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data == {
        "code": "$00.0001",
        "found": False,
        "neighbors": [],
        "error": "Code not found in codebook",
    }


def test_hash_missing_state_exits_non_zero(monkeypatch):
    import openai
    import semhex.core.geohash_v2 as geohash_mod

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            self.n_dims = n_dims
            self.bits_per_dim = bits_per_dim

        def load(self, state_name: str):
            raise FileNotFoundError(state_name)

    monkeypatch.setattr(openai, "OpenAI", lambda: object())
    monkeypatch.setattr(geohash_mod, "SemHasher", DummyHasher)

    result = _invoke("hash", "hello")
    assert result.exit_code == 1
    assert "Trained state 'matryoshka_64d_4b' not found" in result.output
