"""Targeted CLI neighbor-count regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

import semhex.cli as cli


RUNNER = CliRunner()


class DummyCode:
    def __init__(self, value: str, depth: int = 2):
        self.value = value
        self.depth = depth

    def __str__(self) -> str:
        return self.value


def _invoke(*args: str):
    return RUNNER.invoke(cli.main, list(args))


def test_inspect_neighbors_are_configurable(monkeypatch):
    import semhex.core.decoder as decoder_mod
    import semhex.core.format as format_mod

    captured = {}

    def fake_decode(codes, codebook=None, k_neighbors=None):
        captured["k_neighbors"] = k_neighbors
        entry = SimpleNamespace(
            code=DummyCode("$00.0001"),
            label="triage",
            l1_label="support",
            examples=["help fix bug"],
            neighbors=["$00.0002", "$00.0003"],
        )
        return SimpleNamespace(decoded=[entry], summary="triage")

    monkeypatch.setattr(cli, "_get_codebook", lambda version="v0.1": object())
    monkeypatch.setattr(format_mod, "parse_code", lambda code: code)
    monkeypatch.setattr(decoder_mod, "decode", fake_decode)

    result = _invoke("inspect", "$00.0001", "--neighbors", "2", "-j")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert captured["k_neighbors"] == 2
    assert data["requested_neighbors"] == 2
    assert data["neighbors"] == ["$00.0002", "$00.0003"]


def test_unhash_neighbors_are_configurable(monkeypatch):
    import semhex.core.geohash_v2 as geohash_mod

    class DummyHasher:
        def __init__(self, n_dims: int, bits_per_dim: int):
            self.n_dims = n_dims
            self.bits_per_dim = bits_per_dim

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

    result = _invoke("unhash", "DEADBEEF", "--neighbors", "2")
    assert result.exit_code == 0, result.output
    assert "Nearest regions (top 2):" in result.output
    assert result.output.count("sim=") == 2
    assert "AAA111" in result.output
    assert "BBB222" in result.output
    assert "CCC333" not in result.output


@pytest.mark.parametrize("args", [("inspect", "$00.0001", "--neighbors", "0"), ("unhash", "DEADBEEF", "--neighbors", "0")])
def test_neighbor_count_must_be_positive(args):
    result = _invoke(*args)
    assert result.exit_code == 2
    assert "Invalid value for '--neighbors' / '-k'" in result.output
