"""Targeted CLI JSON-output regression tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner

import semhex.cli as cli


class DummyCode:
    def __init__(self, value: str, depth: int = 2):
        self.value = value
        self.depth = depth

    def __str__(self) -> str:
        return self.value


class DummyEvalResult:
    def __init__(self, payload: dict, **attrs):
        self._payload = payload
        for key, value in attrs.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        return dict(self._payload)


RUNNER = CliRunner()


def _invoke_json(*args: str) -> dict:
    result = RUNNER.invoke(cli.main, list(args))
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _entry(
    code: str,
    label: str,
    *,
    category: str = "category",
    depth: int = 2,
    examples: list[str] | None = None,
    neighbors: list[str] | None = None,
):
    return SimpleNamespace(
        code=DummyCode(code, depth=depth),
        label=label,
        l1_label=category,
        examples=examples or [],
        neighbors=neighbors or [],
    )


def _decode_result(entries, summary: str = "decoded summary"):
    return SimpleNamespace(decoded=list(entries), summary=summary)


def test_distance_json(monkeypatch):
    import semhex.core.distance as distance_mod

    monkeypatch.setattr(cli, "_get_codebook", lambda version="v0.1": object())
    monkeypatch.setattr(distance_mod, "distance", lambda code_a, code_b, codebook=None: 0.125)
    monkeypatch.setattr(distance_mod, "similarity", lambda code_a, code_b, codebook=None: 0.875)

    data = _invoke_json("distance", "$00.0000", "$00.0001", "-j")
    assert data == {
        "code_a": "$00.0000",
        "code_b": "$00.0001",
        "distance": 0.125,
        "similarity": 0.875,
    }


def test_blend_json(monkeypatch):
    import semhex.core.blend as blend_mod
    import semhex.core.decoder as decoder_mod

    monkeypatch.setattr(cli, "_get_codebook", lambda version="v0.1": object())
    monkeypatch.setattr(blend_mod, "blend", lambda code_a, code_b, weight=0.5, codebook=None: "$00.00FF")

    def fake_decode(codes, codebook=None, k_neighbors=None):
        mapping = {
            "$00.0001": "source-a",
            "$00.0002": "source-b",
            "$00.00FF": "blend-result",
        }
        return _decode_result([_entry(str(code), mapping[str(code)]) for code in codes])

    monkeypatch.setattr(decoder_mod, "decode", fake_decode)

    data = _invoke_json("blend", "$00.0001", "$00.0002", "--weight", "0.25", "-j")
    assert data["result"] == "$00.00FF"
    assert data["label_result"] == "blend-result"
    assert data["weight"] == 0.25
    assert data["inverse_weight"] == 0.75


def test_inspect_json(monkeypatch):
    import semhex.core.decoder as decoder_mod
    import semhex.core.format as format_mod

    monkeypatch.setattr(cli, "_get_codebook", lambda version="v0.1": object())
    monkeypatch.setattr(format_mod, "parse_code", lambda code: code)
    monkeypatch.setattr(
        decoder_mod,
        "decode",
        lambda codes, codebook=None, k_neighbors=None: _decode_result(
            [
                _entry(
                    "$00.0001",
                    "triage bug",
                    category="support",
                    examples=["help fix bug"],
                    neighbors=["$00.0002", "$00.0003"],
                )
            ]
        ),
    )

    data = _invoke_json("inspect", "$00.0001", "-j")
    assert data["found"] is True
    assert data["label"] == "triage bug"
    assert data["examples"] == ["help fix bug"]
    assert data["neighbors"] == ["$00.0002", "$00.0003"]


def test_roundtrip_json(monkeypatch):
    import semhex.core.decoder as decoder_mod
    import semhex.core.encoder as encoder_mod

    monkeypatch.setattr(cli, "_get_codebook", lambda version="v0.1": object())
    monkeypatch.setattr(cli, "_get_provider", lambda codebook=None: object())
    monkeypatch.setattr(
        encoder_mod,
        "encode",
        lambda text, depth=2, codebook=None, provider=None: SimpleNamespace(
            code_strings=["$00.0001"],
            codes=[DummyCode("$00.0001")],
            chunks=[text],
            distances=[0.1],
            compression_ratio=3.5,
        ),
    )
    monkeypatch.setattr(
        decoder_mod,
        "decode",
        lambda codes, codebook=None, k_neighbors=None: _decode_result([_entry("$00.0001", "triage")], summary="triage"),
    )

    data = _invoke_json("roundtrip", "Investigate the bug", "-j")
    assert data["input"] == "Investigate the bug"
    assert data["codes"] == ["$00.0001"]
    assert data["decoded_summary"] == "triage"
    assert data["decoded"][0]["label"] == "triage"


def test_dict_info_json():
    data = _invoke_json("dict-info", "-j")
    assert data["version"] == "1.0"
    assert data["entries"] > 0
    assert isinstance(data["tiers"], dict)


def test_codebook_info_json(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_get_codebook",
        lambda version="v0.1": SimpleNamespace(version="test", dimensions=64, n_level1=16, n_level2=64),
    )

    data = _invoke_json("codebook", "info", "-j")
    assert data == {
        "version": "test",
        "dimensions": 64,
        "l1_clusters": 16,
        "l2_clusters": 64,
        "total_codes": 80,
    }


def test_codec_commands_json(monkeypatch):
    import semhex.core.codec as codec_mod

    monkeypatch.setattr(codec_mod, "compress", lambda text, quality=2, provider="cerebras": "BUG.HLP")
    monkeypatch.setattr(codec_mod, "decompress", lambda codes, provider="cerebras": "help fix bug")
    monkeypatch.setattr(
        codec_mod,
        "roundtrip",
        lambda text, quality=2, provider="cerebras": {
            "input": text,
            "codes": "BUG.HLP",
            "output": "help fix bug",
            "compression_ratio": 2.4,
            "input_chars": len(text),
            "code_chars": 7,
            "semantic_similarity": 0.91,
            "similarity_error": None,
            "compress_time": 0.01,
            "decompress_time": 0.02,
            "quality": quality,
        },
    )

    compressed = _invoke_json("compress", "help fix bug", "-j")
    assert compressed["codes"] == "BUG.HLP"
    assert compressed["quality"] == 2

    decompressed = _invoke_json("decompress", "BUG.HLP", "-j")
    assert decompressed == {
        "codes": "BUG.HLP",
        "text": "help fix bug",
        "provider": "cerebras",
    }

    roundtrip = _invoke_json("codec-roundtrip", "help fix bug", "-j")
    assert roundtrip["semantic_similarity"] == 0.91
    assert roundtrip["quality"] == 2


def test_codec_roundtrip_reports_similarity_error(monkeypatch):
    import semhex.core.codec as codec_mod

    monkeypatch.setattr(
        codec_mod,
        "roundtrip",
        lambda text, quality=2, provider="cerebras": {
            "input": text,
            "codes": "BUG.HLP",
            "output": "help fix bug",
            "compression_ratio": 2.4,
            "input_chars": len(text),
            "code_chars": 7,
            "semantic_similarity": None,
            "similarity_error": "OPENAI_API_KEY not found",
            "compress_time": 0.01,
            "decompress_time": 0.02,
            "quality": quality,
            "provider": provider,
        },
    )

    roundtrip = RUNNER.invoke(cli.main, ["codec-roundtrip", "help fix bug"])
    assert roundtrip.exit_code == 0, roundtrip.output
    assert "Similarity:" in roundtrip.output
    assert "unavailable" in roundtrip.output
    assert "OPENAI_API_KEY not found" in roundtrip.output


def test_eval_commands_json(monkeypatch):
    import evaluation.benchmark as benchmark_mod
    import evaluation.eval_composition as composition_mod
    import evaluation.eval_distance as distance_mod
    import evaluation.eval_roundtrip as roundtrip_mod

    monkeypatch.setattr(
        roundtrip_mod,
        "eval_roundtrip",
        lambda: DummyEvalResult(
            {"n_sentences": 2, "mean_similarity": 0.8, "min_similarity": 0.6, "std_similarity": 0.1, "elapsed_seconds": 0.02},
            n_sentences=2,
            mean_similarity=0.8,
            min_similarity=0.6,
            std_similarity=0.1,
            elapsed_seconds=0.02,
        ),
    )
    monkeypatch.setattr(
        composition_mod,
        "eval_composition",
        lambda n_pairs=200: DummyEvalResult(
            {"n_pairs": n_pairs, "valid_count": n_pairs - 10, "validity_rate": 0.95, "mean_similarity": 0.72, "elapsed_seconds": 0.03},
            n_pairs=n_pairs,
            valid_count=n_pairs - 10,
            validity_rate=0.95,
            mean_similarity=0.72,
            elapsed_seconds=0.03,
        ),
    )
    monkeypatch.setattr(
        distance_mod,
        "eval_distance_correlation",
        lambda: DummyEvalResult(
            {"n_pairs": 5, "spearman_r": 0.84, "spearman_p": 0.001, "elapsed_seconds": 0.04},
            n_pairs=5,
            spearman_r=0.84,
            spearman_p=0.001,
            elapsed_seconds=0.04,
        ),
    )
    monkeypatch.setattr(
        benchmark_mod,
        "run_benchmark",
        lambda: DummyEvalResult(
            {
                "n_sentences": 10,
                "total_words": 50,
                "total_codes": 20,
                "compression_ratio": 2.5,
                "encode_time_seconds": 0.2,
                "encode_rate_per_second": 50.0,
                "lookup_latency_ms": 0.3,
                "codebook_memory_mb": 1.2,
            },
            n_sentences=10,
            compression_ratio=2.5,
            encode_rate=50.0,
            lookup_time=0.0003,
            codebook_memory_mb=1.2,
        ),
    )

    assert _invoke_json("eval", "roundtrip", "-j")["mean_similarity"] == 0.8
    assert _invoke_json("eval", "composition", "-j")["validity_rate"] == 0.95
    assert _invoke_json("eval", "distance", "-j")["spearman_r"] == 0.84
    assert _invoke_json("eval", "benchmark", "-j")["compression_ratio"] == 2.5

    bundle = _invoke_json("eval", "all", "-j")
    assert set(bundle) == {"roundtrip", "composition", "distance", "benchmark"}
    assert bundle["benchmark"]["compression_ratio"] == 2.5
