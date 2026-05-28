"""Integration tests: full encode→decode roundtrip pipeline."""

import pytest
from pathlib import Path

from semhex.core.codebook import load_codebook
from semhex.core.encoder import encode
from semhex.core.decoder import decode
from semhex.core.distance import distance, similarity
from semhex.core.blend import blend
from semhex.core.format import SemHexCode, parse_code
from semhex.embeddings.mock import MockEmbeddingProvider


@pytest.fixture
def setup():
    provider = MockEmbeddingProvider(dimensions=64)
    codebook = load_codebook("v0.1")
    return codebook, provider


class TestFullPipeline:
    """End-to-end tests for the entire SemHex pipeline."""

    def test_encode_decode_cycle(self, setup):
        codebook, provider = setup
        text = "I need help fixing a bug in my Python code."
        enc = encode(text, codebook=codebook, provider=provider)
        dec = decode(enc.codes, codebook=codebook)
        assert len(dec.decoded) == len(enc.codes)
        assert all(d.label != "[unknown code]" for d in dec.decoded)

    def test_encode_distance_decode(self, setup):
        codebook, provider = setup
        text_a = "The server is crashing repeatedly."
        text_b = "The system keeps going down."
        enc_a = encode(text_a, codebook=codebook, provider=provider)
        enc_b = encode(text_b, codebook=codebook, provider=provider)
        if enc_a.codes and enc_b.codes:
            d = distance(enc_a.codes[0], enc_b.codes[0], codebook=codebook)
            assert 0.0 <= d <= 2.0

    def test_encode_blend_decode(self, setup):
        codebook, provider = setup
        text_a = "I am happy"
        text_b = "I am sad"
        enc_a = encode(text_a, codebook=codebook, provider=provider)
        enc_b = encode(text_b, codebook=codebook, provider=provider)
        if enc_a.codes and enc_b.codes:
            blended = blend(enc_a.codes[0], enc_b.codes[0], codebook=codebook)
            dec = decode([blended], codebook=codebook)
            assert dec.decoded[0].label != "[unknown code]"

    def test_multiple_sentences_pipeline(self, setup):
        codebook, provider = setup
        text = "The first point is about performance. The second point is about reliability. The third is about cost."
        enc = encode(text, codebook=codebook, provider=provider)
        assert len(enc.codes) == 3
        dec = decode(enc.codes, codebook=codebook)
        assert len(dec.decoded) == 3
        assert "|" in dec.summary

    def test_code_string_roundtrip(self, setup):
        codebook, provider = setup
        text = "Test sentence for roundtrip."
        enc = encode(text, codebook=codebook, provider=provider)
        code_str = " ".join(enc.code_strings)
        dec = decode(code_str, codebook=codebook)
        assert len(dec.decoded) == len(enc.codes)

    def test_empty_text_safe(self, setup):
        codebook, provider = setup
        enc = encode("", codebook=codebook, provider=provider)
        dec = decode(enc.codes, codebook=codebook)
        assert enc.codes == []
        assert dec.decoded == []

    def test_very_long_text(self, setup):
        codebook, provider = setup
        text = ". ".join([f"This is sentence number {i} about various topics" for i in range(20)])
        enc = encode(text, codebook=codebook, provider=provider)
        assert len(enc.codes) > 1
        dec = decode(enc.codes, codebook=codebook)
        assert len(dec.decoded) == len(enc.codes)

    def test_compression_is_real(self, setup):
        codebook, provider = setup
        text = "The machine learning pipeline processes data through multiple stages including preprocessing, feature extraction, model training, hyperparameter tuning, and finally deployment to production servers."
        enc = encode(text, codebook=codebook, provider=provider)
        words = len(text.split())
        codes = len(enc.codes)
        assert codes < words  # Must compress

    def test_different_depths(self, setup):
        codebook, provider = setup
        text = "Hello world"
        enc1 = encode(text, depth=1, codebook=codebook, provider=provider)
        enc2 = encode(text, depth=2, codebook=codebook, provider=provider)
        assert enc1.codes[0].depth == 1
        assert enc2.codes[0].depth == 2

    def test_determinism(self, setup):
        codebook, provider = setup
        text = "Reproducibility is important for science."
        r1 = encode(text, codebook=codebook, provider=provider)
        r2 = encode(text, codebook=codebook, provider=provider)
        assert r1.code_strings == r2.code_strings


class TestCLIIntegration:
    """Test CLI commands via subprocess."""

    def test_cli_encode(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "semhex", "encode", "hello world", "-j"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert "codes" in data
        assert len(data["codes"]) > 0

    def test_cli_decode(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "semhex", "decode", "$00.0000", "-j"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert "summary" in data

    def test_cli_codebook_info(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "semhex", "codebook", "info"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "L1 clusters" in result.stdout

    def test_cli_roundtrip(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "semhex", "roundtrip", "The cat sat on the mat."],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "Compression" in result.stdout
