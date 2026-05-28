"""Tests for the decoder: SemHex codes → human-readable output."""

import pytest

from semhex.core.codebook import load_codebook
from semhex.core.decoder import decode, DecodeResult, DecodedCode
from semhex.core.encoder import encode
from semhex.core.format import SemHexCode
from semhex.embeddings.mock import MockEmbeddingProvider


@pytest.fixture
def mock_setup():
    provider = MockEmbeddingProvider(dimensions=64)
    codebook = load_codebook("v0.1")
    return codebook, provider


class TestDecode:
    def test_decode_single_code_object(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook)
        assert isinstance(result, DecodeResult)
        assert len(result.decoded) == 1
        assert isinstance(result.decoded[0], DecodedCode)

    def test_decode_string_codes(self, mock_setup):
        codebook, _ = mock_setup
        result = decode(["$00.0000"], codebook=codebook)
        assert len(result.decoded) == 1

    def test_decode_space_separated_string(self, mock_setup):
        codebook, _ = mock_setup
        result = decode("$00.0000 $01.0000", codebook=codebook)
        assert len(result.decoded) == 2

    def test_decoded_has_label(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook)
        assert result.decoded[0].label != ""
        assert result.decoded[0].label != "[unknown code]"

    def test_decoded_has_l1_label(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook)
        assert result.decoded[0].l1_label != ""

    def test_decoded_has_neighbors(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook, k_neighbors=3)
        assert len(result.decoded[0].neighbors) > 0

    def test_summary(self, mock_setup):
        codebook, _ = mock_setup
        result = decode("$00.0000 $01.0000", codebook=codebook)
        summary = result.summary
        assert "|" in summary  # Two codes separated by |
        assert len(summary) > 0

    def test_detailed(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook)
        detailed = result.detailed
        assert "$00.0000" in detailed

    def test_to_dict(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook)
        d = result.to_dict()
        assert "summary" in d
        assert "codes" in d
        assert len(d["codes"]) == 1
        assert "label" in d["codes"][0]
        assert "category" in d["codes"][0]
        assert "examples" in d["codes"][0]
        assert "neighbors" in d["codes"][0]

    def test_str_representation(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([SemHexCode(0, 0)], codebook=codebook)
        s = str(result)
        assert len(s) > 0

    def test_unknown_code_graceful(self, mock_setup):
        codebook, _ = mock_setup
        # Code with L1 out of range
        result = decode([SemHexCode(255, 65535)], codebook=codebook)
        assert result.decoded[0].label == "[unknown code]"

    def test_empty_input(self, mock_setup):
        codebook, _ = mock_setup
        result = decode([], codebook=codebook)
        assert len(result.decoded) == 0

    def test_empty_string(self, mock_setup):
        codebook, _ = mock_setup
        result = decode("", codebook=codebook)
        assert len(result.decoded) == 0


class TestRoundtrip:
    """Encode text → get codes → decode codes → check output."""

    def test_encode_then_decode(self, mock_setup):
        codebook, provider = mock_setup
        text = "The cat sat on the mat."

        # Encode
        enc = encode(text, codebook=codebook, provider=provider)
        assert len(enc.codes) > 0

        # Decode
        dec = decode(enc.codes, codebook=codebook)
        assert len(dec.decoded) == len(enc.codes)

        # Each decoded entry should have a label
        for d in dec.decoded:
            assert d.label != "[unknown code]"

    def test_roundtrip_preserves_count(self, mock_setup):
        codebook, provider = mock_setup
        text = "First sentence about cats. Second sentence about dogs. Third about birds."

        enc = encode(text, codebook=codebook, provider=provider)
        dec = decode(enc.codes, codebook=codebook)

        assert len(dec.decoded) == len(enc.codes)

    def test_roundtrip_summary_nonempty(self, mock_setup):
        codebook, provider = mock_setup
        text = "I need help debugging this Python error."

        enc = encode(text, codebook=codebook, provider=provider)
        dec = decode(enc.codes, codebook=codebook)

        assert len(dec.summary) > 0
