"""Tests for dictionary encoder/decoder and dict CLI commands."""

import json
import subprocess
from pathlib import Path

import pytest

from semhex.core.dict_encoder import dict_encode
from semhex.core.dict_decoder import dict_decode, dict_decode_detailed

PROJECT_ROOT = str(Path(__file__).parent.parent)


class TestDictEncoder:
    def test_encodes_common_words(self):
        codes = dict_encode("the cat sat on the mat")
        assert isinstance(codes, str)
        assert len(codes) > 0
        assert "." in codes or len(codes) <= 4  # at least one code

    def test_top_words_get_short_codes(self):
        # "the" should be code "00" (most common word)
        codes = dict_encode("the")
        assert "00" in codes

    def test_unknown_words_pass_through_bracketed(self):
        codes = dict_encode("xyzzy_unknown_word_12345")
        assert "[" in codes

    def test_roundtrip_common_text(self):
        text = "I am frustrated with this bug"
        codes = dict_encode(text)
        decoded = dict_decode(codes)
        # Should recover the key words (lowercased)
        assert "frustrated" in decoded
        assert "bug" in decoded

    def test_empty_string(self):
        codes = dict_encode("")
        assert codes == ""

    def test_phrase_encoding(self):
        # "i am" should encode as a single phrase code
        codes = dict_encode("i am happy")
        parts = codes.split(".")
        # Should be fewer codes than words (phrase merging)
        assert len(parts) <= 3

    def test_compression_ratio_positive(self):
        text = "Can you help me debug this async error?"
        codes = dict_encode(text)
        # Should compress (codes shorter than original)
        assert len(codes) < len(text)


class TestDictDecoder:
    def test_decodes_known_code(self):
        # "00" → "the"
        result = dict_decode("00")
        assert result == "the"

    def test_decodes_unknown_code_with_marker(self):
        result = dict_decode("ZZZZ")
        assert "[?ZZZZ]" in result

    def test_decodes_passthrough_brackets(self):
        result = dict_decode("[hello]")
        assert result == "hello"

    def test_detailed_returns_dict(self):
        result = dict_decode_detailed("00.01")
        assert isinstance(result, dict)
        assert "text" in result
        assert "entries" in result
        assert "n_codes" in result
        assert "n_found" in result
        assert "n_unknown" in result

    def test_detailed_counts_correct(self):
        # "00" = "the", "ZZZZ" = unknown
        result = dict_decode_detailed("00.ZZZZ")
        assert result["n_codes"] == 2
        assert result["n_found"] == 1
        assert result["n_unknown"] == 1

    def test_detailed_entries_structure(self):
        result = dict_decode_detailed("00.01")
        for entry in result["entries"]:
            assert "code" in entry
            assert "text" in entry
            assert "found" in entry


class TestDictCLI:
    """Integration tests for dict CLI commands."""

    def _run(self, *args):
        return subprocess.run(
            ["python3", "-m", "semhex", *args],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )

    def test_dict_encode_basic(self):
        result = self._run("dict-encode", "hello world")
        assert result.returncode == 0
        assert "Codes:" in result.stdout

    def test_dict_encode_json(self):
        result = self._run("dict-encode", "hello world", "-j")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "codes" in data
        assert "text" in data
        assert "compression_ratio" in data

    def test_dict_decode_basic(self):
        result = self._run("dict-decode", "00.01")
        assert result.returncode == 0
        assert "Text:" in result.stdout

    def test_dict_decode_json(self):
        result = self._run("dict-decode", "00.01", "-j")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "text" in data
        assert "entries" in data

    def test_dict_decode_detailed(self):
        result = self._run("dict-decode", "00.01", "--detailed")
        assert result.returncode == 0
        assert "Found" in result.stdout

    def test_dict_roundtrip_basic(self):
        result = self._run("dict-roundtrip", "hello world")
        assert result.returncode == 0
        assert "Input:" in result.stdout
        assert "Codes:" in result.stdout
        assert "Output:" in result.stdout

    def test_dict_roundtrip_json(self):
        result = self._run("dict-roundtrip", "hello world", "-j")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "input" in data
        assert "codes" in data
        assert "output" in data
        assert "compression_ratio" in data

    def test_dict_info(self):
        result = self._run("dict-info")
        assert result.returncode == 0
        assert "Entries:" in result.stdout
        assert "73,256" in result.stdout
