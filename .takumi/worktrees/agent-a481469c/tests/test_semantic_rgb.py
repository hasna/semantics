"""Tests for SemanticColor bit-packing, decode, and CLI commands (no LLM needed)."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semhex.core.semantic_rgb import (
    SemanticColor,
    DOMAIN_LABELS,
    AGENT_LABELS,
    INTENT_LABELS,
    decode,
)

PROJECT_ROOT = str(Path(__file__).parent.parent)


class TestSemanticColorBitPacking:
    """to_hex / from_hex must be a perfect round-trip for all valid values."""

    def test_basic_roundtrip(self):
        c = SemanticColor(evaluation=4, potency=5, activity=3, agent=0, domain=8, intent=0, specificity=6)
        assert SemanticColor.from_hex(c.to_hex()) == c

    def test_all_zeros(self):
        c = SemanticColor(0, 0, 0, 0, 0, 0, 0)
        assert c.to_hex() == "$00.00.00"
        assert SemanticColor.from_hex("$00.00.00") == c

    def test_max_values(self):
        c = SemanticColor(15, 7, 7, 7, 15, 7, 15)
        back = SemanticColor.from_hex(c.to_hex())
        assert back == c

    def test_edge_values_exhaustive(self):
        """All combinations of min/max per dimension must round-trip."""
        for ev in (0, 15):
            for po in (0, 7):
                for ac in (0, 7):
                    for ag in (0, 7):
                        for do in (0, 15):
                            for it in (0, 7):
                                for sp in (0, 15):
                                    orig = SemanticColor(ev, po, ac, ag, do, it, sp)
                                    back = SemanticColor.from_hex(orig.to_hex())
                                    assert orig == back, f"Mismatch at {orig}"

    def test_mid_values(self):
        c = SemanticColor(8, 4, 4, 3, 7, 3, 8)
        assert SemanticColor.from_hex(c.to_hex()) == c

    def test_hex_format(self):
        code = SemanticColor(0, 0, 0, 0, 0, 0, 0).to_hex()
        assert code.startswith("$")
        assert code.count(".") == 2
        # $XX.XX.XX = 9 chars
        assert len(code) == 9

    def test_from_hex_strips_dollar(self):
        c1 = SemanticColor.from_hex("$00.00.00")
        c2 = SemanticColor.from_hex("00.00.00")
        assert c1 == c2

    def test_from_hex_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            SemanticColor.from_hex("$ZZ.ZZ.ZZ")

    def test_from_hex_wrong_length_raises(self):
        with pytest.raises((ValueError, Exception)):
            SemanticColor.from_hex("$12.34")


class TestSemanticColorDescribe:
    def test_describe_returns_string(self):
        c = SemanticColor(4, 5, 3, 0, 8, 0, 6)
        desc = c.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_describe_contains_agent_label(self):
        c = SemanticColor(8, 4, 4, 0, 0, 0, 8)  # agent=0 → I/self
        desc = c.describe()
        assert "I/self" in desc

    def test_describe_contains_domain_label(self):
        c = SemanticColor(8, 4, 4, 0, 8, 0, 8)  # domain=8 → technology
        desc = c.describe()
        assert "technology" in desc

    def test_describe_very_negative(self):
        c = SemanticColor(0, 4, 4, 0, 0, 0, 8)
        assert "very negative" in c.describe()

    def test_describe_very_positive(self):
        c = SemanticColor(15, 4, 4, 0, 0, 0, 8)
        assert "very positive" in c.describe()

    def test_describe_neutral(self):
        c = SemanticColor(8, 4, 4, 0, 0, 0, 8)
        assert "neutral" in c.describe()


class TestDecodeFunction:
    def test_decode_returns_string(self):
        result = decode("$00.00.00")
        assert isinstance(result, str)

    def test_decode_matches_describe(self):
        c = SemanticColor(4, 5, 3, 0, 8, 0, 6)
        assert decode(c.to_hex()) == c.describe()


class TestDomainLabels:
    def test_all_domain_labels_exist(self):
        for i in range(16):
            assert i in DOMAIN_LABELS, f"Missing domain label {i}"

    def test_technology_is_8(self):
        assert DOMAIN_LABELS[8] == "technology"

    def test_emotion_is_0(self):
        assert DOMAIN_LABELS[0] == "emotion"


class TestScoreTextMocked:
    """Test score_text with a mocked LLM response."""

    @patch("semhex.core.semantic_rgb._get_cerebras")
    def test_score_text_parses_response(self, mock_get):
        from semhex.core.semantic_rgb import score_text
        client = MagicMock()
        fake_json = '{"evaluation": 3, "potency": 6, "activity": 5, "agent": 0, "domain": 8, "intent": 0, "specificity": 12}'
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=fake_json))]
        )
        mock_get.return_value = client

        result = score_text("I'm frustrated with this bug", provider="cerebras")
        assert result.evaluation == 3
        assert result.potency == 6
        assert result.activity == 5
        assert result.agent == 0
        assert result.domain == 8
        assert result.specificity == 12

    @patch("semhex.core.semantic_rgb._get_cerebras")
    def test_score_text_clamps_values(self, mock_get):
        from semhex.core.semantic_rgb import score_text
        client = MagicMock()
        # Out-of-range values should be clamped
        fake_json = '{"evaluation": 99, "potency": -5, "activity": 4, "agent": 0, "domain": 0, "intent": 0, "specificity": 0}'
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=fake_json))]
        )
        mock_get.return_value = client

        result = score_text("test", provider="cerebras")
        assert result.evaluation == 15  # clamped to max
        assert result.potency == 0      # clamped to min

    @patch("semhex.core.semantic_rgb._get_cerebras")
    def test_encode_detailed_structure(self, mock_get):
        from semhex.core.semantic_rgb import encode_detailed
        client = MagicMock()
        fake_json = '{"evaluation": 8, "potency": 4, "activity": 4, "agent": 0, "domain": 0, "intent": 0, "specificity": 8}'
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=fake_json))]
        )
        mock_get.return_value = client

        result = encode_detailed("hello world")
        assert "input" in result
        assert "code" in result
        assert "description" in result
        assert "dimensions" in result
        assert "compression_ratio" in result
        assert result["code"].startswith("$")
        assert len(result["code"]) == 9


class TestRGBCLI:
    """Integration tests for rgb-encode/rgb-decode CLI commands."""

    def _run(self, *args):
        return subprocess.run(
            ["python3", "-m", "semhex", *args],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )

    def test_rgb_decode_basic(self):
        result = self._run("rgb-decode", "$00.00.00")
        assert result.returncode == 0
        assert "Summary:" in result.stdout

    def test_rgb_decode_json(self):
        result = self._run("rgb-decode", "$00.00.00", "-j")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "code" in data
        assert "description" in data
        assert "dimensions" in data
        assert data["dimensions"]["evaluation"] == 0

    def test_rgb_decode_all_dims_present(self):
        result = self._run("rgb-decode", "$4A.C4.06", "-j")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        dims = data["dimensions"]
        for key in ("evaluation", "potency", "activity", "agent", "domain", "intent", "specificity"):
            assert key in dims
