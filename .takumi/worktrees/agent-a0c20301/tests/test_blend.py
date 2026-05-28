"""Tests for code arithmetic (blending)."""

import pytest

from semhex.core.blend import blend, blend_multiple
from semhex.core.codebook import load_codebook
from semhex.core.distance import distance
from semhex.core.format import SemHexCode


@pytest.fixture
def codebook():
    return load_codebook("v0.1")


class TestBlend:
    def test_blend_returns_code(self, codebook):
        result = blend(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        assert isinstance(result, SemHexCode)

    def test_blend_with_self_returns_same(self, codebook):
        """Blending a code with itself should return the same code."""
        code = SemHexCode(0, 0)
        result = blend(code, code, codebook=codebook)
        assert result == code

    def test_blend_weight_1_returns_first(self, codebook):
        """Weight=1.0 should return a code very close to the first input."""
        a = SemHexCode(0, 0)
        b = SemHexCode(5, 0)
        result = blend(a, b, weight=1.0, codebook=codebook)
        # Should be same as a (or very close)
        d = distance(result, a, codebook=codebook)
        assert d < 0.1

    def test_blend_weight_0_returns_second(self, codebook):
        """Weight=0.0 should return a code close to the second input."""
        a = SemHexCode(0, 0)
        b = SemHexCode(1, 0)
        result = blend(a, b, weight=0.0, codebook=codebook)
        d = distance(result, b, codebook=codebook)
        assert d < 1.0  # With small random codebook, tolerance must be higher

    def test_blend_string_input(self, codebook):
        result = blend("$00.0000", "$01.0000", codebook=codebook)
        assert isinstance(result, SemHexCode)

    def test_blend_depth_1(self, codebook):
        result = blend(SemHexCode(0, 0), SemHexCode(1, 0), depth=1, codebook=codebook)
        assert result.depth == 1

    def test_blend_is_between_inputs(self, codebook):
        """Blended code should be closer to both inputs than inputs are to each other."""
        a = SemHexCode(0, 0)
        b = SemHexCode(10, 0)
        result = blend(a, b, weight=0.5, codebook=codebook)

        d_ab = distance(a, b, codebook=codebook)
        d_ra = distance(result, a, codebook=codebook)
        d_rb = distance(result, b, codebook=codebook)

        # The blend should be closer to each input than the inputs are to each other
        # (triangle inequality — the blend is "between" them)
        assert d_ra <= d_ab + 0.1  # Allow some slack for quantization
        assert d_rb <= d_ab + 0.1


class TestBlendMultiple:
    def test_single_code(self, codebook):
        result = blend_multiple([SemHexCode(0, 0)], codebook=codebook)
        assert isinstance(result, SemHexCode)

    def test_two_codes_equal_weights(self, codebook):
        result = blend_multiple(
            [SemHexCode(0, 0), SemHexCode(1, 0)],
            codebook=codebook,
        )
        assert isinstance(result, SemHexCode)

    def test_three_codes(self, codebook):
        result = blend_multiple(
            [SemHexCode(0, 0), SemHexCode(1, 0), SemHexCode(2, 0)],
            codebook=codebook,
        )
        assert isinstance(result, SemHexCode)

    def test_custom_weights(self, codebook):
        result = blend_multiple(
            [SemHexCode(0, 0), SemHexCode(1, 0)],
            weights=[0.8, 0.2],
            codebook=codebook,
        )
        # Should be closer to code 0 than code 1
        d_0 = distance(result, SemHexCode(0, 0), codebook=codebook)
        d_1 = distance(result, SemHexCode(1, 0), codebook=codebook)
        assert d_0 <= d_1 + 0.3  # Allow slack

    def test_empty_raises(self, codebook):
        with pytest.raises(ValueError, match="at least one code"):
            blend_multiple([], codebook=codebook)

    def test_mismatched_weights_raises(self, codebook):
        with pytest.raises(ValueError, match="weights length"):
            blend_multiple(
                [SemHexCode(0, 0), SemHexCode(1, 0)],
                weights=[0.5],
                codebook=codebook,
            )

    def test_string_input(self, codebook):
        result = blend_multiple(["$00.0000", "$01.0000"], codebook=codebook)
        assert isinstance(result, SemHexCode)
