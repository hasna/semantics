"""Tests for SemHex code format parsing, validation, and formatting."""

import pytest
from semhex.core.format import (
    SemHexCode,
    parse_code,
    format_code,
    is_valid_code,
    code_hamming_distance,
    codes_share_prefix,
    parse_code_sequence,
)


class TestParseCode:
    def test_level1_only(self):
        code = parse_code("$8A")
        assert code.level1 == 0x8A
        assert code.level2 is None
        assert code.level3 is None
        assert code.depth == 1

    def test_level1_and_level2(self):
        code = parse_code("$8A.2100")
        assert code.level1 == 0x8A
        assert code.level2 == 0x2100
        assert code.level3 is None
        assert code.depth == 2

    def test_all_three_levels(self):
        code = parse_code("$8A.2100.03FF00")
        assert code.level1 == 0x8A
        assert code.level2 == 0x2100
        assert code.level3 == 0x03FF00
        assert code.depth == 3

    def test_lowercase_hex(self):
        code = parse_code("$8a.21ff")
        assert code.level1 == 0x8A
        assert code.level2 == 0x21FF

    def test_leading_zeros(self):
        code = parse_code("$00.0001")
        assert code.level1 == 0
        assert code.level2 == 1

    def test_max_values(self):
        code = parse_code("$FF.FFFF.FFFFFF")
        assert code.level1 == 255
        assert code.level2 == 65535
        assert code.level3 == 16777215

    def test_strips_whitespace(self):
        code = parse_code("  $8A.2100  ")
        assert code.level1 == 0x8A

    def test_invalid_no_dollar(self):
        with pytest.raises(ValueError, match="Invalid SemHex code"):
            parse_code("8A.2100")

    def test_invalid_too_short(self):
        with pytest.raises(ValueError, match="Invalid SemHex code"):
            parse_code("$8")

    def test_invalid_too_long_l2(self):
        with pytest.raises(ValueError, match="Invalid SemHex code"):
            parse_code("$8A.21001FF00")  # Too many chars after dot without separator

    def test_invalid_non_hex(self):
        with pytest.raises(ValueError, match="Invalid SemHex code"):
            parse_code("$GG")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid SemHex code"):
            parse_code("")

    def test_invalid_just_dollar(self):
        with pytest.raises(ValueError, match="Invalid SemHex code"):
            parse_code("$")


class TestFormatCode:
    def test_level1(self):
        assert format_code(SemHexCode(0x8A)) == "$8A"

    def test_level2(self):
        assert format_code(SemHexCode(0x8A, 0x2100)) == "$8A.2100"

    def test_level3(self):
        assert format_code(SemHexCode(0x8A, 0x2100, 0x03FF00)) == "$8A.2100.03FF00"

    def test_zero_padding(self):
        assert format_code(SemHexCode(0x00, 0x0001)) == "$00.0001"

    def test_roundtrip_l1(self):
        original = "$8A"
        assert format_code(parse_code(original)) == original

    def test_roundtrip_l2(self):
        original = "$8A.2100"
        assert format_code(parse_code(original)) == original

    def test_roundtrip_l3(self):
        original = "$8A.2100.03FF00"
        assert format_code(parse_code(original)) == original

    def test_str_method(self):
        code = SemHexCode(0x3A, 0xC8F0)
        assert str(code) == "$3A.C8F0"


class TestSemHexCode:
    def test_frozen(self):
        code = SemHexCode(0x8A, 0x2100)
        with pytest.raises(AttributeError):
            code.level1 = 0x00  # type: ignore

    def test_at_depth_1(self):
        code = SemHexCode(0x8A, 0x2100, 0x03FF00)
        truncated = code.at_depth(1)
        assert truncated.depth == 1
        assert truncated.level1 == 0x8A
        assert truncated.level2 is None

    def test_at_depth_2(self):
        code = SemHexCode(0x8A, 0x2100, 0x03FF00)
        truncated = code.at_depth(2)
        assert truncated.depth == 2
        assert truncated.level2 == 0x2100
        assert truncated.level3 is None

    def test_to_bytes_l1(self):
        code = SemHexCode(0x8A)
        assert code.to_bytes() == b'\x8a'

    def test_to_bytes_l2(self):
        code = SemHexCode(0x8A, 0x2100)
        assert code.to_bytes() == b'\x8a\x21\x00'

    def test_from_bytes_l1(self):
        code = SemHexCode.from_bytes(b'\x8a')
        assert code.level1 == 0x8A
        assert code.level2 is None

    def test_from_bytes_l2(self):
        code = SemHexCode.from_bytes(b'\x8a\x21\x00')
        assert code.level1 == 0x8A
        assert code.level2 == 0x2100

    def test_bytes_roundtrip(self):
        original = SemHexCode(0x8A, 0x2100)
        restored = SemHexCode.from_bytes(original.to_bytes())
        assert restored == original

    def test_equality(self):
        a = SemHexCode(0x8A, 0x2100)
        b = SemHexCode(0x8A, 0x2100)
        assert a == b

    def test_inequality(self):
        a = SemHexCode(0x8A, 0x2100)
        b = SemHexCode(0x8A, 0x2400)
        assert a != b

    def test_hash(self):
        a = SemHexCode(0x8A, 0x2100)
        b = SemHexCode(0x8A, 0x2100)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


class TestIsValidCode:
    def test_valid_codes(self):
        assert is_valid_code("$8A")
        assert is_valid_code("$8A.2100")
        assert is_valid_code("$8A.2100.03FF00")
        assert is_valid_code("$00.0000")
        assert is_valid_code("$FF.FFFF")

    def test_invalid_codes(self):
        assert not is_valid_code("")
        assert not is_valid_code("hello")
        assert not is_valid_code("8A.2100")
        assert not is_valid_code("$G0")
        assert not is_valid_code("$8")
        assert not is_valid_code("$8A.")


class TestHammingDistance:
    def test_identical(self):
        a = parse_code("$8A.2100")
        assert code_hamming_distance(a, a) == 0

    def test_one_digit_different_l1(self):
        a = parse_code("$8A")
        b = parse_code("$8B")
        assert code_hamming_distance(a, b) == 1

    def test_all_different_l1(self):
        a = parse_code("$00")
        b = parse_code("$FF")
        assert code_hamming_distance(a, b) == 2

    def test_l2_difference(self):
        a = parse_code("$8A.2100")
        b = parse_code("$8A.2400")
        assert code_hamming_distance(a, b) == 1  # L1 same, L2 differs in 1 digit

    def test_both_levels_different(self):
        a = parse_code("$00.0000")
        b = parse_code("$FF.FFFF")
        assert code_hamming_distance(a, b) == 6  # 2 L1 digits + 4 L2 digits


class TestSharePrefix:
    def test_same_prefix(self):
        a = parse_code("$8A.2100")
        b = parse_code("$8A.2400")
        assert codes_share_prefix(a, b)

    def test_different_prefix(self):
        a = parse_code("$8A.2100")
        b = parse_code("$3A.2100")
        assert not codes_share_prefix(a, b)


class TestParseCodeSequence:
    def test_single_code(self):
        codes = parse_code_sequence("$8A.2100")
        assert len(codes) == 1
        assert codes[0].level1 == 0x8A

    def test_multiple_codes(self):
        codes = parse_code_sequence("$3A.C8F0 $72.B1A0 $4F.2031")
        assert len(codes) == 3
        assert codes[0].level1 == 0x3A
        assert codes[1].level1 == 0x72
        assert codes[2].level1 == 0x4F

    def test_empty_string(self):
        codes = parse_code_sequence("")
        assert codes == []

    def test_mixed_depths(self):
        codes = parse_code_sequence("$8A $8A.2100 $8A.2100.03FF00")
        assert len(codes) == 3
        assert codes[0].depth == 1
        assert codes[1].depth == 2
        assert codes[2].depth == 3
