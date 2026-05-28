"""SemHex code format: parsing, validation, formatting.

Code format: $XX.XXXX.XXXXXX
  Level 1: 2 hex chars (1 byte)  = 256 coarse categories
  Level 2: 4 hex chars (2 bytes) = 65,536 specific meanings per L1
  Level 3: 6 hex chars (3 bytes) = fine-grained (future)

Examples:
  $8A           — Level 1 only (emotion category)
  $8A.2100      — Level 1 + Level 2 (anger)
  $8A.2100.03FF — Level 1 + Level 2 + Level 3 (specific anger variant)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Valid SemHex code pattern
_CODE_PATTERN = re.compile(
    r'^\$([0-9A-Fa-f]{2})'           # L1: $XX
    r'(?:\.([0-9A-Fa-f]{4}))?'       # L2: .XXXX (optional)
    r'(?:\.([0-9A-Fa-f]{4,6}))?$'    # L3: .XXXX-XXXXXX (optional)
)

_HEX_CHARS = set('0123456789ABCDEFabcdef')


@dataclass(frozen=True)
class SemHexCode:
    """A parsed SemHex code with up to 3 levels."""
    level1: int          # 0-255
    level2: Optional[int] = None  # 0-65535
    level3: Optional[int] = None  # 0-16777215

    @property
    def depth(self) -> int:
        if self.level3 is not None:
            return 3
        if self.level2 is not None:
            return 2
        return 1

    @property
    def l1_hex(self) -> str:
        return f"{self.level1:02X}"

    @property
    def l2_hex(self) -> Optional[str]:
        if self.level2 is None:
            return None
        return f"{self.level2:04X}"

    @property
    def l3_hex(self) -> Optional[str]:
        if self.level3 is None:
            return None
        return f"{self.level3:06X}"

    def at_depth(self, depth: int) -> SemHexCode:
        """Return this code truncated to the given depth."""
        if depth <= 1:
            return SemHexCode(self.level1)
        if depth <= 2:
            return SemHexCode(self.level1, self.level2)
        return self

    def __str__(self) -> str:
        return format_code(self)

    def __repr__(self) -> str:
        return f"SemHexCode({str(self)})"

    def to_bytes(self) -> bytes:
        """Serialize to compact bytes representation."""
        result = self.level1.to_bytes(1, 'big')
        if self.level2 is not None:
            result += self.level2.to_bytes(2, 'big')
        if self.level3 is not None:
            result += self.level3.to_bytes(3, 'big')
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> SemHexCode:
        """Deserialize from bytes."""
        if len(data) < 1:
            raise ValueError("Empty bytes")
        l1 = data[0]
        l2 = int.from_bytes(data[1:3], 'big') if len(data) >= 3 else None
        l3 = int.from_bytes(data[3:6], 'big') if len(data) >= 6 else None
        return cls(l1, l2, l3)


def parse_code(code_str: str) -> SemHexCode:
    """Parse a SemHex code string like '$8A.2100' into a SemHexCode."""
    code_str = code_str.strip()
    match = _CODE_PATTERN.match(code_str)
    if not match:
        raise ValueError(f"Invalid SemHex code: {code_str!r}. "
                         f"Expected format: $XX, $XX.XXXX, or $XX.XXXX.XXXXXX")

    l1 = int(match.group(1), 16)
    l2 = int(match.group(2), 16) if match.group(2) else None
    l3 = int(match.group(3), 16) if match.group(3) else None

    return SemHexCode(l1, l2, l3)


def format_code(code: SemHexCode) -> str:
    """Format a SemHexCode into a string like '$8A.2100'."""
    result = f"${code.l1_hex}"
    if code.level2 is not None:
        result += f".{code.l2_hex}"
    if code.level3 is not None:
        result += f".{code.l3_hex}"
    return result


def is_valid_code(code_str: str) -> bool:
    """Check if a string is a valid SemHex code."""
    try:
        parse_code(code_str)
        return True
    except ValueError:
        return False


def code_hamming_distance(a: SemHexCode, b: SemHexCode) -> int:
    """Compute Hamming distance between two codes at their shared depth.

    Returns the number of differing hex digits at the deepest shared level.
    """
    dist = 0
    # L1: compare 2 hex digits
    a_hex = f"{a.level1:02X}"
    b_hex = f"{b.level1:02X}"
    dist += sum(c1 != c2 for c1, c2 in zip(a_hex, b_hex))

    # L2: compare 4 hex digits (if both have L2)
    if a.level2 is not None and b.level2 is not None:
        a_hex = f"{a.level2:04X}"
        b_hex = f"{b.level2:04X}"
        dist += sum(c1 != c2 for c1, c2 in zip(a_hex, b_hex))

    # L3: compare 6 hex digits (if both have L3)
    if a.level3 is not None and b.level3 is not None:
        a_hex = f"{a.level3:06X}"
        b_hex = f"{b.level3:06X}"
        dist += sum(c1 != c2 for c1, c2 in zip(a_hex, b_hex))

    return dist


def codes_share_prefix(a: SemHexCode, b: SemHexCode) -> bool:
    """Check if two codes share the same Level 1 prefix (same coarse category)."""
    return a.level1 == b.level1


def parse_code_sequence(text: str) -> list[SemHexCode]:
    """Parse a space-separated sequence of SemHex codes.

    Example: "$3A.C8F0 $72.B1A0" → [SemHexCode(...), SemHexCode(...)]
    """
    tokens = text.strip().split()
    return [parse_code(t) for t in tokens if t.startswith('$')]
