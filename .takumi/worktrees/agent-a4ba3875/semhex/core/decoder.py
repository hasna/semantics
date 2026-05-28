"""Decoder: SemHex codes → human-readable output.

v0.1: Codebook lookup + concept expansion (no LLM).
  - Looks up each code's centroid
  - Finds nearest concept labels
  - Returns structured output with labels and neighbors

v0.2 (future): LLM expansion
  - Feed codes + labels to a small LLM
  - Generate fluent natural language
"""

from __future__ import annotations

from dataclasses import dataclass, field

from semhex.core.codebook import Codebook, CentroidEntry, load_codebook
from semhex.core.format import SemHexCode, parse_code


@dataclass
class DecodedCode:
    """Decoded information for a single SemHex code."""
    code: SemHexCode
    label: str
    l1_label: str
    examples: list[str]
    neighbors: list[str]

    def __str__(self) -> str:
        return f"{self.code} → {self.label}"


@dataclass
class DecodeResult:
    """Result of decoding a sequence of SemHex codes."""
    decoded: list[DecodedCode]

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        labels = [d.label for d in self.decoded]
        return " | ".join(labels)

    @property
    def detailed(self) -> str:
        """Multi-line detailed description."""
        lines = []
        for d in self.decoded:
            lines.append(f"{d.code} → {d.label} ({d.l1_label})")
            if d.examples:
                lines.append(f"  examples: {', '.join(d.examples[:5])}")
            if d.neighbors:
                lines.append(f"  nearby: {', '.join(d.neighbors[:5])}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Structured output for programmatic use."""
        return {
            "summary": self.summary,
            "codes": [
                {
                    "code": str(d.code),
                    "label": d.label,
                    "category": d.l1_label,
                    "examples": d.examples,
                    "neighbors": d.neighbors,
                }
                for d in self.decoded
            ],
        }

    def __str__(self) -> str:
        return self.summary


# Default singleton
_default_codebook: Codebook | None = None


def decode(
    codes: list[SemHexCode] | list[str] | str,
    codebook: Codebook | None = None,
    k_neighbors: int = 3,
) -> DecodeResult:
    """Decode SemHex codes into human-readable output.

    Args:
        codes: SemHex codes as objects, strings, or a space-separated string.
        codebook: Codebook to use (default: load v0.1).
        k_neighbors: Number of neighbor codes to include.

    Returns:
        DecodeResult with labels, examples, and neighbors.
    """
    global _default_codebook

    if codebook is None:
        if _default_codebook is None:
            _default_codebook = load_codebook("v0.1")
        codebook = _default_codebook

    # Normalize input to list of SemHexCode
    if isinstance(codes, str):
        from semhex.core.format import parse_code_sequence
        parsed = parse_code_sequence(codes)
    elif codes and isinstance(codes[0], str):
        parsed = [parse_code(c) for c in codes]
    else:
        parsed = list(codes)

    decoded = []
    for code in parsed:
        # Look up the code
        try:
            entry = codebook.lookup(code)
        except KeyError:
            decoded.append(DecodedCode(
                code=code,
                label="[unknown code]",
                l1_label="[unknown]",
                examples=[],
                neighbors=[],
            ))
            continue

        # Get L1 label
        l1_code = code.at_depth(1)
        try:
            l1_entry = codebook.lookup(l1_code)
            l1_label = l1_entry.label
        except KeyError:
            l1_label = "[unknown category]"

        # Get neighbors
        try:
            neighbor_entries = codebook.neighbors(code, k=k_neighbors)
            neighbor_labels = [f"{n.code}: {n.label}" for n in neighbor_entries]
        except (KeyError, IndexError):
            neighbor_labels = []

        decoded.append(DecodedCode(
            code=code,
            label=entry.label,
            l1_label=l1_label,
            examples=entry.examples,
            neighbors=neighbor_labels,
        ))

    return DecodeResult(decoded=decoded)
