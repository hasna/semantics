"""Code arithmetic: blend two SemHex codes to produce a new one.

Performs weighted vector averaging in embedding space,
then finds the nearest codebook entry to the blended vector.

Example: blend(anger, small_degree) → annoyance
"""

from __future__ import annotations

import numpy as np

from semhex.core.codebook import Codebook, load_codebook
from semhex.core.format import SemHexCode, parse_code

_default_codebook: Codebook | None = None


def blend(
    code_a: SemHexCode | str,
    code_b: SemHexCode | str,
    weight: float = 0.5,
    depth: int = 2,
    codebook: Codebook | None = None,
) -> SemHexCode:
    """Blend two SemHex codes via weighted vector averaging.

    The result is the nearest codebook entry to the blended vector.
    This is code arithmetic: anger + small = annoyance.

    Args:
        code_a: First SemHex code.
        code_b: Second SemHex code.
        weight: Weight for code_a (code_b gets 1-weight). Default 0.5 = equal blend.
        depth: Depth of output code (1=coarse, 2=fine).
        codebook: Codebook to use.

    Returns:
        The nearest SemHex code to the blended vector.
    """
    global _default_codebook
    if codebook is None:
        if _default_codebook is None:
            _default_codebook = load_codebook("v0.1")
        codebook = _default_codebook

    if isinstance(code_a, str):
        code_a = parse_code(code_a)
    if isinstance(code_b, str):
        code_b = parse_code(code_b)

    entry_a = codebook.lookup(code_a)
    entry_b = codebook.lookup(code_b)

    # Weighted average
    blended = weight * entry_a.vector + (1.0 - weight) * entry_b.vector

    # Normalize
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    # Find nearest codebook entry
    result_code, _ = codebook.nearest(blended, depth=depth)
    return result_code


def blend_multiple(
    codes: list[SemHexCode | str],
    weights: list[float] | None = None,
    depth: int = 2,
    codebook: Codebook | None = None,
) -> SemHexCode:
    """Blend multiple SemHex codes via weighted vector averaging.

    Args:
        codes: List of SemHex codes to blend.
        weights: Optional weights (must sum to ~1.0). Default: equal weights.
        depth: Depth of output code.
        codebook: Codebook to use.

    Returns:
        The nearest SemHex code to the blended vector.
    """
    global _default_codebook
    if codebook is None:
        if _default_codebook is None:
            _default_codebook = load_codebook("v0.1")
        codebook = _default_codebook

    if not codes:
        raise ValueError("Need at least one code to blend")

    parsed = []
    for c in codes:
        if isinstance(c, str):
            parsed.append(parse_code(c))
        else:
            parsed.append(c)

    if weights is None:
        weights = [1.0 / len(parsed)] * len(parsed)

    if len(weights) != len(parsed):
        raise ValueError(f"weights length ({len(weights)}) must match codes length ({len(parsed)})")

    vectors = [codebook.lookup(c).vector for c in parsed]
    blended = sum(w * v for w, v in zip(weights, vectors))

    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    result_code, _ = codebook.nearest(blended, depth=depth)
    return result_code
