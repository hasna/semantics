"""Semantic distance between SemHex codes.

Uses cosine distance in the codebook's embedding space.
"""

from __future__ import annotations

import numpy as np

from semhex.core.codebook import Codebook, load_codebook
from semhex.core.format import SemHexCode, parse_code

_default_codebook: Codebook | None = None


def distance(
    code_a: SemHexCode | str,
    code_b: SemHexCode | str,
    codebook: Codebook | None = None,
) -> float:
    """Compute semantic distance between two SemHex codes.

    Uses cosine distance in embedding space:
      0.0 = identical meaning
      1.0 = orthogonal (unrelated)
      2.0 = opposite meaning

    Args:
        code_a: First SemHex code (object or string like "$8A.2100").
        code_b: Second SemHex code.
        codebook: Codebook to use (default: load v0.1).

    Returns:
        Cosine distance as a float in [0.0, 2.0].
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

    cos_sim = float(np.dot(entry_a.vector, entry_b.vector))
    # Clamp to [-1, 1] for numerical safety
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def similarity(
    code_a: SemHexCode | str,
    code_b: SemHexCode | str,
    codebook: Codebook | None = None,
) -> float:
    """Compute semantic similarity between two SemHex codes.

    Returns cosine similarity:
      1.0 = identical meaning
      0.0 = unrelated
     -1.0 = opposite meaning

    Args:
        code_a: First SemHex code.
        code_b: Second SemHex code.
        codebook: Codebook to use.
    """
    return 1.0 - distance(code_a, code_b, codebook=codebook)
