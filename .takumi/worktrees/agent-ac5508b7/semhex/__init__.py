"""SemHex — Semantic Hexadecimal Encoding.

A universal compact discrete encoding for meaning, like hex codes for colors.

Quick start:
    # Dictionary encoding — local, no API key needed
    from semhex import dict_encode, dict_decode
    codes = dict_encode("I am frustrated with this bug")  # → "D019.1866.DBC7.13F0"
    text  = dict_decode("D019.1866.DBC7.13F0")            # → "i am frustrated with this bug"

    # Semantic RGB — 7 dimensions of meaning in 6 hex chars (requires LLM API key)
    from semhex import rgb_encode, rgb_decode
    code = rgb_encode("I'm frustrated with this bug")  # → "$2A.C4.06"
    desc = rgb_decode("$2A.C4.06")                     # → "very negative, strong, moderate | ..."
"""

__version__ = "0.1.0"

# ── Core format (always available, no ML deps) ──────────────────────────────
from semhex.core.format import SemHexCode, parse_code, format_code

# ── Dictionary encoding (local, no API key, instant) ───────────────────────
from semhex.core.dict_encoder import dict_encode
from semhex.core.dict_decoder import dict_decode, dict_decode_detailed

# ── Semantic RGB (requires LLM API key for encode) ─────────────────────────
from semhex.core.semantic_rgb import (
    SemanticColor,
    encode as rgb_encode,
    decode as rgb_decode,
    encode_detailed as rgb_encode_detailed,
    score_text as rgb_score_text,
    DOMAIN_LABELS,
    AGENT_LABELS,
    INTENT_LABELS,
)


def __getattr__(name: str):
    """Lazy imports for modules that require ML/embedding dependencies."""
    if name in ("Codebook", "load_codebook"):
        from semhex.core.codebook import Codebook, load_codebook
        return Codebook if name == "Codebook" else load_codebook
    if name in ("encode", "encode_batch"):
        from semhex.core.encoder import encode, encode_batch
        return encode if name == "encode" else encode_batch
    if name == "decode":
        from semhex.core.decoder import decode
        return decode
    if name == "distance":
        from semhex.core.distance import distance
        return distance
    if name == "blend":
        from semhex.core.blend import blend
        return blend
    if name == "SemHasher":
        from semhex.core.geohash_v2 import SemHasher
        return SemHasher
    raise AttributeError(f"module 'semhex' has no attribute {name!r}")


__all__ = [
    # Format
    "SemHexCode",
    "parse_code",
    "format_code",
    # Dictionary encoding (local, no API key)
    "dict_encode",
    "dict_decode",
    "dict_decode_detailed",
    # Semantic RGB
    "SemanticColor",
    "rgb_encode",
    "rgb_decode",
    "rgb_encode_detailed",
    "rgb_score_text",
    "DOMAIN_LABELS",
    "AGENT_LABELS",
    "INTENT_LABELS",
    # VQ codebook (lazy)
    "Codebook",
    "load_codebook",
    "encode",
    "encode_batch",
    "decode",
    "distance",
    "blend",
    # Geohash (lazy)
    "SemHasher",
]
