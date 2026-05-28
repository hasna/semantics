"""SemHex MCP Server — expose all encoding/decoding tools as MCP tools.

Tools (no API key needed):
- semhex_dict_encode:   Text → dot-separated hex codes (local dictionary, instant)
- semhex_dict_decode:   Codes → text (local dictionary, instant)
- semhex_rgb_decode:    $XX.XX.XX → 7-dimension breakdown (bit unpacking, instant)
- semhex_distance:      Distance between two VQ codes
- semhex_blend:         VQ code arithmetic
- semhex_codebook_info: Codebook statistics

Tools (require LLM API key):
- semhex_rgb_encode:    Text → $XX.XX.XX Semantic RGB code
- semhex_compress:      Text → compact LLM codes
- semhex_decompress:    Codes → natural language
- semhex_encode:        Text → VQ codebook code
- semhex_decode:        VQ code → meaning labels
- semhex_inspect:       VQ code details + neighbors
- semhex_roundtrip:     VQ encode → decode roundtrip

Tools (require OPENAI_API_KEY):
- semhex_hash:          Text → semantic geohash address

Run: python -m semhex.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "semhex",
    instructions="SemHex — Semantic Hexadecimal Encoding. Encode meaning into compact hex codes, decode them back, measure semantic distance, and perform code arithmetic.",
)

# Lazy-loaded singletons
_codebook = None
_provider = None


def _get_codebook():
    global _codebook
    if _codebook is None:
        from semhex.core.codebook import load_codebook

        try:
            _codebook = load_codebook("v0.1")
        except FileNotFoundError:
            # Keep MCP behavior aligned with CLI fallback in dev checkouts.
            _codebook = load_codebook("test")
    return _codebook


def _get_provider():
    global _provider
    if _provider is None:
        from semhex.embeddings import get_provider
        _provider = get_provider("auto")
    return _provider


@mcp.tool()
def semhex_encode(text: str, depth: int = 2) -> dict:
    """Encode text into SemHex codes. Compression happens at meaning level — a sentence becomes a few codes, not one per word.

    Args:
        text: Input text to encode.
        depth: Code depth (1=coarse 256 categories, 2=fine 65K meanings).
    """
    from semhex.core.encoder import encode

    result = encode(text, depth=depth, codebook=_get_codebook(), provider=_get_provider())
    return {
        "codes": result.code_strings,
        "chunks": result.chunks,
        "distances": [round(d, 4) for d in result.distances],
        "compression_ratio": round(result.compression_ratio, 1),
    }


@mcp.tool()
def semhex_decode(codes: str, k_neighbors: int = 3) -> dict:
    """Decode SemHex codes into human-readable meaning labels.

    Args:
        codes: Space-separated SemHex codes (e.g., "$3A.C8F0 $72.B1A0").
        k_neighbors: Number of neighbor concepts to include.
    """
    from semhex.core.decoder import decode

    result = decode(codes, codebook=_get_codebook(), k_neighbors=k_neighbors)
    return result.to_dict()


@mcp.tool()
def semhex_distance(code_a: str, code_b: str) -> dict:
    """Compute semantic distance between two SemHex codes.

    Args:
        code_a: First SemHex code (e.g., "$8A.2100").
        code_b: Second SemHex code (e.g., "$8A.2400").
    """
    from semhex.core.distance import distance, similarity

    cb = _get_codebook()
    d = distance(code_a, code_b, codebook=cb)
    s = similarity(code_a, code_b, codebook=cb)
    return {
        "code_a": code_a,
        "code_b": code_b,
        "distance": round(d, 4),
        "similarity": round(s, 4),
        "interpretation": "identical" if d < 0.05 else "very close" if d < 0.3 else "related" if d < 0.7 else "different" if d < 1.2 else "very different",
    }


@mcp.tool()
def semhex_blend(code_a: str, code_b: str, weight: float = 0.5) -> dict:
    """Blend two SemHex codes via semantic arithmetic. Like color mixing but for meaning.

    Args:
        code_a: First SemHex code.
        code_b: Second SemHex code.
        weight: Weight for first code (0.0-1.0). Default 0.5 = equal blend.
    """
    from semhex.core.blend import blend
    from semhex.core.decoder import decode

    cb = _get_codebook()
    result = blend(code_a, code_b, weight=weight, codebook=cb)

    dec_a = decode([code_a], codebook=cb)
    dec_b = decode([code_b], codebook=cb)
    dec_r = decode([result], codebook=cb)

    return {
        "input_a": {"code": code_a, "label": dec_a.decoded[0].label if dec_a.decoded else "?"},
        "input_b": {"code": code_b, "label": dec_b.decoded[0].label if dec_b.decoded else "?"},
        "result": {"code": str(result), "label": dec_r.decoded[0].label if dec_r.decoded else "?"},
        "weight": weight,
    }


@mcp.tool()
def semhex_inspect(code: str, k_neighbors: int = 5) -> dict:
    """Inspect a SemHex code — show its meaning, category, examples, and neighbors.

    Args:
        code: SemHex code to inspect (e.g., "$8A.2100").
        k_neighbors: Number of neighbors to show.
    """
    from semhex.core.decoder import decode

    result = decode([code], codebook=_get_codebook(), k_neighbors=k_neighbors)
    if not result.decoded:
        return {"error": f"Code {code} not found in codebook"}

    d = result.decoded[0]
    return {
        "code": str(d.code),
        "label": d.label,
        "category": d.l1_label,
        "depth": d.code.depth,
        "examples": d.examples,
        "neighbors": d.neighbors,
    }


@mcp.tool()
def semhex_roundtrip(text: str, depth: int = 2) -> dict:
    """Encode text to SemHex codes, then decode back — shows both sides and compression ratio.

    Args:
        text: Input text to roundtrip.
        depth: Code depth (1=coarse, 2=fine).
    """
    from semhex.core.encoder import encode
    from semhex.core.decoder import decode

    cb = _get_codebook()
    provider = _get_provider()

    enc = encode(text, depth=depth, codebook=cb, provider=provider)
    dec = decode(enc.codes, codebook=cb)

    return {
        "input": text,
        "codes": enc.code_strings,
        "decoded_summary": dec.summary,
        "compression_ratio": round(enc.compression_ratio, 1),
        "n_words": sum(len(c.split()) for c in enc.chunks),
        "n_codes": len(enc.codes),
        "details": [
            {
                "chunk": chunk,
                "code": code_str,
                "label": d.label,
                "distance": round(dist, 4),
            }
            for chunk, code_str, d, dist in zip(enc.chunks, enc.code_strings, dec.decoded, enc.distances)
        ],
    }


@mcp.tool()
def semhex_codebook_info() -> dict:
    """Show codebook statistics — version, dimensions, cluster counts."""
    cb = _get_codebook()
    return {
        "version": cb.version,
        "dimensions": cb.dimensions,
        "l1_clusters": cb.n_level1,
        "l2_clusters": cb.n_level2,
        "total_codes": cb.n_level1 + cb.n_level2,
    }


@mcp.tool()
def semhex_compress(text: str, quality: int = 2, provider: str = "cerebras") -> dict:
    """Compress text into compact SemHex codes using LLM. Like JPEG for meaning.

    Args:
        text: Text to compress.
        quality: 1 (max compression ~10x) to 4 (near-lossless ~2x).
        provider: "cerebras" (fast, free) or "openai".
    """
    from semhex.core.codec import compress
    codes = compress(text, quality=quality, provider=provider)
    return {
        "codes": codes,
        "input_chars": len(text),
        "code_chars": len(codes),
        "compression_ratio": round(len(text) / max(len(codes), 1), 1),
        "quality": quality,
    }


@mcp.tool()
def semhex_decompress(codes: str, provider: str = "cerebras") -> dict:
    """Decompress SemHex codes back into natural language.

    Args:
        codes: Dot-separated alphanumeric codes (e.g., "FRU.BUG.HLP").
        provider: "cerebras" or "openai".
    """
    from semhex.core.codec import decompress
    text = decompress(codes, provider=provider)
    return {
        "codes": codes,
        "text": text,
    }


@mcp.tool()
def semhex_codec_roundtrip(text: str, quality: int = 2, provider: str = "cerebras") -> dict:
    """Compress then decompress — shows both sides with similarity score.

    Args:
        text: Text to roundtrip.
        quality: 1-4.
        provider: "cerebras" or "openai".
    """
    from semhex.core.codec import roundtrip
    return roundtrip(text, quality=quality, provider=provider)


@mcp.tool()
def semhex_scaling_info() -> dict:
    """Show scaling law results from experiments."""
    import json
    from pathlib import Path

    results = {}
    vq_path = Path("evaluation/results/scaling_results.json")
    if vq_path.exists():
        results["single_vq"] = json.loads(vq_path.read_text())

    rvq_path = Path("evaluation/results/rvq_scaling_results.json")
    if rvq_path.exists():
        results["rvq"] = json.loads(rvq_path.read_text())

    codec_path = Path("evaluation/results/codec_eval.json")
    if codec_path.exists():
        results["codec"] = json.loads(codec_path.read_text())

    results["summary"] = {
        "single_vq_scaling": "error = 0.755 / K^0.09 (slow, impractical)",
        "rvq_scaling": "error = 0.467 × 0.987^L (1.3% per level)",
        "codec_baseline": "6-10x compression, 0.39-0.43 similarity (before fine-tuning)",
        "conclusion": "Pre-computed embedding VQ scales poorly. LLM-native codec with fine-tuning is the path.",
    }

    return results


@mcp.tool()
def semhex_dict_encode(text: str) -> dict:
    """Encode text using the local dictionary — instant, no API key needed.

    Uses a 73K-word dictionary with phrase merging for compression.
    Returns dot-separated hex codes. Faster than LLM codec but less semantic.

    Args:
        text: Input text to encode.
    """
    from semhex.core.dict_encoder import dict_encode
    codes = dict_encode(text)
    ratio = len(text) / max(len(codes), 1)
    return {
        "text": text,
        "codes": codes,
        "compression_ratio": round(ratio, 2),
        "input_chars": len(text),
        "code_chars": len(codes),
    }


@mcp.tool()
def semhex_dict_decode(codes: str, detailed: bool = False) -> dict:
    """Decode dictionary codes back to text — instant, no API key needed.

    Args:
        codes: Dot-separated hex codes (e.g., "D019.1866.DBC7.13F0").
        detailed: If True, include per-code breakdown with found/unknown status.
    """
    from semhex.core.dict_decoder import dict_decode, dict_decode_detailed
    if detailed:
        return dict_decode_detailed(codes)
    return {
        "codes": codes,
        "text": dict_decode(codes),
    }


@mcp.tool()
def semhex_rgb_encode(text: str) -> dict:
    """Encode text as Semantic RGB — 7 dimensions of meaning in 6 hex chars ($XX.XX.XX).

    Like #RRGGBB for colors, $XX.XX.XX captures:
    evaluation (positive/negative), potency (strong/weak), activity (active/passive),
    agent (self/other/abstract), domain (technology/emotion/science/…),
    intent (express/ask/command), specificity (vague/specific).

    Requires CEREBRAS_API_KEY or OPENAI_API_KEY.

    Args:
        text: Input text to encode.
    """
    from semhex.core.semantic_rgb import encode_detailed
    return encode_detailed(text)


@mcp.tool()
def semhex_rgb_decode(code: str) -> dict:
    """Decode a Semantic RGB code ($XX.XX.XX) to dimension values and description.

    No API key needed — pure bit unpacking.

    Args:
        code: Semantic RGB code like "$2A.C4.06".
    """
    from semhex.core.semantic_rgb import SemanticColor, DOMAIN_LABELS, AGENT_LABELS, INTENT_LABELS
    color = SemanticColor.from_hex(code)
    return {
        "code": code,
        "description": color.describe(),
        "dimensions": {
            "evaluation": color.evaluation,
            "potency": color.potency,
            "activity": color.activity,
            "agent": color.agent,
            "agent_label": AGENT_LABELS.get(color.agent, "?"),
            "domain": color.domain,
            "domain_label": DOMAIN_LABELS.get(color.domain, "?"),
            "intent": color.intent,
            "intent_label": INTENT_LABELS.get(color.intent, "?"),
            "specificity": color.specificity,
        },
    }


@mcp.tool()
def semhex_hash(text: str, bits: int = 4) -> dict:
    """Encode text as a semantic geohash — a mathematical address in embedding space.

    Uses PCA + quantization on OpenAI Matryoshka embeddings.
    Similar meanings produce nearby addresses.
    Requires OPENAI_API_KEY.

    Args:
        text: Input text.
        bits: Bits per dimension (2=compact 32 hex chars, 4=precise 64 hex chars).
    """
    import numpy as np
    from openai import OpenAI
    from semhex.core.auth import load_api_key as _load_api_key
    from semhex.core.geohash_v2 import SemHasher

    hasher = SemHasher(n_dims=64, bits_per_dim=bits)
    state_name = f"matryoshka_64d_{bits}b"
    hasher.load(state_name)

    api_key = _load_api_key("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(input=[text], model="text-embedding-3-small", dimensions=64)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    code = hasher.encode(vec)

    return {
        "text": text,
        "code": code,
        "bits": hasher.total_bits,
        "hex_chars": hasher.hex_length,
    }


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
