"""SemHex Codec: compress text → compact codes, decompress codes → text.

The LLM IS the encoder and decoder. Instead of VQ on pre-computed embeddings
(which scales poorly: α=0.09), we use the LLM's native understanding of meaning
to compress and decompress.

The codebook still exists as a REFERENCE — it maps codes to human-readable labels
and provides the "address space" of meaning. But the LLM decides which codes to
assign, not a nearest-neighbor lookup.

Quality levels control compression ratio:
  quality=1: ~10x compression (broad strokes)
  quality=2: ~5x compression (captures intent + topic)
  quality=3: ~3x compression (preserves most nuance)
  quality=4: ~2x compression (near-lossless)
"""

from __future__ import annotations

import json
import re
import time
from openai import OpenAI

from semhex.core.auth import load_api_key as _load_api_key

_cerebras_client: OpenAI | None = None
_openai_client: OpenAI | None = None


def _get_cerebras() -> OpenAI:
    global _cerebras_client
    if _cerebras_client is None:
        api_key = _load_api_key("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not found")
        _cerebras_client = OpenAI(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return _cerebras_client


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = _load_api_key("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


QUALITY_INSTRUCTIONS = {
    1: "Compress into 1-2 codes. Capture only the broad category/intent. Maximum compression.",
    2: "Compress into 2-4 codes. Capture intent, topic, and basic emotion/tone.",
    3: "Compress into 4-8 codes. Preserve most nuance including specifics and context.",
    4: "Compress into 8-16 codes. Near-lossless. Capture every detail, qualifier, and nuance.",
}


def compress(
    text: str,
    quality: int = 2,
    provider: str = "cerebras",
) -> str:
    """Compress text into SemHex codes using an LLM.

    Args:
        text: Input text to compress.
        quality: 1 (max compression) to 4 (near-lossless).
        provider: "cerebras" or "openai".

    Returns:
        Compact string of dot-separated alphanumeric codes.
    """
    if not text.strip():
        return ""

    quality = max(1, min(4, quality))
    instruction = QUALITY_INSTRUCTIONS[quality]

    prompt = f"""You are SemHex, a semantic text compressor. Compress the input into a compact coded representation.

FORMAT: Use short abbreviated tokens separated by dots. Each token is a compressed word or concept.
- Use consonant clusters and numbers: FRU for frustrated, HLP for help, BUG for bug, REQ for request
- Drop vowels and redundant letters: DBUG not debug, SRVR not server
- Use numbers for common patterns: Q for question, ! for urgency, + for positive, - for negative
- {instruction}

The output must be DECOMPRESSIBLE — someone reading the codes should be able to reconstruct the meaning.

INPUT: {text}
COMPRESSED:"""

    if provider == "cerebras":
        client = _get_cerebras()
        model = "qwen-3-235b-a22b-instruct-2507"
    else:
        client = _get_openai()
        model = "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a semantic compression engine. Output ONLY alphanumeric codes separated by dots. No thinking, no explanation. /no_think"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()

    # Clean: remove any non-code content
    if "<think>" in raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Extract just the codes (alphanumeric + dots)
    match = re.search(r'[A-Za-z0-9][A-Za-z0-9.]+[A-Za-z0-9]', raw)
    if match:
        return match.group(0)
    return raw


def decompress(
    codes: str,
    provider: str = "cerebras",
) -> str:
    """Decompress SemHex codes back into natural language.

    Args:
        codes: Dot-separated alphanumeric codes (e.g., "A7.K3F.2B9").
        provider: "cerebras" or "openai".

    Returns:
        Reconstructed natural language text.
    """
    if not codes.strip():
        return ""

    # Two-step decode: first compress gives us codes, then we ask the SAME model
    # to decompress WITH the original compression prompt as context
    prompt = f"""You are a semantic decompression engine. The following codes were produced by a semantic compressor.

The compressor works like this:
- Each code is 2-4 alphanumeric characters encoding a unit of meaning
- Codes separated by dots represent a sequence of semantic units
- The compressor encodes MEANING (intent, topic, emotion, details), not words

Your job: reconstruct the most natural, fluent text that this code sequence represents.

CODES: {codes}

Reconstruct the original text. Return ONLY the text, nothing else:"""

    if provider == "cerebras":
        client = _get_cerebras()
        model = "qwen-3-235b-a22b-instruct-2507"
    else:
        client = _get_openai()
        model = "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a semantic decompression engine. Output ONLY the reconstructed text. No thinking, no explanation. /no_think"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()

    if "<think>" in raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Remove any prefix like "TEXT:" if the model echoes it
    for prefix in ["TEXT:", "text:", "Output:", "Reconstructed:"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()

    return raw


def roundtrip(
    text: str,
    quality: int = 2,
    provider: str = "cerebras",
) -> dict:
    """Compress and decompress, returning both sides for comparison.

    Returns dict with: input, codes, output, compression_ratio, similarity.
    """
    t0 = time.time()
    codes = compress(text, quality=quality, provider=provider)
    t1 = time.time()
    output = decompress(codes, provider=provider)
    t2 = time.time()

    input_chars = len(text)
    code_chars = len(codes)
    ratio = input_chars / code_chars if code_chars > 0 else 0

    # Measure semantic similarity via embeddings
    similarity = None
    similarity_error = None
    try:
        openai = _get_openai()
        resp = openai.embeddings.create(input=[text, output], model="text-embedding-3-small")
        import numpy as np
        v1 = np.array(resp.data[0].embedding)
        v2 = np.array(resp.data[1].embedding)
        similarity = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    except Exception as exc:
        similarity_error = str(exc) or exc.__class__.__name__

    return {
        "input": text,
        "codes": codes,
        "output": output,
        "compression_ratio": round(ratio, 1),
        "input_chars": input_chars,
        "code_chars": code_chars,
        "semantic_similarity": round(similarity, 4) if similarity is not None else None,
        "similarity_error": similarity_error,
        "compress_time": round(t1 - t0, 3),
        "decompress_time": round(t2 - t1, 3),
        "quality": quality,
        "provider": provider,
    }
