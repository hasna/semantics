"""Encoder: text → SemHex code sequences.

Encodes at MEANING level, not word level. A sentence or paragraph becomes
a SHORT sequence of meaning-codes — this is where compression happens.

Strategy:
1. Split input into semantic chunks (sentences/clauses)
2. Embed each chunk as a whole
3. Find nearest codebook entries
4. Return compact sequence of SemHex codes
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from semhex.core.codebook import Codebook, load_codebook
from semhex.core.format import SemHexCode
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings import get_provider

# Sentence splitter: split on . ! ? followed by space or end, or on newlines
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')

# Default singleton instances (lazy-loaded)
_default_codebook: Codebook | None = None
_default_provider: EmbeddingProvider | None = None


@dataclass
class EncodeResult:
    """Result of encoding text to SemHex codes."""
    codes: list[SemHexCode]
    chunks: list[str]
    distances: list[float]

    @property
    def code_strings(self) -> list[str]:
        return [str(c) for c in self.codes]

    @property
    def compression_ratio(self) -> float:
        """Ratio of input words to output codes."""
        total_words = sum(len(chunk.split()) for chunk in self.chunks)
        if len(self.codes) == 0:
            return 0.0
        return total_words / len(self.codes)

    def __str__(self) -> str:
        return " ".join(self.code_strings)


def _split_into_chunks(text: str) -> list[str]:
    """Split text into semantic chunks (sentences/clauses).

    Short inputs (<10 words) are kept as a single chunk.
    Longer inputs are split on sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    # Short text: single chunk
    words = text.split()
    if len(words) <= 10:
        return [text]

    # Split on sentence boundaries
    chunks = _SENTENCE_SPLIT.split(text)
    chunks = [c.strip() for c in chunks if c.strip()]

    # Merge very short chunks (< 3 words) with their predecessor
    merged = []
    for chunk in chunks:
        if merged and len(chunk.split()) < 3:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged if merged else [text]


def encode(
    text: str,
    depth: int = 2,
    codebook: Codebook | None = None,
    provider: EmbeddingProvider | None = None,
) -> EncodeResult:
    """Encode text into a sequence of SemHex codes.

    This is MEANING-LEVEL encoding: a sentence becomes a small number of codes,
    not one code per word.

    Args:
        text: Input text to encode.
        depth: Code depth (1=coarse, 2=fine).
        codebook: Codebook to use (default: load v0.1).
        provider: Embedding provider (default: auto-detect).

    Returns:
        EncodeResult with codes, chunks, and distances.
    """
    global _default_codebook, _default_provider

    if codebook is None:
        if _default_codebook is None:
            _default_codebook = load_codebook("v0.1")
        codebook = _default_codebook

    if provider is None:
        if _default_provider is None:
            _default_provider = get_provider("auto")
        provider = _default_provider

    chunks = _split_into_chunks(text)
    if not chunks:
        return EncodeResult(codes=[], chunks=[], distances=[])

    # Embed all chunks
    embeddings = provider.embed_batch(chunks)

    # Find nearest codebook entry for each chunk
    codes = []
    distances = []
    for i, emb in enumerate(embeddings):
        code, dist = codebook.nearest(emb, depth=depth)
        codes.append(code)
        distances.append(dist)

    return EncodeResult(codes=codes, chunks=chunks, distances=distances)


def encode_batch(
    texts: list[str],
    depth: int = 2,
    codebook: Codebook | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[EncodeResult]:
    """Encode multiple texts into SemHex code sequences.

    Args:
        texts: List of input texts.
        depth: Code depth (1=coarse, 2=fine).
        codebook: Codebook to use.
        provider: Embedding provider.

    Returns:
        List of EncodeResult, one per input text.
    """
    return [encode(t, depth=depth, codebook=codebook, provider=provider) for t in texts]
