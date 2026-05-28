"""SemHex Geohash: encode embedding vectors directly into hex coordinate strings.

Like geohash maps lat/long → "9q8yyf", this maps meaning vectors → "$8A.2F01.C3A7B0"

The code IS the vector, quantized. No codebook lookup table needed.
Any system that knows the quantization scheme can decode.

Format: $XX.XXXX.XXXXXX
  $XX       = 8 bits  = coarse meaning (256 regions)
  .XXXX     = 16 bits = specific concept (65K sub-regions)
  .XXXXXX   = 24 bits = exact nuance (16M points)
  Total     = 48 bits

Each character is hex (0-F) = 4 bits.
Nearby meanings → similar strings (shared prefix = close in meaning space).
Truncating = coarser but still valid address.

Pipeline:
1. Embed text → 1536-dim vector (OpenAI)
2. Reduce to 48 dims via Matryoshka truncation (first 48 dims carry most info)
3. Quantize each dim to 1 bit (above/below trained median)
4. Pack 48 bits into hex string
5. Format as $XX.XXXX.XXXXXX
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Quantization state — learned medians per dimension
_medians: NDArray | None = None
_projection: NDArray | None = None
_state_path = Path(__file__).parent.parent.parent / "codebooks" / "geohash_state.npz"


def _load_state():
    """Load trained quantization state (medians + projection matrix)."""
    global _medians, _projection
    if _medians is not None:
        return

    if _state_path.exists():
        state = np.load(_state_path)
        _medians = state["medians"]
        _projection = state["projection"] if "projection" in state else None
    else:
        # Default: zero medians, no projection (identity)
        _medians = np.zeros(48, dtype=np.float32)
        _projection = None


def train_quantizer(
    embeddings: NDArray[np.float32],
    n_bits: int = 48,
    output_path: str | None = None,
):
    """Train the quantizer from a set of embeddings.

    Learns:
    1. PCA projection matrix (1536d → 48d) that maximizes variance in first dims
    2. Per-dimension medians for binary quantization

    Args:
        embeddings: shape (N, dims), normalized
        n_bits: number of output bits (= number of reduced dimensions)
        output_path: where to save the trained state
    """
    global _medians, _projection

    from sklearn.decomposition import PCA

    n_samples, orig_dims = embeddings.shape
    print(f"Training geohash quantizer: {n_samples} samples, {orig_dims}d → {n_bits} bits")

    # Step 1: PCA to reduce dimensions
    # PCA automatically puts most variance in first components
    pca = PCA(n_components=n_bits, random_state=42)
    reduced = pca.fit_transform(embeddings)

    # The projection matrix: (n_bits, orig_dims)
    _projection = pca.components_.astype(np.float32)

    # Step 2: Compute median per dimension for binary quantization
    _medians = np.median(reduced, axis=0).astype(np.float32)

    # Save
    save_path = Path(output_path) if output_path else _state_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, medians=_medians, projection=_projection)

    # Report quality
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained:.4f} ({explained*100:.1f}%)")
    print(f"  Saved to {save_path}")

    return explained


def _reduce(embedding: NDArray[np.float32]) -> NDArray[np.float32]:
    """Project a 1536-dim embedding down to 48 dims using trained PCA."""
    _load_state()
    if _projection is not None:
        return (embedding @ _projection.T).astype(np.float32)
    # Fallback: just take first 48 dims
    return embedding[:48].astype(np.float32)


def _quantize(reduced: NDArray[np.float32]) -> list[int]:
    """Quantize each dimension to a single bit (above/below median)."""
    _load_state()
    bits = (reduced > _medians).astype(np.int8)
    return bits.tolist()


def _bits_to_hex(bits: list[int]) -> str:
    """Pack a list of bits into a hex string."""
    hex_chars = []
    for i in range(0, len(bits), 4):
        nibble = bits[i:i+4]
        # Pad if needed
        while len(nibble) < 4:
            nibble.append(0)
        val = nibble[0] * 8 + nibble[1] * 4 + nibble[2] * 2 + nibble[3]
        hex_chars.append(f"{val:X}")
    return "".join(hex_chars)


def _hex_to_bits(hex_str: str) -> list[int]:
    """Unpack a hex string into a list of bits."""
    bits = []
    for ch in hex_str:
        val = int(ch, 16)
        bits.extend([
            (val >> 3) & 1,
            (val >> 2) & 1,
            (val >> 1) & 1,
            val & 1,
        ])
    return bits


def _format_semhex(hex_str: str) -> str:
    """Format raw hex into $XX.XXXX.XXXXXX."""
    # Pad to 12 chars (48 bits)
    hex_str = hex_str.ljust(12, "0")
    return f"${hex_str[:2]}.{hex_str[2:6]}.{hex_str[6:12]}"


def _parse_semhex(code: str) -> str:
    """Parse $XX.XXXX.XXXXXX back to raw hex."""
    code = code.strip()
    if code.startswith("$"):
        code = code[1:]
    return code.replace(".", "")


def encode_vector(embedding: NDArray[np.float32]) -> str:
    """Encode an embedding vector into a SemHex geohash string.

    Args:
        embedding: 1536-dim (or whatever the model outputs) normalized vector.

    Returns:
        String like "$8A.2F01.C3A7B0"
    """
    reduced = _reduce(embedding)
    bits = _quantize(reduced)
    hex_str = _bits_to_hex(bits)
    return _format_semhex(hex_str)


def decode_vector(code: str) -> NDArray[np.float32]:
    """Decode a SemHex geohash string back to an approximate embedding vector.

    This is lossy — 48 bits can't fully reconstruct a 1536-dim vector.
    But it gives the CENTROID of the region that code represents.

    Args:
        code: String like "$8A.2F01.C3A7B0"

    Returns:
        Approximate 1536-dim vector (or reduced-dim vector if no projection).
    """
    _load_state()
    hex_str = _parse_semhex(code)
    bits = _hex_to_bits(hex_str)

    # Reconstruct reduced-dim vector: each bit tells us above/below median
    # Use median + small offset for 1, median - small offset for 0
    bits_arr = np.array(bits[:len(_medians)], dtype=np.float32)
    # Approximate: bit=1 means value is above median, bit=0 means below
    # Use median ± standard deviation as approximation
    reduced = _medians + (bits_arr * 2 - 1) * 0.5

    # Project back to original space if we have the projection matrix
    if _projection is not None:
        # Pseudo-inverse of the projection
        reconstructed = reduced @ _projection
        # Normalize
        norm = np.linalg.norm(reconstructed)
        if norm > 0:
            reconstructed = reconstructed / norm
        return reconstructed

    return reduced


def similarity_from_codes(code_a: str, code_b: str) -> float:
    """Estimate semantic similarity from two SemHex geohash codes.

    Uses Hamming distance on the bit representation.
    Shared prefix = similar meaning.

    Returns:
        Estimated similarity in [0, 1]. 1 = identical codes.
    """
    hex_a = _parse_semhex(code_a)
    hex_b = _parse_semhex(code_b)

    bits_a = _hex_to_bits(hex_a)
    bits_b = _hex_to_bits(hex_b)

    # Hamming distance
    n = min(len(bits_a), len(bits_b))
    matching = sum(a == b for a, b in zip(bits_a[:n], bits_b[:n]))
    return matching / n if n > 0 else 0.0


def shared_prefix_length(code_a: str, code_b: str) -> int:
    """How many characters of prefix do two codes share?

    More shared prefix = more similar meaning.
    """
    a = _parse_semhex(code_a)
    b = _parse_semhex(code_b)
    length = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            length += 1
        else:
            break
    return length
