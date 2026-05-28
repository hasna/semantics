"""SemHex Geohash v2: Multi-bit quantization with configurable precision.

Improvements over v1:
- Multi-bit quantization (1-4 bits per dimension instead of just 1)
- Configurable total bits (48, 96, 128, 192, 256)
- Learned quantization thresholds (not just median)
- Better reconstruction via quantile-based dequantization

Format scales with precision:
  48 bits  = $XX.XXXX.XXXXXX                    (12 hex chars)
  96 bits  = $XX.XXXX.XXXXXX.XXXXXXXX           (24 hex chars)
  128 bits = $XX.XXXX.XXXXXX.XXXXXXXX.XXXXXXXX  (32 hex chars)
  192 bits = 48 hex chars
  256 bits = 64 hex chars
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_STATE_DIR = Path(__file__).parent.parent.parent / "codebooks"


class SemHasher:
    """Configurable semantic geohash encoder/decoder.

    Args:
        n_dims: Number of PCA dimensions to use.
        bits_per_dim: Bits per dimension (1, 2, or 4).
        total bits = n_dims × bits_per_dim
    """

    def __init__(self, n_dims: int = 48, bits_per_dim: int = 2):
        self.n_dims = n_dims
        self.bits_per_dim = bits_per_dim
        self.total_bits = n_dims * bits_per_dim
        self.n_levels = 2 ** bits_per_dim  # quantization levels per dim

        self.projection: NDArray | None = None  # PCA components (n_dims, orig_dims)
        self.thresholds: NDArray | None = None   # quantization boundaries (n_dims, n_levels-1)
        self.centroids: NDArray | None = None    # reconstruction values (n_dims, n_levels)
        self.trained = False

    @property
    def hex_length(self) -> int:
        """Number of hex characters in the output."""
        return self.total_bits // 4

    def train(self, embeddings: NDArray[np.float32]):
        """Train PCA projection and quantization thresholds.

        Args:
            embeddings: (N, orig_dims), normalized.
        """
        from sklearn.decomposition import PCA

        n_samples, orig_dims = embeddings.shape
        print(f"Training SemHasher: {n_samples} samples, {orig_dims}d → {self.n_dims}d × {self.bits_per_dim}b = {self.total_bits} bits ({self.hex_length} hex chars)")

        # PCA
        pca = PCA(n_components=self.n_dims, random_state=42)
        reduced = pca.fit_transform(embeddings)
        self.projection = pca.components_.astype(np.float32)

        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA explained variance: {explained:.4f} ({explained*100:.1f}%)")

        # Learn quantization thresholds per dimension using quantiles
        # For bits_per_dim=2: thresholds at 25%, 50%, 75% percentiles
        # For bits_per_dim=4: thresholds at every 1/16 percentile
        n_levels = self.n_levels
        percentiles = np.linspace(0, 100, n_levels + 1)[1:-1]  # e.g., [25, 50, 75] for 4 levels

        self.thresholds = np.zeros((self.n_dims, n_levels - 1), dtype=np.float32)
        self.centroids = np.zeros((self.n_dims, n_levels), dtype=np.float32)

        for d in range(self.n_dims):
            vals = reduced[:, d]
            self.thresholds[d] = np.percentile(vals, percentiles)

            # Compute centroid (mean) for each quantization bin
            for lvl in range(n_levels):
                if lvl == 0:
                    mask = vals <= self.thresholds[d, 0]
                elif lvl == n_levels - 1:
                    mask = vals > self.thresholds[d, -1]
                else:
                    mask = (vals > self.thresholds[d, lvl-1]) & (vals <= self.thresholds[d, lvl])
                if mask.any():
                    self.centroids[d, lvl] = vals[mask].mean()
                else:
                    self.centroids[d, lvl] = self.thresholds[d, min(lvl, len(self.thresholds[d])-1)]

        self.trained = True

        # Measure reconstruction quality
        reconstructed = self._batch_reconstruct(
            self._batch_quantize(reduced)
        )
        # Project back to original space
        recon_full = reconstructed @ self.projection
        recon_norms = np.linalg.norm(recon_full, axis=1, keepdims=True)
        recon_full = recon_full / np.where(recon_norms > 0, recon_norms, 1.0)

        sims = np.sum(embeddings * recon_full, axis=1)
        sims = np.clip(sims, -1, 1)
        print(f"  Reconstruction similarity: mean={np.mean(sims):.4f}, p5={np.percentile(sims,5):.4f}, p95={np.percentile(sims,95):.4f}")

        return explained, float(np.mean(sims))

    def save(self, name: str = "default"):
        """Save trained state."""
        path = _STATE_DIR / f"semhasher_{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path,
            projection=self.projection,
            thresholds=self.thresholds,
            centroids=self.centroids,
            n_dims=self.n_dims,
            bits_per_dim=self.bits_per_dim,
        )
        print(f"  Saved to {path}")

    def load(self, name: str = "default"):
        """Load trained state."""
        path = _STATE_DIR / f"semhasher_{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"No trained state at {path}")
        state = np.load(path)
        self.projection = state["projection"]
        self.thresholds = state["thresholds"]
        self.centroids = state["centroids"]
        self.n_dims = int(state["n_dims"])
        self.bits_per_dim = int(state["bits_per_dim"])
        self.total_bits = self.n_dims * self.bits_per_dim
        self.n_levels = 2 ** self.bits_per_dim
        self.trained = True

    def _reduce(self, embedding: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project to reduced dimensions."""
        return (embedding @ self.projection.T).astype(np.float32)

    def _quantize_dim(self, value: float, dim: int) -> int:
        """Quantize a single value to an integer level."""
        for lvl, thresh in enumerate(self.thresholds[dim]):
            if value <= thresh:
                return lvl
        return self.n_levels - 1

    def _quantize(self, reduced: NDArray[np.float32]) -> list[int]:
        """Quantize all dimensions to integer levels."""
        return [self._quantize_dim(reduced[d], d) for d in range(self.n_dims)]

    def _batch_quantize(self, reduced: NDArray[np.float32]) -> NDArray[np.int8]:
        """Quantize a batch of reduced vectors."""
        n = reduced.shape[0]
        result = np.zeros((n, self.n_dims), dtype=np.int8)
        for d in range(self.n_dims):
            vals = reduced[:, d]
            result[:, d] = self.n_levels - 1  # default to highest
            for lvl in range(self.n_levels - 1):
                mask = vals <= self.thresholds[d, lvl]
                result[:, d] = np.where(mask & (result[:, d] == self.n_levels - 1), lvl, result[:, d])
        return result

    def _dequantize(self, levels: list[int]) -> NDArray[np.float32]:
        """Convert quantization levels back to approximate values."""
        return np.array([self.centroids[d, lvl] for d, lvl in enumerate(levels)], dtype=np.float32)

    def _batch_reconstruct(self, levels: NDArray[np.int8]) -> NDArray[np.float32]:
        """Reconstruct reduced vectors from quantization levels."""
        n = levels.shape[0]
        result = np.zeros((n, self.n_dims), dtype=np.float32)
        for d in range(self.n_dims):
            for lvl in range(self.n_levels):
                mask = levels[:, d] == lvl
                result[mask, d] = self.centroids[d, lvl]
        return result

    def _levels_to_bits(self, levels: list[int]) -> list[int]:
        """Convert quantization levels to a flat bit list."""
        bits = []
        for lvl in levels:
            for b in range(self.bits_per_dim - 1, -1, -1):
                bits.append((lvl >> b) & 1)
        return bits

    def _bits_to_levels(self, bits: list[int]) -> list[int]:
        """Convert flat bit list back to quantization levels."""
        levels = []
        for i in range(0, len(bits), self.bits_per_dim):
            chunk = bits[i:i + self.bits_per_dim]
            val = 0
            for b in chunk:
                val = (val << 1) | b
            levels.append(val)
        return levels

    def _bits_to_hex(self, bits: list[int]) -> str:
        """Pack bits into hex string."""
        hex_chars = []
        for i in range(0, len(bits), 4):
            nibble = bits[i:i+4]
            while len(nibble) < 4:
                nibble.append(0)
            val = nibble[0]*8 + nibble[1]*4 + nibble[2]*2 + nibble[3]
            hex_chars.append(f"{val:X}")
        return "".join(hex_chars)

    def _hex_to_bits(self, hex_str: str) -> list[int]:
        """Unpack hex string to bits."""
        bits = []
        for ch in hex_str:
            val = int(ch, 16)
            bits.extend([(val>>3)&1, (val>>2)&1, (val>>1)&1, val&1])
        return bits

    def format_code(self, hex_str: str) -> str:
        """Format raw hex into $XX.XXXX.XXXXXX... blocks."""
        # Block sizes: 2, 4, 6, then 8 repeating
        blocks = []
        i = 0
        block_sizes = [2, 4, 6]
        for bs in block_sizes:
            if i >= len(hex_str):
                break
            blocks.append(hex_str[i:i+bs])
            i += bs
        while i < len(hex_str):
            blocks.append(hex_str[i:i+8])
            i += 8
        return "$" + ".".join(blocks)

    def parse_code(self, code: str) -> str:
        """Parse $XX.XXXX.XXXXXX... back to raw hex."""
        code = code.strip()
        if code.startswith("$"):
            code = code[1:]
        return code.replace(".", "")

    def encode(self, embedding: NDArray[np.float32]) -> str:
        """Encode an embedding vector to a SemHex code string.

        Args:
            embedding: normalized vector from embedding model.

        Returns:
            String like "$8A.2F01.C3A7B0"
        """
        reduced = self._reduce(embedding)
        levels = self._quantize(reduced)
        bits = self._levels_to_bits(levels)
        hex_str = self._bits_to_hex(bits)
        return self.format_code(hex_str)

    def decode(self, code: str) -> NDArray[np.float32]:
        """Decode a SemHex code back to an approximate embedding vector.

        Args:
            code: String like "$8A.2F01.C3A7B0"

        Returns:
            Approximate embedding vector in original space.
        """
        hex_str = self.parse_code(code)
        bits = self._hex_to_bits(hex_str)
        levels = self._bits_to_levels(bits[:self.n_dims * self.bits_per_dim])
        reduced = self._dequantize(levels)

        # Project back to original space
        reconstructed = reduced @ self.projection
        norm = np.linalg.norm(reconstructed)
        if norm > 0:
            reconstructed = reconstructed / norm
        return reconstructed.astype(np.float32)

    def similarity(self, code_a: str, code_b: str) -> float:
        """Compute similarity between two codes via Hamming distance on bits."""
        ha = self.parse_code(code_a)
        hb = self.parse_code(code_b)
        bits_a = self._hex_to_bits(ha)
        bits_b = self._hex_to_bits(hb)
        n = min(len(bits_a), len(bits_b))
        matching = sum(a == b for a, b in zip(bits_a[:n], bits_b[:n]))
        return matching / n if n > 0 else 0.0


def train_and_compare(embeddings: NDArray[np.float32]):
    """Train multiple configurations and compare accuracy."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    results = []

    configs = [
        (48, 1, "48 bits (v1 baseline)"),
        (48, 2, "96 bits (2b/dim)"),
        (64, 2, "128 bits (2b/dim)"),
        (48, 4, "192 bits (4b/dim)"),
        (64, 4, "256 bits (4b/dim)"),
        (96, 2, "192 bits (96d×2b)"),
        (128, 2, "256 bits (128d×2b)"),
    ]

    for n_dims, bpd, label in configs:
        console.print(f"\n[bold]{label}[/bold]")
        hasher = SemHasher(n_dims=n_dims, bits_per_dim=bpd)
        explained, mean_sim = hasher.train(embeddings)
        hasher.save(f"{n_dims}d_{bpd}b")

        results.append({
            "config": label,
            "n_dims": n_dims,
            "bits_per_dim": bpd,
            "total_bits": n_dims * bpd,
            "hex_chars": n_dims * bpd // 4,
            "explained_var": round(explained, 4),
            "mean_similarity": round(mean_sim, 4),
        })

    table = Table(title="SemHasher Configuration Comparison")
    table.add_column("Config")
    table.add_column("Bits", justify="right")
    table.add_column("Hex Chars", justify="right")
    table.add_column("PCA Var", justify="right")
    table.add_column("Recon Sim", justify="right", style="cyan")

    for r in results:
        table.add_row(
            r["config"],
            str(r["total_bits"]),
            str(r["hex_chars"]),
            f"{r['explained_var']:.4f}",
            f"{r['mean_similarity']:.4f}",
        )

    console.print(table)
    return results


if __name__ == "__main__":
    embeddings = np.load("data/embeddings.npy").astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)
    train_and_compare(embeddings)
