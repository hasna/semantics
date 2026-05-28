"""Frozen codebook: load, query, nearest-neighbor search.

A codebook is a set of centroids in embedding space, each with a hex address.
Level 1: 256 coarse categories ($XX)
Level 2: 256 fine meanings per L1 ($XX.XXXX)

The codebook is frozen once trained. Codes are permanent addresses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from semhex.core.format import SemHexCode

_CODEBOOK_DIR = Path(__file__).parent.parent.parent / "codebooks"


@dataclass
class CentroidEntry:
    """A single entry in the codebook."""
    code: SemHexCode
    label: str
    vector: NDArray[np.float32]
    examples: list[str]


class Codebook:
    """A frozen SemHex codebook for encoding/decoding.

    Contains centroids at two levels:
    - Level 1: 256 coarse categories
    - Level 2: up to 256 fine meanings per L1 category
    """

    def __init__(
        self,
        l1_centroids: NDArray[np.float32],
        l2_centroids: NDArray[np.float32],
        l1_labels: dict[str, str],
        l2_labels: dict[str, str],
        l1_examples: dict[str, list[str]] | None = None,
        l2_examples: dict[str, list[str]] | None = None,
        metadata: dict | None = None,
    ):
        """
        Args:
            l1_centroids: shape (n_l1, dims) — Level 1 centroid vectors
            l2_centroids: shape (n_l2, dims) — Level 2 centroid vectors
            l1_labels: hex_code → human label for L1
            l2_labels: hex_code → human label for L2
            l1_examples: hex_code → example concept strings for L1
            l2_examples: hex_code → example concept strings for L2
            metadata: version info, training params, etc.
        """
        self.l1_centroids = l1_centroids
        self.l2_centroids = l2_centroids
        self.l1_labels = l1_labels
        self.l2_labels = l2_labels
        self.l1_examples = l1_examples or {}
        self.l2_examples = l2_examples or {}
        self.metadata = metadata or {}
        self._n_l1 = l1_centroids.shape[0]
        self._dims = l1_centroids.shape[1]

        # L2 centroids are stored flat: index = l1_idx * 256 + l2_idx
        # So total L2 entries = n_l1 * 256 (if fully populated)
        self._n_l2_per_l1 = l2_centroids.shape[0] // self._n_l1 if l2_centroids.shape[0] > 0 else 0

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def n_level1(self) -> int:
        return self._n_l1

    @property
    def n_level2(self) -> int:
        return self.l2_centroids.shape[0]

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unknown")

    def nearest_l1(self, vector: NDArray[np.float32]) -> tuple[SemHexCode, float]:
        """Find the nearest Level 1 centroid.

        Returns:
            (code, distance) where distance is cosine distance [0=identical, 2=opposite].
        """
        sims = self.l1_centroids @ vector
        idx = int(np.argmax(sims))
        code = SemHexCode(idx)
        dist = 1.0 - float(sims[idx])
        return code, dist

    def nearest_l2(self, vector: NDArray[np.float32], l1_idx: int | None = None) -> tuple[SemHexCode, float]:
        """Find the nearest Level 2 centroid.

        Args:
            vector: Query vector (normalized).
            l1_idx: If provided, search only within this L1 cluster. Otherwise search all.

        Returns:
            (code, distance)
        """
        if l1_idx is not None and self._n_l2_per_l1 > 0:
            start = l1_idx * self._n_l2_per_l1
            end = start + self._n_l2_per_l1
            subset = self.l2_centroids[start:end]
            sims = subset @ vector
            local_idx = int(np.argmax(sims))
            global_idx = start + local_idx
        else:
            sims = self.l2_centroids @ vector
            global_idx = int(np.argmax(sims))
            local_idx = global_idx % self._n_l2_per_l1 if self._n_l2_per_l1 > 0 else global_idx

        l1 = global_idx // self._n_l2_per_l1 if self._n_l2_per_l1 > 0 else 0
        l2 = local_idx

        # Encode L2 index as 2 bytes: high byte = sub-cluster group, low byte = sub-cluster index
        l2_combined = l2
        code = SemHexCode(l1, l2_combined)
        dist = 1.0 - float(self.l2_centroids[global_idx] @ vector)
        return code, dist

    def nearest(self, vector: NDArray[np.float32], depth: int = 2) -> tuple[SemHexCode, float]:
        """Find the nearest codebook entry at the given depth.

        Args:
            vector: Query vector (normalized).
            depth: 1 for coarse, 2 for fine.
        """
        if depth <= 1:
            return self.nearest_l1(vector)

        # Two-stage: find L1 first, then L2 within that cluster
        l1_code, _ = self.nearest_l1(vector)
        return self.nearest_l2(vector, l1_idx=l1_code.level1)

    def lookup(self, code: SemHexCode) -> CentroidEntry:
        """Look up a codebook entry by its code."""
        code_str = str(code)

        if code.depth == 1:
            if code.level1 >= self._n_l1:
                raise KeyError(f"L1 index {code.level1} out of range (max {self._n_l1 - 1})")
            return CentroidEntry(
                code=code,
                label=self.l1_labels.get(code_str, f"cluster_{code.level1}"),
                vector=self.l1_centroids[code.level1],
                examples=self.l1_examples.get(code_str, []),
            )

        # L2 lookup
        if self._n_l2_per_l1 == 0:
            raise KeyError(f"No L2 centroids in this codebook")
        global_idx = code.level1 * self._n_l2_per_l1 + (code.level2 or 0)
        if global_idx >= self.l2_centroids.shape[0]:
            raise KeyError(f"L2 index out of range for code {code_str}")

        return CentroidEntry(
            code=code,
            label=self.l2_labels.get(code_str, f"cluster_{code.level1}_{code.level2}"),
            vector=self.l2_centroids[global_idx],
            examples=self.l2_examples.get(code_str, []),
        )

    def neighbors(self, code: SemHexCode, k: int = 5) -> list[CentroidEntry]:
        """Find k nearest neighbor codes to the given code."""
        entry = self.lookup(code)
        vector = entry.vector

        if code.depth == 1:
            sims = self.l1_centroids @ vector
            top_k = np.argsort(-sims)[1:k + 1]  # Skip self
            results = []
            for idx in top_k:
                c = SemHexCode(int(idx))
                results.append(self.lookup(c))
            return results

        # L2 neighbors within the same L1 cluster
        l1_idx = code.level1
        start = l1_idx * self._n_l2_per_l1
        end = start + self._n_l2_per_l1
        subset = self.l2_centroids[start:end]
        sims = subset @ vector
        top_k = np.argsort(-sims)[1:k + 1]
        results = []
        for local_idx in top_k:
            c = SemHexCode(l1_idx, int(local_idx))
            results.append(self.lookup(c))
        return results


def load_codebook(version: str = "v0.1", codebook_dir: Path | None = None) -> Codebook:
    """Load a frozen codebook from disk.

    Args:
        version: Codebook version directory name.
        codebook_dir: Override codebook directory path.
    """
    base = (codebook_dir or _CODEBOOK_DIR) / version

    l1_path = base / "level1.npy"
    l2_path = base / "level2.npy"
    labels_path = base / "labels.json"
    metadata_path = base / "metadata.json"

    if not l1_path.exists():
        raise FileNotFoundError(
            f"Codebook not found at {base}. "
            f"Run 'python -m semhex.training.build_codebook' to generate it."
        )

    l1_centroids = np.load(l1_path).astype(np.float32)
    l2_centroids = np.load(l2_path).astype(np.float32) if l2_path.exists() else np.empty((0, l1_centroids.shape[1]), dtype=np.float32)

    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    return Codebook(
        l1_centroids=l1_centroids,
        l2_centroids=l2_centroids,
        l1_labels=labels.get("l1", {}),
        l2_labels=labels.get("l2", {}),
        l1_examples=labels.get("l1_examples", {}),
        l2_examples=labels.get("l2_examples", {}),
        metadata=metadata,
    )
