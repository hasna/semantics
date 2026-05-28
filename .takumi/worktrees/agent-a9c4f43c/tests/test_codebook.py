"""Tests for codebook loading, querying, and nearest-neighbor search."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from semhex.core.codebook import Codebook, load_codebook, CentroidEntry
from semhex.core.format import SemHexCode


@pytest.fixture
def small_codebook(tmp_path):
    """Create a small test codebook (4 L1, 4 L2 per L1)."""
    dims = 8
    n_l1 = 4
    n_l2_per_l1 = 4

    rng = np.random.RandomState(42)

    # Create normalized L1 centroids
    l1 = rng.randn(n_l1, dims).astype(np.float32)
    l1 /= np.linalg.norm(l1, axis=1, keepdims=True)

    # Create normalized L2 centroids (n_l1 * n_l2_per_l1 total)
    l2 = rng.randn(n_l1 * n_l2_per_l1, dims).astype(np.float32)
    l2 /= np.linalg.norm(l2, axis=1, keepdims=True)

    # Save to tmp
    np.save(tmp_path / "level1.npy", l1)
    np.save(tmp_path / "level2.npy", l2)

    labels = {
        "l1": {"$00": "category_a", "$01": "category_b", "$02": "category_c", "$03": "category_d"},
        "l2": {"$00.0000": "concept_a1", "$00.0001": "concept_a2", "$01.0000": "concept_b1"},
        "l1_examples": {"$00": ["example_a1", "example_a2"]},
        "l2_examples": {"$00.0000": ["ex_a1_1", "ex_a1_2"]},
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels))

    metadata = {"version": "test", "dimensions": dims, "n_l1": n_l1, "n_l2_per_l1": n_l2_per_l1}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    return tmp_path, l1, l2


class TestCodebookLoad:
    def test_load_from_disk(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Codebook not found"):
            load_codebook("nonexistent", codebook_dir=tmp_path)

    def test_properties(self, small_codebook):
        path, l1, l2 = small_codebook
        # Load using the parent as codebook_dir, and path.name as version
        cb = load_codebook(path.name, codebook_dir=path.parent)
        assert cb.dimensions == 8
        assert cb.n_level1 == 4
        assert cb.version == "test"


class TestCodebookNearest:
    def test_nearest_l1(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        # Query with L1 centroid 0 — should find itself
        code, dist = cb.nearest_l1(l1[0])
        assert code.level1 == 0
        assert dist < 0.01  # Very close (cosine distance)

    def test_nearest_l1_returns_semhexcode(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        code, dist = cb.nearest_l1(l1[0])
        assert isinstance(code, SemHexCode)
        assert code.depth == 1

    def test_nearest_l2(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        # Query with L2 centroid 0 (in L1 cluster 0)
        code, dist = cb.nearest_l2(l2[0], l1_idx=0)
        assert code.level1 == 0
        assert code.level2 is not None
        assert dist < 0.01

    def test_nearest_depth2(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        code, dist = cb.nearest(l2[0], depth=2)
        assert code.depth == 2

    def test_nearest_depth1(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        code, dist = cb.nearest(l1[2], depth=1)
        assert code.depth == 1
        assert code.level1 == 2


class TestCodebookLookup:
    def test_lookup_l1(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        entry = cb.lookup(SemHexCode(0))
        assert isinstance(entry, CentroidEntry)
        assert entry.label == "category_a"
        assert entry.vector.shape == (8,)
        np.testing.assert_array_almost_equal(entry.vector, l1[0])

    def test_lookup_l2(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        entry = cb.lookup(SemHexCode(0, 0))
        assert isinstance(entry, CentroidEntry)
        assert entry.label == "concept_a1"

    def test_lookup_l1_out_of_range(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        with pytest.raises(KeyError):
            cb.lookup(SemHexCode(255))

    def test_lookup_has_examples(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        entry = cb.lookup(SemHexCode(0))
        assert entry.examples == ["example_a1", "example_a2"]


class TestCodebookNeighbors:
    def test_neighbors_l1(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        neighbors = cb.neighbors(SemHexCode(0), k=2)
        assert len(neighbors) == 2
        assert all(isinstance(n, CentroidEntry) for n in neighbors)
        # Neighbors should not include self
        assert all(n.code.level1 != 0 for n in neighbors)

    def test_neighbors_l2(self, small_codebook):
        path, l1, l2 = small_codebook
        cb = load_codebook(path.name, codebook_dir=path.parent)

        neighbors = cb.neighbors(SemHexCode(0, 0), k=2)
        assert len(neighbors) == 2
        # All neighbors should be in the same L1 cluster
        assert all(n.code.level1 == 0 for n in neighbors)


class TestBuiltCodebook:
    """Test the actual codebook built from mock embeddings."""

    def test_load_v01(self):
        """The v0.1 codebook should exist after build_codebook.py runs."""
        cb = load_codebook("v0.1")
        assert cb.n_level1 > 0
        assert cb.dimensions > 0

    def test_roundtrip_nearest(self):
        """Looking up a centroid should find itself."""
        cb = load_codebook("v0.1")
        # Get the first L1 centroid
        entry = cb.lookup(SemHexCode(0))
        # Find nearest — should be itself
        code, dist = cb.nearest_l1(entry.vector)
        assert code.level1 == 0
        assert dist < 0.01

    def test_labels_populated(self):
        cb = load_codebook("v0.1")
        entry = cb.lookup(SemHexCode(0))
        assert entry.label != ""
        assert len(entry.examples) > 0
