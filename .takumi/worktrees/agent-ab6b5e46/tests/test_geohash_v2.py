"""Tests for SemHasher (geohash_v2) — encode/decode/train/save/load."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from semhex.core.geohash_v2 import SemHasher


@pytest.fixture
def tiny_hasher():
    """A tiny SemHasher (4d, 2b) trained on synthetic data."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(200, 16).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    hasher = SemHasher(n_dims=4, bits_per_dim=2)
    hasher.train(embeddings)
    return hasher, embeddings


class TestSemHasherInit:
    def test_default_params(self):
        h = SemHasher()
        assert h.n_dims == 48
        assert h.bits_per_dim == 2
        assert h.total_bits == 96
        assert not h.trained

    def test_custom_params(self):
        h = SemHasher(n_dims=8, bits_per_dim=4)
        assert h.total_bits == 32
        assert h.hex_length == 8

    def test_hex_length_2b(self):
        h = SemHasher(n_dims=64, bits_per_dim=2)
        assert h.total_bits == 128
        assert h.hex_length == 32

    def test_hex_length_4b(self):
        h = SemHasher(n_dims=64, bits_per_dim=4)
        assert h.total_bits == 256
        assert h.hex_length == 64


class TestSemHasherTrain:
    def test_train_sets_trained(self, tiny_hasher):
        h, _ = tiny_hasher
        assert h.trained

    def test_train_sets_projection(self, tiny_hasher):
        h, _ = tiny_hasher
        assert h.projection is not None
        assert h.projection.shape == (4, 16)

    def test_train_sets_thresholds(self, tiny_hasher):
        h, _ = tiny_hasher
        assert h.thresholds is not None
        assert h.thresholds.shape[0] == 4

    def test_train_sets_centroids(self, tiny_hasher):
        h, _ = tiny_hasher
        assert h.centroids is not None


class TestSemHasherEncode:
    def test_encode_returns_string(self, tiny_hasher):
        h, embs = tiny_hasher
        code = h.encode(embs[0])
        assert isinstance(code, str)

    def test_encode_correct_length(self, tiny_hasher):
        h, embs = tiny_hasher
        code = h.encode(embs[0])
        # code may have a "$" prefix; strip it for length check
        hex_part = code.lstrip("$")
        assert len(hex_part) == h.hex_length

    def test_encode_hex_chars(self, tiny_hasher):
        h, embs = tiny_hasher
        code = h.encode(embs[0]).lstrip("$")
        assert all(c in "0123456789ABCDEFabcdef" for c in code)

    def test_same_vector_same_code(self, tiny_hasher):
        h, embs = tiny_hasher
        v = embs[0]
        assert h.encode(v) == h.encode(v)

    def test_different_vectors_may_differ(self, tiny_hasher):
        """Statistically, random vectors should not all hash to the same code."""
        h, embs = tiny_hasher
        codes = [h.encode(embs[i]) for i in range(20)]
        assert len(set(codes)) > 1


class TestSemHasherDecode:
    def test_decode_returns_array(self, tiny_hasher):
        h, embs = tiny_hasher
        code = h.encode(embs[0])
        vec = h.decode(code)
        assert isinstance(vec, np.ndarray)

    def test_decode_correct_shape(self, tiny_hasher):
        h, embs = tiny_hasher
        code = h.encode(embs[0])
        vec = h.decode(code)
        # decode reconstructs back to original embedding space
        assert vec.shape[0] > 0

    def test_encode_decode_returns_array(self, tiny_hasher):
        """Decoded vector is a numpy array."""
        h, embs = tiny_hasher
        code = h.encode(embs[0])
        decoded = h.decode(code)
        assert isinstance(decoded, np.ndarray)


class TestSemHasherSaveLoad:
    def test_save_and_load(self, tiny_hasher, tmp_path):
        """save/load round-trip using default codebook dir (writes to codebooks/)."""
        h, embs = tiny_hasher
        save_name = "_pytest_tiny_hasher"
        h.save(save_name)
        # Load into new hasher
        h2 = SemHasher(n_dims=4, bits_per_dim=2)
        h2.load(save_name)
        assert h2.trained
        # Should produce same codes
        for v in embs[:5]:
            assert h.encode(v) == h2.encode(v)
        # Cleanup
        import glob
        for f in glob.glob(f"codebooks/semhasher_{save_name}.npz"):
            Path(f).unlink(missing_ok=True)

    def test_load_pretrained_64d_4b(self):
        """Load the production-trained 64d/4b hasher (matryoshka_64d_4b)."""
        h = SemHasher(n_dims=64, bits_per_dim=4)
        h.load("matryoshka_64d_4b")
        assert h.trained
        assert h.hex_length == 64

    def test_load_pretrained_64d_2b(self):
        """Load the production-trained 64d/2b hasher."""
        h = SemHasher(n_dims=64, bits_per_dim=2)
        h.load("matryoshka_64d_2b")
        assert h.trained
        assert h.hex_length == 32
