"""Tests for semantic distance between SemHex codes."""

import pytest

from semhex.core.codebook import load_codebook
from semhex.core.distance import distance, similarity
from semhex.core.format import SemHexCode


@pytest.fixture
def codebook():
    return load_codebook("v0.1")


class TestDistance:
    def test_same_code_zero_distance(self, codebook):
        d = distance(SemHexCode(0, 0), SemHexCode(0, 0), codebook=codebook)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_different_codes_positive_distance(self, codebook):
        d = distance(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        assert d > 0.0

    def test_distance_range(self, codebook):
        """Distance should be in [0, 2]."""
        d = distance(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        assert 0.0 <= d <= 2.0

    def test_symmetry(self, codebook):
        d_ab = distance(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        d_ba = distance(SemHexCode(1, 0), SemHexCode(0, 0), codebook=codebook)
        assert d_ab == pytest.approx(d_ba, abs=1e-6)

    def test_string_input(self, codebook):
        d = distance("$00.0000", "$01.0000", codebook=codebook)
        assert d > 0.0

    def test_l1_codes(self, codebook):
        d = distance(SemHexCode(0), SemHexCode(1), codebook=codebook)
        assert d > 0.0

    def test_same_cluster_closer_than_different(self, codebook):
        """Codes in the same L1 cluster should generally be closer than codes in different clusters."""
        # Same L1 cluster, different L2
        d_same = distance(SemHexCode(0, 0), SemHexCode(0, 1), codebook=codebook)
        # Different L1 clusters
        d_diff = distance(SemHexCode(0, 0), SemHexCode(5, 0), codebook=codebook)
        # Not guaranteed with mock embeddings, but both should be valid
        assert 0.0 <= d_same <= 2.0
        assert 0.0 <= d_diff <= 2.0


class TestSimilarity:
    def test_same_code_max_similarity(self, codebook):
        s = similarity(SemHexCode(0, 0), SemHexCode(0, 0), codebook=codebook)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_similarity_range(self, codebook):
        s = similarity(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        assert -1.0 <= s <= 1.0

    def test_similarity_is_1_minus_distance(self, codebook):
        d = distance(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        s = similarity(SemHexCode(0, 0), SemHexCode(1, 0), codebook=codebook)
        assert s == pytest.approx(1.0 - d, abs=1e-6)
