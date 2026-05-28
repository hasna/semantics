"""Tests for the encoder: text → SemHex codes."""

import pytest

from semhex.core.codebook import load_codebook
from semhex.core.encoder import encode, encode_batch, EncodeResult, _split_into_chunks
from semhex.core.format import SemHexCode
from semhex.embeddings.mock import MockEmbeddingProvider


@pytest.fixture
def mock_setup():
    """Setup with mock embeddings and the built codebook."""
    provider = MockEmbeddingProvider(dimensions=64)
    codebook = load_codebook("v0.1")
    return codebook, provider


class TestSplitIntoChunks:
    def test_empty(self):
        assert _split_into_chunks("") == []

    def test_short_text_single_chunk(self):
        chunks = _split_into_chunks("hello world")
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_single_sentence(self):
        chunks = _split_into_chunks("The cat sat on the mat.")
        assert len(chunks) == 1

    def test_two_sentences(self):
        chunks = _split_into_chunks("The cat sat on the mat. The dog lay on the rug nearby and watched.")
        assert len(chunks) == 2

    def test_newline_split(self):
        chunks = _split_into_chunks("First paragraph with enough words to be meaningful.\nSecond paragraph also long enough.")
        assert len(chunks) == 2

    def test_short_fragments_merged(self):
        chunks = _split_into_chunks("This is a longer sentence that makes sense. OK. And another sentence that continues on.")
        # "OK." should be merged since it's < 3 words
        assert all(len(c.split()) >= 3 for c in chunks)


class TestEncode:
    def test_basic_encode(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("hello world", codebook=codebook, provider=provider)

        assert isinstance(result, EncodeResult)
        assert len(result.codes) > 0
        assert all(isinstance(c, SemHexCode) for c in result.codes)

    def test_returns_l2_codes_by_default(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("The cat sat on the mat.", codebook=codebook, provider=provider)
        assert all(c.depth == 2 for c in result.codes)

    def test_depth_1(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("hello world", depth=1, codebook=codebook, provider=provider)
        assert all(c.depth == 1 for c in result.codes)

    def test_compression_ratio(self, mock_setup):
        codebook, provider = mock_setup
        text = "Can you help me fix this async function that is throwing a timeout error when fetching data from the API?"
        result = encode(text, codebook=codebook, provider=provider)

        # Multi-sentence text should produce fewer codes than words
        n_words = len(text.split())
        n_codes = len(result.codes)
        assert n_codes < n_words  # Compression!
        assert result.compression_ratio > 1.0

    def test_short_text_one_code(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("hello", codebook=codebook, provider=provider)
        assert len(result.codes) == 1

    def test_multi_sentence_multiple_codes(self, mock_setup):
        codebook, provider = mock_setup
        text = "The first sentence is about cats. The second sentence is about dogs. The third sentence is about birds."
        result = encode(text, codebook=codebook, provider=provider)
        assert len(result.codes) == 3  # One code per sentence
        assert len(result.chunks) == 3

    def test_empty_text(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("", codebook=codebook, provider=provider)
        assert result.codes == []
        assert result.chunks == []

    def test_distances_populated(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("test sentence", codebook=codebook, provider=provider)
        assert len(result.distances) == len(result.codes)
        assert all(isinstance(d, float) for d in result.distances)

    def test_code_strings(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("hello world", codebook=codebook, provider=provider)
        strings = result.code_strings
        assert len(strings) == len(result.codes)
        assert all(s.startswith("$") for s in strings)

    def test_str_representation(self, mock_setup):
        codebook, provider = mock_setup
        result = encode("hello world", codebook=codebook, provider=provider)
        s = str(result)
        assert s.startswith("$")

    def test_deterministic(self, mock_setup):
        """Same input should always produce the same codes."""
        codebook, provider = mock_setup
        r1 = encode("The quick brown fox", codebook=codebook, provider=provider)
        r2 = encode("The quick brown fox", codebook=codebook, provider=provider)
        assert r1.code_strings == r2.code_strings

    def test_different_inputs_different_codes(self, mock_setup):
        """Different meanings should produce different codes."""
        codebook, provider = mock_setup
        r1 = encode("I am very happy and excited about the good news", codebook=codebook, provider=provider)
        r2 = encode("The quantum physics equations are extremely complex and hard", codebook=codebook, provider=provider)
        # At least one code should differ (mock embeddings are hash-based, so different text = different vectors)
        assert r1.code_strings != r2.code_strings


class TestEncodeBatch:
    def test_batch(self, mock_setup):
        codebook, provider = mock_setup
        texts = ["hello", "world", "test"]
        results = encode_batch(texts, codebook=codebook, provider=provider)
        assert len(results) == 3
        assert all(isinstance(r, EncodeResult) for r in results)

    def test_batch_matches_single(self, mock_setup):
        codebook, provider = mock_setup
        texts = ["hello world", "goodbye world"]
        batch = encode_batch(texts, codebook=codebook, provider=provider)
        singles = [encode(t, codebook=codebook, provider=provider) for t in texts]

        for b, s in zip(batch, singles):
            assert b.code_strings == s.code_strings
