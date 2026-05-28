"""Tests for semhex public API (__init__.py).

Verifies that all top-level imports and lazy imports work correctly.
No LLM API keys required — all tests are pure or use mocks.
"""

import semhex
import pytest


class TestVersion:
    def test_version_exists(self):
        assert hasattr(semhex, "__version__")

    def test_version_is_string(self):
        assert isinstance(semhex.__version__, str)

    def test_version_semver(self):
        parts = semhex.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


class TestFormatAPI:
    def test_semhex_code_importable(self):
        assert semhex.SemHexCode is not None

    def test_parse_code_callable(self):
        code = semhex.parse_code("$1D.0003")
        assert code is not None

    def test_format_code_callable(self):
        code = semhex.parse_code("$1D.0003")
        result = semhex.format_code(code)
        assert isinstance(result, str)


class TestDictAPI:
    def test_dict_encode_importable(self):
        assert callable(semhex.dict_encode)

    def test_dict_decode_importable(self):
        assert callable(semhex.dict_decode)

    def test_dict_decode_detailed_importable(self):
        assert callable(semhex.dict_decode_detailed)

    def test_dict_encode_returns_string(self):
        result = semhex.dict_encode("hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dict_decode_returns_string(self):
        result = semhex.dict_decode("00.01")
        assert isinstance(result, str)

    def test_dict_roundtrip(self):
        text = "I am frustrated with this bug"
        codes = semhex.dict_encode(text)
        decoded = semhex.dict_decode(codes)
        assert "frustrated" in decoded
        assert "bug" in decoded

    def test_dict_decode_detailed_returns_dict(self):
        result = semhex.dict_decode_detailed("00.01")
        assert isinstance(result, dict)
        assert "text" in result
        assert "entries" in result
        assert "n_found" in result


class TestSemanticRGBAPI:
    def test_semantic_color_importable(self):
        assert semhex.SemanticColor is not None

    def test_rgb_encode_importable(self):
        assert callable(semhex.rgb_encode)

    def test_rgb_decode_importable(self):
        assert callable(semhex.rgb_decode)

    def test_rgb_encode_detailed_importable(self):
        assert callable(semhex.rgb_encode_detailed)

    def test_rgb_score_text_importable(self):
        assert callable(semhex.rgb_score_text)

    def test_domain_labels_importable(self):
        assert isinstance(semhex.DOMAIN_LABELS, dict)
        assert 8 in semhex.DOMAIN_LABELS

    def test_agent_labels_importable(self):
        assert isinstance(semhex.AGENT_LABELS, dict)

    def test_intent_labels_importable(self):
        assert isinstance(semhex.INTENT_LABELS, dict)

    def test_semantic_color_construct(self):
        c = semhex.SemanticColor(4, 5, 3, 0, 8, 0, 6)
        assert c.evaluation == 4
        assert c.domain == 8

    def test_rgb_decode_returns_string(self):
        c = semhex.SemanticColor(4, 5, 3, 0, 8, 0, 6)
        desc = semhex.rgb_decode(c.to_hex())
        assert isinstance(desc, str)
        assert "technology" in desc

    def test_rgb_decode_import_without_openai_dependency(self):
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "openai":
                raise ImportError("openai unavailable")
            return real_import(name, globals, locals, fromlist, level)

        with pytest.MonkeyPatch.context() as monkeypatch:
            for name in ["semhex", "semhex.core.semantic_rgb"]:
                sys.modules.pop(name, None)
            monkeypatch.setattr(builtins, "__import__", fake_import)
            module = importlib.import_module("semhex")

        desc = module.rgb_decode("$00.00.00")
        assert isinstance(desc, str)

    def test_rgb_decode_via_init(self):
        desc = semhex.rgb_decode("$00.00.00")
        assert isinstance(desc, str)


class TestLazyImports:
    def test_semhasher_lazy(self):
        SH = semhex.SemHasher
        assert SH is not None
        h = SH(n_dims=4, bits_per_dim=2)
        assert h.n_dims == 4

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            _ = semhex.this_does_not_exist

    def test_codebook_lazy(self):
        CB = semhex.Codebook
        assert CB is not None

    def test_distance_lazy(self):
        dist_fn = semhex.distance
        assert callable(dist_fn)

    def test_blend_lazy(self):
        blend_fn = semhex.blend
        assert callable(blend_fn)


class TestAllExports:
    """Verify __all__ is complete and all names are importable."""

    def test_all_defined(self):
        assert hasattr(semhex, "__all__")

    def test_all_names_exist(self):
        for name in semhex.__all__:
            assert hasattr(semhex, name), f"semhex.{name} not found but listed in __all__"
