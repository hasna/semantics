"""Shared test fixtures for SemHex tests.

Builds a small mock codebook (64 dims, 64 L1, 32 L2) that all tests use.
This avoids depending on the real v0.1 codebook which may be 1536 dims.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from semhex.core.codebook import Codebook


@pytest.fixture(scope="session", autouse=True)
def mock_codebook(tmp_path_factory):
    """Build a test codebook and set it as v0.1."""
    tmp = tmp_path_factory.mktemp("codebooks")
    v01 = tmp / "v0.1"
    v01.mkdir()

    dims = 64
    n_l1 = 16
    n_l2_per_l1 = 8
    rng = np.random.RandomState(42)

    # L1 centroids
    l1 = rng.randn(n_l1, dims).astype(np.float32)
    l1 /= np.linalg.norm(l1, axis=1, keepdims=True)
    np.save(v01 / "level1.npy", l1)

    # L2 centroids
    l2 = rng.randn(n_l1 * n_l2_per_l1, dims).astype(np.float32)
    l2 /= np.linalg.norm(l2, axis=1, keepdims=True)
    np.save(v01 / "level2.npy", l2)

    # Labels
    l1_labels = {f"${i:02X}": f"cat_{i}" for i in range(n_l1)}
    l2_labels = {f"${i:02X}.{j:04X}": f"sub_{i}_{j}" for i in range(n_l1) for j in range(n_l2_per_l1)}
    l1_examples = {f"${i:02X}": [f"ex_{i}_1", f"ex_{i}_2"] for i in range(n_l1)}
    l2_examples = {f"${i:02X}.{j:04X}": [f"ex_{i}_{j}"] for i in range(n_l1) for j in range(n_l2_per_l1)}

    labels = {"l1": l1_labels, "l2": l2_labels, "l1_examples": l1_examples, "l2_examples": l2_examples}
    (v01 / "labels.json").write_text(json.dumps(labels))
    (v01 / "metadata.json").write_text(json.dumps({
        "version": "test", "dimensions": dims, "n_l1": n_l1, "n_l2_per_l1": n_l2_per_l1,
    }))

    # Monkey-patch the default codebook directory
    import semhex.core.codebook as cb_mod
    cb_mod._CODEBOOK_DIR = tmp

    # Clear any cached defaults in encoder/decoder
    import semhex.core.encoder as enc_mod
    import semhex.core.decoder as dec_mod
    enc_mod._default_codebook = None
    enc_mod._default_provider = None
    dec_mod._default_codebook = None

    return tmp
