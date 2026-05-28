"""Build a SemHex codebook from seed concepts.

Pipeline:
1. Load seed concepts
2. Embed all concepts using an embedding model
3. Run KMeans to find Level 1 centroids (256 clusters)
4. For each L1 cluster, run KMeans to find Level 2 sub-centroids (256 per cluster)
5. Save frozen codebook as numpy arrays + JSON labels

Usage:
    python -m training.build_codebook --provider mock --output codebooks/v0.1
    python -m training.build_codebook --provider local --output codebooks/v0.1
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semhex.embeddings import get_provider
from training.concepts import get_all_concepts


def build_codebook(
    provider_name: str = "mock",
    n_l1: int = 256,
    n_l2_per_l1: int = 256,
    output_dir: str = "codebooks/v0.1",
) -> None:
    """Build a 2-level codebook from seed concepts.

    Args:
        provider_name: Embedding provider ("mock", "local", "openai")
        n_l1: Number of Level 1 clusters
        n_l2_per_l1: Number of Level 2 sub-clusters per L1 cluster
        output_dir: Where to save the codebook files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load concepts
    concepts = get_all_concepts()
    print(f"Loaded {len(concepts)} seed concepts")

    # 2. Embed
    provider = get_provider(provider_name)
    print(f"Embedding with {provider.name} ({provider.dimensions} dims)...")
    t0 = time.time()
    embeddings = provider.embed_batch(concepts)
    print(f"  Embedded in {time.time() - t0:.1f}s → shape {embeddings.shape}")

    dims = embeddings.shape[1]

    # Adjust cluster counts if we have fewer concepts than clusters
    actual_n_l1 = min(n_l1, len(concepts))
    print(f"  Using {actual_n_l1} L1 clusters (requested {n_l1})")

    # 3. Level 1: KMeans on all embeddings
    print(f"Training Level 1 ({actual_n_l1} clusters)...")
    t0 = time.time()
    km_l1 = MiniBatchKMeans(n_clusters=actual_n_l1, random_state=42, batch_size=256, n_init=3)
    l1_assignments = km_l1.fit_predict(embeddings)
    l1_centroids = km_l1.cluster_centers_.astype(np.float32)

    # Normalize L1 centroids
    norms = np.linalg.norm(l1_centroids, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    l1_centroids = l1_centroids / norms
    print(f"  L1 done in {time.time() - t0:.1f}s")

    # 4. Level 2: KMeans within each L1 cluster
    print(f"Training Level 2 ({n_l2_per_l1} sub-clusters per L1)...")
    t0 = time.time()

    # Pre-allocate L2 centroids array
    l2_centroids = np.zeros((actual_n_l1 * n_l2_per_l1, dims), dtype=np.float32)
    l2_labels_map = {}
    l1_labels_map = {}
    l1_examples_map = {}
    l2_examples_map = {}

    for l1_idx in range(actual_n_l1):
        mask = l1_assignments == l1_idx
        cluster_embeddings = embeddings[mask]
        cluster_concepts = [concepts[i] for i in range(len(concepts)) if mask[i]]

        l1_code = f"${l1_idx:02X}"
        l1_labels_map[l1_code] = cluster_concepts[0] if cluster_concepts else f"cluster_{l1_idx}"
        l1_examples_map[l1_code] = cluster_concepts[:5]

        actual_n_l2 = min(n_l2_per_l1, len(cluster_concepts))
        if actual_n_l2 == 0:
            continue

        if actual_n_l2 < 2:
            # Only one concept in this cluster — use it as the sole L2 centroid
            start = l1_idx * n_l2_per_l1
            l2_centroids[start] = cluster_embeddings[0]
            l2_code = f"${l1_idx:02X}.{0:04X}"
            l2_labels_map[l2_code] = cluster_concepts[0]
            l2_examples_map[l2_code] = cluster_concepts[:3]
            continue

        km_l2 = MiniBatchKMeans(n_clusters=actual_n_l2, random_state=42, batch_size=max(64, actual_n_l2), n_init=3)
        l2_assignments = km_l2.fit_predict(cluster_embeddings)
        l2_sub_centroids = km_l2.cluster_centers_.astype(np.float32)

        # Normalize
        norms = np.linalg.norm(l2_sub_centroids, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        l2_sub_centroids = l2_sub_centroids / norms

        start = l1_idx * n_l2_per_l1
        l2_centroids[start:start + actual_n_l2] = l2_sub_centroids

        # Label each L2 cluster with its closest concept
        for l2_idx in range(actual_n_l2):
            sub_mask = l2_assignments == l2_idx
            sub_concepts = [cluster_concepts[j] for j in range(len(cluster_concepts)) if sub_mask[j]]
            l2_code = f"${l1_idx:02X}.{l2_idx:04X}"
            l2_labels_map[l2_code] = sub_concepts[0] if sub_concepts else f"sub_{l1_idx}_{l2_idx}"
            l2_examples_map[l2_code] = sub_concepts[:3]

    print(f"  L2 done in {time.time() - t0:.1f}s")

    # 5. Save
    print(f"Saving to {output_path}/")
    np.save(output_path / "level1.npy", l1_centroids)
    np.save(output_path / "level2.npy", l2_centroids)

    labels = {
        "l1": l1_labels_map,
        "l2": l2_labels_map,
        "l1_examples": l1_examples_map,
        "l2_examples": l2_examples_map,
    }
    (output_path / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    metadata = {
        "version": "0.1.0",
        "provider": provider_name,
        "dimensions": dims,
        "n_l1": actual_n_l1,
        "n_l2_per_l1": n_l2_per_l1,
        "n_concepts": len(concepts),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    total_codes = actual_n_l1 + actual_n_l1 * n_l2_per_l1
    print(f"\nCodebook v0.1 built:")
    print(f"  L1 clusters: {actual_n_l1}")
    print(f"  L2 clusters: {actual_n_l1 * n_l2_per_l1}")
    print(f"  Dimensions: {dims}")
    print(f"  Total addressable codes: {total_codes:,}")
    print(f"  Files: level1.npy, level2.npy, labels.json, metadata.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SemHex codebook")
    parser.add_argument("--provider", default="mock", help="Embedding provider: mock, local, openai")
    parser.add_argument("--output", default="codebooks/v0.1", help="Output directory")
    parser.add_argument("--n-l1", type=int, default=256, help="Number of L1 clusters")
    parser.add_argument("--n-l2", type=int, default=256, help="Number of L2 sub-clusters per L1")
    args = parser.parse_args()

    build_codebook(
        provider_name=args.provider,
        n_l1=args.n_l1,
        n_l2_per_l1=args.n_l2,
        output_dir=args.output,
    )
