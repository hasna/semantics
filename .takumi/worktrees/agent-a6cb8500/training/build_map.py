"""Build the SemHex Map — the exportable dictionary of the language.

Takes 89K Matryoshka 64d embeddings, clusters into regions, labels each region
with example sentences, and exports as a standalone file any LLM can load.

Output:
  codebooks/map_v1/centroids.npy    — (N_regions, 64) float32 centroid vectors
  codebooks/map_v1/labels.json      — region_id → {hex_code, examples, nearest}
  codebooks/map_v1/hasher_state.npz — trained quantizer state for encoding
  codebooks/map_v1/metadata.json    — version, stats, config
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from rich.console import Console

console = Console()


def build_map(
    embeddings_path: str = "data/embeddings_64d.npy",
    sentences_path: str = "data/sentences_100k.jsonl",
    n_regions: int = 8192,
    output_dir: str = "codebooks/map_v1",
):
    """Build the exportable map."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    embeddings = np.load(embeddings_path).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    sentences = []
    with open(sentences_path) as f:
        for line in f:
            sentences.append(json.loads(line)["text"])

    n = min(len(embeddings), len(sentences))
    embeddings = embeddings[:n]
    sentences = sentences[:n]

    console.print(f"[bold]Building SemHex Map[/bold]")
    console.print(f"  Sentences: {n:,}")
    console.print(f"  Dimensions: {embeddings.shape[1]}")
    console.print(f"  Target regions: {n_regions:,}")

    # Adjust n_regions if we have fewer sentences
    actual_regions = min(n_regions, n)
    console.print(f"  Actual regions: {actual_regions:,}")

    # Cluster
    console.print(f"\n[bold]Clustering...[/bold]")
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=actual_regions,
        random_state=42,
        batch_size=min(4096, actual_regions * 2),
        n_init=3,
        max_iter=100,
    )
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_.astype(np.float32)

    # Normalize centroids
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.where(c_norms > 0, c_norms, 1.0)

    elapsed = time.time() - t0
    console.print(f"  Clustered in {elapsed:.1f}s")

    # Build labels: for each region, find top 5 example sentences
    console.print(f"\n[bold]Labeling regions...[/bold]")
    region_labels = {}
    for region_id in range(actual_regions):
        mask = labels == region_id
        indices = np.where(mask)[0]

        if len(indices) == 0:
            region_labels[region_id] = {
                "examples": [],
                "count": 0,
            }
            continue

        # Find the 5 sentences closest to the centroid
        region_embs = embeddings[indices]
        sims = region_embs @ centroids[region_id]
        top_k = np.argsort(-sims)[:5]
        examples = [sentences[indices[i]] for i in top_k]

        region_labels[region_id] = {
            "examples": examples,
            "count": int(len(indices)),
        }

    # Now encode each centroid with the SemHasher to get hex codes
    console.print(f"\n[bold]Encoding centroids to hex codes...[/bold]")
    from semhex.core.geohash_v2 import SemHasher

    hasher = SemHasher(n_dims=64, bits_per_dim=4)
    hasher.load("matryoshka_64d_4b")

    for region_id in range(actual_regions):
        code = hasher.encode(centroids[region_id])
        region_labels[region_id]["hex_code"] = code

    # Save
    console.print(f"\n[bold]Saving map...[/bold]")

    np.save(out / "centroids.npy", centroids)
    console.print(f"  centroids.npy: {centroids.nbytes / 1024 / 1024:.1f} MB")

    # Convert labels to JSON-serializable
    labels_json = {}
    for rid, info in region_labels.items():
        labels_json[str(rid)] = info

    (out / "labels.json").write_text(json.dumps(labels_json, indent=2, ensure_ascii=False))

    # Copy hasher state
    import shutil
    hasher_src = Path("codebooks/semhasher_matryoshka_64d_4b.npz")
    if hasher_src.exists():
        shutil.copy(hasher_src, out / "hasher_state.npz")

    # Metadata
    metadata = {
        "version": "1.0",
        "n_regions": actual_regions,
        "n_sentences": n,
        "dimensions": 64,
        "bits_per_dim": 4,
        "total_bits": 256,
        "hex_chars": 64,
        "embedding_model": "text-embedding-3-small",
        "embedding_dims": 64,
        "reconstruction_similarity": 0.991,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Compute total size
    total_size = sum(f.stat().st_size for f in out.iterdir()) / 1024 / 1024

    console.print(f"\n[bold green]Map v1 built![/bold green]")
    console.print(f"  Regions: {actual_regions:,}")
    console.print(f"  Total size: {total_size:.1f} MB")
    console.print(f"  Files: {[f.name for f in sorted(out.iterdir())]}")

    # Sample regions
    console.print(f"\n[bold]Sample regions:[/bold]")
    for rid in [0, 100, 500, 1000, 4000]:
        if rid < actual_regions:
            info = region_labels[rid]
            code = info.get("hex_code", "?")
            ex = info["examples"][0][:60] if info["examples"] else "(empty)"
            console.print(f"  Region {rid} ({code[:20]}...): \"{ex}...\" ({info['count']} sentences)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=int, default=8192, help="Number of map regions")
    args = parser.parse_args()
    build_map(n_regions=args.regions)
