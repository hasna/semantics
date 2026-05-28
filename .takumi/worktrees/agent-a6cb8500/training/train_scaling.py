"""Train VQ codebooks at multiple sizes and measure reconstruction quality.

This is the scaling law experiment:
- For each codebook size K: train KMeans, measure how well centroids reconstruct original embeddings
- Plot codebook_size vs reconstruction_quality
- Find the power law: quality = 1 - C / K^alpha

Usage:
    python -m training.train_scaling
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from rich.console import Console
from rich.table import Table

console = Console()

CODEBOOK_SIZES = [256, 1024, 4096, 16384, 65536]


def train_and_measure(
    embeddings: np.ndarray,
    codebook_sizes: list[int] = CODEBOOK_SIZES,
    output_dir: str = "codebooks/scaling",
    results_path: str = "evaluation/results/scaling_results.json",
):
    """Train codebooks at multiple sizes and measure reconstruction quality."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    n_samples, dims = embeddings.shape
    console.print(f"[bold]Scaling law experiment: {n_samples} embeddings, {dims} dims[/bold]")
    console.print(f"Codebook sizes: {codebook_sizes}")

    results = []

    for K in codebook_sizes:
        console.print(f"\n[bold]Training K={K}...[/bold]")
        t0 = time.time()

        # Train KMeans
        km = MiniBatchKMeans(
            n_clusters=K,
            random_state=42,
            batch_size=min(4096, K * 4),
            n_init=3,
            max_iter=100,
        )
        labels = km.fit_predict(embeddings)
        centroids = km.cluster_centers_.astype(np.float32)

        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        centroids = centroids / norms

        train_time = time.time() - t0

        # Measure reconstruction quality
        # For each embedding, find its assigned centroid and compute cosine similarity
        assigned_centroids = centroids[labels]  # (N, dims)

        # Cosine similarity (both are normalized)
        similarities = np.sum(embeddings * assigned_centroids, axis=1)
        similarities = np.clip(similarities, -1.0, 1.0)

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        min_sim = float(np.min(similarities))
        p5_sim = float(np.percentile(similarities, 5))
        p50_sim = float(np.percentile(similarities, 50))
        p95_sim = float(np.percentile(similarities, 95))
        reconstruction_error = 1.0 - mean_sim

        result = {
            "codebook_size": K,
            "mean_similarity": round(mean_sim, 6),
            "std_similarity": round(std_sim, 6),
            "min_similarity": round(min_sim, 6),
            "p5_similarity": round(p5_sim, 6),
            "p50_similarity": round(p50_sim, 6),
            "p95_similarity": round(p95_sim, 6),
            "reconstruction_error": round(reconstruction_error, 6),
            "train_time_seconds": round(train_time, 2),
            "inertia": round(float(km.inertia_), 2),
        }
        results.append(result)

        console.print(f"  Mean similarity: [cyan]{mean_sim:.6f}[/cyan]")
        console.print(f"  Error (1-sim):   [cyan]{reconstruction_error:.6f}[/cyan]")
        console.print(f"  P5/P50/P95:      {p5_sim:.4f} / {p50_sim:.4f} / {p95_sim:.4f}")
        console.print(f"  Time:            {train_time:.1f}s")

        # Save codebook
        np.save(out / f"codebook_{K}.npy", centroids)

    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]Results saved to {results_path}[/bold green]")

    # Summary table
    table = Table(title="Scaling Law Results")
    table.add_column("K", justify="right")
    table.add_column("Mean Sim", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("P5", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        table.add_row(
            f"{r['codebook_size']:,}",
            f"{r['mean_similarity']:.6f}",
            f"{r['reconstruction_error']:.6f}",
            f"{r['p5_similarity']:.4f}",
            f"{r['p95_similarity']:.4f}",
            f"{r['train_time_seconds']:.1f}s",
        )

    console.print(table)

    # Quick power law estimate
    if len(results) >= 3:
        Ks = np.array([r["codebook_size"] for r in results], dtype=np.float64)
        errors = np.array([r["reconstruction_error"] for r in results], dtype=np.float64)

        # Fit log(error) = log(C) - alpha * log(K)
        log_K = np.log(Ks)
        log_err = np.log(errors)
        # Linear regression in log space
        A = np.vstack([np.ones_like(log_K), log_K]).T
        coeffs = np.linalg.lstsq(A, log_err, rcond=None)[0]
        log_C = coeffs[0]
        alpha = -coeffs[1]
        C = np.exp(log_C)

        console.print(f"\n[bold]Power law fit: error = {C:.4f} / K^{alpha:.4f}[/bold]")
        console.print(f"  Extrapolation:")
        for target_sim in [0.95, 0.99, 0.999]:
            target_err = 1.0 - target_sim
            K_needed = (C / target_err) ** (1 / alpha)
            console.print(f"    {target_sim:.3f} similarity → K = {K_needed:,.0f}")

    return results


if __name__ == "__main__":
    embeddings_path = "data/embeddings.npy"
    console.print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)

    # Normalize just in case
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    console.print(f"  Shape: {embeddings.shape}")

    train_and_measure(embeddings)
