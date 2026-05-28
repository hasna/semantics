"""Train Residual Vector Quantization (RVQ) and measure the REAL scaling law.

Instead of one huge codebook, RVQ uses multiple levels:
- Level 1: find nearest centroid, record code
- Level 2: compute RESIDUAL (original - centroid), find nearest centroid of the residual
- Level 3: residual of residual...

Each level MULTIPLIES the compression quality instead of adding to it.

The scaling law: quality ∝ (1 - base_error)^n_levels
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


def train_rvq(
    embeddings: np.ndarray,
    codebook_size: int = 256,
    n_levels: int = 8,
    output_dir: str = "codebooks/rvq",
    results_path: str = "evaluation/results/rvq_scaling_results.json",
):
    """Train RVQ codebooks and measure quality at each level.

    This reveals the TRUE scaling law for SemHex.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    n_samples, dims = embeddings.shape
    console.print(f"[bold]RVQ Scaling Experiment[/bold]")
    console.print(f"  Samples: {n_samples:,}")
    console.print(f"  Dims: {dims}")
    console.print(f"  Codebook size per level: {codebook_size}")
    console.print(f"  Max levels: {n_levels}")

    results = []
    residuals = embeddings.copy()
    reconstructed = np.zeros_like(embeddings)
    all_codes = []  # list of arrays, one per level

    total_t0 = time.time()

    for level in range(n_levels):
        console.print(f"\n[bold]Level {level + 1}/{n_levels}[/bold]")
        t0 = time.time()

        # Train KMeans on the current residuals
        km = MiniBatchKMeans(
            n_clusters=codebook_size,
            random_state=42 + level,
            batch_size=min(4096, codebook_size * 4),
            n_init=3,
            max_iter=100,
        )
        labels = km.fit_predict(residuals)
        centroids = km.cluster_centers_.astype(np.float32)

        train_time = time.time() - t0

        # Save codebook for this level
        np.save(out / f"level_{level + 1}.npy", centroids)
        all_codes.append(labels)

        # Update reconstruction and residuals
        assigned = centroids[labels]
        reconstructed += assigned
        residuals = embeddings - reconstructed

        # Measure quality: cosine similarity between original and reconstruction
        # Normalize both for cosine similarity
        recon_norms = np.linalg.norm(reconstructed, axis=1, keepdims=True)
        recon_norms = np.where(recon_norms > 0, recon_norms, 1.0)
        recon_normalized = reconstructed / recon_norms

        orig_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        orig_norms = np.where(orig_norms > 0, orig_norms, 1.0)
        orig_normalized = embeddings / orig_norms

        similarities = np.sum(orig_normalized * recon_normalized, axis=1)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Also measure L2 reconstruction error (not just cosine)
        l2_errors = np.linalg.norm(residuals, axis=1)
        mean_l2 = float(np.mean(l2_errors))

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        min_sim = float(np.min(similarities))
        p5_sim = float(np.percentile(similarities, 5))
        p50_sim = float(np.percentile(similarities, 50))
        p95_sim = float(np.percentile(similarities, 95))

        # Total codes used so far
        total_codes = codebook_size * (level + 1)
        # Codes per sentence at this level
        codes_per_sentence = level + 1
        # Each code is log2(codebook_size) bits
        bits_per_code = np.log2(codebook_size)
        total_bits = codes_per_sentence * bits_per_code
        # Compare to original: ~1536 dims × 32 bits = 49,152 bits for full vector
        compression_ratio = 49152 / total_bits

        result = {
            "level": level + 1,
            "codebook_size": codebook_size,
            "total_codebook_entries": total_codes,
            "codes_per_sentence": codes_per_sentence,
            "bits_per_sentence": round(total_bits, 1),
            "compression_ratio_vs_vector": round(compression_ratio, 1),
            "mean_similarity": round(mean_sim, 6),
            "std_similarity": round(std_sim, 6),
            "min_similarity": round(min_sim, 6),
            "p5_similarity": round(p5_sim, 6),
            "p50_similarity": round(p50_sim, 6),
            "p95_similarity": round(p95_sim, 6),
            "mean_l2_error": round(mean_l2, 6),
            "train_time_seconds": round(train_time, 2),
        }
        results.append(result)

        console.print(f"  Cosine similarity: [cyan]{mean_sim:.6f}[/cyan]")
        console.print(f"  L2 error:          {mean_l2:.6f}")
        console.print(f"  P5/P50/P95:        {p5_sim:.4f} / {p50_sim:.4f} / {p95_sim:.4f}")
        console.print(f"  Codes/sentence:    {codes_per_sentence} ({total_bits:.0f} bits)")
        console.print(f"  Compression:       {compression_ratio:.0f}x (vs raw vector)")
        console.print(f"  Time:              {train_time:.1f}s")

    total_time = time.time() - total_t0

    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save the code assignments for the full dataset
    codes_array = np.array(all_codes).T  # shape (n_samples, n_levels)
    np.save(out / "codes.npy", codes_array)

    # Summary table
    table = Table(title="RVQ Scaling Law Results")
    table.add_column("Level", justify="right")
    table.add_column("Codes", justify="right")
    table.add_column("Bits", justify="right")
    table.add_column("Mean Sim", justify="right", style="cyan")
    table.add_column("P5", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Compression", justify="right")

    for r in results:
        table.add_row(
            str(r["level"]),
            str(r["codes_per_sentence"]),
            f"{r['bits_per_sentence']:.0f}",
            f"{r['mean_similarity']:.6f}",
            f"{r['p5_similarity']:.4f}",
            f"{r['p95_similarity']:.4f}",
            f"{r['compression_ratio_vs_vector']:.0f}x",
        )

    console.print(table)
    console.print(f"\nTotal time: {total_time:.1f}s")

    # Fit the RVQ scaling law
    levels = np.array([r["level"] for r in results])
    sims = np.array([r["mean_similarity"] for r in results])
    errors = 1.0 - sims

    # For RVQ, error should decay exponentially: error = a * r^level
    # log(error) = log(a) + level * log(r)
    log_err = np.log(errors + 1e-10)
    A = np.vstack([np.ones_like(levels, dtype=float), levels.astype(float)]).T
    coeffs = np.linalg.lstsq(A, log_err, rcond=None)[0]
    a = np.exp(coeffs[0])
    r = np.exp(coeffs[1])

    console.print(f"\n[bold]RVQ Scaling Law: error = {a:.4f} × {r:.4f}^level[/bold]")
    console.print(f"  Each level reduces error by factor: {r:.4f} ({(1-r)*100:.1f}% reduction)")
    console.print(f"\n  Extrapolations:")
    for target in [0.90, 0.95, 0.99, 0.999]:
        target_err = 1.0 - target
        if target_err > 0 and a > 0 and r > 0 and r < 1:
            levels_needed = np.log(target_err / a) / np.log(r)
            codes_needed = max(1, int(np.ceil(levels_needed)))
            bits_needed = codes_needed * np.log2(codebook_size)
            console.print(f"    sim={target:.3f} → {codes_needed} levels, {bits_needed:.0f} bits/sentence")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook-size", type=int, default=256, help="Entries per level")
    parser.add_argument("--levels", type=int, default=8, help="Max RVQ levels")
    args = parser.parse_args()

    embeddings = np.load("data/embeddings.npy").astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    console.print(f"Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]} dims)")

    train_rvq(
        embeddings,
        codebook_size=args.codebook_size,
        n_levels=args.levels,
    )
