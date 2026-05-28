"""Evaluate the LLM codec: compress → decompress → measure quality.

Runs on a sample of sentences from the downloaded dataset.
Measures semantic similarity and compression ratio at each quality level.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from semhex.core.codec import compress, decompress, roundtrip

console = Console()


def load_sample(n: int = 50, path: str = "data/sentences_100k.jsonl") -> list[str]:
    """Load a random sample of sentences."""
    sentences = []
    with open(path) as f:
        for line in f:
            sentences.append(json.loads(line)["text"])

    rng = np.random.RandomState(42)
    indices = rng.choice(len(sentences), size=min(n, len(sentences)), replace=False)
    return [sentences[i] for i in indices]


def eval_codec(
    n_samples: int = 50,
    qualities: list[int] = [1, 2, 3],
    provider: str = "cerebras",
    output_path: str = "evaluation/results/codec_eval.json",
):
    """Run codec evaluation."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sentences = load_sample(n_samples)
    console.print(f"[bold]Codec Evaluation: {len(sentences)} sentences, qualities {qualities}[/bold]")

    all_results = []

    for q in qualities:
        console.print(f"\n[bold]Quality {q}[/bold]")
        similarities = []
        ratios = []
        failures = 0
        t0 = time.time()

        for i, text in enumerate(sentences):
            try:
                r = roundtrip(text, quality=q, provider=provider)
                sim = r["semantic_similarity"]
                if sim is not None:
                    similarities.append(sim)
                    ratios.append(r["compression_ratio"])
                else:
                    failures += 1
            except Exception as e:
                failures += 1
                if i < 3:
                    console.print(f"  [red]Error on sentence {i}: {e}[/red]")

            if (i + 1) % 10 == 0:
                avg = np.mean(similarities) if similarities else 0
                console.print(f"  [{i+1}/{len(sentences)}] avg_sim={avg:.4f} avg_ratio={np.mean(ratios) if ratios else 0:.1f}x")

        elapsed = time.time() - t0

        result = {
            "quality": q,
            "n_samples": len(sentences),
            "n_successful": len(similarities),
            "n_failures": failures,
            "mean_similarity": round(float(np.mean(similarities)), 4) if similarities else 0,
            "std_similarity": round(float(np.std(similarities)), 4) if similarities else 0,
            "min_similarity": round(float(np.min(similarities)), 4) if similarities else 0,
            "p25_similarity": round(float(np.percentile(similarities, 25)), 4) if similarities else 0,
            "median_similarity": round(float(np.median(similarities)), 4) if similarities else 0,
            "p75_similarity": round(float(np.percentile(similarities, 75)), 4) if similarities else 0,
            "mean_compression": round(float(np.mean(ratios)), 2) if ratios else 0,
            "elapsed_seconds": round(elapsed, 1),
        }
        all_results.append(result)

        console.print(f"  Mean similarity: [cyan]{result['mean_similarity']:.4f}[/cyan]")
        console.print(f"  Median:          [cyan]{result['median_similarity']:.4f}[/cyan]")
        console.print(f"  Mean compression: [cyan]{result['mean_compression']:.1f}x[/cyan]")
        console.print(f"  Failures:        {failures}")

    # Save
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    table = Table(title="Codec Evaluation Results")
    table.add_column("Quality")
    table.add_column("Mean Sim", style="cyan")
    table.add_column("Median Sim")
    table.add_column("Compression", style="cyan")
    table.add_column("Failures")
    table.add_column("Time")

    for r in all_results:
        table.add_row(
            str(r["quality"]),
            f"{r['mean_similarity']:.4f}",
            f"{r['median_similarity']:.4f}",
            f"{r['mean_compression']:.1f}x",
            str(r["n_failures"]),
            f"{r['elapsed_seconds']:.0f}s",
        )

    console.print(table)
    console.print(f"\n[bold green]Results saved to {output_path}[/bold green]")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--provider", default="cerebras")
    args = parser.parse_args()

    eval_codec(n_samples=args.samples, provider=args.provider)
