"""Evaluation: Distance correlation.

Tests whether SemHex code distance tracks actual semantic distance.
For sentence pairs: compute SemHex code distance AND embedding cosine distance,
then measure Spearman rank correlation between the two.

Target: >0.80 correlation (code distance tracks semantic distance).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from semhex.core.codebook import Codebook, load_codebook
from semhex.core.encoder import encode
from semhex.core.distance import distance as code_distance
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings import get_provider

# Sentence pairs for distance evaluation — varying similarity
DISTANCE_PAIRS = [
    # Very similar
    ("I'm happy today", "I feel joyful right now"),
    ("The server crashed", "The system went down"),
    ("Can you help me?", "I need your assistance"),
    ("The code has a bug", "There's an error in the program"),
    ("It's raining outside", "The weather is wet and rainy"),
    # Somewhat similar
    ("I'm writing Python code", "I'm developing software"),
    ("The meeting starts at noon", "The conference begins at midday"),
    ("She enjoys reading books", "He likes to read novels"),
    ("The car is fast", "The vehicle has high speed"),
    ("I'm cooking dinner", "I'm preparing the evening meal"),
    # Moderately different
    ("The stock market rose today", "I went for a walk in the park"),
    ("She plays the piano beautifully", "The database needs optimization"),
    ("The weather is sunny", "The algorithm is complex"),
    ("I love chocolate ice cream", "The project deadline is Friday"),
    ("The cat sleeps on the couch", "Quantum computing is advancing"),
    # Very different
    ("Fix this JavaScript error", "The sunset was beautiful over the ocean"),
    ("Deploy the microservice", "My grandmother makes excellent cookies"),
    ("The neural network converged", "The flowers bloom in spring"),
    ("Optimize the SQL query", "The children played in the park happily"),
    ("The API rate limit exceeded", "Philosophy explores the meaning of existence"),
]


@dataclass
class DistanceCorrelationResult:
    """Results of distance correlation evaluation."""
    n_pairs: int = 0
    spearman_r: float = 0.0
    spearman_p: float = 0.0
    code_distances: list[float] = field(default_factory=list)
    embedding_distances: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
            "elapsed_seconds": self.elapsed_seconds,
        }


def eval_distance_correlation(
    pairs: list[tuple[str, str]] | None = None,
    codebook: Codebook | None = None,
    provider: EmbeddingProvider | None = None,
) -> DistanceCorrelationResult:
    """Run distance correlation evaluation."""
    if pairs is None:
        pairs = DISTANCE_PAIRS
    if codebook is None:
        codebook = load_codebook("v0.1")
    if provider is None:
        provider = get_provider("auto")

    result = DistanceCorrelationResult(n_pairs=len(pairs))
    t0 = time.time()

    for sent_a, sent_b in pairs:
        # Embedding-space distance (ground truth)
        emb_a = provider.embed(sent_a)
        emb_b = provider.embed(sent_b)
        emb_dist = 1.0 - float(np.dot(emb_a, emb_b))
        result.embedding_distances.append(emb_dist)

        # SemHex code distance
        enc_a = encode(sent_a, codebook=codebook, provider=provider)
        enc_b = encode(sent_b, codebook=codebook, provider=provider)

        if enc_a.codes and enc_b.codes:
            # Use distance between first codes
            cd = code_distance(enc_a.codes[0], enc_b.codes[0], codebook=codebook)
            result.code_distances.append(cd)
        else:
            result.code_distances.append(1.0)

    # Compute Spearman rank correlation
    if len(result.code_distances) > 2:
        corr = stats.spearmanr(result.code_distances, result.embedding_distances)
        result.spearman_r = float(corr.statistic)
        result.spearman_p = float(corr.pvalue)

    result.elapsed_seconds = time.time() - t0
    return result


if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    console.print("[bold]Running distance correlation evaluation...[/bold]")
    result = eval_distance_correlation()

    console.print(f"\n[bold]Results ({result.n_pairs} pairs):[/bold]")
    console.print(f"  Spearman r:  [cyan]{result.spearman_r:.4f}[/cyan] (target: >0.80)")
    console.print(f"  p-value:     {result.spearman_p:.6f}")
    console.print(f"  Time:        {result.elapsed_seconds:.2f}s")
