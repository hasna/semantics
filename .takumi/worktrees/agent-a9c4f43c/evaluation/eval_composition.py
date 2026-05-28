"""Evaluation: Compositionality (nero's test).

Tests whether code arithmetic produces semantically valid compositions.
For concept pairs (A, B): blend(A, B) should produce a code whose label
is semantically related to the composition of A and B.

Approach:
- For each pair (A, B), compute blend(A, B)
- Embed labels of A, B, and blend result
- Check: is the blend's embedding closer to a valid composition than to random?
- Score: cosine similarity between blend embedding and average(A, B) embeddings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from semhex.core.blend import blend
from semhex.core.codebook import Codebook, load_codebook
from semhex.core.format import SemHexCode
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings import get_provider

# Concept pairs to test composition
COMPOSITION_PAIRS = [
    # (code_a_l1, code_a_l2, code_b_l1, code_b_l2) — using codebook indices
    # We test all unique pairs from the first N L1 clusters
]


@dataclass
class CompositionResult:
    """Results of composition evaluation."""
    n_pairs: int = 0
    valid_count: int = 0
    similarities: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def validity_rate(self) -> float:
        return self.valid_count / self.n_pairs if self.n_pairs > 0 else 0.0

    @property
    def mean_similarity(self) -> float:
        return float(np.mean(self.similarities)) if self.similarities else 0.0

    def to_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "valid_count": self.valid_count,
            "validity_rate": self.validity_rate,
            "mean_similarity": self.mean_similarity,
            "elapsed_seconds": self.elapsed_seconds,
        }


def eval_composition(
    n_pairs: int = 200,
    codebook: Codebook | None = None,
    provider: EmbeddingProvider | None = None,
    similarity_threshold: float = 0.3,
) -> CompositionResult:
    """Run composition evaluation.

    Generates random pairs of L1 codes, blends them, and checks if the result
    is semantically "between" the inputs.

    A blend is "valid" if the blended code's vector is closer to the average
    of the two input vectors than to a random code's vector.
    """
    if codebook is None:
        codebook = load_codebook("v0.1")
    if provider is None:
        provider = get_provider("auto")

    result = CompositionResult()
    t0 = time.time()

    n_l1 = codebook.n_level1
    rng = np.random.RandomState(42)

    # Generate random pairs
    pairs = []
    for _ in range(n_pairs):
        a = rng.randint(0, n_l1)
        b = rng.randint(0, n_l1)
        if a != b:
            pairs.append((a, b))

    result.n_pairs = len(pairs)

    for a_l1, b_l1 in pairs:
        code_a = SemHexCode(a_l1, 0)
        code_b = SemHexCode(b_l1, 0)

        try:
            entry_a = codebook.lookup(code_a)
            entry_b = codebook.lookup(code_b)
        except KeyError:
            continue

        # Expected: midpoint between A and B
        expected = 0.5 * entry_a.vector + 0.5 * entry_b.vector
        norm = np.linalg.norm(expected)
        if norm > 0:
            expected = expected / norm

        # Actual: blend result
        blended_code = blend(code_a, code_b, codebook=codebook)
        try:
            blended_entry = codebook.lookup(blended_code)
        except KeyError:
            continue

        # Similarity between blended vector and expected midpoint
        sim = float(np.dot(blended_entry.vector, expected))
        sim = max(-1.0, min(1.0, sim))
        result.similarities.append(sim)

        # Random baseline: pick a random code
        random_l1 = rng.randint(0, n_l1)
        random_code = SemHexCode(random_l1, 0)
        try:
            random_entry = codebook.lookup(random_code)
            random_sim = float(np.dot(random_entry.vector, expected))
        except KeyError:
            random_sim = 0.0

        # Valid if blend is closer to expected than random
        if sim > random_sim:
            result.valid_count += 1

    result.elapsed_seconds = time.time() - t0
    return result


if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    console.print("[bold]Running composition evaluation (nero's test)...[/bold]")
    result = eval_composition(n_pairs=500)

    console.print(f"\n[bold]Results ({result.n_pairs} pairs):[/bold]")
    console.print(f"  Validity rate:    [cyan]{result.validity_rate:.1%}[/cyan] (target: >75%)")
    console.print(f"  Valid pairs:      {result.valid_count}/{result.n_pairs}")
    console.print(f"  Mean similarity:  [cyan]{result.mean_similarity:.4f}[/cyan]")
    console.print(f"  Time:             {result.elapsed_seconds:.2f}s")
