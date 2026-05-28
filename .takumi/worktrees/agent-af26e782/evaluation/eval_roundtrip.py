"""Evaluation: Roundtrip quality.

Measures how well encode→decode preserves meaning.
For each sentence: embed original, encode to SemHex, look up centroid vector,
compute cosine similarity between original embedding and centroid.

Metrics:
- Mean cosine similarity (target: >0.85 with real embeddings)
- Worst-case similarity
- Distribution histogram
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semhex.core.codebook import Codebook, load_codebook
from semhex.core.encoder import encode
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings import get_provider

# Diverse test sentences spanning many domains and styles
EVAL_SENTENCES = [
    # Requests
    "Can you help me with this problem?",
    "Please review my code and suggest improvements.",
    "I need someone to explain how this algorithm works.",
    # Emotions
    "I'm so frustrated that nothing is working correctly.",
    "This is absolutely wonderful news!",
    "I feel anxious about the upcoming deadline.",
    "I'm grateful for all the help you've provided.",
    # Technical
    "The database query is running slower than expected.",
    "We need to refactor the authentication middleware.",
    "The API endpoint returns a 404 error for valid requests.",
    "Memory usage increases linearly with each request.",
    # Knowledge
    "Photosynthesis converts sunlight into chemical energy.",
    "The Pythagorean theorem relates the sides of a right triangle.",
    "Inflation occurs when the general price level rises.",
    # Instructions
    "First, install the required dependencies using pip.",
    "Navigate to the settings page and click on security.",
    "Run the test suite before submitting your pull request.",
    # Opinions
    "I think the microservice architecture is overkill for this project.",
    "The new design is much more intuitive than the previous version.",
    "Performance optimization should be our top priority.",
    # Spatial/physical
    "The server is located in the data center on the third floor.",
    "Move the button to the top right corner of the screen.",
    # Temporal
    "The deployment is scheduled for next Tuesday at midnight.",
    "This bug has been present since the last release three weeks ago.",
    # Comparisons
    "Python is more readable than Java for data processing tasks.",
    "The new approach is significantly faster but uses more memory.",
    # Conditional
    "If the cache is stale, fetch fresh data from the database.",
    "The feature should only be enabled for premium users.",
    # Negation
    "This is not the correct way to handle authentication.",
    "The system should never expose user credentials in logs.",
    # Short
    "Help!",
    "Thank you.",
    "I disagree.",
    # Long
    "The machine learning model we trained on the customer feedback dataset is showing promising results with an accuracy of 94 percent on the validation set, but we need to investigate the false positive rate before deploying to production.",
    "After careful consideration of the trade-offs between consistency and availability in our distributed system, we decided to implement eventual consistency with a conflict resolution strategy based on last-writer-wins semantics.",
]


@dataclass
class RoundtripResult:
    """Results of roundtrip evaluation."""
    n_sentences: int = 0
    similarities: list[float] = field(default_factory=list)
    worst_cases: list[tuple[str, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def mean_similarity(self) -> float:
        return float(np.mean(self.similarities)) if self.similarities else 0.0

    @property
    def min_similarity(self) -> float:
        return float(np.min(self.similarities)) if self.similarities else 0.0

    @property
    def std_similarity(self) -> float:
        return float(np.std(self.similarities)) if self.similarities else 0.0

    def to_dict(self) -> dict:
        return {
            "n_sentences": self.n_sentences,
            "mean_similarity": self.mean_similarity,
            "min_similarity": self.min_similarity,
            "std_similarity": self.std_similarity,
            "elapsed_seconds": self.elapsed_seconds,
            "worst_cases": [{"sentence": s, "similarity": sim} for s, sim in self.worst_cases[:10]],
        }


def eval_roundtrip(
    sentences: list[str] | None = None,
    codebook: Codebook | None = None,
    provider: EmbeddingProvider | None = None,
) -> RoundtripResult:
    """Run roundtrip evaluation.

    For each sentence:
    1. Embed the original sentence
    2. Encode to SemHex codes
    3. Look up the centroid vectors for each code
    4. Average the centroid vectors
    5. Compute cosine similarity between original and averaged centroid
    """
    if sentences is None:
        sentences = EVAL_SENTENCES
    if codebook is None:
        codebook = load_codebook("v0.1")
    if provider is None:
        provider = get_provider("auto")

    result = RoundtripResult(n_sentences=len(sentences))
    t0 = time.time()

    for sentence in sentences:
        # Original embedding
        orig_emb = provider.embed(sentence)

        # Encode
        enc = encode(sentence, codebook=codebook, provider=provider)
        if not enc.codes:
            result.similarities.append(0.0)
            result.worst_cases.append((sentence, 0.0))
            continue

        # Reconstruct: average centroid vectors
        centroid_vecs = []
        for code in enc.codes:
            try:
                entry = codebook.lookup(code)
                centroid_vecs.append(entry.vector)
            except KeyError:
                pass

        if not centroid_vecs:
            result.similarities.append(0.0)
            result.worst_cases.append((sentence, 0.0))
            continue

        reconstructed = np.mean(centroid_vecs, axis=0)
        norm = np.linalg.norm(reconstructed)
        if norm > 0:
            reconstructed = reconstructed / norm

        # Cosine similarity
        sim = float(np.dot(orig_emb, reconstructed))
        sim = max(-1.0, min(1.0, sim))
        result.similarities.append(sim)

        if sim < 0.5:
            result.worst_cases.append((sentence, sim))

    result.elapsed_seconds = time.time() - t0
    result.worst_cases.sort(key=lambda x: x[1])

    return result


if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    console.print("[bold]Running roundtrip evaluation...[/bold]")
    result = eval_roundtrip()

    console.print(f"\n[bold]Results ({result.n_sentences} sentences):[/bold]")
    console.print(f"  Mean similarity: [cyan]{result.mean_similarity:.4f}[/cyan]")
    console.print(f"  Min similarity:  [cyan]{result.min_similarity:.4f}[/cyan]")
    console.print(f"  Std deviation:   [cyan]{result.std_similarity:.4f}[/cyan]")
    console.print(f"  Time:            {result.elapsed_seconds:.2f}s")

    if result.worst_cases:
        console.print(f"\n[bold]Worst cases:[/bold]")
        for sent, sim in result.worst_cases[:5]:
            console.print(f"  {sim:.4f} — {sent[:80]}")
