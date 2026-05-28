"""Benchmark: Speed and compression metrics.

Measures:
- Encoding speed (sentences/second)
- Compression ratio (input words / output codes)
- Codebook lookup latency
- Memory usage of codebook
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

from semhex.core.codebook import Codebook, load_codebook
from semhex.core.encoder import encode
from semhex.core.format import SemHexCode
from semhex.embeddings.base import EmbeddingProvider
from semhex.embeddings import get_provider

BENCHMARK_SENTENCES = [
    "Can you help me debug this async function?",
    "The database query is running too slowly.",
    "I feel frustrated with the lack of progress.",
    "Please deploy the new version to production.",
    "The machine learning model shows 94% accuracy on validation.",
    "We need to refactor the authentication system.",
    "The API returns inconsistent results for the same input.",
    "I think we should switch to a microservice architecture.",
    "The meeting is scheduled for next Tuesday at 3pm.",
    "This approach is significantly more efficient than the baseline.",
] * 10  # 100 sentences


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    n_sentences: int
    total_words: int
    total_codes: int
    compression_ratio: float
    encode_time: float
    encode_rate: float  # sentences/second
    lookup_time: float  # single lookup latency
    codebook_memory_mb: float

    def to_dict(self) -> dict:
        return {
            "n_sentences": self.n_sentences,
            "total_words": self.total_words,
            "total_codes": self.total_codes,
            "compression_ratio": self.compression_ratio,
            "encode_time_seconds": self.encode_time,
            "encode_rate_per_second": self.encode_rate,
            "lookup_latency_ms": self.lookup_time * 1000,
            "codebook_memory_mb": self.codebook_memory_mb,
        }


def run_benchmark(
    sentences: list[str] | None = None,
    codebook: Codebook | None = None,
    provider: EmbeddingProvider | None = None,
) -> BenchmarkResult:
    """Run performance benchmark."""
    if sentences is None:
        sentences = BENCHMARK_SENTENCES
    if codebook is None:
        codebook = load_codebook("v0.1")
    if provider is None:
        provider = get_provider("auto")

    # Encoding speed
    total_words = 0
    total_codes = 0
    t0 = time.time()
    for sentence in sentences:
        result = encode(sentence, codebook=codebook, provider=provider)
        total_words += sum(len(c.split()) for c in result.chunks)
        total_codes += len(result.codes)
    encode_time = time.time() - t0

    # Lookup latency (1000 lookups)
    t0 = time.time()
    for _ in range(1000):
        codebook.lookup(SemHexCode(0, 0))
    lookup_time = (time.time() - t0) / 1000

    # Memory
    l1_mem = codebook.l1_centroids.nbytes
    l2_mem = codebook.l2_centroids.nbytes
    total_mem = (l1_mem + l2_mem) / (1024 * 1024)

    return BenchmarkResult(
        n_sentences=len(sentences),
        total_words=total_words,
        total_codes=total_codes,
        compression_ratio=total_words / total_codes if total_codes > 0 else 0,
        encode_time=encode_time,
        encode_rate=len(sentences) / encode_time if encode_time > 0 else 0,
        lookup_time=lookup_time,
        codebook_memory_mb=total_mem,
    )


if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    console.print("[bold]Running benchmark...[/bold]")
    result = run_benchmark()

    console.print(f"\n[bold]Results ({result.n_sentences} sentences):[/bold]")
    console.print(f"  Compression:    [cyan]{result.compression_ratio:.1f}x[/cyan] ({result.total_words} words → {result.total_codes} codes)")
    console.print(f"  Encode speed:   [cyan]{result.encode_rate:.0f}[/cyan] sentences/sec")
    console.print(f"  Encode total:   {result.encode_time:.3f}s")
    console.print(f"  Lookup latency: [cyan]{result.lookup_time * 1000:.3f}ms[/cyan]")
    console.print(f"  Codebook RAM:   {result.codebook_memory_mb:.2f} MB")
