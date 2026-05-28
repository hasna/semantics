"""Embed all downloaded sentences using OpenAI text-embedding-3-small.

Produces:
- data/embeddings.npy — numpy array shape (N, 1536)
- data/sentences_indexed.jsonl — sentences with their embedding index
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress

console = Console()


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        secrets_path = Path.home() / ".secrets"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if line.startswith("export OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


def embed_sentences(
    input_path: str = "data/sentences_100k.jsonl",
    output_embeddings: str = "data/embeddings.npy",
    output_indexed: str = "data/sentences_indexed.jsonl",
    batch_size: int = 2000,
):
    """Embed all sentences via OpenAI API."""
    client = get_openai_client()

    # Load sentences
    sentences = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            sentences.append(data["text"])

    console.print(f"[bold]Embedding {len(sentences)} sentences via OpenAI text-embedding-3-small...[/bold]")
    console.print(f"  Estimated cost: ${len(sentences) * 5 / 1_000_000:.4f} (at $0.02/1M tokens, ~5 tokens/sentence avg)")

    all_embeddings = []
    t0 = time.time()

    with Progress() as progress:
        task = progress.add_task("Embedding...", total=len(sentences))

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Retry logic
            for attempt in range(3):
                try:
                    response = client.embeddings.create(
                        input=batch,
                        model="text-embedding-3-small",
                    )
                    vecs = [d.embedding for d in response.data]
                    all_embeddings.extend(vecs)
                    break
                except Exception as e:
                    if attempt < 2:
                        console.print(f"  [yellow]Retry {attempt + 1}: {e}[/yellow]")
                        time.sleep(2 ** attempt)
                    else:
                        raise

            progress.update(task, advance=len(batch))

    elapsed = time.time() - t0
    console.print(f"  Embedded in {elapsed:.1f}s ({len(sentences) / elapsed:.0f} sentences/sec)")

    # Convert to numpy and normalize
    embeddings = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    console.print(f"  Shape: {embeddings.shape}")
    console.print(f"  Size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Save
    np.save(output_embeddings, embeddings)
    console.print(f"  Saved embeddings to {output_embeddings}")

    # Save indexed sentences
    with open(output_indexed, "w") as f:
        for idx, text in enumerate(sentences):
            f.write(json.dumps({"idx": idx, "text": text}, ensure_ascii=False) + "\n")
    console.print(f"  Saved indexed sentences to {output_indexed}")

    return embeddings


if __name__ == "__main__":
    embed_sentences()
