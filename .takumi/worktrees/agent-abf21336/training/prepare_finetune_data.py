"""Generate fine-tuning data for SemHex encoder/decoder models.

Creates JSONL files in OpenAI chat format:
1. Encoder training: text → codes
2. Decoder training: codes → text

Uses Cerebras to generate the compress/decompress pairs,
then formats them for fine-tuning via brains MCP.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from semhex.core.codec import compress

console = Console()

ENCODER_SYSTEM = "You are SemHex, a semantic text compressor. Compress the user's message into compact dot-separated alphanumeric codes that preserve meaning. Use consonant clusters: FRU=frustrated, HLP=help, BUG=bug, DB=database, PERF=performance, Q=question. Output ONLY the codes."

DECODER_SYSTEM = "You are a SemHex decoder. The user gives you compressed semantic codes (dot-separated alphanumeric tokens). Expand them back into natural, fluent text. Output ONLY the reconstructed text."


def prepare_finetune_data(
    input_path: str = "data/sentences_100k.jsonl",
    output_encoder: str = "data/finetune_encoder.jsonl",
    output_decoder: str = "data/finetune_decoder.jsonl",
    n_samples: int = 500,
    quality: int = 2,
    provider: str = "cerebras",
):
    """Generate encoder and decoder training pairs."""
    Path(output_encoder).parent.mkdir(parents=True, exist_ok=True)

    # Load sentences
    sentences = []
    with open(input_path) as f:
        for line in f:
            sentences.append(json.loads(line)["text"])

    # Take a diverse sample
    import numpy as np
    rng = np.random.RandomState(123)
    indices = rng.choice(len(sentences), size=min(n_samples, len(sentences)), replace=False)
    sample = [sentences[i] for i in indices]

    console.print(f"[bold]Generating {len(sample)} training pairs via {provider}...[/bold]")

    encoder_pairs = []
    decoder_pairs = []
    failures = 0
    t0 = time.time()

    with Progress() as progress:
        task = progress.add_task("Compressing...", total=len(sample))

        for i, text in enumerate(sample):
            try:
                codes = compress(text, quality=quality, provider=provider)

                if not codes or len(codes) < 2:
                    failures += 1
                    progress.update(task, advance=1)
                    continue

                # Encoder pair: text → codes
                encoder_pairs.append({
                    "messages": [
                        {"role": "system", "content": ENCODER_SYSTEM},
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": codes},
                    ]
                })

                # Decoder pair: codes → text
                decoder_pairs.append({
                    "messages": [
                        {"role": "system", "content": DECODER_SYSTEM},
                        {"role": "user", "content": codes},
                        {"role": "assistant", "content": text},
                    ]
                })

            except Exception as e:
                failures += 1
                if failures <= 3:
                    console.print(f"  [red]Error {i}: {e}[/red]")

            progress.update(task, advance=1)

            # Rate limit: ~2 requests/sec for Cerebras
            if provider == "cerebras" and i % 50 == 49:
                time.sleep(1)

    elapsed = time.time() - t0

    # Save
    with open(output_encoder, "w") as f:
        for pair in encoder_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    with open(output_decoder, "w") as f:
        for pair in decoder_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
    console.print(f"  Encoder pairs: {len(encoder_pairs)} → {output_encoder}")
    console.print(f"  Decoder pairs: {len(decoder_pairs)} → {output_decoder}")
    console.print(f"  Failures: {failures}")

    # Show samples
    console.print(f"\n[bold]Sample encoder pairs:[/bold]")
    for pair in encoder_pairs[:5]:
        text = pair["messages"][1]["content"][:60]
        codes = pair["messages"][2]["content"]
        console.print(f"  \"{text}...\" → {codes}")

    return len(encoder_pairs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--quality", type=int, default=2)
    parser.add_argument("--provider", default="cerebras")
    args = parser.parse_args()

    prepare_finetune_data(n_samples=args.samples, quality=args.quality, provider=args.provider)
