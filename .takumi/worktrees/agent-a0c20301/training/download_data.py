"""Download diverse sentences from HuggingFace for codebook training.

Sources:
- ag_news: news headlines (topic diversity)
- emotion: tweets with emotion labels (emotional diversity)
- rotten_tomatoes: movie reviews (opinion diversity)
- financial_phrasebank: financial sentences (domain diversity)
- wiki_qa: question-answer pairs from Wikipedia (Q&A diversity)

Target: ~100K diverse sentences, filtered for quality.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

def download_sentences(output_path: str = "data/sentences_100k.jsonl", target: int = 100_000):
    from datasets import load_dataset
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    sentences = []

    def add_sentences(texts: list[str], source: str, limit: int):
        count = 0
        for text in texts:
            if len(sentences) >= target:
                break
            text = text.strip()
            # Quality filters
            if len(text.split()) < 5 or len(text.split()) > 80:
                continue
            if len(text) < 20 or len(text) > 500:
                continue
            # Deduplicate
            h = hashlib.md5(text.lower().encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            sentences.append({"text": text, "source": source})
            count += 1
            if count >= limit:
                break
        return count

    with Progress() as progress:
        task = progress.add_task("Downloading...", total=target)

        # 1. AG News — news headlines (25K)
        console.print("[bold]Loading ag_news...[/bold]")
        try:
            ds = load_dataset("ag_news", split="train", trust_remote_code=True)
            added = add_sentences([row["text"] for row in ds], "ag_news", 25000)
            console.print(f"  ag_news: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]ag_news failed: {e}[/red]")

        # 2. Emotion — tweets with emotion labels (20K)
        console.print("[bold]Loading emotion...[/bold]")
        try:
            ds = load_dataset("dair-ai/emotion", split="train", trust_remote_code=True)
            added = add_sentences([row["text"] for row in ds], "emotion", 20000)
            console.print(f"  emotion: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]emotion failed: {e}[/red]")

        # 3. Rotten Tomatoes — movie reviews (10K)
        console.print("[bold]Loading rotten_tomatoes...[/bold]")
        try:
            ds = load_dataset("rotten_tomatoes", split="train", trust_remote_code=True)
            added = add_sentences([row["text"] for row in ds], "rotten_tomatoes", 10000)
            console.print(f"  rotten_tomatoes: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]rotten_tomatoes failed: {e}[/red]")

        # 4. Financial PhraseBank (5K)
        console.print("[bold]Loading financial_phrasebank...[/bold]")
        try:
            ds = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
            added = add_sentences([row["sentence"] for row in ds], "financial_phrasebank", 5000)
            console.print(f"  financial_phrasebank: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]financial_phrasebank failed: {e}[/red]")

        # 5. STS Benchmark — sentence pairs for similarity eval (10K)
        console.print("[bold]Loading stsb...[/bold]")
        try:
            ds = load_dataset("mteb/stsbenchmark-sts", split="train", trust_remote_code=True)
            texts = []
            for row in ds:
                texts.append(row["sentence1"])
                texts.append(row["sentence2"])
            added = add_sentences(texts, "stsb", 10000)
            console.print(f"  stsb: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]stsb failed: {e}[/red]")

        # 6. WikiQA — questions (10K)
        console.print("[bold]Loading wiki_qa...[/bold]")
        try:
            ds = load_dataset("wiki_qa", split="train", trust_remote_code=True)
            texts = [row["question"] for row in ds] + [row["answer"] for row in ds]
            added = add_sentences(texts, "wiki_qa", 10000)
            console.print(f"  wiki_qa: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]wiki_qa failed: {e}[/red]")

        # 7. Tweet Eval — sentiment tweets (20K)
        console.print("[bold]Loading tweet_eval...[/bold]")
        try:
            ds = load_dataset("tweet_eval", "sentiment", split="train", trust_remote_code=True)
            added = add_sentences([row["text"] for row in ds], "tweet_eval", 20000)
            console.print(f"  tweet_eval: {added} sentences")
            progress.update(task, advance=added)
        except Exception as e:
            console.print(f"  [red]tweet_eval failed: {e}[/red]")

        progress.update(task, completed=len(sentences))

    # Save
    with open(out, "w") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    console.print(f"\n[bold green]Saved {len(sentences)} sentences to {out}[/bold green]")

    # Source breakdown
    sources = {}
    for s in sentences:
        sources[s["source"]] = sources.get(s["source"], 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        console.print(f"  {src}: {count}")

    return len(sentences)


if __name__ == "__main__":
    download_sentences()
