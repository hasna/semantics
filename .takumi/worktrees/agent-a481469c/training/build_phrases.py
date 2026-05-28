"""Find frequent multi-word phrases and add them to the dictionary.

Scans training sentences for common n-grams (2-5 words).
Phrases that appear frequently enough get their own single code —
shorter than encoding each word separately.

"help me" appears 500 times → gets one 3-char code instead of AA.0C
"I'm sorry" appears 300 times → gets one code
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from rich.console import Console

console = Console()


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer — lowercase, strip punctuation."""
    words = text.lower().split()
    clean = []
    for w in words:
        w = w.strip(".,!?;:\"'()[]{}…–—-/\\")
        if w:
            clean.append(w)
    return clean


def build_phrases(
    sentences_path: str = "data/sentences_100k.jsonl",
    dict_path: str = "codebooks/dictionary_v1.json",
    min_count: int = 5,
    max_ngram: int = 4,
    max_phrases: int = 20000,
):
    """Find frequent phrases and add to dictionary."""

    console.print("[bold]Building phrase dictionary[/bold]")

    # Load existing dictionary
    dictionary = json.loads(Path(dict_path).read_text())
    word_to_code = dictionary["word_to_code"]
    code_to_word = dictionary["code_to_word"]
    existing_words = set(word_to_code.keys())

    # Count n-grams
    console.print(f"\nCounting n-grams (2-{max_ngram} words) from sentences...")
    ngram_counts: dict[int, Counter] = {n: Counter() for n in range(2, max_ngram + 1)}
    n_sentences = 0

    with open(sentences_path) as f:
        for line in f:
            data = json.loads(line)
            words = tokenize(data["text"])
            for n in range(2, max_ngram + 1):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    ngram_counts[n][phrase] += 1
            n_sentences += 1

    # Filter: only phrases that appear enough AND save space
    console.print(f"\nFiltering phrases (min_count={min_count})...")
    good_phrases = []

    for n in range(2, max_ngram + 1):
        for phrase, count in ngram_counts[n].most_common():
            if count < min_count:
                break

            words = phrase.split()
            # All words must be in dictionary
            if not all(w in word_to_code for w in words):
                continue

            # Calculate savings: encoding words separately vs one phrase code
            separate_len = sum(len(word_to_code[w]) for w in words) + len(words) - 1  # codes + dots
            # A phrase code will be 3-4 chars
            phrase_code_len = 3 if len(good_phrases) < 3840 else 4
            savings = separate_len - phrase_code_len

            if savings > 0:
                good_phrases.append({
                    "phrase": phrase,
                    "count": count,
                    "n_words": n,
                    "savings_per_use": savings,
                    "total_savings": savings * count,
                })

    # Sort by total savings (most beneficial first)
    good_phrases.sort(key=lambda x: -x["total_savings"])
    good_phrases = good_phrases[:max_phrases]

    console.print(f"  Found {len(good_phrases)} beneficial phrases")

    # Assign codes — continue from where word dictionary left off
    # Find the next available code
    max_existing = max(int(c, 16) for c in code_to_word.keys())
    next_code = max_existing + 1

    phrase_to_code = {}
    n_added = 0

    for p in good_phrases:
        phrase = p["phrase"]
        code = f"{next_code:04X}" if next_code >= 0x1000 else f"{next_code:03X}"
        phrase_to_code[phrase] = code
        word_to_code[phrase] = code
        code_to_word[code] = phrase
        next_code += 1
        n_added += 1

    console.print(f"  Added {n_added} phrase codes")

    # Update dictionary
    dictionary["word_to_code"] = word_to_code
    dictionary["code_to_word"] = code_to_word
    dictionary["n_words"] = len(word_to_code)
    dictionary["n_phrases"] = n_added
    dictionary["phrase_stats"] = {
        "total_phrases": n_added,
        "min_count": min_count,
        "max_ngram": max_ngram,
    }

    Path(dict_path).write_text(json.dumps(dictionary, ensure_ascii=False, indent=2))
    size_mb = Path(dict_path).stat().st_size / 1024 / 1024
    console.print(f"\n[bold green]Updated dictionary: {dict_path} ({size_mb:.1f} MB)[/bold green]")
    console.print(f"  Total entries: {len(word_to_code)} (words + phrases)")

    # Show top phrases
    console.print(f"\n[bold]Top 30 phrases by savings:[/bold]")
    for p in good_phrases[:30]:
        code = phrase_to_code[p["phrase"]]
        words = p["phrase"].split()
        separate = ".".join(word_to_code.get(w, "?") for w in words)
        console.print(f'  "{p["phrase"]}" (×{p["count"]}) → {code} (was: {separate}, saves {p["savings_per_use"]} chars/use)')

    return n_added


if __name__ == "__main__":
    build_phrases()
