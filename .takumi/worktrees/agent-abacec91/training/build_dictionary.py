"""Build the SemHex dictionary: every word → short hex code.

Assigns codes by frequency (Huffman-like):
  Top 256 words     → 2 hex chars (00-FF)
  Next 4,096 words  → 3 hex chars (100-FFF)
  Next 61,440 words → 4 hex chars (1000-FFFF)

Total: ~65K words covered with 2-4 hex char codes.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from rich.console import Console

console = Console()


def build_dictionary(
    n_words: int = 50000,
    sentences_path: str = "data/sentences_100k.jsonl",
    output_path: str = "codebooks/dictionary_v1.json",
):
    """Build the word→code dictionary."""
    from wordfreq import top_n_list, zipf_frequency

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Building SemHex Dictionary[/bold]")

    # Step 1: Get top words by frequency
    console.print(f"\nGetting top {n_words} English words by frequency...")
    words = top_n_list("en", n_words)
    console.print(f"  Got {len(words)} words")

    # Step 2: Also extract words from our actual training sentences
    console.print(f"\nExtracting words from training sentences...")
    sentence_words = Counter()
    n_sentences = 0
    with open(sentences_path) as f:
        for line in f:
            data = json.loads(line)
            tokens = data["text"].lower().split()
            for t in tokens:
                # Clean: strip punctuation
                clean = t.strip(".,!?;:\"'()[]{}…–—-/\\")
                if clean and len(clean) > 1 and clean.isalpha():
                    sentence_words[clean] += 1
            n_sentences += 1

    console.print(f"  {len(sentence_words)} unique words from {n_sentences} sentences")

    # Merge: wordfreq list + sentence words, deduplicated
    all_words = list(dict.fromkeys(words))  # preserve order from wordfreq (by frequency)
    # Add sentence-only words not in wordfreq list
    for w, count in sentence_words.most_common():
        if w not in set(all_words) and count >= 3:
            all_words.append(w)

    console.print(f"  Total unique words: {len(all_words)}")

    # Step 3: Assign hex codes by frequency rank
    console.print(f"\nAssigning hex codes...")

    word_to_code = {}
    code_to_word = {}
    code_idx = 0

    # Tier 1: top 256 → 2 hex chars (00-FF)
    tier1_end = min(256, len(all_words))
    for i in range(tier1_end):
        code = f"{i:02X}"
        word_to_code[all_words[i]] = code
        code_to_word[code] = all_words[i]
        code_idx += 1

    # Tier 2: next 3840 → 3 hex chars (100-FFF)
    tier2_start = 256
    tier2_end = min(256 + 3840, len(all_words))
    for i in range(tier2_start, tier2_end):
        hex_val = 0x100 + (i - tier2_start)
        code = f"{hex_val:03X}"
        word_to_code[all_words[i]] = code
        code_to_word[code] = all_words[i]
        code_idx += 1

    # Tier 3: rest → 4 hex chars (1000-FFFF)
    tier3_start = tier2_end
    tier3_end = min(tier3_start + 61440, len(all_words))
    for i in range(tier3_start, tier3_end):
        hex_val = 0x1000 + (i - tier3_start)
        code = f"{hex_val:04X}"
        word_to_code[all_words[i]] = code
        code_to_word[code] = all_words[i]
        code_idx += 1

    console.print(f"  Tier 1 (2 hex): {tier1_end} words")
    console.print(f"  Tier 2 (3 hex): {tier2_end - tier2_start} words")
    console.print(f"  Tier 3 (4 hex): {tier3_end - tier3_start} words")
    console.print(f"  Total mapped: {len(word_to_code)}")

    # Step 4: Show samples
    console.print(f"\n[bold]Sample codes:[/bold]")
    samples = ["the", "is", "a", "of", "to", "and", "in", "it", "you", "that",
               "help", "frustrated", "database", "error", "love", "happy",
               "beautiful", "question", "answer", "computer"]
    for w in samples:
        code = word_to_code.get(w, "?")
        console.print(f"  {w:20s} → {code}")

    # Step 5: Save
    dictionary = {
        "version": "1.0",
        "n_words": len(word_to_code),
        "tiers": {
            "2_hex": tier1_end,
            "3_hex": tier2_end - tier2_start,
            "4_hex": tier3_end - tier3_start,
        },
        "word_to_code": word_to_code,
        "code_to_word": code_to_word,
    }

    out.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2))
    size_mb = out.stat().st_size / 1024 / 1024
    console.print(f"\n[bold green]Dictionary saved: {out} ({size_mb:.1f} MB)[/bold green]")

    return dictionary


if __name__ == "__main__":
    build_dictionary()
