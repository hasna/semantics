"""Dictionary encoder: text → SemHex code sequence.

Looks up words and phrases in the dictionary, outputs dot-separated hex codes.
Uses greedy longest-match: tries longest phrase first, falls back to single words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from functools import lru_cache

_DICT_PATH = Path(__file__).parent.parent.parent / "codebooks" / "dictionary_v1.json"
_dictionary: dict | None = None


def _load_dict() -> dict:
    global _dictionary
    if _dictionary is None:
        _dictionary = json.loads(_DICT_PATH.read_text())
    return _dictionary


def _tokenize(text: str) -> list[str]:
    """Split text into words, preserving punctuation as separate tokens."""
    # Split on whitespace, then separate punctuation
    tokens = []
    for word in text.split():
        # Pull off leading/trailing punctuation
        clean = word.strip(".,!?;:\"'()[]{}…–—-/\\")
        leading = word[:len(word) - len(word.lstrip(".,!?;:\"'()[]{}…–—-/\\"))]
        trailing = word[len(clean) + len(leading):]
        if leading:
            tokens.append(leading)
        if clean:
            tokens.append(clean)
        if trailing:
            tokens.append(trailing)
    return tokens


def dict_encode(text: str) -> str:
    """Encode text to SemHex code sequence using dictionary lookup.

    Uses greedy longest-match for phrases, falls back to single words.
    Unknown words are passed through as-is wrapped in brackets.

    Args:
        text: Input text.

    Returns:
        Dot-separated hex code string like "D019.1866.0D.09.AAA"
    """
    d = _load_dict()
    w2c = d["word_to_code"]

    tokens = _tokenize(text)
    lower_tokens = [t.lower() for t in tokens]

    codes = []
    i = 0
    while i < len(lower_tokens):
        matched = False

        # Try longest phrase first (4 words down to 2)
        for n in range(min(4, len(lower_tokens) - i), 1, -1):
            phrase = " ".join(lower_tokens[i:i+n])
            if phrase in w2c:
                codes.append(w2c[phrase])
                i += n
                matched = True
                break

        if not matched:
            # Single word lookup
            word = lower_tokens[i]
            if word in w2c:
                codes.append(w2c[word])
            elif word in {".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-"}:
                # Skip punctuation (grammar reconstructed by decoder)
                pass
            else:
                # Unknown word — pass through
                codes.append(f"[{tokens[i]}]")
            i += 1

    return ".".join(codes)


def dict_encode_detailed(text: str) -> dict:
    """Encode with detailed breakdown showing what each code maps to."""
    d = _load_dict()
    w2c = d["word_to_code"]
    c2w = d["code_to_word"]

    tokens = _tokenize(text)
    lower_tokens = [t.lower() for t in tokens]

    entries = []
    i = 0
    while i < len(lower_tokens):
        matched = False

        for n in range(min(4, len(lower_tokens) - i), 1, -1):
            phrase = " ".join(lower_tokens[i:i+n])
            if phrase in w2c:
                code = w2c[phrase]
                entries.append({"text": phrase, "code": code, "type": "phrase" if n > 1 else "word"})
                i += n
                matched = True
                break

        if not matched:
            word = lower_tokens[i]
            if word in w2c:
                entries.append({"text": word, "code": w2c[word], "type": "word"})
            elif word in {".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "-"}:
                entries.append({"text": word, "code": None, "type": "punct"})
            else:
                entries.append({"text": tokens[i], "code": f"[{tokens[i]}]", "type": "unknown"})
            i += 1

    code_str = ".".join(e["code"] for e in entries if e["code"] is not None)

    return {
        "input": text,
        "codes": code_str,
        "entries": entries,
        "input_chars": len(text),
        "code_chars": len(code_str),
        "compression_ratio": round(len(text) / max(len(code_str), 1), 2),
        "n_tokens": len(tokens),
        "n_codes": len([e for e in entries if e["code"] is not None]),
    }
