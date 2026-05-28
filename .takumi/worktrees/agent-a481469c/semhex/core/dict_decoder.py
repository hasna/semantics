"""Dictionary decoder: SemHex code sequence → text.

Looks up each code in the dictionary and joins into text.
"""

from __future__ import annotations

import json
from pathlib import Path

_DICT_PATH = Path(__file__).parent.parent.parent / "codebooks" / "dictionary_v1.json"
_dictionary: dict | None = None


def _load_dict() -> dict:
    global _dictionary
    if _dictionary is None:
        _dictionary = json.loads(_DICT_PATH.read_text())
    return _dictionary


def dict_decode(codes: str) -> str:
    """Decode a dot-separated SemHex code sequence back to text.

    Args:
        codes: Dot-separated hex codes like "D019.1866.0D.09.AAA"

    Returns:
        Decoded text like "i am frustrated with this error"
    """
    d = _load_dict()
    c2w = d["code_to_word"]

    parts = codes.split(".")
    words = []

    for code in parts:
        code = code.strip()
        if not code:
            continue
        if code.startswith("[") and code.endswith("]"):
            # Passthrough unknown word
            words.append(code[1:-1])
        elif code in c2w:
            words.append(c2w[code])
        else:
            words.append(f"[?{code}]")

    return " ".join(words)


def dict_decode_detailed(codes: str) -> dict:
    """Decode with detailed breakdown."""
    d = _load_dict()
    c2w = d["code_to_word"]

    parts = codes.split(".")
    entries = []

    for code in parts:
        code = code.strip()
        if not code:
            continue
        if code.startswith("[") and code.endswith("]"):
            entries.append({"code": code, "text": code[1:-1], "found": False})
        elif code in c2w:
            entries.append({"code": code, "text": c2w[code], "found": True})
        else:
            entries.append({"code": code, "text": f"[?{code}]", "found": False})

    text = " ".join(e["text"] for e in entries)

    return {
        "codes": codes,
        "text": text,
        "entries": entries,
        "n_codes": len(entries),
        "n_found": sum(1 for e in entries if e["found"]),
        "n_unknown": sum(1 for e in entries if not e["found"]),
    }
