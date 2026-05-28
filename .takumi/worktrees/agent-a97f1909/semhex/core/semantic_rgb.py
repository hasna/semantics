"""Semantic RGB: The color language for meaning.

Like RGB encodes any color in 3 dimensions (R, G, B) → 6 hex chars,
Semantic RGB encodes any meaning in 7 dimensions → 6 hex chars.

Dimensions (24 bits total):
  Evaluation  (good↔bad):        4 bits [0-15]
  Potency     (strong↔weak):     3 bits [0-7]
  Activity    (active↔passive):  3 bits [0-7]
  Agent       (self→other→abstract): 3 bits [0-7]
  Domain      (topic category):  4 bits [0-15]
  Intent      (purpose):         3 bits [0-7]
  Specificity (vague→precise):   4 bits [0-15]

Format: $XX.XX.XX (6 hex chars, 3 bytes, like #RRGGBB)

Usage:
  encode("I'm frustrated with this bug") → $4B.C4.2F
  decode("$4B.C4.2F") → negative, strong, active, self, technology, express, specific
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from semhex.core.auth import load_api_key as _load_api_key

_cerebras_client: OpenAI | None = None
_openai_client: OpenAI | None = None


def _require_openai():
    if OpenAI is None:
        raise ImportError(
            "openai is required for Semantic RGB encoding. Install with: pip install semhex[openai]"
        )
    return OpenAI


def _get_cerebras() -> OpenAI:
    global _cerebras_client
    if _cerebras_client is None:
        api_key = _load_api_key("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not found")
        _cerebras_client = _require_openai()(base_url="https://api.cerebras.ai/v1", api_key=api_key)
    return _cerebras_client


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = _load_api_key("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        _openai_client = _require_openai()(api_key=api_key)
    return _openai_client


def _get_client(provider: str = "auto") -> tuple[OpenAI, str]:
    """Return (client, model) for the given provider. Falls back: cerebras → openai."""
    if provider in ("cerebras", "auto"):
        try:
            return _get_cerebras(), "qwen-3-235b-a22b-instruct-2507"
        except (ImportError, ValueError):
            if provider == "cerebras":
                raise
    # openai fallback
    return _get_openai(), "gpt-4o-mini"


# Dimension definitions
DIMENSIONS = {
    "evaluation": {"bits": 4, "min_label": "very negative", "max_label": "very positive", "scale": "0=very negative, 7-8=neutral, 15=very positive"},
    "potency": {"bits": 3, "min_label": "very weak", "max_label": "very strong", "scale": "0=very weak/gentle, 3-4=moderate, 7=very strong/intense"},
    "activity": {"bits": 3, "min_label": "very passive/static", "max_label": "very active/dynamic", "scale": "0=completely passive, 3-4=moderate, 7=very active"},
    "agent": {"bits": 3, "min_label": "first person/self", "max_label": "abstract/impersonal", "scale": "0=I/self, 1=you, 2=we, 3=specific person, 4=group, 5=thing, 6=concept, 7=universal/no agent"},
    "domain": {"bits": 4, "min_label": "personal/emotion", "max_label": "abstract/meta", "scale": "0=emotion, 1=body/health, 2=social, 3=family, 4=work, 5=money, 6=food, 7=nature, 8=technology, 9=science, 10=art, 11=politics, 12=education, 13=travel, 14=time, 15=abstract/philosophy"},
    "intent": {"bits": 3, "min_label": "express/state", "max_label": "command/demand", "scale": "0=express feeling, 1=state fact, 2=ask question, 3=request, 4=suggest, 5=warn, 6=instruct, 7=command"},
    "specificity": {"bits": 4, "min_label": "extremely vague", "max_label": "extremely specific", "scale": "0=maximally vague/generic, 7-8=moderate detail, 15=maximally specific/detailed"},
}

DIM_ORDER = ["evaluation", "potency", "activity", "agent", "domain", "intent", "specificity"]

DOMAIN_LABELS = {
    0: "emotion", 1: "body/health", 2: "social", 3: "family", 4: "work",
    5: "money", 6: "food", 7: "nature", 8: "technology", 9: "science",
    10: "art", 11: "politics", 12: "education", 13: "travel", 14: "time",
    15: "abstract",
}

INTENT_LABELS = {
    0: "express", 1: "state", 2: "question", 3: "request",
    4: "suggest", 5: "warn", 6: "instruct", 7: "command",
}

AGENT_LABELS = {
    0: "I/self", 1: "you", 2: "we", 3: "specific person",
    4: "group", 5: "thing", 6: "concept", 7: "universal",
}


@dataclass
class SemanticColor:
    """A meaning encoded as 7 dimensions, 24 bits."""
    evaluation: int   # 0-15
    potency: int      # 0-7
    activity: int     # 0-7
    agent: int        # 0-7
    domain: int       # 0-15
    intent: int       # 0-7
    specificity: int  # 0-15

    def to_hex(self) -> str:
        """Pack into 24-bit hex: $XX.XX.XX"""
        # Byte 1: evaluation(4) + potency(3) + activity_high(1)
        b1 = (self.evaluation << 4) | (self.potency << 1) | (self.activity >> 2)
        # Byte 2: activity_low(2) + agent(3) + domain_high(3)
        b2 = ((self.activity & 0x3) << 6) | (self.agent << 3) | (self.domain >> 1)
        # Byte 3: domain_low(1) + intent(3) + specificity(4)
        b3 = ((self.domain & 0x1) << 7) | (self.intent << 4) | self.specificity

        return f"${b1:02X}.{b2:02X}.{b3:02X}"

    @classmethod
    def from_hex(cls, code: str) -> SemanticColor:
        """Unpack from $XX.XX.XX"""
        code = code.strip()
        if code.startswith("$"):
            code = code[1:]
        parts = code.replace(".", "")
        if len(parts) != 6:
            raise ValueError(f"Expected 6 hex chars, got {len(parts)}: {code}")

        b1 = int(parts[0:2], 16)
        b2 = int(parts[2:4], 16)
        b3 = int(parts[4:6], 16)

        evaluation = (b1 >> 4) & 0xF
        potency = (b1 >> 1) & 0x7
        activity = ((b1 & 0x1) << 2) | ((b2 >> 6) & 0x3)
        agent = (b2 >> 3) & 0x7
        domain = ((b2 & 0x7) << 1) | ((b3 >> 7) & 0x1)
        intent = (b3 >> 4) & 0x7
        specificity = b3 & 0xF

        return cls(evaluation, potency, activity, agent, domain, intent, specificity)

    def describe(self) -> str:
        """Human-readable description of this semantic color."""
        ev = "very negative" if self.evaluation < 4 else "negative" if self.evaluation < 7 else "neutral" if self.evaluation < 9 else "positive" if self.evaluation < 12 else "very positive"
        po = "weak" if self.potency < 3 else "moderate" if self.potency < 5 else "strong"
        ac = "passive" if self.activity < 3 else "moderate" if self.activity < 5 else "active"
        ag = AGENT_LABELS.get(self.agent, "?")
        do = DOMAIN_LABELS.get(self.domain, "?")
        it = INTENT_LABELS.get(self.intent, "?")
        sp = "vague" if self.specificity < 5 else "moderate" if self.specificity < 10 else "specific"

        return f"{ev}, {po}, {ac} | {ag} | {do} | {it} | {sp}"


def score_text(text: str, provider: str = "auto") -> SemanticColor:
    """Use LLM to score a text on all 7 meaning dimensions.

    Tries Cerebras first, falls back to OpenAI on rate-limit errors.
    """
    import time

    dim_descriptions = "\n".join(
        f"- {name}: {d['scale']}"
        for name, d in DIMENSIONS.items()
    )

    prompt = f"""Score this text on 7 dimensions of meaning. Return ONLY a JSON object with integer scores.

Dimensions:
{dim_descriptions}

Text: "{text}"

Return JSON: {{"evaluation": N, "potency": N, "activity": N, "agent": N, "domain": N, "intent": N, "specificity": N}}
ONLY the JSON, nothing else."""

    providers_to_try: list[tuple[str, str]] = []
    if provider in ("cerebras", "auto"):
        providers_to_try.append(("cerebras", "qwen-3-235b-a22b-instruct-2507"))
    if provider in ("openai", "auto"):
        providers_to_try.append(("openai", "gpt-4o-mini"))

    last_error: Exception | None = None
    for pname, model in providers_to_try:
        try:
            client = _get_cerebras() if pname == "cerebras" else _get_openai()
            system_msg = "Return ONLY valid JSON. No thinking. /no_think" if pname == "cerebras" else "Return ONLY valid JSON."
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            if "<think>" in raw:
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

            scores = json.loads(raw)
            return SemanticColor(
                evaluation=max(0, min(15, int(scores.get("evaluation", 8)))),
                potency=max(0, min(7, int(scores.get("potency", 4)))),
                activity=max(0, min(7, int(scores.get("activity", 4)))),
                agent=max(0, min(7, int(scores.get("agent", 0)))),
                domain=max(0, min(15, int(scores.get("domain", 0)))),
                intent=max(0, min(7, int(scores.get("intent", 0)))),
                specificity=max(0, min(15, int(scores.get("specificity", 8)))),
            )
        except Exception as e:
            last_error = e
            continue  # try next provider

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


def encode(text: str) -> str:
    """Encode text to a 6-char semantic hex code. Like #FF0000 for colors."""
    color = score_text(text)
    return color.to_hex()


def decode(code: str) -> str:
    """Decode a 6-char semantic hex code to a human-readable description."""
    color = SemanticColor.from_hex(code)
    return color.describe()


def encode_detailed(text: str) -> dict:
    """Encode with full breakdown of each dimension."""
    color = score_text(text)
    code = color.to_hex()
    return {
        "input": text,
        "code": code,
        "description": color.describe(),
        "dimensions": {
            "evaluation": color.evaluation,
            "potency": color.potency,
            "activity": color.activity,
            "agent": color.agent,
            "domain": color.domain,
            "intent": color.intent,
            "specificity": color.specificity,
        },
        "input_chars": len(text),
        "code_chars": 9,  # $XX.XX.XX
        "compression_ratio": round(len(text) / 9, 1),
    }
