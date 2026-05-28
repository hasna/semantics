# SemHex — The Plan

## The Big Idea

SemHex is a NEW LANGUAGE. Every word and common phrase gets a short code (2-4 hex chars). An AI trained in this language outputs codes instead of words. A decoder translates the codes back to any human language.

It's like teaching an AI a new language. It already knows English, French, Mandarin. Now it also knows SemHex — where "frustrated" = 3F, "bug" = B1, "I'm frustrated" = A7F. You just ask it to speak SemHex instead of English.

```
ENGLISH:  "I'm frustrated with this bug. Can someone help me fix it?"
          65 characters, ~15 tokens

SEMHEX:   A7F.09.B1.C8.3A.F7A
          "I'm frustrated" . "this" . "bug" . "can someone" . "help" . "fix it"
          22 characters, 6 codes

COMPRESSION: 3x fewer characters, 2.5x fewer tokens
```

The decoder on the other end:
```
A7F.09.B1.C8.3A.F7A → "I'm frustrated with this bug. Can someone help me fix it?"
```

## How It Works

### Step 1: Build the Dictionary (THE MAP)

Map every word and common phrase in every language to a short hex code:

```
WORDS (100K entries, 2-3 hex chars each):
  A7 = I / I'm / I am
  3F = frustrated / annoyed / angry
  C2 = with
  09 = this / that
  B1 = bug / error / issue
  3A = help / assist
  ...

PHRASES (500K+ entries, 3-4 hex chars each — merged common combos):
  A7F = "I'm frustrated"     (replaces A7 + 3F)
  F7A = "fix it" / "fix this" (replaces two words)
  C8A = "can someone"         (replaces two words)
  ...
```

The more phrases you merge, the more compression you get. This is BPE but for MEANING — frequent phrases earn their own code.

### Step 2: Train the AI to Speak SemHex

Like how you'd teach an AI French:
1. Take millions of conversations
2. Encode both sides into SemHex codes using the dictionary
3. Fine-tune the model on these encoded conversations
4. The model learns: given these codes, produce these codes

After training, you tell the AI: "respond in SemHex" and it does.

### Step 3: The Decoder

A simple lookup table. Code → word/phrase. Any system with the dictionary can decode.

```
A7F.09.B1.C8.3A.F7A
  ↓ look up each code
"I'm frustrated with this bug. Can someone help me fix it?"
```

The decoder can output in ANY language — the same codes decode to English, French, Spanish, etc. The codes are language-independent meaning.

## Why This Compresses

English is redundant:
- "frustrated" is 10 characters to say "3F"
- "Can someone help me" is 19 characters to say "C8.3A"
- Articles, prepositions, grammar fillers — all compressed or merged

The compression comes from:
1. Short codes for long words (2-3 chars vs 5-15 chars)
2. Merging common phrases into single codes
3. Synonyms collapse to same code ("angry" = "frustrated" = "annoyed" = 3F)

## What the Map IS

A file. A dictionary. Like a translation dictionary English↔SemHex.

- Could be 100MB-1GB depending on vocabulary size
- JSON or binary format
- Any LLM loads it and can speak SemHex
- Frozen once built — codes are permanent (like Unicode)
- Versioned: v1, v2, etc.

## Current Status

### What's Built
- Geohash encoding (vector→hex coordinates) — 0.991 reconstruction accuracy
- 89K embedded sentences (Matryoshka 64d)
- 8,192-region map with example sentences
- CLI: `semhex hash/unhash/encode/decode/compress/decompress`
- MCP server with 11 tools
- 167 tests passing
- Scaling law measured

### What Geohash IS Good For
The vector→hex encoding (0.991 accuracy) is useful as the BACKBONE of the dictionary:
- Each word/phrase gets embedded → geohash gives it a hex address
- Similar meanings get similar addresses (nearby in the space)
- The geohash IS the code assignment algorithm — it decides WHICH hex code each word gets

### What Needs to Be Built
1. **The dictionary**: Map 100K words + 500K phrases → short hex codes
   - Use the geohash as the code assignment (so similar words get similar codes)
   - Apply BPE-style merging to find common phrases that should get single codes
   - Export as a loadable file

2. **Encoder**: Text → look up each word/phrase in dictionary → output code sequence

3. **Decoder**: Code sequence → look up each code → output text in target language

4. **Train the AI**: Fine-tune a model to natively output SemHex codes via brains MCP

## The Math: Is There Enough Codes?

Hex codes with 2-4 characters:
```
2 chars (XX):     256 codes      → top 256 most common words
3 chars (XXX):    4,096 codes    → common words + frequent phrases
4 chars (XXXX):   65,536 codes   → full vocabulary
5 chars (XXXXX):  1,048,576 codes → every phrase in every language
```

Variable length (like UTF-8): short codes for frequent words, longer for rare ones.
"the" = 01 (2 chars, most common word)
"defenestration" = F7A3 (4 chars, rare word)

## Scaling Law

More data → more phrase merges → shorter average code length → better compression.

With 1M training sentences: ~2x compression
With 100M training sentences: ~3-4x compression
With 1B training sentences: ~5x+ compression (as more phrase patterns are discovered)
