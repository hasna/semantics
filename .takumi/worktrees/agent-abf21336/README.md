# SemHex

Semantic Hexadecimal Encoding — a universal compact discrete encoding for meaning, like hex codes for colors.

> "invent the same language as colors" — find the dimensions of meaning, make every meaning a coordinate in that space.

Like `#FF0000` gives you any color in 6 hex chars, `$XX.XX.XX` gives you any meaning in 6 hex chars.

## Install

```bash
pip install semhex
```

## Quick Start

```python
import semhex

# Semantic RGB — 7 dimensions of meaning → 6 hex chars
from semhex.core.semantic_rgb import encode, decode
code = encode("I'm frustrated with this bug")   # → "$2A.C4.06"
desc = decode("$2A.C4.06")  # → "very negative, strong, moderate | I/self | technology | express | moderate"

# Dictionary encoding — local, no API key, instant
from semhex.core.dict_encoder import dict_encode
from semhex.core.dict_decoder import dict_decode
codes = dict_encode("Can you help me debug this?")  # → "E251.EC61.8699.0E"
text  = dict_decode("E251.EC61.8699.0E")            # → "can you help me debug this"
```

## CLI

### Semantic RGB — 7 dimensions of meaning in 6 hex chars

```bash
# Encode text → $XX.XX.XX (requires CEREBRAS_API_KEY or OPENAI_API_KEY)
semhex rgb-encode "I'm frustrated with this bug"
# Code:        $2A.C4.06
# Description: very negative, strong, moderate | I/self | technology | express | moderate
# Compression: 3.1x

# Decode code → dimension table + summary
semhex rgb-decode '$2A.C4.06'
semhex rgb-decode '$2A.C4.06' --json-output
```

### Dictionary Encoding — local, no API key, instant

```bash
semhex dict-encode "I am frustrated with this bug"
# → Codes: D019.1866.DBC7.13F0  (1.5x compression)

semhex dict-decode "D019.1866.DBC7.13F0"
# → i am frustrated with this bug

semhex dict-decode "D019.1866.DBC7.13F0" --detailed   # per-code table: Code | Text | Found ✓/✗
semhex dict-roundtrip "Can you help me debug this?"    # encode + decode in one shot
semhex dict-info                                        # 73,256 entries, 20,000 phrases
```

### Geohash — mathematical semantic address (requires OPENAI_API_KEY)

```bash
semhex hash "Can you help me debug this async error?"
# → $14.345E.5874F8...  (256 bits, 64 hex chars)

semhex unhash '$14.345E.5874F8...'
# → nearest regions in embedding space
```

### LLM Codec — compress/decompress via LLM (requires CEREBRAS_API_KEY or OPENAI_API_KEY)

```bash
semhex compress "Can you help me debug this async error?"
# → HLP.DBUG.ASYNC.ERR  (2.2x)

semhex decompress "HLP.DBUG.ASYNC.ERR"
# → Help debugging an asynchronous error.

semhex codec-roundtrip "text"    # compress → decompress + similarity score
```

### VQ Codebook

```bash
semhex encode "text"             # text → nearest codebook code
semhex decode '$1D.0003'         # code → label + category
semhex distance '$A' '$B'        # semantic distance (0=identical, 2=opposite)
semhex blend '$A' '$B'           # code arithmetic: blend two meanings
semhex inspect '$1D.0003'        # code details + 5 nearest neighbors
semhex roundtrip "text"          # encode then decode
semhex codebook info             # codebook statistics
```

### Evaluation

```bash
semhex eval roundtrip            # encode→decode similarity score
semhex eval composition          # code arithmetic validity
semhex eval distance             # distance correlation with embedding space
semhex eval benchmark            # speed, compression ratio, memory
semhex eval all                  # run all evaluations
```

## The 7 Dimensions of Meaning (Semantic RGB)

| Dimension | Bits | Range | Meaning |
|-----------|------|-------|---------|
| Evaluation | 4 | 0–15 | very negative → very positive |
| Potency | 3 | 0–7 | weak → strong |
| Activity | 3 | 0–7 | passive → active |
| Agent | 3 | 0–7 | I/self → universal/impersonal |
| Domain | 4 | 0–15 | emotion, technology, science, … |
| Intent | 3 | 0–7 | express → command |
| Specificity | 4 | 0–15 | vague → specific |

**Total: 24 bits → 6 hex chars → `$XX.XX.XX`**

## Architecture

```
Text → LLM scoring → 7 dimension scores → 24-bit packing → $XX.XX.XX
$XX.XX.XX → 24-bit unpacking → dimension values → human description
```

## Requirements

- Python 3.10+, numpy, scikit-learn, click, rich
- `CEREBRAS_API_KEY` or `OPENAI_API_KEY` for: `rgb-encode`, `compress`, `hash`
- No API key needed for: `dict-encode/decode/roundtrip/info`, `rgb-decode`

## License

Apache 2.0
