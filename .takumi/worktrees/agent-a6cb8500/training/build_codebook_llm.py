"""Build SemHex codebook using an LLM (Cerebras) as the semantic brain.

Instead of embeddings + clustering, we ask a large LLM to:
1. Organize concepts into meaningful categories
2. Assign hex codes to each category
3. Place similar concepts near each other

The LLM IS the semantic understanding — it decides what goes where.

Usage:
    python -m training.build_codebook_llm
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.concepts import get_all_concepts

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def get_client() -> OpenAI:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        # Try loading from ~/.secrets
        secrets_path = Path.home() / ".secrets"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if line.startswith("export CEREBRAS_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY not found in env or ~/.secrets")
    return OpenAI(base_url=CEREBRAS_BASE_URL, api_key=api_key)


def build_category_mapping(client: OpenAI, concepts: list[str], n_categories: int = 64) -> dict:
    """Ask the LLM to organize concepts into categories."""

    concept_list = "\n".join(f"- {c}" for c in concepts)

    prompt = f"""You are building a semantic encoding system called SemHex — like hex color codes but for meaning.

I have {len(concepts)} concepts that need to be organized into exactly {n_categories} categories.

RULES:
1. Each category gets a hex code from $00 to ${n_categories-1:02X}
2. Semantically similar concepts MUST be in the same category
3. Every concept must be assigned to exactly one category
4. Each category needs a short label (1-3 words)
5. Categories should cover: emotions, intents/speech acts, topics/domains, actions, descriptions, logical/temporal, spatial, quantitative

Here are the concepts:
{concept_list}

Return a JSON object with this EXACT structure:
{{
  "categories": [
    {{
      "hex": "$00",
      "label": "category name",
      "concepts": ["concept1", "concept2", ...]
    }},
    ...
  ]
}}

IMPORTANT:
- Return ONLY valid JSON, no markdown, no explanation
- Every single concept from the list above must appear in exactly one category
- Use all {n_categories} category slots
- Group by MEANING similarity, not alphabetical order"""

    print(f"Asking LLM to organize {len(concepts)} concepts into {n_categories} categories...")
    t0 = time.time()

    response = client.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": "You are a precise JSON generator. Return ONLY valid JSON, no thinking, no explanation. /no_think"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=32000,
    )

    elapsed = time.time() - t0
    content = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

    # Strip think tags if present (Qwen sometimes adds these)
    if "<think>" in content:
        # Remove everything between <think> and </think>
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    print(f"  LLM responded in {elapsed:.1f}s ({len(content)} chars)")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  First 500 chars: {content[:500]}")
        # Try to find JSON within the response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
        else:
            raise

    return data


def build_subcategories(client: OpenAI, category: dict, n_subcategories: int = 32) -> list[dict]:
    """Ask the LLM to split a category into subcategories."""

    concepts = category["concepts"]
    if len(concepts) <= 1:
        return [{"hex": "0000", "label": concepts[0] if concepts else category["label"], "concepts": concepts}]

    actual_n = min(n_subcategories, len(concepts))

    prompt = f"""Split these {len(concepts)} concepts from the "{category['label']}" category into exactly {actual_n} subcategories.

Concepts:
{chr(10).join(f'- {c}' for c in concepts)}

Return JSON:
{{
  "subcategories": [
    {{"index": 0, "label": "subcategory name", "concepts": ["concept1", ...]}},
    ...
  ]
}}

RULES:
- Return ONLY valid JSON
- Every concept must appear in exactly one subcategory
- Similar concepts in the same subcategory
- Use all {actual_n} slots"""

    response = client.chat.completions.create(
        model="qwen-3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON, no thinking. /no_think"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=8000,
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

    if "<think>" in content:
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
        else:
            # Fallback: one subcategory per concept
            return [{"index": i, "label": c, "concepts": [c]} for i, c in enumerate(concepts[:actual_n])]

    return data.get("subcategories", [])


def generate_real_embeddings(
    client: OpenAI,
    categories: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Generate REAL embeddings using OpenAI text-embedding-3-small.

    For each category: embed the label + concepts → average = L1 centroid.
    For each subcategory: embed the label + concepts → average = L2 centroid.
    """
    import os

    # Use OpenAI for embeddings (Cerebras doesn't have embeddings endpoint)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        secrets_path = Path.home() / ".secrets"
        if secrets_path.exists():
            for line in secrets_path.read_text().splitlines():
                if line.startswith("export OPENAI_API_KEY="):
                    openai_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not openai_key:
        raise ValueError("OPENAI_API_KEY required for real embeddings")

    embed_client = OpenAI(api_key=openai_key)

    def embed_texts(texts: list[str]) -> np.ndarray:
        """Embed a batch of texts via OpenAI."""
        # OpenAI allows max 2048 inputs per call
        all_vecs = []
        for i in range(0, len(texts), 2000):
            batch = texts[i:i + 2000]
            resp = embed_client.embeddings.create(input=batch, model="text-embedding-3-small")
            vecs = [d.embedding for d in resp.data]
            all_vecs.extend(vecs)
        arr = np.array(all_vecs, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return arr / norms

    n_l1 = len(categories)
    n_l2_per_l1 = max(
        len(c.get("subcategories", c.get("concepts", [])))
        for c in categories
    ) if categories else 16
    n_l2_per_l1 = max(n_l2_per_l1, 1)

    # Collect all texts to embed in one batch (cheaper, faster)
    l1_texts = []
    l2_texts = []
    l2_info = []  # (l1_idx, l2_local_idx, label, examples)

    l1_labels = {}
    l2_labels = {}
    l1_examples = {}
    l2_examples = {}

    for i, cat in enumerate(categories):
        hex_l1 = f"${i:02X}"
        cat_label = cat["label"]
        cat_concepts = cat.get("concepts", [])
        l1_labels[hex_l1] = cat_label
        l1_examples[hex_l1] = cat_concepts[:5]

        # L1 text: category label + top concepts
        l1_text = f"{cat_label}: {', '.join(cat_concepts[:10])}" if cat_concepts else cat_label
        l1_texts.append(l1_text)

        subcats = cat.get("subcategories", [])
        for j in range(n_l2_per_l1):
            hex_l2 = f"${i:02X}.{j:04X}"
            if j < len(subcats):
                sc = subcats[j]
                sc_label = sc.get("label", f"sub_{j}")
                sc_concepts = sc.get("concepts", [])
                l2_labels[hex_l2] = sc_label
                l2_examples[hex_l2] = sc_concepts[:3]
                l2_text = f"{sc_label}: {', '.join(sc_concepts[:5])}" if sc_concepts else sc_label
            elif j < len(cat_concepts):
                # Use individual concepts as L2 entries
                l2_labels[hex_l2] = cat_concepts[j]
                l2_examples[hex_l2] = [cat_concepts[j]]
                l2_text = cat_concepts[j]
            else:
                l2_labels[hex_l2] = f"{cat_label}_{j}"
                l2_examples[hex_l2] = []
                l2_text = cat_label  # Fall back to category label
            l2_texts.append(l2_text)

    # Embed everything in two batches
    print(f"  Embedding {len(l1_texts)} L1 texts + {len(l2_texts)} L2 texts via OpenAI...")
    t0 = time.time()
    l1_centroids = embed_texts(l1_texts)
    l2_centroids = embed_texts(l2_texts)
    dims = l1_centroids.shape[1]
    print(f"  Embedded in {time.time() - t0:.1f}s → {dims} dimensions")

    labels = {
        "l1": l1_labels,
        "l2": l2_labels,
        "l1_examples": l1_examples,
        "l2_examples": l2_examples,
    }

    return l1_centroids, l2_centroids, labels, {"n_l2_per_l1": n_l2_per_l1, "dims": dims}


def build_codebook_llm(
    n_categories: int = 64,
    n_subcategories: int = 32,
    output_dir: str = "codebooks/v0.1",
    dims: int = 64,
):
    """Build codebook using LLM as the semantic brain."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = get_client()
    concepts = get_all_concepts()
    print(f"Loaded {len(concepts)} concepts")

    # Step 1: Get L1 category mapping from LLM
    mapping = build_category_mapping(client, concepts, n_categories)
    categories = mapping.get("categories", [])
    print(f"  LLM created {len(categories)} categories")

    # Step 2: Get L2 subcategories for each category
    print(f"Getting subcategories for {len(categories)} categories...")
    t0 = time.time()
    for i, cat in enumerate(categories):
        if len(cat.get("concepts", [])) > 1:
            actual_n_sub = min(n_subcategories, len(cat["concepts"]))
            subcats = build_subcategories(client, cat, actual_n_sub)
            cat["subcategories"] = subcats
            print(f"  ${i:02X} {cat['label']}: {len(subcats)} subcategories")
        else:
            cat["subcategories"] = []
            print(f"  ${i:02X} {cat['label']}: 1 concept (no split needed)")
    print(f"  Subcategorization done in {time.time() - t0:.1f}s")

    # Step 3: Generate REAL embeddings via OpenAI
    print("Generating real embeddings via OpenAI...")
    l1_centroids, l2_centroids, labels, extra = generate_real_embeddings(
        client, categories
    )
    n_l2_per_l1 = extra["n_l2_per_l1"]
    dims = extra["dims"]

    # Step 4: Save
    print(f"Saving to {output_path}/")
    np.save(output_path / "level1.npy", l1_centroids)
    np.save(output_path / "level2.npy", l2_centroids)
    (output_path / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    # Save the raw LLM mapping for reference
    (output_path / "llm_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    metadata = {
        "version": "0.1.0",
        "method": "llm-cerebras-qwen-235b",
        "dimensions": dims,
        "n_l1": len(categories),
        "n_l2_per_l1": n_l2_per_l1,
        "n_concepts": len(concepts),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Print summary
    total_assigned = sum(len(c.get("concepts", [])) for c in categories)
    print(f"\nCodebook v0.1 built (LLM method):")
    print(f"  Categories: {len(categories)}")
    print(f"  Concepts assigned: {total_assigned}/{len(concepts)}")
    print(f"  Dimensions: {dims}")
    print(f"  Method: Cerebras Qwen-3 235B")

    # Print first few categories
    print(f"\nSample categories:")
    for cat in categories[:10]:
        concepts_preview = ", ".join(cat.get("concepts", [])[:4])
        print(f"  {cat.get('hex', '??')} {cat['label']}: {concepts_preview}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SemHex codebook using LLM")
    parser.add_argument("--categories", type=int, default=64, help="Number of L1 categories")
    parser.add_argument("--subcategories", type=int, default=32, help="Max L2 subcategories per category")
    parser.add_argument("--output", default="codebooks/v0.1", help="Output directory")
    parser.add_argument("--dims", type=int, default=64, help="Embedding dimensions")
    args = parser.parse_args()

    build_codebook_llm(
        n_categories=args.categories,
        n_subcategories=args.subcategories,
        output_dir=args.output,
        dims=args.dims,
    )
