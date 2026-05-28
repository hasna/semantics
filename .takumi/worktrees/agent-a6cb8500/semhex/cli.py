"""SemHex CLI — encode, decode, distance, blend, inspect, roundtrip.

Usage:
    semhex encode "Can you help me debug this?"
    semhex decode "$3A.C8F0 $72.B1A0"
    semhex distance "$8A.2100" "$8A.2400"
    semhex blend "$8A.2100" "$60.3000"
    semhex inspect "$8A.2100"
    semhex roundtrip "The cat sat on the mat."
    semhex codebook info
"""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def _echo_json(payload) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _decoded_entry_to_dict(decoded) -> dict:
    return {
        "code": str(decoded.code),
        "label": decoded.label,
        "category": decoded.l1_label,
        "depth": decoded.code.depth,
        "examples": list(decoded.examples) if decoded.examples else [],
        "neighbors": list(decoded.neighbors) if decoded.neighbors else [],
    }


def _validate_bits(_ctx, _param, value: int) -> int:
    if value not in {2, 4}:
        raise click.BadParameter("bits must be 2 or 4")
    return value


def _fail(message: str, *, json_payload: dict | None = None) -> None:
    if json_payload is not None:
        payload = dict(json_payload)
        payload["error"] = message
        _echo_json(payload)
        raise click.exceptions.Exit(1)
    raise click.ClickException(message)


def _get_codebook(version: str = "v0.1"):
    from semhex.core.codebook import load_codebook
    from pathlib import Path

    # Check if the requested version has the required numpy arrays
    base = Path(__file__).parent.parent / "codebooks" / version
    has_npy = (base / "level1.npy").exists()

    if not has_npy and version == "v0.1":
        # v0.1 has labels but missing .npy centroids — fall back to test codebook
        err_console.print(f"[yellow]Note: codebook {version} missing centroid arrays, using test codebook.[/yellow]")
        err_console.print("[dim]Run 'python -m training.build_codebook' to generate the full codebook.[/dim]")
        try:
            return load_codebook("test")
        except FileNotFoundError as exc:
            raise click.ClickException("No usable codebook found.") from exc

    try:
        return load_codebook(version)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Codebook {version} not found. Run: python -m training.build_codebook"
        ) from exc


def _get_provider(codebook=None):
    from semhex.embeddings import get_provider

    provider = get_provider("auto")
    # If codebook dimensions don't match the provider, fall back to mock
    if codebook is not None and provider.dimensions != codebook.dimensions:
        from semhex.embeddings.mock import MockEmbeddingProvider

        err_console.print(
            f"[yellow]Note: provider dims ({provider.dimensions}) != codebook dims ({codebook.dimensions}), using mock provider.[/yellow]"
        )
        return MockEmbeddingProvider(dimensions=codebook.dimensions)
    return provider


@click.group()
@click.version_option(version="0.1.0", prog_name="semhex")
def main():
    """SemHex — Semantic Hexadecimal Encoding."""
    pass


@main.command()
@click.argument("text")
@click.option("--depth", "-d", default=2, type=click.IntRange(1, 2), help="Code depth: 1=coarse, 2=fine")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def encode(text: str, depth: int, json_output: bool):
    """Encode text into SemHex codes."""
    from semhex.core.encoder import encode as do_encode

    codebook = _get_codebook()
    provider = _get_provider(codebook)
    result = do_encode(text, depth=depth, codebook=codebook, provider=provider)

    if json_output:
        _echo_json(
            {
                "codes": result.code_strings,
                "chunks": result.chunks,
                "distances": result.distances,
                "compression_ratio": result.compression_ratio,
            }
        )
        return

    table = Table(title="SemHex Encoding")
    table.add_column("Chunk", style="dim")
    table.add_column("Code", style="bold cyan")
    table.add_column("Distance", justify="right")

    for chunk, code, dist in zip(result.chunks, result.code_strings, result.distances):
        table.add_row(
            chunk[:60] + "..." if len(chunk) > 60 else chunk,
            code,
            f"{dist:.4f}",
        )

    console.print(table)
    console.print(f"\n[bold]Codes:[/bold] {' '.join(result.code_strings)}")
    console.print(
        f"[bold]Compression:[/bold] {result.compression_ratio:.1f}x ({len(result.chunks)} codes for {sum(len(c.split()) for c in result.chunks)} words)"
    )


@main.command()
@click.argument("codes")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def decode(codes: str, json_output: bool):
    """Decode SemHex codes to meaning."""
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    result = do_decode(codes, codebook=codebook)

    if json_output:
        _echo_json(result.to_dict())
        return

    table = Table(title="SemHex Decoding")
    table.add_column("Code", style="bold cyan")
    table.add_column("Label", style="bold")
    table.add_column("Category", style="dim")
    table.add_column("Examples", style="dim")

    for d in result.decoded:
        table.add_row(
            str(d.code),
            d.label,
            d.l1_label,
            ", ".join(d.examples[:3]) if d.examples else "-",
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {result.summary}")


@main.command()
@click.argument("code_a")
@click.argument("code_b")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def distance(code_a: str, code_b: str, json_output: bool):
    """Compute semantic distance between two codes."""
    from semhex.core.distance import distance as do_distance, similarity as do_similarity

    codebook = _get_codebook()
    d = do_distance(code_a, code_b, codebook=codebook)
    s = do_similarity(code_a, code_b, codebook=codebook)

    if json_output:
        _echo_json({"code_a": code_a, "code_b": code_b, "distance": d, "similarity": s})
        return

    console.print(f"[bold]{code_a}[/bold] ↔ [bold]{code_b}[/bold]")
    console.print(f"  Distance:   [cyan]{d:.4f}[/cyan] (0=identical, 2=opposite)")
    console.print(f"  Similarity: [cyan]{s:.4f}[/cyan] (1=identical, -1=opposite)")


@main.command()
@click.argument("code_a")
@click.argument("code_b")
@click.option("--weight", "-w", default=0.5, type=click.FloatRange(0.0, 1.0), help="Weight for first code (0-1)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def blend(code_a: str, code_b: str, weight: float, json_output: bool):
    """Blend two codes via semantic arithmetic."""
    from semhex.core.blend import blend as do_blend
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    result = do_blend(code_a, code_b, weight=weight, codebook=codebook)

    # Decode all three for display
    dec_a = do_decode([code_a], codebook=codebook)
    dec_b = do_decode([code_b], codebook=codebook)
    dec_r = do_decode([result], codebook=codebook)

    label_a = dec_a.decoded[0].label if dec_a.decoded else "?"
    label_b = dec_b.decoded[0].label if dec_b.decoded else "?"
    label_r = dec_r.decoded[0].label if dec_r.decoded else "?"

    if json_output:
        _echo_json(
            {
                "code_a": code_a,
                "label_a": label_a,
                "code_b": code_b,
                "label_b": label_b,
                "result": result,
                "label_result": label_r,
                "weight": weight,
                "inverse_weight": 1 - weight,
            }
        )
        return

    console.print(f"[bold]{code_a}[/bold] ({label_a}) + [bold]{code_b}[/bold] ({label_b})")
    console.print(f"  = [bold cyan]{result}[/bold cyan] ({label_r})")
    console.print(f"  weight: {weight:.2f} / {1 - weight:.2f}")


@main.command()
@click.argument("code")
@click.option("--neighbors", "-k", default=5, type=click.IntRange(1, None), help="Number of nearest neighbors to show")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def inspect(code: str, neighbors: int, json_output: bool):
    """Inspect a SemHex code — show centroid details and neighbors."""
    from semhex.core.format import parse_code
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    parsed = parse_code(code)
    result = do_decode([parsed], codebook=codebook, k_neighbors=neighbors)

    if not result.decoded:
        _fail(
            "Code not found in codebook",
            json_payload={"code": code, "found": False, "neighbors": []} if json_output else None,
        )

    d = result.decoded[0]

    if json_output:
        payload = _decoded_entry_to_dict(d)
        payload["found"] = True
        payload["requested_neighbors"] = neighbors
        _echo_json(payload)
        return

    console.print(f"[bold]Code:[/bold] {d.code}")
    console.print(f"[bold]Label:[/bold] {d.label}")
    console.print(f"[bold]Category:[/bold] {d.l1_label}")
    console.print(f"[bold]Depth:[/bold] {d.code.depth}")
    if d.examples:
        console.print(f"[bold]Examples:[/bold] {', '.join(d.examples)}")
    if d.neighbors:
        console.print(f"[bold]Neighbors (top {neighbors}):[/bold]")
        for n in d.neighbors:
            console.print(f"  {n}")


@main.command()
@click.argument("text")
@click.option("--depth", "-d", default=2, type=click.IntRange(1, 2), help="Code depth: 1=coarse, 2=fine")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def roundtrip(text: str, depth: int, json_output: bool):
    """Encode text, then decode the codes — show both sides."""
    from semhex.core.encoder import encode as do_encode
    from semhex.core.decoder import decode as do_decode

    codebook = _get_codebook()
    provider = _get_provider(codebook)

    # Encode
    enc = do_encode(text, depth=depth, codebook=codebook, provider=provider)
    # Decode
    dec = do_decode(enc.codes, codebook=codebook)

    if json_output:
        _echo_json(
            {
                "input": text,
                "depth": depth,
                "codes": enc.code_strings,
                "chunks": enc.chunks,
                "distances": enc.distances,
                "compression_ratio": enc.compression_ratio,
                "decoded_summary": dec.summary,
                "decoded": [_decoded_entry_to_dict(entry) for entry in dec.decoded],
            }
        )
        return

    console.print(f"[bold]Input:[/bold] {text}")
    console.print(f"[bold]Codes:[/bold] {' '.join(enc.code_strings)}")
    console.print(f"[bold]Decoded:[/bold] {dec.summary}")
    console.print(f"[bold]Compression:[/bold] {enc.compression_ratio:.1f}x")

    # Detail table
    table = Table(title="Roundtrip Detail")
    table.add_column("Chunk", style="dim")
    table.add_column("Code", style="cyan bold")
    table.add_column("Decoded Label")
    table.add_column("Distance", justify="right")

    for chunk, code_str, d, dist in zip(enc.chunks, enc.code_strings, dec.decoded, enc.distances):
        table.add_row(
            chunk[:50] + "..." if len(chunk) > 50 else chunk,
            code_str,
            d.label,
            f"{dist:.4f}",
        )

    console.print(table)


@main.command(name="hash")
@click.argument("text")
@click.option("--bits", "-b", default=4, type=int, callback=_validate_bits, help="Bits per dimension: 2 (compact) or 4 (precise)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def hash_cmd(text: str, bits: int, json_output: bool):
    """Encode text into a SemHex geohash — a mathematical semantic address."""
    import numpy as np
    from openai import OpenAI
    from semhex.core.auth import load_api_key as _load_api_key
    from semhex.core.geohash_v2 import SemHasher

    hasher = SemHasher(n_dims=64, bits_per_dim=bits)
    state_name = f"matryoshka_64d_{bits}b"
    try:
        hasher.load(state_name)
    except FileNotFoundError as exc:
        _fail(
            f"Trained state '{state_name}' not found. Run training first.",
            json_payload={"state": state_name, "bits_per_dimension": bits} if json_output else None,
        )

    api_key = _load_api_key("OPENAI_API_KEY")
    if not api_key:
        _fail(
            "OPENAI_API_KEY not found",
            json_payload={"state": state_name, "bits_per_dimension": bits} if json_output else None,
        )

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(input=[text], model="text-embedding-3-small", dimensions=64)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)

    code = hasher.encode(vec)
    hex_chars = hasher.hex_length
    total_bits = hasher.total_bits

    if json_output:
        _echo_json(
            {
                "input": text,
                "code": code,
                "bits": total_bits,
                "bits_per_dimension": bits,
                "hex_chars": hex_chars,
                "state": state_name,
            }
        )
        return

    console.print(f"[bold]Input:[/bold] {text}")
    console.print(f"[bold]Code:[/bold]  [cyan]{code}[/cyan]")
    console.print(f"[bold]Bits:[/bold]  {total_bits} ({hex_chars} hex chars)")


@main.command(name="unhash")
@click.argument("code")
@click.option("--bits", "-b", default=4, type=int, callback=_validate_bits, help="Bits per dimension used during encoding")
@click.option("--neighbors", "-k", default=5, type=click.IntRange(1, None), help="Number of nearest regions to show")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def unhash_cmd(code: str, bits: int, neighbors: int, json_output: bool):
    """Decode a SemHex geohash back to the nearest concepts from the map."""
    import json
    import numpy as np
    from semhex.core.geohash_v2 import SemHasher

    hasher = SemHasher(n_dims=64, bits_per_dim=bits)
    state_name = f"matryoshka_64d_{bits}b"
    try:
        hasher.load(state_name)
    except FileNotFoundError as exc:
        _fail(
            "Trained state not found.",
            json_payload={"code": code, "bits_per_dimension": bits, "state": state_name} if json_output else None,
        )

    # Decode to vector
    vec = hasher.decode(code)

    # Find nearest regions in map
    from pathlib import Path

    map_dir = Path("codebooks/map_v1")
    if not (map_dir / "centroids.npy").exists():
        _fail(
            f"Map not found at {map_dir}. Run build_map first.",
            json_payload={"code": code, "map_path": str(map_dir), "bits_per_dimension": bits} if json_output else None,
        )

    centroids = np.load(map_dir / "centroids.npy").astype(np.float32)
    labels = json.loads((map_dir / "labels.json").read_text())

    # Find nearest centroids
    sims = centroids @ vec
    top_indices = np.argsort(-sims)[:neighbors]

    nearest_regions = []
    for idx in top_indices:
        info = labels.get(str(idx), {})
        examples = info.get("examples", [])
        region_code = info.get("hex_code", "?")
        sim = float(sims[idx])
        nearest_regions.append(
            {
                "index": int(idx),
                "hex_code": region_code,
                "similarity": round(sim, 4),
                "examples": examples,
            }
        )

    if json_output:
        _echo_json(
            {
                "code": code,
                "bits_per_dimension": bits,
                "neighbors": neighbors,
                "state": state_name,
                "nearest_regions": nearest_regions,
            }
        )
        return

    console.print(f"[bold]Code:[/bold] {code}")
    console.print(f"[bold]Nearest regions (top {neighbors}):[/bold]")
    for region in nearest_regions:
        ex = region["examples"][0][:80] if region["examples"] else "(no examples)"
        console.print(f"  [cyan]{region['hex_code'][:25]}...[/cyan] (sim={region['similarity']:.4f})")
        console.print(f"    \"{ex}\"")


@main.command(name="rgb-encode")
@click.argument("text")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def rgb_encode_cmd(text: str, json_output: bool):
    """Encode text as a Semantic RGB code — 7 dimensions of meaning in 6 hex chars.

    Like #RRGGBB for colors, $XX.XX.XX captures meaning:
    evaluation, potency, activity, agent, domain, intent, specificity.
    """
    from semhex.core.semantic_rgb import encode, encode_detailed

    if json_output:
        result = encode_detailed(text)
        _echo_json(result)
        return
    result = encode_detailed(text)
    console.print(f"[bold]Input:[/bold]       {text}")
    console.print(f"[bold]Code:[/bold]        [cyan]{result['code']}[/cyan]")
    console.print(f"[bold]Description:[/bold] {result['description']}")
    console.print(f"[bold]Compression:[/bold] {result['compression_ratio']}x ({result['input_chars']} → {result['code_chars']} chars)")


@main.command(name="rgb-decode")
@click.argument("code")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def rgb_decode_cmd(code: str, json_output: bool):
    """Decode a Semantic RGB code to human-readable description."""
    from semhex.core.semantic_rgb import AGENT_LABELS, DOMAIN_LABELS, INTENT_LABELS, SemanticColor, decode

    color = SemanticColor.from_hex(code)
    desc = color.describe()
    if json_output:
        _echo_json(
            {
                "code": code,
                "description": desc,
                "dimensions": {
                    "evaluation": color.evaluation,
                    "potency": color.potency,
                    "activity": color.activity,
                    "agent": color.agent,
                    "domain": color.domain,
                    "intent": color.intent,
                    "specificity": color.specificity,
                },
            }
        )
        return
    table = Table(title=f"Semantic RGB: {code}")
    table.add_column("Dimension", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Meaning")
    table.add_row("Evaluation", str(color.evaluation), "very negative→very positive (0-15)")
    table.add_row("Potency", str(color.potency), "weak→strong (0-7)")
    table.add_row("Activity", str(color.activity), "passive→active (0-7)")
    table.add_row("Agent", str(color.agent), AGENT_LABELS.get(color.agent, "?"))
    table.add_row("Domain", str(color.domain), DOMAIN_LABELS.get(color.domain, "?"))
    table.add_row("Intent", str(color.intent), INTENT_LABELS.get(color.intent, "?"))
    table.add_row("Specificity", str(color.specificity), "vague→specific (0-15)")
    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {desc}")


@main.command(name="compress")
@click.argument("text")
@click.option("--quality", "-q", default=2, type=click.IntRange(1, 4), help="Quality: 1 (max compress) to 4 (near-lossless)")
@click.option("--provider", "-p", default="cerebras", type=click.Choice(["cerebras", "openai"], case_sensitive=False), help="LLM provider: cerebras or openai")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def compress_cmd(text: str, quality: int, provider: str, json_output: bool):
    """Compress text into SemHex codes using LLM."""
    from semhex.core.codec import compress

    codes = compress(text, quality=quality, provider=provider)
    ratio = len(text) / max(len(codes), 1)

    if json_output:
        _echo_json(
            {
                "input": text,
                "codes": codes,
                "compression_ratio": round(ratio, 2),
                "input_chars": len(text),
                "code_chars": len(codes),
                "quality": quality,
                "provider": provider,
            }
        )
        return

    console.print(f"[bold]Input:[/bold]  {text}")
    console.print(f"[bold]Codes:[/bold]  [cyan]{codes}[/cyan]")
    console.print(f"[bold]Ratio:[/bold]  {ratio:.1f}x ({len(text)} → {len(codes)} chars)")


@main.command(name="decompress")
@click.argument("codes")
@click.option("--provider", "-p", default="cerebras", type=click.Choice(["cerebras", "openai"], case_sensitive=False), help="LLM provider: cerebras or openai")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def decompress_cmd(codes: str, provider: str, json_output: bool):
    """Decompress SemHex codes back to text using LLM."""
    from semhex.core.codec import decompress

    text = decompress(codes, provider=provider)

    if json_output:
        _echo_json({"codes": codes, "text": text, "provider": provider})
        return

    console.print(f"[bold]Codes:[/bold]  {codes}")
    console.print(f"[bold]Text:[/bold]   {text}")


@main.command(name="codec-roundtrip")
@click.argument("text")
@click.option("--quality", "-q", default=2, type=click.IntRange(1, 4), help="Quality: 1 to 4")
@click.option("--provider", "-p", default="cerebras", type=click.Choice(["cerebras", "openai"], case_sensitive=False))
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def codec_roundtrip_cmd(text: str, quality: int, provider: str, json_output: bool):
    """Compress → decompress roundtrip with similarity measurement."""
    from semhex.core.codec import roundtrip as codec_roundtrip

    r = codec_roundtrip(text, quality=quality, provider=provider)

    if json_output:
        _echo_json(r)
        return

    console.print(f"[bold]Input:[/bold]    {r['input']}")
    console.print(f"[bold]Codes:[/bold]    [cyan]{r['codes']}[/cyan]")
    console.print(f"[bold]Output:[/bold]   {r['output']}")
    console.print(f"[bold]Ratio:[/bold]    {r['compression_ratio']}x")
    if r["semantic_similarity"] is not None:
        sim = r["semantic_similarity"]
        color = "green" if sim > 0.7 else "yellow" if sim > 0.5 else "red"
        console.print(f"[bold]Similarity:[/bold] [{color}]{sim:.4f}[/{color}]")
    elif r.get("similarity_error"):
        console.print(f"[bold]Similarity:[/bold] [yellow]unavailable[/yellow] ({r['similarity_error']})")
    console.print(f"[bold]Time:[/bold]    compress {r['compress_time']}s + decompress {r['decompress_time']}s")


@main.command(name="dict-encode")
@click.argument("text")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def dict_encode_cmd(text: str, json_output: bool):
    """Encode text using the local dictionary (no API key needed, instant)."""
    from semhex.core.dict_encoder import dict_encode

    codes = dict_encode(text)
    ratio = len(text) / max(len(codes), 1)
    if json_output:
        _echo_json({"text": text, "codes": codes, "compression_ratio": round(ratio, 2)})
        return
    console.print(f"[bold]Input:[/bold]  {text}")
    console.print(f"[bold]Codes:[/bold]  [cyan]{codes}[/cyan]")
    console.print(f"[bold]Ratio:[/bold]  {ratio:.1f}x ({len(text)} → {len(codes)} chars)")


@main.command(name="dict-decode")
@click.argument("codes")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.option("--detailed", "-d", is_flag=True, help="Show per-code breakdown")
def dict_decode_cmd(codes: str, json_output: bool, detailed: bool):
    """Decode dictionary codes back to text (no API key needed, instant)."""
    from semhex.core.dict_decoder import dict_decode, dict_decode_detailed

    if json_output or detailed:
        result = dict_decode_detailed(codes)
        if json_output:
            _echo_json(result)
            return
        # detailed table
        table = Table(title="Dictionary Decode")
        table.add_column("Code", style="cyan bold")
        table.add_column("Text")
        table.add_column("Found", justify="center")
        for entry in result["entries"]:
            table.add_row(entry["code"], entry["text"], "✓" if entry["found"] else "✗")
        console.print(table)
        console.print(f"\n[bold]Text:[/bold]    {result['text']}")
        console.print(f"[bold]Found:[/bold]   {result['n_found']}/{result['n_codes']} codes")
        return
    text = dict_decode(codes)
    console.print(f"[bold]Codes:[/bold]  {codes}")
    console.print(f"[bold]Text:[/bold]   {text}")


@main.command(name="dict-info")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def dict_info_cmd(json_output: bool):
    """Show dictionary statistics."""
    import json as _json
    from pathlib import Path

    dict_path = Path(__file__).parent.parent / "codebooks" / "dictionary_v1.json"
    if not dict_path.exists():
        _fail(
            "dictionary_v1.json not found",
            json_payload={"path": str(dict_path)} if json_output else None,
        )
    d = _json.loads(dict_path.read_text())
    payload = {
        "version": d.get("version", "unknown"),
        "entries": d.get("n_words", 0),
        "phrases": d.get("n_phrases", 0),
        "tiers": d.get("tiers", {}),
    }
    if json_output:
        _echo_json(payload)
        return
    console.print(f"[bold]Dictionary version:[/bold] {payload['version']}")
    console.print(f"[bold]Entries:[/bold]        {payload['entries']:,}")
    console.print(f"[bold]Phrases:[/bold]        {payload['phrases']:,}")
    for tier, count in payload["tiers"].items():
        console.print(f"  [cyan]{tier}:[/cyan] {count:,} codes")


@main.command(name="dict-roundtrip")
@click.argument("text")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def dict_roundtrip_cmd(text: str, json_output: bool):
    """Dictionary encode → decode roundtrip (no API key needed)."""
    from semhex.core.dict_encoder import dict_encode
    from semhex.core.dict_decoder import dict_decode

    codes = dict_encode(text)
    decoded = dict_decode(codes)
    ratio = len(text) / max(len(codes), 1)
    if json_output:
        _echo_json({"input": text, "codes": codes, "output": decoded, "compression_ratio": round(ratio, 2)})
        return
    console.print(f"[bold]Input:[/bold]   {text}")
    console.print(f"[bold]Codes:[/bold]   [cyan]{codes}[/cyan]")
    console.print(f"[bold]Output:[/bold]  {decoded}")
    console.print(f"[bold]Ratio:[/bold]   {ratio:.1f}x")


@main.group()
def codebook():
    """Codebook management commands."""
    pass


@codebook.command(name="info")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def codebook_info(json_output: bool):
    """Show codebook statistics."""
    cb = _get_codebook()
    payload = {
        "version": cb.version,
        "dimensions": cb.dimensions,
        "l1_clusters": cb.n_level1,
        "l2_clusters": cb.n_level2,
        "total_codes": cb.n_level1 + cb.n_level2,
    }

    if json_output:
        _echo_json(payload)
        return

    console.print(f"[bold]Version:[/bold] {payload['version']}")
    console.print(f"[bold]Dimensions:[/bold] {payload['dimensions']}")
    console.print(f"[bold]L1 clusters:[/bold] {payload['l1_clusters']}")
    console.print(f"[bold]L2 clusters:[/bold] {payload['l2_clusters']}")
    console.print(f"[bold]Total codes:[/bold] {payload['total_codes']:,}")


@main.group()
def eval():
    """Run evaluation suites."""
    pass


@eval.command(name="roundtrip")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def eval_roundtrip_cmd(json_output: bool):
    """Evaluate encode→decode roundtrip quality."""
    from evaluation.eval_roundtrip import eval_roundtrip

    result = eval_roundtrip()

    if json_output:
        _echo_json(result.to_dict())
        return

    console.print(f"\n[bold]Roundtrip ({result.n_sentences} sentences):[/bold]")
    console.print(f"  Mean similarity: [cyan]{result.mean_similarity:.4f}[/cyan]")
    console.print(f"  Min similarity:  [cyan]{result.min_similarity:.4f}[/cyan]")
    console.print(f"  Std deviation:   {result.std_similarity:.4f}")
    console.print(f"  Time:            {result.elapsed_seconds:.2f}s")


@eval.command(name="composition")
@click.option("--n-pairs", default=200, type=click.IntRange(1, None), help="Number of pairs to test")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def eval_composition_cmd(n_pairs: int, json_output: bool):
    """Evaluate code arithmetic compositionality (nero's test)."""
    from evaluation.eval_composition import eval_composition

    result = eval_composition(n_pairs=n_pairs)

    if json_output:
        _echo_json(result.to_dict())
        return

    console.print(f"\n[bold]Composition ({result.n_pairs} pairs):[/bold]")
    console.print(f"  Validity rate:   [cyan]{result.validity_rate:.1%}[/cyan] (target: >75%)")
    console.print(f"  Mean similarity: [cyan]{result.mean_similarity:.4f}[/cyan]")
    console.print(f"  Time:            {result.elapsed_seconds:.2f}s")


@eval.command(name="distance")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def eval_distance_cmd(json_output: bool):
    """Evaluate distance correlation with embedding space."""
    from evaluation.eval_distance import eval_distance_correlation

    result = eval_distance_correlation()

    if json_output:
        _echo_json(result.to_dict())
        return

    console.print(f"\n[bold]Distance Correlation ({result.n_pairs} pairs):[/bold]")
    console.print(f"  Spearman r:  [cyan]{result.spearman_r:.4f}[/cyan] (target: >0.80)")
    console.print(f"  p-value:     {result.spearman_p:.6f}")
    console.print(f"  Time:        {result.elapsed_seconds:.2f}s")


@eval.command(name="benchmark")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def eval_benchmark_cmd(json_output: bool):
    """Run performance benchmark."""
    from evaluation.benchmark import run_benchmark

    result = run_benchmark()

    if json_output:
        _echo_json(result.to_dict())
        return

    console.print(f"\n[bold]Benchmark ({result.n_sentences} sentences):[/bold]")
    console.print(f"  Compression:    [cyan]{result.compression_ratio:.1f}x[/cyan]")
    console.print(f"  Encode speed:   [cyan]{result.encode_rate:.0f}[/cyan] sentences/sec")
    console.print(f"  Lookup latency: [cyan]{result.lookup_time * 1000:.3f}ms[/cyan]")
    console.print(f"  Codebook RAM:   {result.codebook_memory_mb:.2f} MB")


@eval.command(name="all")
@click.option("--n-pairs", default=200, type=click.IntRange(1, None), help="Number of composition pairs")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def eval_all_cmd(n_pairs: int, json_output: bool):
    """Run all evaluations."""
    from evaluation.benchmark import run_benchmark
    from evaluation.eval_composition import eval_composition
    from evaluation.eval_distance import eval_distance_correlation
    from evaluation.eval_roundtrip import eval_roundtrip

    if not json_output:
        console.print("[bold]Running all evaluations...[/bold]\n")

    rt = eval_roundtrip()
    comp = eval_composition(n_pairs=n_pairs)
    dist = eval_distance_correlation()
    bench = run_benchmark()

    if json_output:
        _echo_json(
            {
                "roundtrip": rt.to_dict(),
                "composition": comp.to_dict(),
                "distance": dist.to_dict(),
                "benchmark": bench.to_dict(),
            }
        )
        return

    console.print(f"[green]Roundtrip:[/green] mean_sim={rt.mean_similarity:.4f} min={rt.min_similarity:.4f}")
    console.print(f"[green]Composition:[/green] validity={comp.validity_rate:.1%} mean_sim={comp.mean_similarity:.4f}")
    console.print(f"[green]Distance:[/green] spearman_r={dist.spearman_r:.4f} p={dist.spearman_p:.4f}")
    console.print(f"[green]Benchmark:[/green] {bench.compression_ratio:.1f}x compression, {bench.encode_rate:.0f} sent/s")

    console.print("\n[bold]All evaluations complete.[/bold]")


if __name__ == "__main__":
    main()
