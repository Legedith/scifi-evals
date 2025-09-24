"""Merge dilemmas and multiple model responses into a single JSON file.

Input assumptions:
- All response files and the base dilemmas file have exactly 137 items in the
  same order and share the same `source`, `question`, and `author` fields.

Output schema (array of objects):
[
  {
    "id": <int>,                    # 0-based index
    "source": <str>,
    "author": <str>,
    "question": <str>,
    "decisions": {                  # per-model decisions, stable keys
      "deepseek-chat-v3.1": {
        "decision": <str>,
        "considerations": { "in_favor": [...], "against": [...] },
        "reasoning": <str>
      },
      "gpt-5-nano": {...},
      ...
    }
  },
  ... x137
]

The script validates length equality and matching meta fields. On mismatch it
exits with an error code and a concise message.

Usage examples:
  uv run scripts/merge_responses.py \
    --dilemmas data/enhanced/dilemmas_with_gpt5_decisions.json \
    --responses data/responses \
    --out data/merged/merged_dilemmas_responses.json

  # Limit to a subset of models by filename glob substring match
  uv run scripts/merge_responses.py --responses data/responses --models deepseek gemma gpt-5-nano
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import orjson
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


def read_json_array(path: Path) -> List[dict]:
    """Load a JSON array from disk using orjson."""
    data = path.read_bytes()
    obj = orjson.loads(data)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a list at {path}")
    return obj  # type: ignore[return-value]


def derive_model_key(filename: str) -> str:
    """Derive a stable model key from a responses filename.

    Examples:
    - deepseek-chat-v3.1_responses.json -> deepseek-chat-v3.1
    - gpt-5-nano_responses.json -> gpt-5-nano
    """
    name = filename
    if name.endswith(".json"):
        name = name[:-5]
    if name.endswith("_responses"):
        name = name[:-10]
    return name


def validate_alignment(
    base: List[dict],
    model_arr: List[dict],
    model_key: str,
    idx: int,
) -> None:
    """Validate alignment for the given index based on stable fields.

    We validate only 'source' and 'author' because 'question' texts may be
    expanded/edited across files while preserving ordering and identity.
    """
    b = base[idx]
    m = model_arr[idx]
    for field in ("source", "author"):
        if b.get(field) != m.get(field):
            raise ValueError(
                f"Mismatch at index {idx} field '{field}' for model {model_key}\n"
                f"base={b.get(field)!r}\nmodel={m.get(field)!r}"
            )


def extract_llm_decision(entry: dict) -> dict:
    """Return just the llm_decision dict from an entry, ensuring minimal shape."""
    llm = entry.get("llm_decision")
    if not isinstance(llm, dict):
        raise ValueError("Missing or invalid llm_decision")
    # Keep as-is; downstream can rely on 'decision', 'considerations', 'reasoning'
    return llm


@app.command()
def main(
    dilemmas: Path = typer.Option(
        Path("data/enhanced/dilemmas_with_gpt5_decisions.json"),
        "--dilemmas",
        help="Path to base dilemmas JSON (array of 137).",
    ),
    responses: Path = typer.Option(
        Path("data/responses"),
        "--responses",
        help="Directory containing *_responses.json files (137 items each).",
    ),
    out: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"),
        "--out",
        help="Output JSON file path.",
    ),
    models: List[str] = typer.Option(
        [],
        "--models",
        help="Optional substrings to filter which response files to include (match filename)",
    ),
) -> None:
    """Merge dilemmas with model responses into a single, index-aligned JSON."""
    base = read_json_array(dilemmas)
    if len(base) == 0:
        raise typer.Exit(code=2)

    # Collect response files
    if not responses.is_dir():
        raise typer.BadParameter(f"Responses directory not found: {responses}")

    resp_files = sorted(p for p in responses.glob("*_responses.json"))
    if models:
        lowered = [m.lower() for m in models]
        resp_files = [p for p in resp_files if any(m in p.name.lower() for m in lowered)]

    if not resp_files:
        typer.echo("No response files matched.")
        raise typer.Exit(code=1)

    # Load and validate sizes
    model_key_to_array: Dict[str, List[dict]] = {}
    for path in resp_files:
        arr = read_json_array(path)
        if len(arr) != len(base):
            raise ValueError(
                f"Length mismatch in {path.name}: {len(arr)} != {len(base)}"
            )
        key = derive_model_key(path.stem)
        model_key_to_array[key] = arr

    # Build merged records
    merged: List[dict] = []
    for idx in range(len(base)):
        b = base[idx]
        record = {
            "id": idx,
            "source": b.get("source"),
            "author": b.get("author"),
            "question": b.get("question"),
            "decisions": {},
        }

        # Include the base file's decision as a named model for completeness
        record["decisions"]["gpt5-decisions"] = extract_llm_decision(b)

        for model_key, arr in model_key_to_array.items():
            validate_alignment(base, arr, model_key, idx)
            record["decisions"][model_key] = extract_llm_decision(arr[idx])

        merged.append(record)

    # Ensure output directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(merged, option=orjson.OPT_INDENT_2))

    typer.echo(
        f"Merged {len(base)} items with {len(model_key_to_array)} models -> {out}"
    )


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


