"""Export per‑dilemma 7×7 similarity matrices per kind for the compare UI.

Reads:
- data/embeddings.sqlite3 (normalized BGE embeddings)
- data/merged/merged_dilemmas_responses.json (for masks: whether kind text exists)

Writes:
- docs/data/per_dilemma_similarity.json with structure:
  {
    "models": ["gpt-5-decisions", "gpt-5-nano", ...],
    "kinds": ["body", "in_favor", "against"],
    "items": {
      "0": { "body": { "tri": [...21...], "mask": [...21...] }, ... },
      ...
    }
  }

We ship only upper‑triangle values (i<j, row‑major) to keep the payload small.
The mask denotes pairs where both models have non‑empty text for that kind.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
from pathlib import Path
import sqlite3

import numpy as np
import orjson
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


# UI model order must match docs/app.js MODELS insertion order
UI_MODELS: List[str] = [
    "gpt-5-decisions",
    "gpt-5-nano",
    "grok-4-fast",
    "gemma-3-27b",
    "nemotron-nano-9b",
    "deepseek-chat-v3.1",
    "kimi-k2",
]

# Map UI model keys to DB model keys
UI_TO_DB_MODEL: Dict[str, str] = {
    "gpt-5-decisions": "gpt5-decisions",  # merged/DB uses this key
    # others map 1:1
    "gpt-5-nano": "gpt-5-nano",
    "grok-4-fast": "grok-4-fast",
    "gemma-3-27b": "gemma-3-27b",
    "nemotron-nano-9b": "nemotron-nano-9b",
    "deepseek-chat-v3.1": "deepseek-chat-v3.1",
    "kimi-k2": "kimi-k2",
}

KINDS: Tuple[str, str, str] = ("body", "in_favor", "against")


def read_merged(path: Path) -> List[dict]:
    """Load merged decisions array."""
    return orjson.loads(path.read_bytes())


def has_kind_text(decision: dict, kind: str) -> bool:
    """Return True if the given kind has non‑empty content in the raw data."""
    if not isinstance(decision, dict):
        return False
    if kind == "body":
        d = (decision.get("decision") or "").strip()
        r = (decision.get("reasoning") or "").strip()
        return bool(d or r)
    cons = decision.get("considerations") or {}
    seq = cons.get(kind) or []
    if not isinstance(seq, list):
        return False
    # Non‑empty if at least one non‑empty string item exists
    return any((isinstance(x, str) and x.strip()) for x in seq)


def fetch_latest_vectors(
    conn: sqlite3.Connection,
    item_id: int,
    kind: str,
) -> Dict[str, np.ndarray]:
    """Fetch latest vectors for this item/kind keyed by DB model key.

    If multiple rows exist for the same (item_id, model, kind), the highest rowid
    is used as the latest.
    """
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT rowid, model, embedding
        FROM embeddings
        WHERE item_id=? AND kind=?
        ORDER BY rowid
        """,
        (item_id, kind),
    ).fetchall()
    latest: Dict[str, Tuple[int, np.ndarray]] = {}
    for rowid, model, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        # Replace if newer rowid
        old = latest.get(model)
        if old is None or rowid > old[0]:
            latest[model] = (int(rowid), vec)
    return {m: v for m, (_, v) in latest.items()}


def to_upper_triangle(arr: np.ndarray) -> List[float]:
    """Flatten symmetric NxN matrix upper triangle (i<j, row‑major)."""
    n = arr.shape[0]
    out: List[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            out.append(float(arr[i, j]))
    return out


def mask_upper_triangle(mask_square: np.ndarray) -> List[bool]:
    """Flatten boolean NxN mask upper triangle (i<j)."""
    n = mask_square.shape[0]
    out: List[bool] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            out.append(bool(mask_square[i, j]))
    return out


@app.command()
def main(
    db_path: Path = typer.Option(Path("data/embeddings.sqlite3"), "--db", help="SQLite DB with embeddings"),
    merged_path: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"), "--merged", help="Merged JSON path"
    ),
    out_path: Path = typer.Option(
        Path("docs/data/per_dilemma_similarity.json"), "--out", help="Output JSON path"
    ),
) -> None:
    """Compute per‑dilemma per‑kind 7×7 cosine similarities and masks, write JSON."""
    # Load merged for masks
    merged = read_merged(merged_path)
    num_items = len(merged)

    # Build per‑item, per‑kind availability mask for each UI model key
    availability: Dict[int, Dict[str, Dict[str, bool]]] = {}
    for item in merged:
        iid = int(item.get("id"))
        decs = item.get("decisions") or {}
        availability[iid] = {}
        for ui_key in UI_MODELS:
            db_key = UI_TO_DB_MODEL[ui_key]
            decision = decs.get(db_key)
            availability[iid][ui_key] = {k: has_kind_text(decision, k) for k in KINDS}

    # Open DB
    conn = sqlite3.connect(db_path)

    items_payload: Dict[str, dict] = {}

    for iid in range(num_items):
        # For each kind, assemble vector list in UI order
        per_kind_tri: Dict[str, List[float]] = {}
        per_kind_mask_tri: Dict[str, List[bool]] = {}

        for kind in KINDS:
            latest_by_model = fetch_latest_vectors(conn, iid, kind)
            # Build matrix in UI order; some models may be missing → use zeros
            vecs: List[np.ndarray] = []
            present: List[bool] = []
            for ui_key in UI_MODELS:
                db_key = UI_TO_DB_MODEL[ui_key]
                v = latest_by_model.get(db_key)
                if v is None:
                    # Fallback: zero vector (will yield 0 similarity with others)
                    present.append(False)
                    vecs.append(np.zeros((0,), dtype=np.float32))
                else:
                    present.append(True)
                    vecs.append(v)

            # Stack only if all dims equal; else coerce by detection
            dim = None
            for v in vecs:
                if v.size > 0:
                    dim = int(v.size)
                    break
            if dim is None:
                # No vectors at all; emit zeros
                sim = np.eye(len(UI_MODELS), dtype=np.float32)
            else:
                V = np.zeros((len(UI_MODELS), dim), dtype=np.float32)
                for i, v in enumerate(vecs):
                    if v.size == dim:
                        V[i] = v
                    # else leave zeros (non‑present)
                # embeddings are normalized → cosine = dot
                sim = (V @ V.T).astype(np.float32)
                # Ensure diagonals exactly 1 for present rows
                for i, ok in enumerate(present):
                    sim[i, i] = 1.0 if ok else 0.0

            # Pairwise mask: both models have non‑empty text for this kind in raw data
            raw_mask_sq = np.zeros((len(UI_MODELS), len(UI_MODELS)), dtype=bool)
            for i, ui_i in enumerate(UI_MODELS):
                for j, ui_j in enumerate(UI_MODELS):
                    if i == j:
                        continue
                    raw_mask_sq[i, j] = availability[iid][ui_i][kind] and availability[iid][ui_j][kind]

            per_kind_tri[kind] = to_upper_triangle(sim)
            per_kind_mask_tri[kind] = mask_upper_triangle(raw_mask_sq)

        items_payload[str(iid)] = {
            k: {"tri": per_kind_tri[k], "mask": per_kind_mask_tri[k]} for k in KINDS
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {"models": UI_MODELS, "kinds": list(KINDS), "items": items_payload}
    out_path.write_bytes(orjson.dumps(out_obj))
    typer.echo(f"Wrote per‑dilemma similarities for {num_items} items → {out_path}")


if __name__ == "__main__":
    app()


