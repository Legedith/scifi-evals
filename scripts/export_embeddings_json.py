"""Export embeddings + 2D coords + neighbor lists to docs/data JSON.

Reads:
- data/embeddings.sqlite3 (embeddings table)
- data/merged/merged_dilemmas_responses.json (for texts/metadata)

Writes:
- docs/data/embedding_points.json

Each point includes idx, item_id, model, kind, x, y, sim (topK neighbors), text.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import orjson
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_merged(path: Path) -> List[dict]:
    return orjson.loads(path.read_bytes())


def get_text(decision: dict, kind: str) -> str:
    if kind == "body":
        d = decision.get("decision") or ""
        r = decision.get("reasoning") or ""
        txt = (d + "\n\n" + r).strip()
    else:
        cons = decision.get("considerations") or {}
        seq = cons.get(kind) or []
        txt = "\n".join([x.strip() for x in seq if isinstance(x, str)]).strip()
    return txt or "[none]"


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    # Center
    X = vectors.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD
    # Using economy SVD; take first 2 components
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    # Normalize to [-1, 1]
    mx = np.abs(coords).max(axis=0)
    mx[mx == 0] = 1.0
    coords = coords / mx
    return coords.astype(np.float32)


def topk_neighbors(vectors: np.ndarray, k: int = 30, batch: int = 256) -> List[List[Tuple[int, float]]]:
    # vectors assumed L2-normalized → dot = cosine
    N, D = vectors.shape
    neigh: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    Vt = vectors.T.copy()
    for start in range(0, N, batch):
        end = min(N, start + batch)
        # similarities for this block vs all
        sims = vectors[start:end] @ Vt  # (B, N)
        # Remove self-similarity by masking diag
        for i in range(start, end):
            sims[i - start, i] = -1.0
        # Take topk
        idxs = np.argpartition(-sims, kth=min(k, N - 1) - 1, axis=1)[:, :k]
        # sort each row
        row_sorted = np.take_along_axis(sims, idxs, axis=1)
        order = np.argsort(-row_sorted, axis=1)
        idxs = np.take_along_axis(idxs, order, axis=1)
        row_sorted = np.take_along_axis(row_sorted, order, axis=1)
        for r in range(end - start):
            neigh[start + r] = [(int(j), float(s)) for j, s in zip(idxs[r], row_sorted[r])]
    return neigh


@app.command()
def main(
    db_path: Path = typer.Option(Path("data/embeddings.sqlite3"), "--db", help="Path to SQLite db"),
    merged_path: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"), "--merged", help="Merged JSON"
    ),
    out_path: Path = typer.Option(
        Path("docs/data/embedding_points.json"), "--out", help="Output JSON path"
    ),
    topk: int = typer.Option(30, "--topk", help="Neighbors per point"),
) -> None:
    # Load embeddings
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT item_id, model, kind, dim, embedding FROM embeddings ORDER BY item_id, model, kind"
    ).fetchall()
    N = len(rows)
    if N == 0:
        typer.echo("No embeddings found.")
        raise typer.Exit(code=1)

    dims = set(r[3] for r in rows)
    if len(dims) != 1:
        typer.echo(f"Mixed dims found: {dims}")
    dim = rows[0][3]

    vectors = np.empty((N, dim), dtype=np.float32)
    meta: List[Tuple[int, str, str]] = []
    for i, (item_id, model, kind, d, blob) in enumerate(rows):
        vec = np.frombuffer(blob, dtype=np.float32)
        vectors[i] = vec
        meta.append((int(item_id), str(model), str(kind)))

    # Compute 2D coords and neighbors
    coords = pca_2d(vectors)
    neigh = topk_neighbors(vectors, k=topk)

    # Build text lookup
    merged = load_merged(merged_path)
    # index: (item_id, model) -> decision
    dec_map: Dict[Tuple[int, str], dict] = {}
    for item in merged:
        iid = int(item.get("id"))
        for model, decision in (item.get("decisions") or {}).items():
            dec_map[(iid, model)] = decision

    points = []
    for idx, ((item_id, model, kind), (x, y), nb) in enumerate(zip(meta, coords, neigh)):
        decision = dec_map.get((item_id, model), {})
        text = get_text(decision, kind)
        points.append(
            {
                "idx": idx,
                "item_id": item_id,
                "model": model,
                "kind": kind,
                "x": float(x),
                "y": float(y),
                "sim": nb,  # list of [idx, sim]
                "text": text,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"points": points}
    out_path.write_bytes(orjson.dumps(payload))
    typer.echo(f"Wrote {len(points)} points to {out_path}")


if __name__ == "__main__":
    app()


