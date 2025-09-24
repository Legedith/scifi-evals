
"""Compute and cache BGE embeddings for merged decisions.

Kinds per model decision:
- body: decision + reasoning (joined)
- in_favor: joined list items
- against: joined list items

Cache: SQLite file (default: data/embeddings.sqlite3)
Key: (item_id, model, kind, sha256(text)) ensures recomputation only when text changes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import orjson
import typer
from sentence_transformers import SentenceTransformer


app = typer.Typer(add_completion=False, no_args_is_help=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            item_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            kind TEXT NOT NULL,
            sha TEXT NOT NULL,
            dim INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            PRIMARY KEY (item_id, model, kind, sha)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_embeddings_model_kind ON embeddings(model, kind)")
    conn.commit()


def load_merged(path: Path) -> List[dict]:
    return orjson.loads(path.read_bytes())


def get_texts_for_decision(decision: dict) -> Dict[str, str]:
    decision_text = decision.get("decision", "") or ""
    reasoning_text = decision.get("reasoning", "") or ""
    cons = decision.get("considerations") or {}
    in_favor_list = cons.get("in_favor") or []
    against_list = cons.get("against") or []

    # Build kinds
    body = (decision_text + "\n\n" + reasoning_text).strip()
    in_favor = "\n".join(x.strip() for x in in_favor_list if isinstance(x, str)).strip()
    against = "\n".join(x.strip() for x in against_list if isinstance(x, str)).strip()
    return {"body": body, "in_favor": in_favor, "against": against}


def rows_to_compute(
    merged: List[dict],
    models: List[str] | None,
    placeholder_empty: str,
) -> Iterable[Tuple[int, str, str, str]]:
    for item in merged:
        item_id = int(item.get("id"))
        decisions = item.get("decisions") or {}
        for model_name, decision in decisions.items():
            if models and model_name not in models:
                continue
            texts = get_texts_for_decision(decision)
            for kind, text in texts.items():
                if not text and placeholder_empty:
                    text = placeholder_empty
                if not text:
                    continue
                yield (item_id, model_name, kind, text)


def fetch_existing(conn: sqlite3.Connection, keys: List[Tuple[int, str, str, str]]) -> set[Tuple[int, str, str, str]]:
    want = [(iid, m, k, sha256_text(t)) for iid, m, k, t in keys]
    if not want:
        return set()
    qmarks = ",".join(["(?, ?, ?, ?)"] * len(want))
    # Query each tuple by PRIMARY KEY
    cur = conn.cursor()
    existing: set[Tuple[int, str, str, str]] = set()
    for (iid, m, k, s) in want:
        row = cur.execute(
            "SELECT 1 FROM embeddings WHERE item_id=? AND model=? AND kind=? AND sha=?",
            (iid, m, k, s),
        ).fetchone()
        if row:
            existing.add((iid, m, k, s))
    return existing


def insert_batch(
    conn: sqlite3.Connection,
    records: List[Tuple[int, str, str, str, np.ndarray]],
) -> None:
    if not records:
        return
    cur = conn.cursor()
    for item_id, model_name, kind, sha, vec in records:
        cur.execute(
            "INSERT OR IGNORE INTO embeddings (item_id, model, kind, sha, dim, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, model_name, kind, sha, int(vec.shape[0]), vec.astype(np.float32).tobytes()),
        )
    conn.commit()


@app.command()
def main(
    merged_path: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"),
        "--merged",
        help="Path to merged JSON.",
    ),
    db_path: Path = typer.Option(
        Path("data/embeddings.sqlite3"),
        "--db",
        help="SQLite file to cache embeddings.",
    ),
    model_name: str = typer.Option(
        "BAAI/bge-large-en-v1.5", "--bge-model", help="Sentence-Transformers model name"
    ),
    limit: int = typer.Option(0, "--limit", help="Max items to process (0=all)."),
    only_models: List[str] = typer.Option(
        [], "--only-model", help="Process only these decision model keys (repeatable)."
    ),
    placeholder_empty: str = typer.Option(
        "", "--placeholder-empty", help="If set, use this text when a field is empty."
    ),
) -> None:
    merged = load_merged(merged_path)
    conn = sqlite3.connect(db_path)
    ensure_schema(conn)

    # Collect keys
    iterator = rows_to_compute(merged, only_models or None, placeholder_empty)
    keys: List[Tuple[int, str, str, str]] = []
    for tup in iterator:
        keys.append(tup)
        if limit and len(keys) >= limit:
            break

    # Filter already cached
    existing = fetch_existing(conn, keys)
    to_do: List[Tuple[int, str, str, str]] = []
    for iid, m, k, text in keys:
        signature = (iid, m, k, sha256_text(text))
        if signature not in existing:
            to_do.append((iid, m, k, text))

    typer.echo(f"Total tasks: {len(keys)}; to compute: {len(to_do)}")
    if not to_do:
        return

    # Encode in small batches to control memory
    model = SentenceTransformer(model_name)
    BATCH = 64
    i = 0
    while i < len(to_do):
        batch = to_do[i : i + BATCH]
        texts = [t for (_, _, _, t) in batch]
        embs = model.encode(
            texts,
            batch_size=min(32, len(texts)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        records: List[Tuple[int, str, str, str, np.ndarray]] = []
        for (iid, m, k, t), vec in zip(batch, embs):
            records.append((iid, m, k, sha256_text(t), vec))
        insert_batch(conn, records)
        i += len(batch)
        typer.echo(f"Cached {i}/{len(to_do)}")


if __name__ == "__main__":
    app()
