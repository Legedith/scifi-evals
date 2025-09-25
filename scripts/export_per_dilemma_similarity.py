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

from typing import Dict, Iterable, List, Tuple, Optional
from pathlib import Path
import sqlite3

import numpy as np
import orjson
import typer


app = typer.Typer(add_completion=False, no_args_is_help=True)


# UI model order must match docs/app.js MODELS insertion order
UI_MODELS: List[str] = [
    "gpt-5-nano",
    "grok-4-fast",
    "gemma-3-27b",
    "nemotron-nano-9b",
    "deepseek-chat-v3.1",
    "kimi-k2",
]

# Map UI model keys to DB model keys
UI_TO_DB_MODEL: Dict[str, str] = {
    # UI → DB model keys (decisions exclude gpt5-decisions)
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


def compute_global_top_pcs(
    conn: sqlite3.Connection, kind: str, top_m: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and top-m principal components for given kind.

    Returns (mean_vec, pcs) where pcs shape is (top_m, dim).
    """
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT embedding FROM embeddings WHERE kind=? ORDER BY item_id, model",
        (kind,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No embeddings found for kind={kind}")
    dim = len(np.frombuffer(rows[0][0], dtype=np.float32))
    X = np.empty((len(rows), dim), dtype=np.float32)
    for i, (blob,) in enumerate(rows):
        X[i] = np.frombuffer(blob, dtype=np.float32)
    mu = X.mean(axis=0, dtype=np.float32)
    Xc = X - mu
    # Economy SVD for principal axes; components are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[: top_m].astype(np.float32)
    return mu.astype(np.float32), pcs


def apply_all_but_top(vecs: np.ndarray, mu: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Subtract projections onto leading PCs and re-normalize row-wise.

    vecs: (N, D) row-major embedding matrix.
    mu: (D,) mean vector
    pcs: (M, D) principal components
    """
    if vecs.size == 0:
        return vecs
    X = (vecs - mu).astype(np.float32, copy=False)
    # Project and subtract: for each pc, X -= (X @ pc)[:,None] * pc
    for k in range(pcs.shape[0]):
        pc = pcs[k]
        proj = X @ pc
        X -= proj[:, None] * pc[None, :]
    # Re-normalize rows to unit length when possible
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return X.astype(np.float32)


def csls_matrix(sim: np.ndarray, k: int) -> np.ndarray:
    """Compute CSLS-adjusted similarity from a symmetric similarity matrix.

    sim: (N,N) with diagonal ~1
    k: neighborhood size
    """
    N = sim.shape[0]
    k_eff = max(1, min(k, N - 1))
    r = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        # exclude diagonal
        row = np.array([sim[i, j] for j in range(N) if j != i], dtype=np.float32)
        if row.size == 0:
            r[i] = 0.0
        else:
            idx = np.argpartition(-row, k_eff - 1)[:k_eff]
            r[i] = float(np.mean(row[idx]))
    out = np.empty_like(sim)
    for i in range(N):
        for j in range(N):
            if i == j:
                out[i, j] = 1.0
            else:
                out[i, j] = 2.0 * sim[i, j] - r[i] - r[j]
    return out.astype(np.float32)


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
    pc_remove_top: int = typer.Option(2, "--pc-remove-top", help="Remove top-M PCs per kind (ABTT)"),
    csls_k: int = typer.Option(2, "--csls-k", help="Neighborhood size for CSLS"),
    reranker_model: Optional[str] = typer.Option(
        None,
        "--reranker-model",
        help="Optional cross-encoder model name for pairwise reranking (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2)",
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

    # Precompute ABTT bases (mean + top PCs) per kind
    abtt: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if pc_remove_top > 0:
        for kind in KINDS:
            mu, pcs = compute_global_top_pcs(conn, kind, top_m=pc_remove_top)
            abtt[kind] = (mu, pcs)

    # Optional cross-encoder
    ce_model = None
    if reranker_model:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            ce_model = CrossEncoder(reranker_model, device="cuda" if _has_cuda() else None)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to load cross-encoder {reranker_model}: {e}")

    for iid in range(num_items):
        # For each kind, assemble vector list in UI order
        per_kind_tri: Dict[str, List[float]] = {}
        per_kind_mask_tri: Dict[str, List[bool]] = {}

        for kind in KINDS:
            latest_by_model = fetch_latest_vectors(conn, iid, kind)
            # Build matrix in UI order; some models may be missing → keep zeros
            vecs: List[np.ndarray] = []
            present: List[bool] = []
            for ui_key in UI_MODELS:
                db_key = UI_TO_DB_MODEL[ui_key]
                v = latest_by_model.get(db_key)
                if v is None:
                    present.append(False)
                    vecs.append(np.zeros((0,), dtype=np.float32))
                else:
                    present.append(True)
                    vecs.append(v)

            # Determine dim and stack
            dim = None
            for v in vecs:
                if v.size > 0:
                    dim = int(v.size)
                    break
            if dim is None:
                sim_abtt = np.eye(len(UI_MODELS), dtype=np.float32)
            else:
                V = np.zeros((len(UI_MODELS), dim), dtype=np.float32)
                for i, v in enumerate(vecs):
                    if v.size == dim:
                        V[i] = v
                # Apply ABTT if configured
                if pc_remove_top > 0 and kind in abtt:
                    mu, pcs = abtt[kind]
                    Vt = apply_all_but_top(V, mu, pcs)
                else:
                    # Ensure row-normalized
                    norms = np.linalg.norm(V, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    Vt = (V / norms).astype(np.float32)
                # Cosine on transformed vectors
                sim_abtt = (Vt @ Vt.T).astype(np.float32)
                for i, ok in enumerate(present):
                    sim_abtt[i, i] = 1.0 if ok else 0.0

            # CSLS on ABTT matrix
            sim_csls = csls_matrix(sim_abtt, k=csls_k)

            # Pairwise mask: both models have non‑empty text for this kind in raw data
            raw_mask_sq = np.zeros((len(UI_MODELS), len(UI_MODELS)), dtype=bool)
            for i, ui_i in enumerate(UI_MODELS):
                for j, ui_j in enumerate(UI_MODELS):
                    if i == j:
                        continue
                    raw_mask_sq[i, j] = availability[iid][ui_i][kind] and availability[iid][ui_j][kind]

            # Base cosine (ABTT) tri + CSLS tri
            per_kind_tri[kind] = {"tri": to_upper_triangle(sim_abtt), "tri_csls": to_upper_triangle(sim_csls)}
            per_kind_mask_tri[kind] = mask_upper_triangle(raw_mask_sq)

        # Optional cross-encoder rerank on decision/consideration texts
        if ce_model is not None:
            # Build texts by model/kind
            decs = merged[iid].get("decisions") or {}
            texts: Dict[str, Dict[str, str]] = {}
            for ui_key in UI_MODELS:
                db_key = UI_TO_DB_MODEL[ui_key]
                d = decs.get(db_key) or {}
                cons = d.get("considerations") or {}
                texts[ui_key] = {
                    "body": f"{d.get('decision') or ''}\n\n{d.get('reasoning') or ''}".strip() or "",
                    "in_favor": "\n".join([x.strip() for x in (cons.get("in_favor") or []) if isinstance(x, str)]).strip(),
                    "against": "\n".join([x.strip() for x in (cons.get("against") or []) if isinstance(x, str)]).strip(),
                }
            for kind in KINDS:
                pairs: List[Tuple[str, str]] = []
                idx_pairs: List[Tuple[int, int]] = []
                for a in range(len(UI_MODELS) - 1):
                    for b in range(a + 1, len(UI_MODELS)):
                        ua, ub = UI_MODELS[a], UI_MODELS[b]
                        ta, tb = texts[ua][kind], texts[ub][kind]
                        if ta and tb:
                            pairs.append((ta, tb))
                        else:
                            pairs.append(("", ""))  # placeholder; will mark NaN
                        idx_pairs.append((a, b))
                if pairs:
                    # Predict in small batch; CrossEncoder expects list of (s1, s2)
                    scores = ce_model.predict(pairs)  # type: ignore[assignment]
                    # Fill tri_ce, mark pairs with missing text as NaN to be ignored by UI
                    tri_ce: List[float] = []
                    for (a, b), s, (ua, ub) in zip(idx_pairs, scores, idx_pairs):
                        ta, tb = pairs[idx_pairs.index((a, b))]
                        if not ta or not tb:
                            tri_ce.append(float("nan"))
                        else:
                            tri_ce.append(float(s))
                    # Attach
                    per_kind_tri[kind]["tri_ce"] = tri_ce

        # Flatten per kind payload preserving mask and multiple matrices
        item_entry: Dict[str, dict] = {}
        for k in KINDS:
            matrices = per_kind_tri[k]
            item_entry[k] = {"mask": per_kind_mask_tri[k]}
            # Merge all matrices into entry
            for key, tri_vals in matrices.items():
                item_entry[k][key] = tri_vals
        items_payload[str(iid)] = item_entry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {"models": UI_MODELS, "kinds": list(KINDS), "items": items_payload}
    out_path.write_bytes(orjson.dumps(out_obj))
    typer.echo(f"Wrote per‑dilemma similarities for {num_items} items → {out_path}")


if __name__ == "__main__":
    app()


