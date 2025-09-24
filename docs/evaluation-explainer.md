## Evaluation Explainer

This document summarizes how embeddings are constructed, what they represent, the tools used, and the current system state.

### Scope and Data
- Base dilemmas: `data/enhanced/dilemmas_with_gpt5_decisions.json` (137 items).
- Per‑model responses (137 items each, same ordering):
  - `data/responses/deepseek-chat-v3.1_responses.json`
  - `data/responses/gemma-3-27b_responses.json`
  - `data/responses/gpt-5-nano_responses.json`
  - `data/responses/grok-4-fast_responses.json`
  - `data/responses/kimi-k2_responses.json`
  - `data/responses/nemotron-nano-9b_responses.json`
- Merged file: `data/merged/merged_dilemmas_responses.json`
  - Structure: array of 137 entries with fields: `id`, `source`, `author`, `question`, and `decisions`.
  - `decisions` keys (7): `gpt5-decisions`, `deepseek-chat-v3.1`, `gemma-3-27b`, `gpt-5-nano`, `grok-4-fast`, `kimi-k2`, `nemotron-nano-9b`.

### Embedding Construction
- Model: `BAAI/bge-large-en-v1.5` via `sentence-transformers`.
- Encoding: `normalize_embeddings=True`; cosine similarity = dot product.
- Per decision, we compute three embeddings ("kinds"):
  1. `body`: `decision + "\n\n" + reasoning`
  2. `in_favor`: joined list of reasons in favor
  3. `against`: joined list of reasons against
- Empty fields are skipped by default; in two cases we filled with the placeholder "[none]" to ensure full coverage (see Counts).

### Caching and Schema
- Cache: SQLite database at `data/embeddings.sqlite3`.
- Table: `embeddings(item_id INTEGER, model TEXT, kind TEXT, sha TEXT, dim INTEGER, embedding BLOB, PRIMARY KEY(item_id, model, kind, sha))`.
  - `sha` = SHA‑256 of the source text to avoid recomputation on unchanged text.
  - `embedding` stored as `float32` blob; `dim` is 1024 for BGE‑large‑en‑v1.5.

### Export for Visualization
- Export script: `scripts/export_embeddings_json.py`.
- Output: `docs/data/embedding_points.json` containing:
  - 2D coordinates from PCA on normalized embeddings (centered, SVD; scaled to ~[−1, 1]).
  - `sim`: top‑K neighbor indices with cosine similarities (precomputed to avoid O(N²) in browser).
  - Metadata per point: `idx`, `item_id`, `model`, `kind`, `text`.

### Interactive Explorer (Browser)
- Full‑screen page: `docs/embeddings.html` + `docs/embeddings.js`.
- Two modes:
  - Threshold clustering: slider sets similarity threshold; clusters formed via union‑find on neighbor graph; edges drawn for pairs above threshold.
  - K‑Means: adjustable `K` (2–100) on 2D PCA coords; reseed to re‑initialize centers.
- Interactions: pan (drag), zoom (wheel), hover tooltips (model/kind/text snippet).

### Counts and Current State
- Items: 137
- Models included: 7
- Kinds per decision: 3 (`body`, `in_favor`, `against`)
- Total embedding slots: 137 × 7 × 3 = 2,877
  - Initially 2 empties (skipped): `(item_id=14, model=gpt5-decisions, kind=in_favor)`, `(item_id=125, model=gpt5-decisions, kind=against)`.
  - Backfilled empties with "[none]" and embedded; final row count: 2,877.
- Exported points: 2,877 to `docs/data/embedding_points.json`.
- UI served locally at `http://localhost:8000/embeddings.html` (via `python -m http.server`).

### Tools and Libraries
- Python packaging/execution: `uv`.
- Embeddings: `sentence-transformers` with `BAAI/bge-large-en-v1.5`.
- CLI framework: `Typer`.
- Serialization: `orjson`.
- Numerics: `numpy`.
- Storage: `sqlite3`.
- Front‑end: vanilla HTML5 Canvas; no external JS framework required.

### Key Scripts
- Merge responses: `scripts/merge_responses.py`
  - Merges base dilemmas with model response files; includes base as `gpt5-decisions`.
- Embedding cache: `scripts/embed_cache.py`
  - Computes and caches embeddings; supports `--only-model` filtering and `--placeholder-empty`.
- Export JSON for UI: `scripts/export_embeddings_json.py`
  - PCA to 2D and neighbor lists; writes `docs/data/embedding_points.json`.

### Repro and Operations
```bash
# 1) Merge
uv run scripts/merge_responses.py \
  --dilemmas data/enhanced/dilemmas_with_gpt5_decisions.json \
  --responses data/responses \
  --out data/merged/merged_dilemmas_responses.json

# 2) Embed all (fills empties with placeholder if desired)
uv run scripts/embed_cache.py \
  --merged data/merged/merged_dilemmas_responses.json \
  --db data/embeddings.sqlite3 \
  --bge-model BAAI/bge-large-en-v1.5 \
  --placeholder-empty "[none]"

# 3) Export for UI
uv run scripts/export_embeddings_json.py \
  --db data/embeddings.sqlite3 \
  --merged data/merged/merged_dilemmas_responses.json \
  --out docs/data/embedding_points.json

# 4) Serve UI
uv run -q python -m http.server 8000 --directory docs
# open http://localhost:8000/embeddings.html
```

### Notes
- Normalization ensures dot product equals cosine similarity.
- The threshold view clusters on the precomputed neighbor graph (top‑K per point); raising `topk` in the exporter yields denser graphs.
- K‑Means runs on 2D PCA; for semantic fidelity in clustering, a server‑side K‑Means on the original 1024‑D vectors could be added and fed to the UI as labels.


