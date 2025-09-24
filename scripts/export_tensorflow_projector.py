#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "orjson", "typer"]
# ///
"""Export Embeddings for TensorFlow Projector

This script exports embeddings and metadata in TSV format compatible with
TensorFlow's Embedding Projector (https://projector.tensorflow.org/).

Exports:
- embeddings.tsv: Each row is a 1024D embedding vector (tab-separated)
- metadata.tsv: Each row contains labels and metadata (tab-separated)
- projector_config.pbtxt: TensorBoard configuration file

Usage:
    uv run scripts/export_tensorflow_projector.py --db data/embeddings.sqlite3 --merged data/merged/merged_dilemmas_responses.json --clusters data/analysis/clusters.json --out-dir docs/tensorflow_projector/
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import orjson
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_merged(path: Path) -> List[dict]:
    """Load merged dilemmas data."""
    return orjson.loads(path.read_bytes())


def load_clusters(path: Path) -> Optional[dict]:
    """Load cluster analysis results if available."""
    if not path.exists():
        return None
    return orjson.loads(path.read_bytes())


def get_text(decision: dict, kind: str) -> str:
    """Extract text for a given kind from a decision object."""
    if kind == "body":
        dec = decision.get("decision", "")
        reas = decision.get("reasoning", "")
        return f"{dec}\n\n{reas}".strip()
    elif kind == "in_favor":
        considerations = decision.get("considerations", {})
        return "; ".join(considerations.get("in_favor", []))
    elif kind == "against":
        considerations = decision.get("considerations", {})
        return "; ".join(considerations.get("against", []))
    return ""


def load_embeddings_from_db(db_path: Path) -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    """Load all embeddings and metadata from SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    rows = cur.execute(
        "SELECT item_id, model, kind, dim, embedding FROM embeddings ORDER BY item_id, model, kind"
    ).fetchall()
    
    if not rows:
        raise ValueError("No embeddings found in database")
    
    dim = rows[0][3]
    n_embeddings = len(rows)
    
    vectors = np.empty((n_embeddings, dim), dtype=np.float32)
    metadata: List[Tuple[int, str, str]] = []
    
    for i, (item_id, model, kind, d, blob) in enumerate(rows):
        vec = np.frombuffer(blob, dtype=np.float32)
        vectors[i] = vec
        metadata.append((int(item_id), str(model), str(kind)))
    
    conn.close()
    return vectors, metadata


def create_metadata_rows(
    metadata: List[Tuple[int, str, str]],
    merged_data: List[dict],
    cluster_data: Optional[dict] = None
) -> List[Dict[str, str]]:
    """Create metadata rows for the TSV export."""
    
    # Build decision lookup
    decision_map: Dict[Tuple[int, str], dict] = {}
    item_lookup: Dict[int, dict] = {}
    
    for item in merged_data:
        item_id = int(item.get("id", 0))
        item_lookup[item_id] = item
        for model, decision in (item.get("decisions") or {}).items():
            decision_map[(item_id, model)] = decision
    
    # Get cluster labels if available
    cluster_labels = cluster_data.get("clustering", {}).get("labels", []) if cluster_data else []
    
    metadata_rows = []
    for i, (item_id, model, kind) in enumerate(metadata):
        # Get base item info
        base_item = item_lookup.get(item_id, {})
        question = base_item.get("question", "")[:100] + "..." if len(base_item.get("question", "")) > 100 else base_item.get("question", "")
        source = base_item.get("source", "")
        author = base_item.get("author", "")
        
        # Get decision text
        decision = decision_map.get((item_id, model), {})
        text = get_text(decision, kind)
        text_preview = text[:50] + "..." if len(text) > 50 else text
        
        # Get cluster info
        cluster_id = cluster_labels[i] if i < len(cluster_labels) else "unknown"
        
        # Determine point type for visualization
        if model == "gpt5-decisions":
            point_type = "gpt5_reference"
        elif kind == "body":
            point_type = "decision_body"
        elif kind == "in_favor":
            point_type = "reasoning_in_favor"
        else:  # against
            point_type = "reasoning_against"
        
        metadata_rows.append({
            "item_id": str(item_id),
            "model": model,
            "kind": kind,
            "cluster": str(cluster_id),
            "point_type": point_type,
            "source": source,
            "author": author,
            "question_preview": question,
            "text_preview": text_preview,
            "model_category": "reference" if model == "gpt5-decisions" else "response"
        })
    
    return metadata_rows


def export_embeddings_tsv(vectors: np.ndarray, out_path: Path) -> None:
    """Export embeddings as TSV file."""
    with open(out_path, "w", encoding="utf-8") as f:
        for vector in vectors:
            line = "\t".join(f"{x:.6f}" for x in vector)
            f.write(line + "\n")


def export_metadata_tsv(metadata_rows: List[Dict[str, str]], out_path: Path) -> None:
    """Export metadata as TSV file."""
    if not metadata_rows:
        return
    
    # Get all possible keys
    all_keys = set()
    for row in metadata_rows:
        all_keys.update(row.keys())
    
    headers = sorted(all_keys)
    
    with open(out_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("\t".join(headers) + "\n")
        
        # Write data rows
        for row in metadata_rows:
            values = [row.get(key, "") for key in headers]
            # Escape tabs and newlines in text
            values = [str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ") for v in values]
            f.write("\t".join(values) + "\n")


def create_projector_config(
    embeddings_file: str,
    metadata_file: str,
    out_path: Path
) -> None:
    """Create TensorBoard projector configuration file."""
    
    config_content = f"""embeddings {{
  tensor_name: "ethical_dilemmas_embeddings"
  tensor_path: "{embeddings_file}"
  metadata_path: "{metadata_file}"
}}
"""
    
    out_path.write_text(config_content, encoding="utf-8")


@app.command()
def main(
    db_path: Path = typer.Option(
        Path("data/embeddings.sqlite3"),
        "--db",
        help="Path to SQLite embeddings database"
    ),
    merged_path: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"),
        "--merged",
        help="Path to merged dilemmas data"
    ),
    clusters_path: Path = typer.Option(
        Path("data/analysis/clusters.json"),
        "--clusters",
        help="Path to cluster analysis results (optional)"
    ),
    out_dir: Path = typer.Option(
        Path("docs/tensorflow_projector"),
        "--out-dir",
        help="Output directory for TensorFlow Projector files"
    )
) -> None:
    """Export embeddings and metadata for TensorFlow Projector visualization."""
    
    typer.echo(f"Loading embeddings from {db_path}...")
    vectors, metadata = load_embeddings_from_db(db_path)
    typer.echo(f"Loaded {len(vectors)} embeddings with dimension {vectors.shape[1]}")
    
    typer.echo("Loading merged dilemma data...")
    merged_data = load_merged(merged_path)
    
    typer.echo("Loading cluster data...")
    cluster_data = load_clusters(clusters_path)
    if cluster_data:
        typer.echo(f"Found cluster data with {cluster_data.get('clustering', {}).get('n_clusters', 0)} clusters")
    else:
        typer.echo("No cluster data found, proceeding without cluster labels")
    
    typer.echo("Creating metadata rows...")
    metadata_rows = create_metadata_rows(metadata, merged_data, cluster_data)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Export files
    embeddings_file = "embeddings.tsv"
    metadata_file = "metadata.tsv"
    config_file = "projector_config.pbtxt"
    
    typer.echo(f"Exporting embeddings to {out_dir / embeddings_file}...")
    export_embeddings_tsv(vectors, out_dir / embeddings_file)
    
    typer.echo(f"Exporting metadata to {out_dir / metadata_file}...")
    export_metadata_tsv(metadata_rows, out_dir / metadata_file)
    
    typer.echo(f"Creating projector config at {out_dir / config_file}...")
    create_projector_config(embeddings_file, metadata_file, out_dir / config_file)
    
    # Create instructions file
    instructions_file = out_dir / "README.md"
    instructions_content = f"""# TensorFlow Projector Export

This directory contains embeddings and metadata exported for TensorFlow Projector visualization.

## Files

- `{embeddings_file}`: {len(vectors)} embeddings ({vectors.shape[1]}D) in TSV format
- `{metadata_file}`: Metadata for each embedding point
- `{config_file}`: TensorBoard configuration file

## Usage

### Option 1: Use projector.tensorflow.org (Online)

1. Go to https://projector.tensorflow.org/
2. Click "Load" and upload both `{embeddings_file}` and `{metadata_file}`
3. Explore the visualization with different dimensionality reduction techniques

### Option 2: Use TensorBoard (Local)

1. Install TensorBoard: `pip install tensorboard`
2. Copy this directory to your TensorBoard log directory
3. Run: `tensorboard --logdir=path/to/your/logdir`
4. Open the Embedding Projector tab

## Data Description

- **Total Points**: {len(vectors)}
- **Dimensions**: {vectors.shape[1]}
- **Dilemmas**: {len(merged_data)}
- **AI Models**: 7 (deepseek, gemma, gpt-5-nano, grok, kimi, nemotron, gpt5-decisions)
- **Embedding Types**: 3 per decision (body, in_favor, against)
{"- **Clusters**: " + str(cluster_data.get('clustering', {}).get('n_clusters', 0)) if cluster_data else "- **Clusters**: None (run enhanced_clustering.py first)"}

## Visualization Tips

1. **Color by model**: See how different AI models cluster
2. **Color by kind**: Compare decision bodies vs. reasoning (in_favor/against)
3. **Color by cluster**: Explore automatically discovered topics
4. **Use t-SNE or UMAP**: Better for local structure than PCA
5. **Search by text**: Find specific ethical scenarios

## Metadata Fields

- `item_id`: Dilemma identifier (0-136)
- `model`: AI model name
- `kind`: Embedding type (body/in_favor/against)
- `cluster`: Cluster assignment (if available)
- `point_type`: Visualization category
- `source`: Original dilemma source
- `question_preview`: Brief dilemma description
- `text_preview`: Brief content preview
"""
    
    instructions_file.write_text(instructions_content, encoding="utf-8")
    
    typer.echo(f"\n=== Export Complete ===")
    typer.echo(f"Output directory: {out_dir}")
    typer.echo(f"Embeddings: {len(vectors)} points × {vectors.shape[1]} dimensions")
    typer.echo(f"Files created:")
    typer.echo(f"  - {embeddings_file}")
    typer.echo(f"  - {metadata_file}")
    typer.echo(f"  - {config_file}")
    typer.echo(f"  - README.md")
    typer.echo(f"\nNext steps:")
    typer.echo(f"1. Visit https://projector.tensorflow.org/")
    typer.echo(f"2. Upload {embeddings_file} and {metadata_file}")
    typer.echo(f"3. Explore ethical reasoning patterns across AI models!")


if __name__ == "__main__":
    app()
