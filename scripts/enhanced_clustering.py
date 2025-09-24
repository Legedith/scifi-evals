#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "orjson", "typer"]
# ///
"""Enhanced Clustering for Ethical Dilemma Embeddings

This script implements sophisticated clustering analysis on the full 1024D embeddings,
inspired by the movie quotes clustering approach but adapted for ethical dilemmas.

Features:
- BisectingKMeans clustering on full 1024D embeddings
- Multiple clustering strategies (global, by-model, by-kind)
- Cluster quality metrics (silhouette scores)
- Representative text selection for each cluster
- Export cluster assignments for downstream analysis

Usage:
    uv run scripts/enhanced_clustering.py --db data/embeddings.sqlite3 --merged data/merged/merged_dilemmas_responses.json --out data/analysis/clusters.json --n-clusters 25
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import orjson
import typer
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score, silhouette_samples

app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_merged(path: Path) -> List[dict]:
    """Load merged dilemmas data."""
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
    
    # Check dimensions are consistent
    dims = set(r[3] for r in rows)
    if len(dims) != 1:
        raise ValueError(f"Mixed dimensions found: {dims}")
    
    dim = rows[0][3]
    n_embeddings = len(rows)
    
    # Load vectors and metadata
    vectors = np.empty((n_embeddings, dim), dtype=np.float32)
    metadata: List[Tuple[int, str, str]] = []
    
    for i, (item_id, model, kind, d, blob) in enumerate(rows):
        vec = np.frombuffer(blob, dtype=np.float32)
        vectors[i] = vec
        metadata.append((int(item_id), str(model), str(kind)))
    
    conn.close()
    return vectors, metadata


def perform_clustering(
    vectors: np.ndarray, 
    n_clusters: int, 
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform BisectingKMeans clustering and return results with quality metrics."""
    
    # Initialize clustering model
    cluster_model = BisectingKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=1000,
        random_state=random_state
    )
    
    # Fit the model
    labels = cluster_model.fit_predict(vectors)
    
    # Calculate quality metrics
    silhouette_avg = silhouette_score(vectors, labels)
    silhouette_per_sample = silhouette_samples(vectors, labels)
    
    # Find closest point to each centroid (representative examples)
    distances = np.linalg.norm(vectors[:, np.newaxis] - cluster_model.cluster_centers_, axis=2)
    representative_indices = np.argmin(distances, axis=0)
    
    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette_avg),
        "silhouette_per_sample": silhouette_per_sample.tolist(),
        "representative_indices": representative_indices.tolist(),
        "cluster_centers": cluster_model.cluster_centers_.tolist(),
        "inertia": float(cluster_model.inertia_)
    }


def analyze_clusters_by_metadata(
    labels: List[int], 
    metadata: List[Tuple[int, str, str]]
) -> Dict[str, Any]:
    """Analyze cluster composition by model and kind."""
    
    cluster_stats = {}
    n_clusters = max(labels) + 1
    
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_metadata = [metadata[i] for i in cluster_indices]
        
        # Count by model
        model_counts = {}
        kind_counts = {}
        item_counts = {}
        
        for item_id, model, kind in cluster_metadata:
            model_counts[model] = model_counts.get(model, 0) + 1
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            item_counts[item_id] = item_counts.get(item_id, 0) + 1
        
        cluster_stats[str(cluster_id)] = {
            "size": len(cluster_indices),
            "model_distribution": model_counts,
            "kind_distribution": kind_counts,
            "unique_items": len(item_counts),
            "indices": cluster_indices
        }
    
    return cluster_stats


def get_representative_texts(
    representative_indices: List[int],
    metadata: List[Tuple[int, str, str]],
    merged_data: List[dict]
) -> List[Dict[str, Any]]:
    """Get the actual text content for representative examples of each cluster."""
    
    # Build decision lookup
    decision_map: Dict[Tuple[int, str], dict] = {}
    for item in merged_data:
        item_id = int(item.get("id", 0))
        for model, decision in (item.get("decisions") or {}).items():
            decision_map[(item_id, model)] = decision
    
    representatives = []
    for cluster_id, idx in enumerate(representative_indices):
        item_id, model, kind = metadata[idx]
        decision = decision_map.get((item_id, model), {})
        text = get_text(decision, kind)
        
        # Get the base dilemma question
        base_item = next((item for item in merged_data if int(item.get("id", 0)) == item_id), {})
        question = base_item.get("question", "")
        source = base_item.get("source", "")
        author = base_item.get("author", "")
        
        representatives.append({
            "cluster_id": cluster_id,
            "representative_index": idx,
            "item_id": item_id,
            "model": model,
            "kind": kind,
            "text": text,
            "question": question,
            "source": source,
            "author": author
        })
    
    return representatives


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
    out_path: Path = typer.Option(
        Path("data/analysis/clusters.json"),
        "--out",
        help="Output path for cluster analysis"
    ),
    n_clusters: int = typer.Option(
        25,
        "--n-clusters",
        help="Number of clusters for BisectingKMeans"
    ),
    random_state: int = typer.Option(
        42,
        "--random-state", 
        help="Random state for reproducible clustering"
    )
) -> None:
    """Perform enhanced clustering analysis on ethical dilemma embeddings."""
    
    typer.echo(f"Loading embeddings from {db_path}...")
    vectors, metadata = load_embeddings_from_db(db_path)
    typer.echo(f"Loaded {len(vectors)} embeddings with dimension {vectors.shape[1]}")
    
    typer.echo("Loading merged dilemma data...")
    merged_data = load_merged(merged_path)
    typer.echo(f"Loaded {len(merged_data)} dilemmas")
    
    typer.echo(f"Performing BisectingKMeans clustering with {n_clusters} clusters...")
    clustering_results = perform_clustering(vectors, n_clusters, random_state)
    typer.echo(f"Clustering complete. Silhouette score: {clustering_results['silhouette_score']:.3f}")
    
    typer.echo("Analyzing cluster composition...")
    cluster_stats = analyze_clusters_by_metadata(clustering_results["labels"], metadata)
    
    typer.echo("Extracting representative texts...")
    representative_texts = get_representative_texts(
        clustering_results["representative_indices"],
        metadata,
        merged_data
    )
    
    # Compile final results
    analysis_results = {
        "clustering": clustering_results,
        "cluster_statistics": cluster_stats,
        "representative_texts": representative_texts,
        "metadata": {
            "n_embeddings": len(vectors),
            "embedding_dimension": vectors.shape[1],
            "n_dilemmas": len(merged_data),
            "n_clusters": n_clusters,
            "random_state": random_state
        }
    }
    
    # Save results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(analysis_results, option=orjson.OPT_INDENT_2))
    typer.echo(f"Analysis results saved to {out_path}")
    
    # Print summary
    typer.echo("\n=== Clustering Summary ===")
    typer.echo(f"Total embeddings: {len(vectors)}")
    typer.echo(f"Number of clusters: {n_clusters}")
    typer.echo(f"Silhouette score: {clustering_results['silhouette_score']:.3f}")
    typer.echo(f"Inertia: {clustering_results['inertia']:.2f}")
    
    typer.echo("\n=== Cluster Size Distribution ===")
    sizes = [cluster_stats[str(i)]["size"] for i in range(n_clusters)]
    typer.echo(f"Min size: {min(sizes)}, Max size: {max(sizes)}, Mean size: {np.mean(sizes):.1f}")


if __name__ == "__main__":
    app()
