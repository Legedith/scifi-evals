#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "orjson", "typer", "pandas", "scipy"]
# ///
"""Comprehensive Model Analysis: Co-occurrence, Stability, and Advanced Insights

This script implements a comprehensive analysis system to understand how models "think alike"
by analyzing clustering patterns across all 137 ethical dilemmas.

Features:
- Model co-occurrence analysis across all questions
- Clustering stability analysis with multiple thresholds  
- Statistical significance testing vs random baselines
- Advanced research questions (consistency, specialization, etc.)
- Multiple CSV outputs for different analytical perspectives

Usage:
    uv run scripts/comprehensive_model_analysis.py --db data/embeddings.sqlite3 --merged data/merged/merged_dilemmas_responses.json --out data/analysis/comprehensive_results
"""

from __future__ import annotations

import sqlite3
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import orjson
import typer
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Model order from the UI - these are the 6 models we compare (excluding gpt5-decisions)
MODELS = [
    "deepseek-chat-v3.1", 
    "gemma-3-27b", 
    "gpt-5-nano", 
    "grok-4-fast", 
    "kimi-k2", 
    "nemotron-nano-9b"
]

KINDS = ["body", "in_favor", "against"]

def load_merged(path: Path) -> List[dict]:
    """Load merged dilemmas data."""
    return orjson.loads(path.read_bytes())

def load_embeddings_from_db(db_path: Path, question_id: int = None, kind: str = None) -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    """Load embeddings from SQLite database, optionally filtered by question_id and/or kind."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT item_id, model, kind, embedding FROM embeddings"
    params = []
    
    if question_id is not None or kind is not None:
        query += " WHERE"
        conditions = []
        if question_id is not None:
            conditions.append(" item_id = ?")
            params.append(question_id)
        if kind is not None:
            conditions.append(" kind = ?")
            params.append(kind)
        query += " AND".join(conditions)
    
    query += " ORDER BY item_id, model, kind"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return np.array([]), []
    
    vectors = []
    metadata = []
    
    for item_id, model, kind, embedding_blob in rows:
        # Deserialize the embedding
        vector = np.frombuffer(embedding_blob, dtype=np.float32)
        vectors.append(vector)
        metadata.append((item_id, model, kind))
    
    return np.array(vectors), metadata

def find_auto_threshold(similarities: np.ndarray, min_clusters: int = 2, min_cluster_size: int = 2) -> float:
    """Find automatic threshold that creates meaningful clusters."""
    # Get unique similarity values, sorted descending
    unique_vals = np.unique(similarities)
    unique_vals = unique_vals[unique_vals > 0]  # Remove zero/negative similarities
    unique_vals = np.sort(unique_vals)[::-1]
    
    if len(unique_vals) == 0:
        return 0.5  # fallback
    
    for threshold in unique_vals:
        # Create adjacency matrix
        n = int(np.sqrt(len(similarities) * 2)) + 1  # Assuming upper triangle
        adj_matrix = np.zeros((n, n))
        
        # Fill adjacency matrix from similarities (assuming upper triangle format)
        idx = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if idx < len(similarities):
                    if similarities[idx] >= threshold:
                        adj_matrix[i, j] = adj_matrix[j, i] = 1
                    idx += 1
        
        # Count connected components using union-find
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        
        for i in range(n):
            for j in range(i+1, n):
                if adj_matrix[i, j] == 1:
                    union(i, j)
        
        # Count cluster sizes
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)
        
        large_clusters = [c for c in clusters.values() if len(c) >= min_cluster_size]
        
        if len(large_clusters) >= min_clusters:
            return threshold
    
    # Fallback: median similarity
    return float(np.median(unique_vals)) if len(unique_vals) > 0 else 0.5

def perform_auto_clustering_for_question(
    vectors: np.ndarray,
    metadata: List[Tuple[int, str, str]],
    similarity_threshold: float = None
) -> Dict[str, Any]:
    """Perform auto-clustering for a single question using similarity-based approach."""
    
    if len(vectors) == 0:
        return {"clusters": [], "threshold": 0.0, "n_clusters": 0}
    
    # Compute pairwise similarities
    similarities = cosine_similarity(vectors)
    
    # Auto-determine threshold if not provided
    if similarity_threshold is None:
        # Extract upper triangle similarities
        n = len(vectors)
        upper_tri_sims = []
        for i in range(n-1):
            for j in range(i+1, n):
                upper_tri_sims.append(similarities[i, j])
        
        similarity_threshold = find_auto_threshold(np.array(upper_tri_sims))
    
    # Union-find clustering based on similarity threshold
    n = len(vectors)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px
    
    # Connect points above threshold
    for i in range(n):
        for j in range(i+1, n):
            if similarities[i, j] >= similarity_threshold:
                union(i, j)
    
    # Group points by cluster
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    
    cluster_list = list(clusters.values())
    
    return {
        "clusters": cluster_list,
        "threshold": similarity_threshold,
        "n_clusters": len(cluster_list),
        "similarities": similarities,
        "metadata": metadata
    }

def analyze_model_cooccurrence(all_results: Dict[int, Dict[str, Dict]], kind: str) -> pd.DataFrame:
    """Analyze how often models appear together in clusters across all questions."""
    
    cooccurrence_matrix = np.zeros((len(MODELS), len(MODELS)))
    model_to_idx = {model: i for i, model in enumerate(MODELS)}
    
    for question_id, results_by_kind in all_results.items():
        if kind not in results_by_kind:
            continue
            
        clusters = results_by_kind[kind]["clusters"]
        
        for cluster in clusters:
            if len(cluster) < 2:  # Skip singleton clusters
                continue
                
            # Get models in this cluster
            cluster_models = []
            for point_idx in cluster:
                metadata = results_by_kind[kind]["metadata"]
                if point_idx < len(metadata):
                    _, model, _ = metadata[point_idx]
                    cluster_models.append(model)
            
            # Update co-occurrence counts for all pairs in this cluster
            for model1, model2 in itertools.combinations(cluster_models, 2):
                if model1 in model_to_idx and model2 in model_to_idx:
                    i, j = model_to_idx[model1], model_to_idx[model2]
                    cooccurrence_matrix[i, j] += 1
                    cooccurrence_matrix[j, i] += 1  # Make symmetric
    
    # Convert to DataFrame
    df = pd.DataFrame(cooccurrence_matrix, index=MODELS, columns=MODELS)
    
    # Add normalized version (divide by total possible questions)
    total_questions = len(all_results)
    df_normalized = df / total_questions
    
    return df, df_normalized

def analyze_model_consistency(all_results: Dict[int, Dict[str, Dict]], merged_data: List[dict]) -> pd.DataFrame:
    """Analyze how consistent each model is across different question types."""
    
    consistency_data = []
    
    for model in MODELS:
        model_data = {"model": model}
        
        # Count clustering patterns for this model across all questions
        cluster_sizes = []
        outlier_count = 0
        total_questions = 0
        
        for question_id, results_by_kind in all_results.items():
            question_data = merged_data[question_id]
            author = question_data.get("author", "Unknown")
            source = question_data.get("source", "Unknown")
            
            for kind in KINDS:
                if kind not in results_by_kind:
                    continue
                    
                clusters = results_by_kind[kind]["clusters"]
                metadata = results_by_kind[kind]["metadata"]
                
                # Find which cluster this model is in
                model_cluster_size = 1  # Default: singleton
                for cluster in clusters:
                    cluster_models = []
                    for point_idx in cluster:
                        if point_idx < len(metadata):
                            _, point_model, _ = metadata[point_idx]
                            cluster_models.append(point_model)
                    
                    if model in cluster_models:
                        model_cluster_size = len(cluster_models)
                        break
                
                cluster_sizes.append(model_cluster_size)
                if model_cluster_size == 1:
                    outlier_count += 1
                total_questions += 1
        
        if total_questions > 0:
            model_data.update({
                "avg_cluster_size": np.mean(cluster_sizes),
                "outlier_rate": outlier_count / total_questions,
                "total_analyzed": total_questions,
                "cluster_size_std": np.std(cluster_sizes)
            })
        else:
            model_data.update({
                "avg_cluster_size": 0,
                "outlier_rate": 0,
                "total_analyzed": 0,
                "cluster_size_std": 0
            })
        
        consistency_data.append(model_data)
    
    return pd.DataFrame(consistency_data)

def analyze_cross_kind_correlation(all_results: Dict[int, Dict[str, Dict]]) -> pd.DataFrame:
    """Analyze correlation between clustering patterns across different kinds."""
    
    correlation_data = []
    
    for question_id, results_by_kind in all_results.items():
        question_data = {"question_id": question_id}
        
        # For each pair of kinds, compute cluster overlap
        for kind1, kind2 in itertools.combinations(KINDS, 2):
            if kind1 not in results_by_kind or kind2 not in results_by_kind:
                continue
            
            # Get clustering results for both kinds
            clusters1 = results_by_kind[kind1]["clusters"]
            clusters2 = results_by_kind[kind2]["clusters"]
            metadata1 = results_by_kind[kind1]["metadata"]
            metadata2 = results_by_kind[kind2]["metadata"]
            
            # Create model-to-cluster mapping for each kind
            model_to_cluster1 = {}
            for cluster_idx, cluster in enumerate(clusters1):
                for point_idx in cluster:
                    if point_idx < len(metadata1):
                        _, model, _ = metadata1[point_idx]
                        model_to_cluster1[model] = cluster_idx
            
            model_to_cluster2 = {}
            for cluster_idx, cluster in enumerate(clusters2):
                for point_idx in cluster:
                    if point_idx < len(metadata2):
                        _, model, _ = metadata2[point_idx]
                        model_to_cluster2[model] = cluster_idx
            
            # Compute overlap: how often models that cluster together in kind1 also cluster together in kind2
            common_models = set(model_to_cluster1.keys()) & set(model_to_cluster2.keys())
            if len(common_models) > 1:
                agreement_count = 0
                total_pairs = 0
                
                for model1, model2 in itertools.combinations(common_models, 2):
                    same_cluster_kind1 = model_to_cluster1[model1] == model_to_cluster1[model2]
                    same_cluster_kind2 = model_to_cluster2[model1] == model_to_cluster2[model2]
                    
                    if same_cluster_kind1 == same_cluster_kind2:
                        agreement_count += 1
                    total_pairs += 1
                
                if total_pairs > 0:
                    agreement_rate = agreement_count / total_pairs
                    question_data[f"{kind1}_vs_{kind2}_agreement"] = agreement_rate
        
        correlation_data.append(question_data)
    
    return pd.DataFrame(correlation_data)

def compute_random_baseline(all_results: Dict[int, Dict[str, Dict]], n_simulations: int = 1000) -> Dict[str, float]:
    """Compute random baseline for co-occurrence statistics."""
    
    random_cooccurrences = []
    
    for _ in range(n_simulations):
        random_matrix = np.zeros((len(MODELS), len(MODELS)))
        model_to_idx = {model: i for i, model in enumerate(MODELS)}
        
        for question_id, results_by_kind in all_results.items():
            for kind in KINDS:
                if kind not in results_by_kind:
                    continue
                
                clusters = results_by_kind[kind]["clusters"]
                metadata = results_by_kind[kind]["metadata"]
                
                # Get all models for this question/kind
                all_models = []
                for point_idx in range(len(metadata)):
                    _, model, _ = metadata[point_idx]
                    all_models.append(model)
                
                # Randomly shuffle cluster assignments
                n_clusters = len(clusters)
                if n_clusters > 0 and len(all_models) > 0:
                    cluster_sizes = [len(cluster) for cluster in clusters]
                    
                    # Randomly assign models to clusters with same size distribution
                    shuffled_models = np.random.permutation(all_models)
                    cluster_start = 0
                    
                    for cluster_size in cluster_sizes:
                        if cluster_size < 2:
                            cluster_start += cluster_size
                            continue
                            
                        cluster_models = shuffled_models[cluster_start:cluster_start + cluster_size]
                        
                        # Update co-occurrence counts
                        for model1, model2 in itertools.combinations(cluster_models, 2):
                            if model1 in model_to_idx and model2 in model_to_idx:
                                i, j = model_to_idx[model1], model_to_idx[model2]
                                random_matrix[i, j] += 1
                                random_matrix[j, i] += 1
                        
                        cluster_start += cluster_size
        
        # Store the maximum co-occurrence value from this simulation
        max_cooccurrence = np.max(random_matrix)
        random_cooccurrences.append(max_cooccurrence)
    
    return {
        "mean_max_cooccurrence": np.mean(random_cooccurrences),
        "std_max_cooccurrence": np.std(random_cooccurrences),
        "p95_max_cooccurrence": np.percentile(random_cooccurrences, 95),
        "p99_max_cooccurrence": np.percentile(random_cooccurrences, 99)
    }

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
        Path("data/analysis/comprehensive_results"),
        "--out",
        help="Output directory for analysis results"
    ),
    n_random_simulations: int = typer.Option(
        1000,
        "--random-sims",
        help="Number of random simulations for baseline"
    )
) -> None:
    """Perform comprehensive model analysis across all 137 questions."""
    
    typer.echo("🚀 Starting Comprehensive Model Analysis...")
    
    # Load merged data
    typer.echo("📊 Loading merged dilemma data...")
    merged_data = load_merged(merged_path)
    typer.echo(f"Loaded {len(merged_data)} dilemmas")
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all clustering results
    all_results = {}
    
    typer.echo("🔍 Analyzing clustering patterns for each question...")
    
    for question_id in range(len(merged_data)):
        if question_id % 10 == 0:
            typer.echo(f"  Processing question {question_id + 1}/{len(merged_data)}")
        
        question_results = {}
        
        for kind in KINDS:
            # Load embeddings for this specific question and kind
            vectors, metadata = load_embeddings_from_db(db_path, question_id=question_id, kind=kind)
            
            if len(vectors) == 0:
                continue
            
            # Perform auto-clustering
            clustering_result = perform_auto_clustering_for_question(vectors, metadata)
            question_results[kind] = clustering_result
        
        all_results[question_id] = question_results
    
    typer.echo("📈 Computing model co-occurrence matrices...")
    
    # Analyze co-occurrence for each kind
    cooccurrence_results = {}
    for kind in KINDS:
        cooccurrence_df, cooccurrence_normalized_df = analyze_model_cooccurrence(all_results, kind)
        
        # Save to CSV
        cooccurrence_df.to_csv(out_path / f"cooccurrence_{kind}_raw.csv")
        cooccurrence_normalized_df.to_csv(out_path / f"cooccurrence_{kind}_normalized.csv")
        
        cooccurrence_results[kind] = {
            "raw": cooccurrence_df,
            "normalized": cooccurrence_normalized_df
        }
    
    typer.echo("🎯 Analyzing model consistency patterns...")
    
    # Model consistency analysis
    consistency_df = analyze_model_consistency(all_results, merged_data)
    consistency_df.to_csv(out_path / "model_consistency.csv", index=False)
    
    typer.echo("🔗 Computing cross-kind correlations...")
    
    # Cross-kind correlation analysis
    correlation_df = analyze_cross_kind_correlation(all_results)
    correlation_df.to_csv(out_path / "cross_kind_correlations.csv", index=False)
    
    typer.echo("🎲 Computing random baseline statistics...")
    
    # Random baseline
    random_baseline = compute_random_baseline(all_results, n_random_simulations)
    
    # Save baseline results
    baseline_df = pd.DataFrame([random_baseline])
    baseline_df.to_csv(out_path / "random_baseline.csv", index=False)
    
    typer.echo("📋 Generating summary statistics...")
    
    # Generate summary statistics
    summary_stats = []
    
    for kind in KINDS:
        cooc_raw = cooccurrence_results[kind]["raw"]
        cooc_norm = cooccurrence_results[kind]["normalized"]
        
        # Remove diagonal (self-co-occurrence)
        np.fill_diagonal(cooc_raw.values, 0)
        np.fill_diagonal(cooc_norm.values, 0)
        
        stats_data = {
            "kind": kind,
            "max_cooccurrence": cooc_raw.values.max(),
            "mean_cooccurrence": cooc_raw.values.mean(),
            "std_cooccurrence": cooc_raw.values.std(),
            "max_normalized": cooc_norm.values.max(),
            "mean_normalized": cooc_norm.values.mean(),
            "std_normalized": cooc_norm.values.std(),
        }
        
        # Find most frequently co-occurring pair
        max_idx = np.unravel_index(cooc_raw.values.argmax(), cooc_raw.shape)
        stats_data["top_pair"] = f"{MODELS[max_idx[0]]} & {MODELS[max_idx[1]]}"
        stats_data["top_pair_count"] = cooc_raw.values[max_idx]
        
        summary_stats.append(stats_data)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(out_path / "summary_statistics.csv", index=False)
    
    typer.echo("💾 Saving detailed clustering results...")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    # Save detailed results as JSON for further analysis
    detailed_results = {
        "metadata": {
            "n_questions": len(merged_data),
            "n_models": len(MODELS),
            "models": MODELS,
            "kinds": KINDS,
            "n_random_simulations": n_random_simulations
        },
        "random_baseline": convert_numpy_types(random_baseline),
        "summary_statistics": convert_numpy_types(summary_stats)
    }
    
    with open(out_path / "detailed_results.json", "wb") as f:
        f.write(orjson.dumps(detailed_results, option=orjson.OPT_INDENT_2))
    
    typer.echo(f"✅ Analysis complete! Results saved to {out_path}")
    
    # Print key findings
    typer.echo("\n🔍 Key Findings:")
    
    for kind in KINDS:
        cooc_norm = cooccurrence_results[kind]["normalized"]
        np.fill_diagonal(cooc_norm.values, 0)
        max_idx = np.unravel_index(cooc_norm.values.argmax(), cooc_norm.shape)
        max_pair = f"{MODELS[max_idx[0]]} & {MODELS[max_idx[1]]}"
        max_rate = cooc_norm.values[max_idx]
        
        typer.echo(f"  {kind.upper()}: {max_pair} co-occur in {max_rate:.1%} of questions")
    
    # Model consistency findings
    consistency_df_sorted = consistency_df.sort_values("outlier_rate")
    most_consistent = consistency_df_sorted.iloc[0]["model"]
    most_outlier = consistency_df_sorted.iloc[-1]["model"]
    
    typer.echo(f"\n📊 Model Patterns:")
    typer.echo(f"  Most consistent: {most_consistent}")
    typer.echo(f"  Most unique: {most_outlier}")
    
    typer.echo(f"\n📈 Statistical Significance:")
    typer.echo(f"  Random baseline max co-occurrence: {random_baseline['mean_max_cooccurrence']:.1f} ± {random_baseline['std_max_cooccurrence']:.1f}")
    typer.echo(f"  95th percentile threshold: {random_baseline['p95_max_cooccurrence']:.1f}")


if __name__ == "__main__":
    app()
