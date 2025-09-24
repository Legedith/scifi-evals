#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi", "uvicorn", "numpy", "scikit-learn", "orjson", "typer"]
# ///
"""Dynamic Clustering API for Real-Time Cluster Adjustment

This FastAPI server provides endpoints for dynamic clustering with adjustable k values,
outlier detection, and real-time visualization updates.

Features:
- Adjustable cluster count (k) via API
- Real-time clustering with BisectingKMeans
- Outlier detection using silhouette scores
- Cluster quality metrics
- JSON export for visualization

Usage:
    uv run scripts/dynamic_clustering_api.py --port 8080
    
Then open: http://localhost:8080/docs for API documentation
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import orjson
import typer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# Global variables to cache data
_embeddings: Optional[np.ndarray] = None
_metadata: Optional[List[Tuple[int, str, str]]] = None
_merged_data: Optional[List[dict]] = None
_pca_2d: Optional[np.ndarray] = None

app = FastAPI(title="Dynamic Clustering API", description="Real-time clustering for ethical dilemma embeddings")

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def load_merged_data(path: Path) -> List[dict]:
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


def perform_dynamic_clustering(
    vectors: np.ndarray, 
    n_clusters: int,
    outlier_threshold: float = -0.1
) -> Dict[str, Any]:
    """Perform clustering and identify outliers."""
    
    # Ensure valid cluster count
    n_clusters = max(2, min(n_clusters, len(vectors) - 1))
    
    # Perform clustering
    cluster_model = BisectingKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=1000,
        random_state=42
    )
    
    labels = cluster_model.fit_predict(vectors)
    
    # Calculate quality metrics
    silhouette_avg = silhouette_score(vectors, labels)
    silhouette_per_sample = silhouette_samples(vectors, labels)
    
    # Identify outliers based on silhouette scores
    outliers = np.where(silhouette_per_sample < outlier_threshold)[0].tolist()
    
    # Find cluster representatives (closest to centroids)
    distances = np.linalg.norm(vectors[:, np.newaxis] - cluster_model.cluster_centers_, axis=2)
    representative_indices = np.argmin(distances, axis=0)
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_silhouettes = silhouette_per_sample[cluster_indices]
        
        cluster_stats[str(cluster_id)] = {
            "size": len(cluster_indices),
            "avg_silhouette": float(np.mean(cluster_silhouettes)),
            "min_silhouette": float(np.min(cluster_silhouettes)),
            "max_silhouette": float(np.max(cluster_silhouettes)),
            "representative_idx": int(representative_indices[cluster_id]),
            "outlier_count": int(np.sum(cluster_silhouettes < outlier_threshold))
        }
    
    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette_avg),
        "silhouette_per_sample": silhouette_per_sample.tolist(),
        "outliers": outliers,
        "outlier_threshold": outlier_threshold,
        "representative_indices": representative_indices.tolist(),
        "cluster_stats": cluster_stats,
        "inertia": float(cluster_model.inertia_)
    }


@app.on_event("startup")
async def load_data():
    """Load embeddings and data on startup."""
    global _embeddings, _metadata, _merged_data, _pca_2d
    
    db_path = Path("data/embeddings.sqlite3")
    merged_path = Path("data/merged/merged_dilemmas_responses.json")
    
    if not db_path.exists():
        raise RuntimeError(f"Database not found: {db_path}")
    if not merged_path.exists():
        raise RuntimeError(f"Merged data not found: {merged_path}")
    
    print(f"Loading embeddings from {db_path}...")
    _embeddings, _metadata = load_embeddings_from_db(db_path)
    print(f"Loaded {len(_embeddings)} embeddings")
    
    print("Loading merged data...")
    _merged_data = load_merged_data(merged_path)
    print(f"Loaded {len(_merged_data)} dilemmas")
    
    print("Computing 2D PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    _pca_2d = pca.fit_transform(_embeddings)
    print("Data loading complete!")


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "Dynamic Clustering API for Ethical Dilemmas",
        "endpoints": {
            "cluster": "/cluster/{k}",
            "outliers": "/outliers/{k}",
            "data_info": "/info",
            "visualization": "/viz"
        },
        "data_loaded": _embeddings is not None
    }


@app.get("/info")
async def data_info():
    """Get information about loaded data."""
    if _embeddings is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "n_embeddings": len(_embeddings),
        "embedding_dim": _embeddings.shape[1],
        "n_dilemmas": len(_merged_data) if _merged_data else 0,
        "models": list(set(model for _, model, _ in _metadata)) if _metadata else [],
        "kinds": list(set(kind for _, _, kind in _metadata)) if _metadata else []
    }


@app.get("/cluster/{k}")
async def cluster_data(k: int, outlier_threshold: float = -0.1):
    """Perform clustering with k clusters and return results."""
    if _embeddings is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if k < 2 or k > len(_embeddings) - 1:
        raise HTTPException(status_code=400, detail=f"k must be between 2 and {len(_embeddings) - 1}")
    
    # Perform clustering
    clustering_results = perform_dynamic_clustering(_embeddings, k, outlier_threshold)
    
    # Add 2D coordinates for visualization
    points = []
    for i, ((item_id, model, kind), (x, y), label, silhouette) in enumerate(
        zip(_metadata, _pca_2d, clustering_results["labels"], clustering_results["silhouette_per_sample"])
    ):
        is_outlier = i in clustering_results["outliers"]
        
        # Get text preview
        text_preview = ""
        if _merged_data:
            # Build decision lookup
            for item in _merged_data:
                if int(item.get("id", 0)) == item_id:
                    decision = item.get("decisions", {}).get(model, {})
                    text = get_text(decision, kind)
                    text_preview = text[:100] + "..." if len(text) > 100 else text
                    break
        
        points.append({
            "idx": i,
            "item_id": item_id,
            "model": model,
            "kind": kind,
            "x": float(x),
            "y": float(y),
            "cluster": int(label),
            "silhouette": float(silhouette),
            "is_outlier": is_outlier,
            "text_preview": text_preview
        })
    
    return {
        "clustering": clustering_results,
        "points": points,
        "success": True
    }


@app.get("/outliers/{k}")
async def get_outliers(k: int, outlier_threshold: float = -0.1):
    """Get outlier information for k clusters."""
    if _embeddings is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    clustering_results = perform_dynamic_clustering(_embeddings, k, outlier_threshold)
    outlier_indices = clustering_results["outliers"]
    
    outlier_details = []
    for idx in outlier_indices:
        item_id, model, kind = _metadata[idx]
        silhouette = clustering_results["silhouette_per_sample"][idx]
        
        # Get full text
        text = ""
        question = ""
        if _merged_data:
            for item in _merged_data:
                if int(item.get("id", 0)) == item_id:
                    question = item.get("question", "")
                    decision = item.get("decisions", {}).get(model, {})
                    text = get_text(decision, kind)
                    break
        
        outlier_details.append({
            "idx": idx,
            "item_id": item_id,
            "model": model,
            "kind": kind,
            "silhouette_score": float(silhouette),
            "cluster": int(clustering_results["labels"][idx]),
            "question": question,
            "text": text
        })
    
    return {
        "outliers": outlier_details,
        "outlier_count": len(outlier_details),
        "outlier_threshold": outlier_threshold,
        "total_points": len(_embeddings)
    }


@app.get("/similarity/{idx}")
async def get_similarity_neighbors(idx: int, threshold: float = 0.5):
    """Get similarity neighbors for a specific point."""
    if _embeddings is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    if idx < 0 or idx >= len(_embeddings):
        raise HTTPException(status_code=400, detail="Invalid point index")
    
    # Calculate similarities to all other points
    target_embedding = _embeddings[idx]
    similarities = np.dot(_embeddings, target_embedding)  # Already normalized
    
    # Find neighbors above threshold
    neighbors = []
    for i, sim in enumerate(similarities):
        if i != idx and sim >= threshold:
            neighbors.append({
                "idx": i,
                "similarity": float(sim),
                "item_id": _metadata[i][0],
                "model": _metadata[i][1],
                "kind": _metadata[i][2]
            })
    
    # Sort by similarity (highest first)
    neighbors.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "target_idx": idx,
        "threshold": threshold,
        "neighbors": neighbors,
        "neighbor_count": len(neighbors)
    }


@app.get("/network/{k}")
async def get_network_data(k: int, similarity_threshold: float = 0.5, max_connections: int = 5):
    """Get optimized network data for force-directed visualization."""
    if _embeddings is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Get clustering data
    clustering_results = perform_dynamic_clustering(_embeddings, k, -0.1)
    
    # Create nodes
    nodes = []
    for i, ((item_id, model, kind), label, silhouette) in enumerate(
        zip(_metadata, clustering_results["labels"], clustering_results["silhouette_per_sample"])
    ):
        # Get text preview
        text_preview = ""
        if _merged_data:
            for item in _merged_data:
                if int(item.get("id", 0)) == item_id:
                    decision = item.get("decisions", {}).get(model, {})
                    text = get_text(decision, kind)
                    text_preview = text[:100] + "..." if len(text) > 100 else text
                    break
        
        nodes.append({
            "id": i,
            "item_id": item_id,
            "model": model,
            "kind": kind,
            "cluster": int(label),
            "silhouette": float(silhouette),
            "is_outlier": i in clustering_results["outliers"],
            "text_preview": text_preview,
            "importance": float(abs(silhouette))
        })
    
    # Calculate similarity-based connections efficiently
    links = []
    similarity_matrix = np.dot(_embeddings, _embeddings.T)  # All pairwise similarities
    
    for i in range(len(nodes)):
        # Get similarities for this node
        node_similarities = similarity_matrix[i]
        
        # Find top similar nodes above threshold
        similar_indices = np.where(node_similarities >= similarity_threshold)[0]
        similar_indices = similar_indices[similar_indices != i]  # Remove self
        
        # Sort by similarity and take top connections
        if len(similar_indices) > 0:
            similarities_subset = node_similarities[similar_indices]
            sorted_indices = similar_indices[np.argsort(similarities_subset)[::-1]]
            top_connections = sorted_indices[:max_connections]
            
            for j in top_connections:
                if i < j:  # Avoid duplicate links
                    links.append({
                        "source": i,
                        "target": int(j),
                        "similarity": float(similarity_matrix[i, j])
                    })
    
    return {
        "nodes": nodes,
        "links": links,
        "clustering": {
            "k": k,
            "silhouette_score": clustering_results["silhouette_score"],
            "inertia": clustering_results["inertia"]
        },
        "similarity_threshold": similarity_threshold,
        "max_connections": max_connections
    }


@app.get("/viz", response_class=HTMLResponse)
async def visualization():
    """Serve the interactive force-directed network visualization."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Network Clustering - Ethical Dilemmas</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            overflow: hidden;
        }
        .container {
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            text-align: center;
            flex-shrink: 0;
        }
        .header h1 {
            margin: 0;
            font-size: 1.8em;
            font-weight: 300;
        }
        .controls {
            padding: 15px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
            flex-shrink: 0;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider {
            width: 150px;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        .slider:hover {
            opacity: 1;
        }
        .slider::-webkit-slider-thumb {
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        .metrics {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .metric {
            background: white;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            min-width: 100px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.75em;
            color: #666;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }
        .visualization {
            flex: 1;
            position: relative;
            overflow: hidden;
        }
        .network-container {
            width: 100%;
            height: 100%;
            position: relative;
        }
        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 1.1em;
            color: #666;
            z-index: 100;
        }
        .tooltip {
            position: absolute;
            padding: 10px 15px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 280px;
            z-index: 1000;
            line-height: 1.4;
        }
        .selection-panel {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            max-width: 300px;
            max-height: 60%;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            display: none;
        }
        .selection-header {
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .selected-item {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 8px;
            margin: 5px 0;
            font-size: 0.85em;
            line-height: 1.3;
        }
        .node {
            cursor: pointer;
            stroke-width: 1.5px;
        }
        .node.selected {
            stroke: #ff6b6b;
            stroke-width: 3px;
        }
        .node.connected {
            stroke: #4ecdc4;
            stroke-width: 2px;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1px;
        }
        .link.active {
            stroke: #4ecdc4;
            stroke-width: 2px;
            stroke-opacity: 0.8;
        }
        .controls-toggle {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 0.9em;
            z-index: 200;
        }
        .network-controls {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .network-controls h4 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            color: #333;
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8em;
            margin: 2px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .btn.secondary {
            background: #6c757d;
        }
        .btn.secondary:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Interactive Network Clustering Explorer</h1>
            <p>Explore connections and clustering through similarity networks</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="clusterSlider"><strong>Clusters (k):</strong></label>
                <input type="range" id="clusterSlider" class="slider" min="2" max="50" value="15">
                <span id="clusterValue">15</span>
            </div>
            
            <div class="control-group">
                <label for="similaritySlider"><strong>Similarity Threshold:</strong></label>
                <input type="range" id="similaritySlider" class="slider" min="0.1" max="0.9" step="0.05" value="0.5">
                <span id="similarityValue">0.5</span>
            </div>
            
            <div class="control-group">
                <label for="forceStrength"><strong>Force Strength:</strong></label>
                <input type="range" id="forceStrength" class="slider" min="0.1" max="2.0" step="0.1" value="0.8">
                <span id="forceValue">0.8</span>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Clusters</div>
                    <div class="metric-value" id="clusterCount">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Connections</div>
                    <div class="metric-value" id="connectionCount">-</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Selected</div>
                    <div class="metric-value" id="selectedCount">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Isolated</div>
                    <div class="metric-value" id="isolatedCount">-</div>
                </div>
            </div>
        </div>
        
        <div class="visualization">
            <div id="loadingMessage" class="loading">Loading network data...</div>
            
            <div class="controls-toggle" id="controlsToggle">
                🎛️ Controls
            </div>
            
            <div class="network-container">
                <svg id="network"></svg>
                <canvas id="networkCanvas" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
            </div>
            
            <div class="tooltip" id="tooltip"></div>
            
            <div class="selection-panel" id="selectionPanel">
                <div class="selection-header" id="selectionHeader">No Selection</div>
                <div id="selectionContent"></div>
            </div>
            
            <div class="network-controls">
                <h4>🎯 Network Controls</h4>
                <button class="btn" onclick="restartSimulation()">🔄 Restart</button>
                <button class="btn" onclick="centerView()">🎯 Center</button>
                <button class="btn" onclick="clearSelection()">❌ Clear</button>
                <button class="btn secondary" onclick="togglePhysics()">⚡ Physics</button>
                <button class="btn secondary" onclick="exportSelection()">💾 Export</button>
            </div>
        </div>
    </div>

    <script>
        // Configuration
        const width = window.innerWidth - 40;
        const height = window.innerHeight - 200;
        
        // D3 setup
        const svg = d3.select("#network")
            .attr("width", width)
            .attr("height", height);
            
        const canvas = d3.select("#networkCanvas")
            .attr("width", width)
            .attr("height", height);
            
        const context = canvas.node().getContext("2d");
        const tooltip = d3.select("#tooltip");
        const selectionPanel = d3.select("#selectionPanel");
        
        // Color scales
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        const sizeScale = d3.scaleLinear().domain([0, 1]).range([3, 8]);
        
        // State
        let nodes = [];
        let links = [];
        let simulation;
        let currentData = null;
        let selectedNodes = new Set();
        let physicsEnabled = true;
        
        // Controls
        const clusterSlider = document.getElementById('clusterSlider');
        const clusterValue = document.getElementById('clusterValue');
        const similaritySlider = document.getElementById('similaritySlider');
        const similarityValue = document.getElementById('similarityValue');
        const forceSlider = document.getElementById('forceStrength');
        const forceValue = document.getElementById('forceValue');
        
        // Event listeners
        clusterSlider.addEventListener('input', (e) => {
            clusterValue.textContent = e.target.value;
            updateClustering();
        });
        
        similaritySlider.addEventListener('input', (e) => {
            similarityValue.textContent = e.target.value;
            updateSimilarityNetwork();
        });
        
        forceSlider.addEventListener('input', (e) => {
            forceValue.textContent = e.target.value;
            updateForceStrength();
        });
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            setupSimulation();
            updateClustering();
        });
        
        function setupSimulation() {
            simulation = d3.forceSimulation()
                .force("link", d3.forceLink().id(d => d.id).distance(50))
                .force("charge", d3.forceManyBody().strength(-100))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => sizeScale(d.importance || 0.5) + 2))
                .on("tick", ticked);
                
            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 10])
                .on("zoom", zoomed);
                
            svg.call(zoom);
            canvas.call(zoom);
            
            // Click handler for canvas
            canvas.on("click", handleCanvasClick);
        }
        
        function zoomed(event) {
            const transform = event.transform;
            svg.attr("transform", transform);
            
            // Redraw canvas with transform
            context.save();
            context.clearRect(0, 0, width, height);
            context.translate(transform.x, transform.y);
            context.scale(transform.k, transform.k);
            drawCanvas();
            context.restore();
        }
        
        function ticked() {
            drawCanvas();
        }
        
        function drawCanvas() {
            context.clearRect(0, 0, width, height);
            
            // Draw links
            context.globalAlpha = 0.6;
            links.forEach(link => {
                context.beginPath();
                context.moveTo(link.source.x, link.source.y);
                context.lineTo(link.target.x, link.target.y);
                context.strokeStyle = link.active ? "#4ecdc4" : "#999";
                context.lineWidth = link.active ? 2 : 1;
                context.stroke();
            });
            
            // Draw nodes
            context.globalAlpha = 0.8;
            nodes.forEach(node => {
                context.beginPath();
                context.arc(node.x, node.y, sizeScale(node.importance || 0.5), 0, 2 * Math.PI);
                context.fillStyle = colorScale(node.cluster);
                context.fill();
                
                // Draw selection highlight
                if (selectedNodes.has(node.id)) {
                    context.strokeStyle = "#ff6b6b";
                    context.lineWidth = 3;
                    context.stroke();
                } else if (node.connected) {
                    context.strokeStyle = "#4ecdc4";
                    context.lineWidth = 2;
                    context.stroke();
                }
            });
        }
        
        function handleCanvasClick(event) {
            const [x, y] = d3.pointer(event);
            const clickedNode = findNodeAt(x, y);
            
            if (clickedNode) {
                if (event.ctrlKey || event.metaKey) {
                    // Multi-select
                    if (selectedNodes.has(clickedNode.id)) {
                        selectedNodes.delete(clickedNode.id);
                    } else {
                        selectedNodes.add(clickedNode.id);
                    }
                } else {
                    // Single select
                    selectedNodes.clear();
                    selectedNodes.add(clickedNode.id);
                }
                
                updateSelection();
                highlightConnections(clickedNode);
            } else {
                // Clear selection
                clearSelection();
            }
        }
        
        function findNodeAt(x, y) {
            for (let node of nodes) {
                const distance = Math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2);
                if (distance <= sizeScale(node.importance || 0.5)) {
                    return node;
                }
            }
            return null;
        }
        
        async function updateClustering() {
            const k = clusterSlider.value;
            
            document.getElementById('loadingMessage').style.display = 'block';
            
            try {
                const response = await fetch(`/cluster/${k}?outlier_threshold=-0.1`);
                const data = await response.json();
                
                if (data.success) {
                    currentData = data;
                    
                    // Convert points to nodes
                    nodes = data.points.map(point => ({
                        id: point.idx,
                        x: Math.random() * width,
                        y: Math.random() * height,
                        cluster: point.cluster,
                        model: point.model,
                        kind: point.kind,
                        text_preview: point.text_preview,
                        silhouette: point.silhouette,
                        is_outlier: point.is_outlier,
                        importance: Math.abs(point.silhouette),
                        item_id: point.item_id
                    }));
                    
                    document.getElementById('clusterCount').textContent = k;
                    document.getElementById('loadingMessage').style.display = 'none';
                    
                    updateSimilarityNetwork();
                }
            } catch (error) {
                console.error('Error fetching clustering data:', error);
                document.getElementById('loadingMessage').textContent = 'Error loading data';
            }
        }
        
        async function updateSimilarityNetwork() {
            const threshold = parseFloat(similaritySlider.value);
            const k = clusterSlider.value;
            
            try {
                // Fetch real similarity-based network data
                const response = await fetch(`/network/${k}?similarity_threshold=${threshold}&max_connections=5`);
                const networkData = await response.json();
                
                // Update nodes and links with real data
                nodes = networkData.nodes.map(node => ({
                    ...node,
                    x: nodes.find(n => n.id === node.id)?.x || Math.random() * width,
                    y: nodes.find(n => n.id === node.id)?.y || Math.random() * height
                }));
                
                links = networkData.links.map(link => ({
                    ...link,
                    active: true
                }));
                
                // Update simulation
                simulation.nodes(nodes);
                simulation.force("link").links(links);
                simulation.alpha(0.5).restart();
                
                // Update metrics
                document.getElementById('connectionCount').textContent = links.length;
                
                const isolatedNodes = nodes.filter(node => 
                    !links.some(link => link.source === node.id || link.target === node.id)
                );
                document.getElementById('isolatedCount').textContent = isolatedNodes.length;
                
            } catch (error) {
                console.error('Error fetching network data:', error);
                // Fallback to simple clustering if network fails
                updateSimpleNetwork(threshold);
            }
        }
        
        function updateSimpleNetwork(threshold) {
            // Fallback network based on clustering only
            links = [];
            
            for (let i = 0; i < Math.min(nodes.length, 500); i++) {
                const node = nodes[i];
                
                const similarNodes = nodes.filter(other => 
                    other.id !== node.id &&
                    other.cluster === node.cluster &&
                    Math.abs(other.silhouette - node.silhouette) < (1 - threshold)
                );
                
                similarNodes.slice(0, 3).forEach(similar => {
                    const similarity = 1 - Math.abs(similar.silhouette - node.silhouette);
                    if (similarity >= threshold) {
                        links.push({
                            source: node.id,
                            target: similar.id,
                            similarity: similarity,
                            active: true
                        });
                    }
                });
            }
            
            simulation.nodes(nodes);
            simulation.force("link").links(links);
            simulation.alpha(0.5).restart();
            
            document.getElementById('connectionCount').textContent = links.length;
            
            const isolatedNodes = nodes.filter(node => 
                !links.some(link => link.source.id === node.id || link.target.id === node.id)
            );
            document.getElementById('isolatedCount').textContent = isolatedNodes.length;
        }
        
        function updateForceStrength() {
            const strength = parseFloat(forceSlider.value);
            simulation.force("charge").strength(-50 * strength);
            simulation.force("link").strength(0.1 * strength);
            simulation.alpha(0.3).restart();
        }
        
        function highlightConnections(targetNode) {
            // Reset all connections
            links.forEach(link => link.active = false);
            nodes.forEach(node => node.connected = false);
            
            // Highlight connections
            const connectedNodeIds = new Set();
            links.forEach(link => {
                if (link.source.id === targetNode.id || link.target.id === targetNode.id) {
                    link.active = true;
                    connectedNodeIds.add(link.source.id);
                    connectedNodeIds.add(link.target.id);
                }
            });
            
            nodes.forEach(node => {
                node.connected = connectedNodeIds.has(node.id);
            });
        }
        
        function updateSelection() {
            const selectedNodeData = nodes.filter(node => selectedNodes.has(node.id));
            
            document.getElementById('selectedCount').textContent = selectedNodes.size;
            
            if (selectedNodes.size > 0) {
                selectionPanel.style("display", "block");
                
                const header = selectedNodes.size === 1 ? 
                    `Selected Node: ${selectedNodeData[0].model}` :
                    `Selected Nodes: ${selectedNodes.size}`;
                    
                document.getElementById('selectionHeader').textContent = header;
                
                const content = selectedNodeData.map(node => `
                    <div class="selected-item">
                        <strong>${node.model}</strong> (${node.kind})<br>
                        <strong>Cluster:</strong> ${node.cluster}<br>
                        <strong>Silhouette:</strong> ${node.silhouette.toFixed(3)}<br>
                        <strong>Text:</strong> ${node.text_preview}
                    </div>
                `).join('');
                
                document.getElementById('selectionContent').innerHTML = content;
            } else {
                selectionPanel.style("display", "none");
            }
        }
        
        function clearSelection() {
            selectedNodes.clear();
            nodes.forEach(node => node.connected = false);
            links.forEach(link => link.active = false);
            updateSelection();
        }
        
        function restartSimulation() {
            simulation.alpha(1).restart();
        }
        
        function centerView() {
            const zoom = d3.zoom();
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(width / 2, height / 2).scale(1)
            );
        }
        
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            if (physicsEnabled) {
                simulation.alpha(0.3).restart();
            } else {
                simulation.stop();
            }
        }
        
        function exportSelection() {
            if (selectedNodes.size === 0) {
                alert('No nodes selected');
                return;
            }
            
            const selectedData = nodes.filter(node => selectedNodes.has(node.id));
            const dataStr = JSON.stringify(selectedData, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `selected_nodes_${selectedNodes.size}.json`;
            link.click();
            URL.revokeObjectURL(url);
        }
        
        // Add hover tooltips to canvas
        canvas.on("mousemove", function(event) {
            const [x, y] = d3.pointer(event);
            const hoveredNode = findNodeAt(x, y);
            
            if (hoveredNode) {
                tooltip.style("opacity", 1)
                    .html(`
                        <strong>Model:</strong> ${hoveredNode.model}<br>
                        <strong>Type:</strong> ${hoveredNode.kind}<br>
                        <strong>Cluster:</strong> ${hoveredNode.cluster}<br>
                        <strong>Silhouette:</strong> ${hoveredNode.silhouette.toFixed(3)}<br>
                        <strong>Outlier:</strong> ${hoveredNode.is_outlier ? 'Yes' : 'No'}<br>
                        <strong>Connections:</strong> ${links.filter(l => l.source.id === hoveredNode.id || l.target.id === hoveredNode.id).length}<br>
                        <strong>Text:</strong> ${hoveredNode.text_preview}
                    `)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            } else {
                tooltip.style("opacity", 0);
            }
        });
        
        canvas.on("mouseleave", () => {
            tooltip.style("opacity", 0);
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


def main(
    port: int = typer.Option(8080, "--port", help="Port to run the server on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development")
):
    """Run the dynamic clustering API server."""
    print(f"🚀 Starting Dynamic Clustering API on {host}:{port}")
    print(f"📊 Interactive interface: http://{host}:{port}/viz")
    print(f"📖 API docs: http://{host}:{port}/docs")
    
    uvicorn.run(
        "dynamic_clustering_api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    typer.run(main)
