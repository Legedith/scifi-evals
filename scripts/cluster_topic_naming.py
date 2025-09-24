#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "orjson", "typer", "httpx"]
# ///
"""Automatic Topic Naming for Ethical Dilemma Clusters

This script generates meaningful topic names for clusters using LLM analysis,
inspired by the movie quotes topic naming approach but adapted for ethical dilemmas.

Features:
- Extracts representative texts from each cluster
- Uses LLM to generate descriptive topic names
- Supports different LLM providers (OpenAI-compatible APIs)
- Exports enhanced cluster data with topic names

Usage:
    uv run scripts/cluster_topic_naming.py --clusters data/analysis/clusters.json --merged data/merged/merged_dilemmas_responses.json --out data/analysis/named_clusters.json --api-key YOUR_API_KEY
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import orjson
import typer
import httpx

app = typer.Typer(add_completion=False, no_args_is_help=True)


def load_clusters(path: Path) -> dict:
    """Load cluster analysis results."""
    return orjson.loads(path.read_bytes())


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


def extract_cluster_examples(
    cluster_data: dict,
    merged_data: List[dict],
    examples_per_cluster: int = 5
) -> Dict[int, List[Dict[str, Any]]]:
    """Extract representative examples for each cluster."""
    
    # Build decision lookup
    decision_map: Dict[tuple[int, str], dict] = {}
    item_lookup: Dict[int, dict] = {}
    
    for item in merged_data:
        item_id = int(item.get("id", 0))
        item_lookup[item_id] = item
        for model, decision in (item.get("decisions") or {}).items():
            decision_map[(item_id, model)] = decision
    
    # Get clustering results
    labels = cluster_data["clustering"]["labels"]
    silhouette_scores = cluster_data["clustering"]["silhouette_per_sample"]
    cluster_stats = cluster_data["cluster_statistics"]
    
    cluster_examples = {}
    n_clusters = max(labels) + 1
    
    for cluster_id in range(n_clusters):
        # Get indices for this cluster, sorted by silhouette score (quality)
        cluster_indices = [
            (i, silhouette_scores[i]) 
            for i, label in enumerate(labels) 
            if label == cluster_id
        ]
        cluster_indices.sort(key=lambda x: x[1], reverse=True)  # Best examples first
        
        # Extract top examples
        examples = []
        for i, (idx, silhouette_score) in enumerate(cluster_indices[:examples_per_cluster]):
            # Get metadata from representative_texts if available
            rep_text = None
            for rep in cluster_data.get("representative_texts", []):
                if rep["representative_index"] == idx:
                    rep_text = rep
                    break
            
            if rep_text:
                examples.append({
                    "index": idx,
                    "item_id": rep_text["item_id"],
                    "model": rep_text["model"],
                    "kind": rep_text["kind"],
                    "text": rep_text["text"],
                    "question": rep_text["question"],
                    "source": rep_text["source"],
                    "silhouette_score": float(silhouette_score),
                    "rank_in_cluster": i + 1
                })
        
        cluster_examples[str(cluster_id)] = examples
    
    return cluster_examples


async def call_llm_for_topic_names(
    cluster_examples: Dict[int, List[Dict[str, Any]]],
    api_base: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """Call LLM to generate topic names for clusters."""
    
    # Prepare prompt with cluster examples
    clusters_text = []
    for cluster_id, examples in cluster_examples.items():
        cluster_text = f"Cluster {cluster_id} examples:\n"
        for i, example in enumerate(examples[:3]):  # Top 3 examples
            cluster_text += f"  {i+1}. Question: {example['question'][:100]}...\n"
            cluster_text += f"     {example['model']} ({example['kind']}): {example['text'][:150]}...\n"
        clusters_text.append(cluster_text)
    
    prompt = f"""Analyze these {len(cluster_examples)} clusters of ethical dilemma responses from AI models.
Each cluster contains similar reasoning patterns or ethical approaches.

Generate concise 2-4 word topic names for each cluster that capture the main ethical theme or reasoning pattern.
Return a JSON array of exactly {len(cluster_examples)} topic names in order.

{chr(10).join(clusters_text)}

Focus on:
- Ethical frameworks (utilitarian, deontological, virtue ethics)
- Decision patterns (risk assessment, stakeholder analysis, etc.)
- Moral themes (autonomy, justice, beneficence, etc.)
- Reasoning styles (pragmatic, principled, consequentialist, etc.)

Return only a JSON array of strings."""

    # Make API call
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            },
            timeout=60.0
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Extract JSON from response
        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            topic_names = json.loads(json_match.group(0))
        else:
            topic_names = json.loads(content)
        
        return topic_names


async def generate_cluster_descriptions(
    cluster_examples: Dict[int, List[Dict[str, Any]]],
    topic_names: List[str],
    api_base: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """Generate detailed descriptions for each cluster."""
    
    descriptions = []
    
    for cluster_id, topic_name in enumerate(topic_names):
        examples = cluster_examples.get(str(cluster_id), [])
        if not examples:
            descriptions.append("")
            continue
        
        # Create prompt for this specific cluster
        cluster_text = f"Topic: {topic_name}\n\nExamples:\n"
        for i, example in enumerate(examples[:3]):
            cluster_text += f"{i+1}. Dilemma: {example['question']}\n"
            cluster_text += f"   Response ({example['model']}, {example['kind']}): {example['text'][:200]}...\n\n"
        
        prompt = f"""Analyze this cluster of ethical dilemma responses labeled "{topic_name}".

{cluster_text}

Write a 2-3 sentence description explaining:
1. What ethical approach or reasoning pattern unifies this cluster
2. How the AI models in this cluster tend to approach ethical decisions
3. What makes this cluster distinct from others

Be concise and focus on the ethical reasoning patterns."""

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 200
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                description = result["choices"][0]["message"]["content"].strip()
                descriptions.append(description)
                
            except Exception as e:
                typer.echo(f"Warning: Failed to generate description for cluster {cluster_id}: {e}")
                descriptions.append("")
    
    return descriptions


@app.command()
def main(
    clusters_path: Path = typer.Option(
        Path("data/analysis/clusters.json"),
        "--clusters",
        help="Path to cluster analysis results"
    ),
    merged_path: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"),
        "--merged",
        help="Path to merged dilemmas data"
    ),
    out_path: Path = typer.Option(
        Path("data/analysis/named_clusters.json"),
        "--out",
        help="Output path for clusters with topic names"
    ),
    api_base: str = typer.Option(
        "https://api.openai.com/v1",
        "--api-base",
        help="LLM API base URL"
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        help="LLM API key"
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        help="LLM model name"
    ),
    examples_per_cluster: int = typer.Option(
        5,
        "--examples-per-cluster",
        help="Number of examples to extract per cluster"
    )
) -> None:
    """Generate topic names for ethical dilemma clusters using LLM analysis."""
    
    if not api_key:
        typer.echo("Error: --api-key is required")
        raise typer.Exit(code=1)
    
    typer.echo(f"Loading cluster data from {clusters_path}...")
    cluster_data = load_clusters(clusters_path)
    n_clusters = cluster_data["clustering"]["n_clusters"]
    typer.echo(f"Found {n_clusters} clusters")
    
    typer.echo("Loading merged dilemma data...")
    merged_data = load_merged(merged_path)
    
    typer.echo(f"Extracting {examples_per_cluster} examples per cluster...")
    cluster_examples = extract_cluster_examples(cluster_data, merged_data, examples_per_cluster)
    
    typer.echo(f"Generating topic names using {model}...")
    
    async def run_topic_naming():
        try:
            topic_names = await call_llm_for_topic_names(
                cluster_examples, api_base, api_key, model
            )
            
            if len(topic_names) != n_clusters:
                typer.echo(f"Warning: Expected {n_clusters} topic names, got {len(topic_names)}")
                # Pad or truncate as needed
                while len(topic_names) < n_clusters:
                    topic_names.append(f"Cluster {len(topic_names)}")
                topic_names = topic_names[:n_clusters]
            
            typer.echo("Generated topic names:")
            for i, name in enumerate(topic_names):
                typer.echo(f"  Cluster {i}: {name}")
            
            typer.echo("Generating cluster descriptions...")
            descriptions = await generate_cluster_descriptions(
                cluster_examples, topic_names, api_base, api_key, model
            )
            
            return topic_names, descriptions
            
        except Exception as e:
            typer.echo(f"Error calling LLM API: {e}")
            # Fallback to generic names
            topic_names = [f"Ethical Cluster {i}" for i in range(n_clusters)]
            descriptions = ["" for _ in range(n_clusters)]
            return topic_names, descriptions
    
    # Run async function
    import asyncio
    topic_names, descriptions = asyncio.run(run_topic_naming())
    
    # Enhance cluster data with topic names
    enhanced_data = cluster_data.copy()
    enhanced_data["topic_names"] = topic_names
    enhanced_data["topic_descriptions"] = descriptions
    enhanced_data["cluster_examples"] = cluster_examples
    enhanced_data["topic_naming_metadata"] = {
        "api_base": api_base,
        "model": model,
        "examples_per_cluster": examples_per_cluster,
        "generation_timestamp": None  # Could add timestamp
    }
    
    # Save enhanced results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(enhanced_data, option=orjson.OPT_INDENT_2))
    typer.echo(f"Enhanced cluster data saved to {out_path}")
    
    # Print summary
    typer.echo("\n=== Topic Naming Summary ===")
    for i, (name, desc) in enumerate(zip(topic_names, descriptions)):
        cluster_size = cluster_data["cluster_statistics"][str(i)]["size"]
        typer.echo(f"\nCluster {i}: {name} ({cluster_size} points)")
        if desc:
            typer.echo(f"  {desc}")


if __name__ == "__main__":
    app()
