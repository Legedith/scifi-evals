#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer", "httpx"]
# ///
"""Enhanced Analysis Workflow for Ethical Dilemma Embeddings

This script orchestrates the complete enhanced analysis workflow:
1. Enhanced clustering with BisectingKMeans
2. Automatic topic naming using LLM
3. TensorFlow Projector export
4. Summary report generation

Usage:
    # Run complete analysis with topic naming
    uv run scripts/run_enhanced_analysis.py --api-key YOUR_API_KEY --n-clusters 25

    # Run without topic naming (skip LLM calls)
    uv run scripts/run_enhanced_analysis.py --skip-topic-naming --n-clusters 25
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    typer.echo(f"\n🔄 {description}")
    typer.echo(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            typer.echo(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Error: {e}")
        if e.stderr:
            typer.echo(f"Error details: {e.stderr}")
        return False


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
    n_clusters: int = typer.Option(
        25,
        "--n-clusters",
        help="Number of clusters for analysis"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="LLM API key for topic naming (optional)"
    ),
    api_base: str = typer.Option(
        "https://api.openai.com/v1",
        "--api-base",
        help="LLM API base URL"
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        help="LLM model for topic naming"
    ),
    skip_topic_naming: bool = typer.Option(
        False,
        "--skip-topic-naming",
        help="Skip topic naming step"
    ),
    output_dir: Path = typer.Option(
        Path("data/analysis"),
        "--output-dir",
        help="Base output directory"
    )
) -> None:
    """Run the complete enhanced analysis workflow."""
    
    # Validate inputs
    if not db_path.exists():
        typer.echo(f"❌ Database not found: {db_path}")
        typer.echo("Run: uv run scripts/embed_cache.py first")
        raise typer.Exit(code=1)
    
    if not merged_path.exists():
        typer.echo(f"❌ Merged data not found: {merged_path}")
        typer.echo("Run: uv run scripts/merge_responses.py first")
        raise typer.Exit(code=1)
    
    if not skip_topic_naming and not api_key:
        typer.echo("❌ API key required for topic naming (use --api-key or --skip-topic-naming)")
        raise typer.Exit(code=1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    tf_projector_dir = Path("docs/tensorflow_projector")
    tf_projector_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    clusters_file = output_dir / "clusters.json"
    named_clusters_file = output_dir / "named_clusters.json"
    
    typer.echo("🚀 Starting Enhanced Analysis Workflow")
    typer.echo(f"Database: {db_path}")
    typer.echo(f"Merged data: {merged_path}")
    typer.echo(f"Clusters: {n_clusters}")
    typer.echo(f"Output: {output_dir}")
    
    success_count = 0
    total_steps = 3 + (0 if skip_topic_naming else 1)
    
    # Step 1: Enhanced Clustering
    clustering_cmd = [
        "uv", "run", "scripts/enhanced_clustering.py",
        "--db", str(db_path),
        "--merged", str(merged_path),
        "--out", str(clusters_file),
        "--n-clusters", str(n_clusters)
    ]
    
    if run_command(clustering_cmd, f"Step 1/{total_steps}: Enhanced Clustering"):
        success_count += 1
        typer.echo("✅ Clustering complete")
    else:
        typer.echo("❌ Clustering failed")
        raise typer.Exit(code=1)
    
    # Step 2: Topic Naming (optional)
    clusters_for_export = clusters_file
    if not skip_topic_naming:
        topic_naming_cmd = [
            "uv", "run", "scripts/cluster_topic_naming.py",
            "--clusters", str(clusters_file),
            "--merged", str(merged_path),
            "--out", str(named_clusters_file),
            "--api-base", api_base,
            "--api-key", api_key,
            "--model", model
        ]
        
        if run_command(topic_naming_cmd, f"Step 2/{total_steps}: Topic Naming"):
            success_count += 1
            typer.echo("✅ Topic naming complete")
            clusters_for_export = named_clusters_file
        else:
            typer.echo("❌ Topic naming failed, continuing without names")
            clusters_for_export = clusters_file
    else:
        typer.echo("⏭️  Skipping topic naming")
    
    # Step 3: TensorFlow Projector Export  
    step_num = 3 if skip_topic_naming else 3
    export_cmd = [
        "uv", "run", "scripts/export_tensorflow_projector.py",
        "--db", str(db_path),
        "--merged", str(merged_path),
        "--clusters", str(clusters_for_export),
        "--out-dir", str(tf_projector_dir)
    ]
    
    if run_command(export_cmd, f"Step {step_num}/{total_steps}: TensorFlow Projector Export"):
        success_count += 1
        typer.echo("✅ TensorFlow Projector export complete")
    else:
        typer.echo("❌ Export failed")
    
    # Step 4: Generate summary report
    typer.echo(f"\n🔄 Step {total_steps}/{total_steps}: Generating Summary Report")
    
    # Create a summary markdown file
    summary_file = output_dir / "analysis_summary.md"
    summary_content = f"""# Enhanced Analysis Summary

## Overview
- **Analysis Date**: Generated automatically
- **Embeddings**: {db_path}
- **Source Data**: {merged_path}  
- **Clusters**: {n_clusters}
- **Topic Naming**: {"✅ Enabled" if not skip_topic_naming else "❌ Skipped"}

## Files Generated

### Clustering Analysis
- `{clusters_file.name}`: Core clustering results with BisectingKMeans
{f"- `{named_clusters_file.name}`: Enhanced with LLM-generated topic names" if not skip_topic_naming else ""}

### TensorFlow Projector
- `{tf_projector_dir}/embeddings.tsv`: 1024D embeddings for visualization
- `{tf_projector_dir}/metadata.tsv`: Point metadata and cluster labels
- `{tf_projector_dir}/projector_config.pbtxt`: TensorBoard configuration
- `{tf_projector_dir}/README.md`: Detailed usage instructions

## Next Steps

### 1. Explore with TensorFlow Projector
```bash
# Option A: Online (Recommended)
# 1. Go to https://projector.tensorflow.org/
# 2. Upload embeddings.tsv and metadata.tsv from {tf_projector_dir}/

# Option B: Local TensorBoard
pip install tensorboard
tensorboard --logdir={tf_projector_dir.parent}
```

### 2. Analyze Results
- Color by **model** to see AI system differences
- Color by **kind** to compare decision types (body/in_favor/against)
- Color by **cluster** to explore ethical reasoning patterns
{"- Review **topic names** for semantic understanding" if not skip_topic_naming else ""}
- Use **t-SNE or UMAP** for better local structure visualization

### 3. Further Analysis
- Compare cluster distributions across AI models
- Identify outlier ethical positions
- Analyze consensus vs. disagreement patterns
- Export specific clusters for detailed study

## Command Reference

```bash
# Re-run clustering with different parameters
uv run scripts/enhanced_clustering.py --n-clusters 30

# Add topic naming to existing clusters  
uv run scripts/cluster_topic_naming.py --api-key YOUR_KEY

# Export for different visualization tools
uv run scripts/export_tensorflow_projector.py

# Complete workflow
uv run scripts/run_enhanced_analysis.py --api-key YOUR_KEY --n-clusters 25
```
"""

    summary_file.write_text(summary_content, encoding="utf-8")
    typer.echo(f"📋 Summary report: {summary_file}")
    
    # Final status
    typer.echo(f"\n🎉 Workflow Complete!")
    typer.echo(f"✅ {success_count}/{total_steps} steps successful")
    
    if success_count == total_steps:
        typer.echo(f"\n🔗 Quick Start:")
        typer.echo(f"1. Visit: https://projector.tensorflow.org/")
        typer.echo(f"2. Upload: {tf_projector_dir}/embeddings.tsv")
        typer.echo(f"3. Upload: {tf_projector_dir}/metadata.tsv") 
        typer.echo(f"4. Explore ethical reasoning patterns!")
        typer.echo(f"\n📁 Output directory: {output_dir}")
        typer.echo(f"📁 TensorFlow Projector files: {tf_projector_dir}")
    else:
        typer.echo("⚠️  Some steps failed - check outputs above")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
