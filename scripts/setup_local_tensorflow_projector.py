#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["typer", "httpx"]
# ///
"""Setup Local TensorFlow Projector

This script helps you set up TensorFlow's Embedding Projector locally
for visualizing embeddings without relying on the online version.

Usage:
    uv run scripts/setup_local_tensorflow_projector.py --setup
    uv run scripts/setup_local_tensorflow_projector.py --serve
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import typer
import httpx

app = typer.Typer(add_completion=False, no_args_is_help=True)


def run_command(command: list[str], cwd: Path = None) -> bool:
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, check=True, cwd=cwd, capture_output=True, text=True)
        if result.stdout:
            typer.echo(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Error: {e}")
        if e.stderr:
            typer.echo(f"Error details: {e.stderr}")
        return False


@app.command()
def setup(
    target_dir: Path = typer.Option(
        Path("tensorflow_projector_local"),
        "--target-dir",
        help="Directory to clone the standalone projector"
    )
) -> None:
    """Set up the standalone TensorFlow Projector locally."""
    
    typer.echo("🔄 Setting up local TensorFlow Projector...")
    
    # Check if git is available
    if not run_command(["git", "--version"]):
        typer.echo("❌ Git is required but not found. Please install Git first.")
        raise typer.Exit(code=1)
    
    # Clone the standalone projector repository
    if target_dir.exists():
        typer.echo(f"📁 Directory {target_dir} already exists. Pulling latest changes...")
        if not run_command(["git", "pull"], cwd=target_dir):
            typer.echo("❌ Failed to pull latest changes")
            raise typer.Exit(code=1)
    else:
        typer.echo(f"📥 Cloning TensorFlow Embedding Projector to {target_dir}...")
        if not run_command([
            "git", "clone", 
            "https://github.com/tensorflow/embedding-projector-standalone.git", 
            str(target_dir)
        ]):
            typer.echo("❌ Failed to clone repository")
            raise typer.Exit(code=1)
    
    typer.echo("✅ Local TensorFlow Projector setup complete!")
    typer.echo(f"📁 Location: {target_dir.absolute()}")
    typer.echo("\nNext steps:")
    typer.echo(f"1. uv run scripts/setup_local_tensorflow_projector.py --serve --projector-dir {target_dir}")
    typer.echo("2. Open http://localhost:8000 in your browser")
    typer.echo("3. Load your TSV files from docs/tensorflow_projector/")


@app.command()
def serve(
    projector_dir: Path = typer.Option(
        Path("tensorflow_projector_local"),
        "--projector-dir",
        help="Directory containing the TensorFlow Projector"
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to serve on"
    ),
    copy_data: bool = typer.Option(
        True,
        "--copy-data/--no-copy-data",
        help="Copy TSV files to projector directory"
    )
) -> None:
    """Serve the local TensorFlow Projector."""
    
    if not projector_dir.exists():
        typer.echo(f"❌ Projector directory not found: {projector_dir}")
        typer.echo("Run: uv run scripts/setup_local_tensorflow_projector.py --setup")
        raise typer.Exit(code=1)
    
    # Copy our data files to the projector directory
    if copy_data:
        source_dir = Path("docs/tensorflow_projector")
        if source_dir.exists():
            typer.echo("📋 Copying TSV files to projector directory...")
            
            # Create a data subdirectory
            data_dir = projector_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Copy TSV files
            for file_name in ["embeddings.tsv", "metadata.tsv"]:
                source_file = source_dir / file_name
                if source_file.exists():
                    target_file = data_dir / file_name
                    target_file.write_bytes(source_file.read_bytes())
                    typer.echo(f"  ✅ Copied {file_name}")
                else:
                    typer.echo(f"  ⚠️  {file_name} not found in {source_dir}")
            
            # Create a simple index.html that redirects to the data
            index_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ethical Dilemmas - TensorFlow Projector</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        .file-links {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .file-links a {{ display: block; margin: 10px 0; padding: 10px; background: white; text-decoration: none; border-radius: 4px; }}
        .file-links a:hover {{ background: #e3f2fd; }}
    </style>
</head>
<body>
    <h1>🧠 Ethical Dilemmas - TensorFlow Projector</h1>
    
    <p>Welcome to the local TensorFlow Projector for ethical dilemma embeddings!</p>
    
    <h2>📁 Data Files</h2>
    <div class="file-links">
        <a href="data/embeddings.tsv">📊 embeddings.tsv - Embedding vectors (2,877 × 1024)</a>
        <a href="data/metadata.tsv">🏷️ metadata.tsv - Point metadata and labels</a>
    </div>
    
    <h2>🚀 Instructions</h2>
    <ol>
        <li>Open the <strong>TensorFlow Projector</strong> in your browser</li>
        <li>Click <strong>"Load"</strong> and upload both TSV files above</li>
        <li>Explore the visualization with different reduction techniques (PCA, t-SNE, UMAP)</li>
        <li>Color by different metadata fields (model, cluster, kind)</li>
    </ol>
    
    <h2>🔗 Quick Links</h2>
    <div class="file-links">
        <a href="index.html" onclick="window.open('index.html', '_blank'); return false;">🎯 Open TensorFlow Projector</a>
        <a href="https://projector.tensorflow.org/" target="_blank">🌐 Online TensorFlow Projector (Alternative)</a>
    </div>
    
    <h2>💡 Visualization Tips</h2>
    <ul>
        <li><strong>Color by model:</strong> See how different AI systems cluster</li>
        <li><strong>Color by cluster:</strong> Explore ethical reasoning patterns</li>
        <li><strong>Color by kind:</strong> Compare decision bodies vs. reasoning</li>
        <li><strong>Use t-SNE/UMAP:</strong> Better for discovering local structure</li>
        <li><strong>Search text:</strong> Find specific ethical scenarios</li>
    </ul>
</body>
</html>"""
            
            landing_page = projector_dir / "landing.html"
            landing_page.write_text(index_content, encoding="utf-8")
            typer.echo(f"  ✅ Created landing page at {landing_page}")
        else:
            typer.echo("⚠️  No TSV files found. Run the export script first:")
            typer.echo("uv run scripts/export_tensorflow_projector.py")
    
    # Start the server
    typer.echo(f"🚀 Starting local server on port {port}...")
    typer.echo(f"📁 Serving from: {projector_dir.absolute()}")
    typer.echo(f"🌐 Open: http://localhost:{port}")
    if copy_data:
        typer.echo(f"📊 Data landing page: http://localhost:{port}/landing.html")
    
    try:
        # Use Python's built-in HTTP server
        subprocess.run([
            sys.executable, "-m", "http.server", str(port)
        ], cwd=projector_dir, check=True)
    except KeyboardInterrupt:
        typer.echo("\n👋 Server stopped")
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Server failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def tensorboard_setup() -> None:
    """Instructions for using TensorBoard's Embedding Projector."""
    
    typer.echo("📊 TensorBoard Embedding Projector Setup")
    typer.echo("=" * 50)
    
    # Check if tensorboard is installed
    try:
        subprocess.run(["tensorboard", "--version"], check=True, capture_output=True)
        tensorboard_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        tensorboard_available = False
    
    if not tensorboard_available:
        typer.echo("📦 Installing TensorBoard...")
        typer.echo("Run: pip install tensorboard")
        typer.echo("Or: uv add tensorboard")
    else:
        typer.echo("✅ TensorBoard is available")
    
    typer.echo("\n🔧 Setup Steps:")
    typer.echo("1. Create a TensorBoard log directory:")
    typer.echo("   mkdir -p tb_logs/ethical_dilemmas")
    
    typer.echo("\n2. Copy our projector files:")
    typer.echo("   cp docs/tensorflow_projector/* tb_logs/ethical_dilemmas/")
    
    typer.echo("\n3. Start TensorBoard:")
    typer.echo("   tensorboard --logdir=tb_logs")
    
    typer.echo("\n4. Open the Projector tab in your browser")
    typer.echo("   Usually at: http://localhost:6006/#projector")
    
    typer.echo("\n💡 Advantages of TensorBoard:")
    typer.echo("   - Integrated with ML workflows")
    typer.echo("   - Better performance for large datasets")
    typer.echo("   - Advanced projector features")
    typer.echo("   - Can load from checkpoint files")


if __name__ == "__main__":
    app()
