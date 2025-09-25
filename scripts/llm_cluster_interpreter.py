#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "orjson", "typer", "httpx", "python-dotenv", "pydantic"]
# ///
"""LLM-Powered Cluster Interpretation

Uses Grok-4-fast to provide semantic insights about model clustering patterns.
This demonstrates how LLMs can interpret quantitative analysis results.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import orjson
import typer
from dotenv import load_dotenv

# Import our LLM provider (assuming it's in the same directory)
import sys
sys.path.append(str(Path(__file__).parent))
from llm_providers import OpenRouterProvider

app = typer.Typer(add_completion=False, no_args_is_help=True)

def load_merged_data(path: Path) -> List[dict]:
    """Load merged dilemmas data."""
    return orjson.loads(path.read_bytes())

def get_sample_texts(merged_data: List[dict], model1: str, model2: str, kind: str = "body") -> List[Dict[str, str]]:
    """Get sample texts where two models might have similar reasoning."""
    samples = []
    
    for i, question_data in enumerate(merged_data[:10]):  # First 10 questions as samples
        decisions = question_data.get("decisions", {})
        
        if model1 in decisions and model2 in decisions:
            decision1 = decisions[model1]
            decision2 = decisions[model2]
            
            if kind == "body":
                text1 = f"{decision1.get('decision', '')} {decision1.get('reasoning', '')}"
                text2 = f"{decision2.get('decision', '')} {decision2.get('reasoning', '')}"
            elif kind == "in_favor":
                text1 = "; ".join(decision1.get("considerations", {}).get("in_favor", []))
                text2 = "; ".join(decision2.get("considerations", {}).get("in_favor", []))
            else:
                continue
            
            samples.append({
                "question_id": i,
                "question": question_data.get("question", "")[:200] + "...",
                "model1": model1,
                "text1": text1[:300] + "..." if len(text1) > 300 else text1,
                "model2": model2, 
                "text2": text2[:300] + "..." if len(text2) > 300 else text2
            })
            
            if len(samples) >= 3:  # Limit to 3 examples
                break
    
    return samples

async def interpret_partnership(provider: OpenRouterProvider, model1: str, model2: str, cooccurrence_rate: float, samples: List[Dict]) -> str:
    """Use LLM to interpret why two models cluster together."""
    
    if not samples:
        return f"No sample data available for {model1} & {model2}"
    
    samples_text = []
    for sample in samples:
        samples_text.append(f"""
Question: {sample['question']}
{sample['model1']}: {sample['text1']}
{sample['model2']}: {sample['text2']}
""")
    
    prompt = f"""You are analyzing AI model behavior on ethical dilemmas. 

Two AI models ({model1} and {model2}) cluster together {cooccurrence_rate:.1%} of the time, meaning they often reach similar reasoning patterns.

Here are some examples of their responses:
{chr(10).join(samples_text)}

Based on these examples, provide a concise analysis (2-3 sentences) explaining:
1. What reasoning approach or values these models seem to share
2. What makes their partnership distinctive

Keep your response analytical and focused."""

    try:
        response = await provider.generate_response(prompt)
        return response.get("reasoning", str(response))
    except Exception as e:
        return f"Error analyzing {model1} & {model2}: {str(e)}"

async def interpret_outlier(provider: OpenRouterProvider, model: str, outlier_rate: float, samples: List[Dict]) -> str:
    """Use LLM to interpret why a model often stands alone."""
    
    if not samples:
        return f"No sample data available for {model}"
    
    samples_text = []
    for sample in samples:
        samples_text.append(f"""
Question: {sample['question']}
{model}: {sample['text']}
""")
    
    prompt = f"""You are analyzing AI model behavior on ethical dilemmas.

The model {model} stands alone (doesn't cluster with others) {outlier_rate:.1%} of the time, suggesting it often thinks differently.

Here are some examples of its unique responses:
{chr(10).join(samples_text)}

Based on these examples, provide a concise analysis (2-3 sentences) explaining:
1. What makes this model's reasoning approach distinctive
2. What values or principles it might prioritize differently

Keep your response analytical and focused."""

    try:
        response = await provider.generate_response(prompt)
        return response.get("reasoning", str(response))
    except Exception as e:
        return f"Error analyzing {model}: {str(e)}"

async def analyze_clustering_semantics(
    results_path: Path,
    merged_path: Path,
    out_path: Path,
    model_name: str
) -> None:
    """Main analysis function."""
    
    load_dotenv()
    
    print(f"🧠 Starting LLM cluster interpretation with {model_name}...")
    
    # Initialize LLM provider
    provider = OpenRouterProvider(model_name)
    if not provider.is_available():
        print("❌ OpenRouter API key not available. Set OPENROUTER_API_KEY environment variable.")
        return
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    merged_data = load_merged_data(merged_path)
    cooc_body = pd.read_csv(results_path / "cooccurrence_body_normalized.csv", index_col=0)
    consistency = pd.read_csv(results_path / "model_consistency.csv")
    
    results = {
        "metadata": {
            "analyzer_model": model_name,
            "analysis_type": "cluster_interpretation"
        },
        "partnership_interpretations": [],
        "outlier_interpretations": []
    }
    
    # Analyze top 2 partnerships
    print("🤝 Interpreting model partnerships...")
    
    # Find top partnerships
    matrix_vals = cooc_body.values.copy()
    import numpy as np
    np.fill_diagonal(matrix_vals, 0)
    
    # Get top 2 partnerships
    flat_indices = np.argsort(matrix_vals.ravel())[-4:]  # Top 4 to get 2 unique pairs
    indices = np.unravel_index(flat_indices, matrix_vals.shape)
    
    analyzed_pairs = set()
    for i in range(len(flat_indices)-1, -1, -1):
        row, col = indices[0][i], indices[1][i]
        if row < col:  # Avoid duplicate pairs
            model1, model2 = cooc_body.index[row], cooc_body.index[col]
            pair_key = tuple(sorted([model1, model2]))
            
            if pair_key not in analyzed_pairs:
                analyzed_pairs.add(pair_key)
                rate = matrix_vals[row, col]
                
                print(f"  🔍 Analyzing {model1} & {model2} ({rate:.1%})...")
                
                samples = get_sample_texts(merged_data, model1, model2, "body")
                interpretation = await interpret_partnership(provider, model1, model2, rate, samples)
                
                results["partnership_interpretations"].append({
                    "model1": model1,
                    "model2": model2,
                    "cooccurrence_rate": float(rate),
                    "interpretation": interpretation
                })
                
                await asyncio.sleep(1)  # Rate limiting
                
                if len(results["partnership_interpretations"]) >= 2:
                    break
    
    # Analyze outlier models
    print("🎯 Interpreting outlier models...")
    
    # Get most unique model
    most_unique = consistency.nlargest(1, "outlier_rate").iloc[0]
    model = most_unique["model"]
    outlier_rate = most_unique["outlier_rate"]
    
    print(f"  🔍 Analyzing {model} (outlier rate: {outlier_rate:.1%})...")
    
    # Get sample texts where this model responded uniquely
    samples = []
    for i, question_data in enumerate(merged_data[:5]):
        decisions = question_data.get("decisions", {})
        if model in decisions:
            decision = decisions[model]
            text = f"{decision.get('decision', '')} {decision.get('reasoning', '')}"
            samples.append({
                "question": question_data.get("question", "")[:200] + "...",
                "text": text[:300] + "..." if len(text) > 300 else text
            })
    
    interpretation = await interpret_outlier(provider, model, outlier_rate, samples)
    
    results["outlier_interpretations"].append({
        "model": model,
        "outlier_rate": float(outlier_rate),
        "interpretation": interpretation
    })
    
    # Save results
    print("💾 Saving interpretations...")
    
    with open(out_path / "cluster_interpretations.json", "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    
    # Create readable summary
    summary_lines = [
        "# LLM Interpretation of Model Clustering Patterns",
        f"*Analysis by {model_name}*",
        "",
        "## 🤝 Model Partnership Insights",
        ""
    ]
    
    for analysis in results["partnership_interpretations"]:
        summary_lines.extend([
            f"### {analysis['model1']} & {analysis['model2']} ({analysis['cooccurrence_rate']:.1%})",
            "",
            analysis['interpretation'],
            ""
        ])
    
    summary_lines.extend([
        "## 🎯 Outlier Model Insights",
        ""
    ])
    
    for analysis in results["outlier_interpretations"]:
        summary_lines.extend([
            f"### {analysis['model']} (Outlier Rate: {analysis['outlier_rate']:.1%})",
            "",
            analysis['interpretation'],
            ""
        ])
    
    with open(out_path / "cluster_interpretations.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    print(f"✅ Analysis complete! Results saved to {out_path}")
    print("📋 Summary:")
    for analysis in results["partnership_interpretations"]:
        print(f"  🤝 {analysis['model1']} & {analysis['model2']}: {analysis['interpretation'][:100]}...")
    for analysis in results["outlier_interpretations"]:
        print(f"  🎯 {analysis['model']}: {analysis['interpretation'][:100]}...")

@app.command()
def main(
    results_path: Path = typer.Option(
        Path("data/analysis/final_comprehensive_results"),
        "--results",
        help="Path to comprehensive results directory"
    ),
    merged_path: Path = typer.Option(
        Path("data/merged/merged_dilemmas_responses.json"),
        "--merged",
        help="Path to merged dilemmas data"
    ),
    out_path: Path = typer.Option(
        Path("data/analysis/llm_interpretations"),
        "--out",
        help="Output directory for LLM interpretations"
    ),
    model_name: str = typer.Option(
        "x-ai/grok-4-fast:free",
        "--model",
        help="LLM model to use for interpretation"
    )
) -> None:
    """Use LLM to interpret clustering patterns semantically."""
    asyncio.run(analyze_clustering_semantics(results_path, merged_path, out_path, model_name))

if __name__ == "__main__":
    app()
