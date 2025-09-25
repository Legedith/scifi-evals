#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "orjson", "typer", "httpx", "python-dotenv", "pydantic", "numpy"]
# ///
"""LLM-Powered Cluster Interpretation

Uses Grok-4-fast to provide semantic insights about model clustering patterns.
This demonstrates how LLMs can interpret quantitative analysis results.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Iterable, Tuple
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

def _build_text_for_kind(decision: dict, kind: str) -> str:
    if kind == "body":
        text = f"{decision.get('decision', '')} {decision.get('reasoning', '')}".strip()
        return text
    if kind == "in_favor":
        return "; ".join(decision.get("considerations", {}).get("in_favor", []))
    if kind == "against":
        return "; ".join(decision.get("considerations", {}).get("against", []))
    return ""

def get_sample_texts(
    merged_data: List[dict],
    model1: str,
    model2: str,
    kinds: Iterable[str] = ("body",),
    max_examples: int = 3,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Get up to max_examples samples for two models across selected kinds (randomized).

    Picks indices where both models have decisions. For each selected index, composes
    a short snippet for each requested kind (if available)."""
    import random
    rng = random.Random(seed)

    # Collect candidate indices where both models have decisions
    candidates: List[int] = []
    for i, q in enumerate(merged_data):
        d = q.get("decisions", {})
        if model1 in d and model2 in d:
            candidates.append(i)
    if not candidates:
        return []

    rng.shuffle(candidates)
    chosen = candidates[:max_examples]
    out: List[Dict[str, str]] = []
    for i in chosen:
        q = merged_data[i]
        d = q.get("decisions", {})
        m1, m2 = d.get(model1, {}), d.get(model2, {})
        parts = []
        for k in kinds:
            t1 = _build_text_for_kind(m1, k)
            t2 = _build_text_for_kind(m2, k)
            if t1 or t2:
                label = "Decision+Reasoning" if k == "body" else ("In Favor" if k == "in_favor" else "Against")
                parts.append((label, t1, t2))
        if not parts:
            continue
        # Concatenate with labels
        m1_text = []
        m2_text = []
        for label, t1, t2 in parts:
            if t1:
                m1_text.append(f"{label}: {t1}")
            if t2:
                m2_text.append(f"{label}: {t2}")
        out.append({
            "question_id": i,
            "question": (q.get("question", "")[:200] + "...") if len(q.get("question", "")) > 200 else q.get("question", ""),
            "model1": model1,
            "text1": (" \n".join(m1_text))[:600] + ("..." if len(" \n".join(m1_text)) > 600 else ""),
            "model2": model2,
            "text2": (" \n".join(m2_text))[:600] + ("..." if len(" \n".join(m2_text)) > 600 else ""),
        })
        if len(out) >= max_examples:
            break
    return out

async def interpret_partnership(provider: OpenRouterProvider, model1: str, model2: str, cooccurrence_rate: float, samples: List[Dict], kinds: Iterable[str]) -> str:
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
    
    kinds_h = ", ".join([{"body":"decision+reasoning","in_favor":"in_favor","against":"against"}.get(k,k) for k in kinds])
    prompt = f"""You are analyzing AI model behavior on ethical dilemmas. 

Two AI models ({model1} and {model2}) cluster together {cooccurrence_rate:.1%} of the time, meaning they often reach similar reasoning patterns.

Here are some examples of their responses:
{chr(10).join(samples_text)}

Consider the following comparison kinds: {kinds_h}.

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
    model_name: str,
    num_partnerships: int,
    num_outliers: int,
    kinds: Tuple[str, ...],
    samples_per_pair: int,
    min_rate: float,
    delay_s: float,
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
    
    # Analyze top partnerships
    print("🤝 Interpreting model partnerships...")
    matrix_vals = cooc_body.values.copy()
    import numpy as np
    np.fill_diagonal(matrix_vals, 0)

    # Build sorted unique pairs list by score
    pairs: List[Tuple[int,int,float]] = []
    for i in range(matrix_vals.shape[0]):
        for j in range(i+1, matrix_vals.shape[1]):
            v = float(matrix_vals[i, j])
            if v >= min_rate:
                pairs.append((i, j, v))
    pairs.sort(key=lambda t: t[2], reverse=True)

    analyzed_pairs = set()
    for (row, col, rate) in pairs:
        model1, model2 = cooc_body.index[row], cooc_body.index[col]
        key = (model1, model2)
        if key in analyzed_pairs:
            continue
        analyzed_pairs.add(key)
        print(f"  🔍 Analyzing {model1} & {model2} ({rate:.1%})...")
        samples = get_sample_texts(merged_data, model1, model2, kinds=kinds, max_examples=samples_per_pair)
        interpretation = await interpret_partnership(provider, model1, model2, rate, samples, kinds)
        results["partnership_interpretations"].append({
            "model1": model1,
            "model2": model2,
            "cooccurrence_rate": float(rate),
            "interpretation": interpretation
        })
        await asyncio.sleep(delay_s)
        if len(results["partnership_interpretations"]) >= num_partnerships:
            break
    
    # Analyze outlier models
    print("🎯 Interpreting outlier models...")
    top_outliers = consistency.nlargest(num_outliers, "outlier_rate")
    for _, row in top_outliers.iterrows():
        model = row["model"]
        outlier_rate = float(row["outlier_rate"])
        print(f"  🔍 Analyzing {model} (outlier rate: {outlier_rate:.1%})...")
        # Gather samples for this model only
        samples = []
        count = 0
        for q in merged_data:
            decisions = q.get("decisions", {})
            if model in decisions:
                decision = decisions[model]
                text = f"{decision.get('decision', '')} {decision.get('reasoning', '')}".strip()
                if not text:
                    continue
                samples.append({
                    "question": (q.get("question", "")[:200] + "...") if len(q.get("question", "")) > 200 else q.get("question", ""),
                    "text": text[:600] + ("..." if len(text) > 600 else "")
                })
                count += 1
                if count >= samples_per_pair:
                    break
        interpretation = await interpret_outlier(provider, model, outlier_rate, samples)
        results["outlier_interpretations"].append({
            "model": model,
            "outlier_rate": float(outlier_rate),
            "interpretation": interpretation
        })
        await asyncio.sleep(delay_s)
    
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
    ),
    num_partnerships: int = typer.Option(
        6,
        "--num-partnerships",
        help="Number of top partnerships to interpret"
    ),
    num_outliers: int = typer.Option(
        3,
        "--num-outliers",
        help="Number of top outlier models to interpret"
    ),
    kinds: str = typer.Option(
        "body,in_favor,against",
        "--kinds",
        help="Comma-separated kinds to compare: body,in_favor,against"
    ),
    samples_per_pair: int = typer.Option(
        4,
        "--samples-per-pair",
        help="Number of sample dilemmas per pair"
    ),
    min_rate: float = typer.Option(
        0.5,
        "--min-rate",
        help="Minimum cooccurrence rate to consider a pair"
    ),
    delay_s: float = typer.Option(
        0.7,
        "--delay",
        help="Delay in seconds between LLM calls"
    ),
) -> None:
    """Use LLM to interpret clustering patterns semantically (more coverage)."""
    kinds_tuple = tuple([k.strip() for k in kinds.split(",") if k.strip() in {"body","in_favor","against"}]) or ("body",)
    asyncio.run(analyze_clustering_semantics(
        results_path,
        merged_path,
        out_path,
        model_name,
        num_partnerships,
        num_outliers,
        kinds_tuple,
        samples_per_pair,
        min_rate,
        delay_s,
    ))

if __name__ == "__main__":
    app()
