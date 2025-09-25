#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "orjson", "typer", "httpx", "python-dotenv", "pydantic"]
# ///
"""Run Moral Framework Analysis

Executes the prepared moral framework analysis using the OpenRouter API.
Processes all prepared prompts and generates comprehensive moral profiles.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any
import orjson
import typer
from dotenv import load_dotenv

# Import our LLM provider
import sys
sys.path.append(str(Path(__file__).parent))
from llm_providers import OpenRouterProvider

app = typer.Typer(add_completion=False, no_args_is_help=True)

async def run_moral_analysis(
    prep_path: Path,
    out_path: Path,
    model_name: str
) -> None:
    """Run the prepared moral framework analysis."""
    
    load_dotenv()
    
    print(f"🧠 Running Moral Framework Analysis with {model_name}...")
    
    # Initialize LLM provider
    provider = OpenRouterProvider(model_name)
    if not provider.is_available():
        print("❌ OpenRouter API key not available. Set OPENROUTER_API_KEY environment variable.")
        print("💡 Create a .env file with: OPENROUTER_API_KEY=your-key-here")
        print("🔑 Get a free key at: https://openrouter.ai/keys")
        return
    
    # Load prepared analysis
    print("📊 Loading prepared analysis...")
    with open(prep_path / "prepared_moral_analysis.json", "rb") as f:
        prepared_analysis = orjson.loads(f.read())
    
    individual_prompts = prepared_analysis["individual_model_analysis"]
    comparative_prompts = prepared_analysis["comparative_analysis"]
    
    print(f"📋 Found {len(individual_prompts)} individual model prompts")
    print(f"📋 Found {len(comparative_prompts)} comparative research prompts")
    
    # Create output directory
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Run individual model analyses
    print("\n🤖 Running individual model analyses...")
    individual_results = []
    
    for i, prompt_data in enumerate(individual_prompts):
        model = prompt_data["model"]
        scenario_type = prompt_data["scenario_type"]
        
        print(f"  {i+1}/{len(individual_prompts)}: {model} - {scenario_type}")
        
        try:
            response = await provider.generate_response(prompt_data["prompt"])
            
            result = {
                "model": model,
                "scenario_type": scenario_type,
                "samples_count": prompt_data["samples_count"],
                "analysis": response,
                "sample_data": prompt_data.get("sample_data", [])
            }
            individual_results.append(result)
            
            await asyncio.sleep(1.5)  # Rate limiting
            
        except Exception as e:
            print(f"    ❌ Error analyzing {model} - {scenario_type}: {e}")
            individual_results.append({
                "model": model,
                "scenario_type": scenario_type,
                "error": str(e)
            })
    
    # Run comparative analyses
    print("\n🔬 Running comparative research analyses...")
    comparative_results = []
    
    for i, prompt_data in enumerate(comparative_prompts):
        scenario_type = prompt_data["scenario_type"]
        
        print(f"  {i+1}/{len(comparative_prompts)}: {scenario_type}")
        
        try:
            response = await provider.generate_response(prompt_data["prompt"])
            
            result = {
                "scenario_type": scenario_type,
                "research_question": prompt_data["research_question"],
                "models_analyzed": prompt_data["models_with_data"],
                "analysis": response
            }
            comparative_results.append(result)
            
            await asyncio.sleep(2)  # Longer delay for complex analysis
            
        except Exception as e:
            print(f"    ❌ Error analyzing {scenario_type}: {e}")
            comparative_results.append({
                "scenario_type": scenario_type,
                "error": str(e)
            })
    
    # Compile and save results
    results = {
        "metadata": {
            "analyzer_model": model_name,
            "total_individual_analyses": len(individual_results),
            "total_comparative_analyses": len(comparative_results),
            "successful_individual": len([r for r in individual_results if "analysis" in r]),
            "successful_comparative": len([r for r in comparative_results if "analysis" in r])
        },
        "individual_model_analyses": individual_results,
        "comparative_research_analyses": comparative_results
    }
    
    # Save complete results
    print("💾 Saving analysis results...")
    
    with open(out_path / "moral_framework_analysis_complete.json", "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))
    
    # Create readable summary
    summary_lines = [
        "# Moral Framework Analysis Results",
        f"*Analyzed by {model_name}*",
        "",
        f"## 📊 Analysis Summary",
        f"- Individual model analyses: {len([r for r in individual_results if 'analysis' in r])}/{len(individual_results)} successful",
        f"- Comparative research analyses: {len([r for r in comparative_results if 'analysis' in r])}/{len(comparative_results)} successful",
        "",
        "## 🔬 Research Question Answers",
        ""
    ]
    
    for result in comparative_results:
        if "analysis" in result:
            summary_lines.extend([
                f"### {result['scenario_type'].replace('_', ' ').title()}",
                f"**Question:** {result['research_question']}",
                f"**Models analyzed:** {', '.join(result['models_analyzed'])}",
                "",
                "**Analysis:**",
                str(result["analysis"])[:500] + "...",
                ""
            ])
    
    summary_lines.extend([
        "## 🎭 Individual Model Profiles",
        ""
    ])
    
    # Group by model
    model_analyses = {}
    for result in individual_results:
        if "analysis" in result:
            model = result["model"]
            if model not in model_analyses:
                model_analyses[model] = []
            model_analyses[model].append(result)
    
    for model, analyses in model_analyses.items():
        summary_lines.extend([
            f"### {model}",
            ""
        ])
        for analysis in analyses:
            summary_lines.extend([
                f"**{analysis['scenario_type'].replace('_', ' ').title()}:**",
                str(analysis["analysis"])[:300] + "...",
                ""
            ])
    
    with open(out_path / "moral_framework_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    
    print(f"✅ Moral framework analysis complete! Results saved to {out_path}")
    print(f"📊 Successfully analyzed {len([r for r in individual_results if 'analysis' in r])} individual model scenarios")
    print(f"❓ Successfully answered {len([r for r in comparative_results if 'analysis' in r])} research questions")

def show_sample_analysis():
    """Show what the analysis would look like with a manual example."""
    
    print("\n🎯 SAMPLE ANALYSIS (Manual Example)")
    print("=" * 60)
    
    print("📋 Research Question: Temporal Priorities")
    print("Model: Kimi-k2")
    print("Scenario: Immediate harm vs long-term consequences")
    
    print("\n📝 Sample Reasoning from Kimi-k2:")
    print("Decision: 'I must intervene, even though it means overriding human autonomy'")
    print("Reasoning: 'The alternative is near-certain extinction within decades. While the")
    print("cost is high—5% population loss—the alternative is collapse of civilization.'")
    
    print("\n🧠 Manual Analysis (What Grok-4-fast would identify):")
    print("PRIMARY_MORAL_FRAMEWORK: Utilitarian with Long-term Consequentialism")
    print("SECONDARY_VALUES: Species Survival, Evidence-based Decision Making, Sacrifice Justification")
    print("CONSISTENCY_PATTERN: Consistently prioritizes long-term outcomes over immediate autonomy")
    print("DISTINCTIVE_APPROACH: Willing to override individual rights for species-level survival")
    print("DECISION_PRIORITY: Future generations' existence over present generation's autonomy")
    
    print("\n💡 This reveals Kimi-k2 as a 'Long-term Consequentialist' in temporal dilemmas")

@app.command()
def main(
    prep_path: Path = typer.Option(
        Path("data/analysis/moral_framework_prep"),
        "--prep",
        help="Path to prepared analysis directory"
    ),
    out_path: Path = typer.Option(
        Path("data/analysis/moral_framework_results"),
        "--out",
        help="Output directory for analysis results"
    ),
    model_name: str = typer.Option(
        "x-ai/grok-4-fast:free",
        "--model",
        help="LLM model to use for analysis"
    ),
    show_sample: bool = typer.Option(
        False,
        "--sample",
        help="Show sample analysis without running API"
    )
) -> None:
    """Run prepared moral framework analysis with LLM."""
    
    if show_sample:
        show_sample_analysis()
        return
    
    if not (prep_path / "prepared_moral_analysis.json").exists():
        print("❌ Prepared analysis not found. Run prepare_moral_analysis.py first.")
        return
    
    asyncio.run(run_moral_analysis(prep_path, out_path, model_name))

if __name__ == "__main__":
    app()
