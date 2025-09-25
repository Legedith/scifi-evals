#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "orjson", "typer"]
# ///
"""Export an overall report JSON for the web UI.

Consolidates quantitative analysis, cluster interpretations, and moral framework
results into a single JSON at docs/data/overall_report.json.

Inputs (expected):
- data/analysis/final_comprehensive_results/summary_statistics.csv
- data/analysis/final_comprehensive_results/model_consistency.csv
- data/analysis/llm_interpretations/cluster_interpretations.json
- data/analysis/moral_framework_results/moral_framework_analysis_complete.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import pandas as pd
import orjson
import typer
from datetime import datetime, timezone

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _convert(obj: Any) -> Any:
    # Ensure JSON-serializable output (convert pandas/numpy types)
    if isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


@app.command()
def main(
    quantitative_dir: Path = typer.Option(
        Path("data/analysis/final_comprehensive_results"),
        "--quant-dir",
        help="Directory with quantitative outputs",
    ),
    cluster_json: Path = typer.Option(
        Path("data/analysis/llm_interpretations/cluster_interpretations.json"),
        "--cluster-json",
        help="Cluster interpretations JSON",
    ),
    moral_json: Path = typer.Option(
        Path("data/analysis/moral_framework_results/moral_framework_analysis_complete.json"),
        "--moral-json",
        help="Moral framework analysis JSON",
    ),
    out_path: Path = typer.Option(
        Path("docs/data/overall_report.json"),
        "--out",
        help="Output overall report JSON path",
    ),
) -> None:
    # Load quantitative
    ss_csv = quantitative_dir / "summary_statistics.csv"
    mc_csv = quantitative_dir / "model_consistency.csv"
    if not ss_csv.exists() or not mc_csv.exists():
        raise FileNotFoundError("Quantitative CSVs not found. Run comprehensive_model_analysis.py first.")

    ss = pd.read_csv(ss_csv)
    mc = pd.read_csv(mc_csv)

    # Summarize quantitative
    quantitative: Dict[str, Any] = {
        "summary_statistics": ss.to_dict(orient="records"),
        "model_consistency": mc.to_dict(orient="records"),
    }

    # Load matrices and models
    detailed = _load_json(quantitative_dir / "detailed_results.json") if (quantitative_dir / "detailed_results.json").exists() else {}
    models = detailed.get("metadata", {}).get("models", [])

    def _matrix_from_csv(csv_path: Path) -> Dict[str, Any]:
        if not csv_path.exists():
            return {"models": models, "matrix": []}
        df = pd.read_csv(csv_path, index_col=0)
        # Ensure order matches df columns/rows
        labels = list(df.columns)
        mat = df.to_numpy().tolist()
        return {"models": labels, "matrix": mat}

    cooccurrence = {
        "body": _matrix_from_csv(quantitative_dir / "cooccurrence_body_normalized.csv"),
        "in_favor": _matrix_from_csv(quantitative_dir / "cooccurrence_in_favor_normalized.csv"),
        "against": _matrix_from_csv(quantitative_dir / "cooccurrence_against_normalized.csv"),
    }

    # Load LLM outputs
    clusters = _load_json(cluster_json) if cluster_json.exists() else {}
    moral = _load_json(moral_json) if moral_json.exists() else {}

    # Reduce moral to essential
    moral_out = {
        "metadata": moral.get("metadata", {}),
        "comparative_research": moral.get("comparative_research_analyses", []),
        "model_profiles": moral.get("individual_model_analyses", []),
    }

    # Construct report
    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "quantitative": quantitative,
        "cooccurrence": cooccurrence,
        "clusterInterpretations": clusters,
        "moralFramework": moral_out,
    }

    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(_convert(report), option=orjson.OPT_INDENT_2))
    typer.echo(f"Exported overall report to {out_path}")


if __name__ == "__main__":
    app()


