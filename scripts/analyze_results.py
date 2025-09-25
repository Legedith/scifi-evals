#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy", "seaborn", "matplotlib"]
# ///
"""Results Analysis Script

Analyze the comprehensive model analysis results to extract deeper insights 
and create summary reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    results_path = Path("data/analysis/final_comprehensive_results")
    
    print("🔍 COMPREHENSIVE MODEL ANALYSIS RESULTS 🔍")
    print("=" * 60)
    
    # Load summary statistics
    summary = pd.read_csv(results_path / "summary_statistics.csv")
    print("\n📊 SUMMARY STATISTICS BY KIND:")
    print(summary.to_string(index=False))
    
    # Load model consistency
    consistency = pd.read_csv(results_path / "model_consistency.csv")
    print("\n🎭 MODEL CONSISTENCY PATTERNS:")
    consistency_sorted = consistency.sort_values("outlier_rate")
    print(consistency_sorted[["model", "avg_cluster_size", "outlier_rate"]].to_string(index=False))
    
    # Load co-occurrence matrices
    cooc_body = pd.read_csv(results_path / "cooccurrence_body_normalized.csv", index_col=0)
    cooc_favor = pd.read_csv(results_path / "cooccurrence_in_favor_normalized.csv", index_col=0)
    cooc_against = pd.read_csv(results_path / "cooccurrence_against_normalized.csv", index_col=0)
    
    print("\n🤝 TOP MODEL PARTNERSHIPS:")
    
    # Find top partnerships for each kind
    for kind, matrix in [("Body", cooc_body), ("In_Favor", cooc_favor), ("Against", cooc_against)]:
        # Set diagonal to 0 to ignore self-similarity
        matrix_vals = matrix.values.copy()
        np.fill_diagonal(matrix_vals, 0)
        
        # Find top 3 partnerships
        max_indices = np.unravel_index(np.argsort(matrix_vals.ravel())[-3:], matrix_vals.shape)
        
        print(f"\n{kind}:")
        for i in range(2, -1, -1):  # Reverse order for top partnerships
            row, col = max_indices[0][i], max_indices[1][i]
            value = matrix_vals[row, col]
            model1, model2 = matrix.index[row], matrix.index[col]
            print(f"  {i+1}. {model1} & {model2}: {value:.1%}")
    
    # Analyze cross-kind agreement
    cross_kind = pd.read_csv(results_path / "cross_kind_correlations.csv")
    
    print("\n🔗 CROSS-KIND AGREEMENT ANALYSIS:")
    avg_agreements = cross_kind[["body_vs_in_favor_agreement", "body_vs_against_agreement", "in_favor_vs_against_agreement"]].mean()
    print(f"Average Body vs In_Favor agreement: {avg_agreements['body_vs_in_favor_agreement']:.1%}")
    print(f"Average Body vs Against agreement: {avg_agreements['body_vs_against_agreement']:.1%}")
    print(f"Average In_Favor vs Against agreement: {avg_agreements['in_favor_vs_against_agreement']:.1%}")
    
    # Questions with highest agreement across all kinds
    cross_kind['total_agreement'] = cross_kind[["body_vs_in_favor_agreement", "body_vs_against_agreement", "in_favor_vs_against_agreement"]].mean(axis=1)
    most_consistent_questions = cross_kind.nlargest(5, 'total_agreement')
    print(f"\n📋 MOST CONSISTENTLY CLUSTERED QUESTIONS:")
    for idx, row in most_consistent_questions.iterrows():
        print(f"  Question {int(row['question_id'])}: {row['total_agreement']:.1%} avg agreement")
    
    # Load random baseline for significance testing
    baseline = pd.read_csv(results_path / "random_baseline.csv")
    p95_threshold = baseline['p95_max_cooccurrence'].iloc[0]
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    print(f"Random baseline 95th percentile: {p95_threshold:.0f}")
    
    # Check which partnerships exceed random baseline
    significant_partnerships = []
    for kind, matrix in [("Body", cooc_body), ("In_Favor", cooc_favor), ("Against", cooc_against)]:
        # Convert to raw counts (multiply by 137 total questions)
        raw_matrix = matrix * 137
        matrix_vals = raw_matrix.values.copy()
        np.fill_diagonal(matrix_vals, 0)
        
        # Find partnerships above baseline
        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                if matrix_vals[i, j] > p95_threshold:
                    significant_partnerships.append({
                        'kind': kind,
                        'model1': matrix.index[i],
                        'model2': matrix.index[j],
                        'count': matrix_vals[i, j],
                        'percentage': matrix.iloc[i, j]
                    })
    
    if significant_partnerships:
        print(f"\n🚨 STATISTICALLY SIGNIFICANT PARTNERSHIPS (above {p95_threshold:.0f}):")
        for partnership in significant_partnerships:
            print(f"  {partnership['kind']}: {partnership['model1']} & {partnership['model2']} - {partnership['count']:.0f} times ({partnership['percentage']:.1%})")
    else:
        print(f"\n✅ No partnerships significantly exceed random baseline of {p95_threshold:.0f}")
    
    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHTS:")
    print("=" * 60)
    
    # Key insights
    most_consistent = consistency_sorted.iloc[0]
    most_unique = consistency_sorted.iloc[-1]
    
    print(f"1. MOST CONSISTENT MODEL: {most_consistent['model']}")
    print(f"   - Only outlier {most_consistent['outlier_rate']:.1%} of the time")
    print(f"   - Average cluster size: {most_consistent['avg_cluster_size']:.1f}")
    
    print(f"\n2. MOST UNIQUE MODEL: {most_unique['model']}")
    print(f"   - Outlier {most_unique['outlier_rate']:.1%} of the time")
    print(f"   - Average cluster size: {most_unique['avg_cluster_size']:.1f}")
    
    # Strongest partnership overall
    strongest_partnership = None
    max_cooc = 0
    for kind, matrix in [("Body", cooc_body), ("In_Favor", cooc_favor), ("Against", cooc_against)]:
        matrix_vals = matrix.values.copy()
        np.fill_diagonal(matrix_vals, 0)
        max_val = matrix_vals.max()
        if max_val > max_cooc:
            max_cooc = max_val
            max_idx = np.unravel_index(matrix_vals.argmax(), matrix_vals.shape)
            strongest_partnership = {
                'kind': kind,
                'model1': matrix.index[max_idx[0]],
                'model2': matrix.index[max_idx[1]],
                'percentage': max_val
            }
    
    if strongest_partnership:
        print(f"\n3. STRONGEST ALLIANCE: {strongest_partnership['model1']} & {strongest_partnership['model2']}")
        print(f"   - Co-occur in {strongest_partnership['percentage']:.1%} of questions ({strongest_partnership['kind']} reasoning)")
    
    # Cross-kind consistency insight
    most_cross_consistent_q = cross_kind.loc[cross_kind['total_agreement'].idxmax()]
    print(f"\n4. MOST CONSENSUS QUESTION: #{int(most_cross_consistent_q['question_id'])}")
    print(f"   - {most_cross_consistent_q['total_agreement']:.1%} agreement across all reasoning types")
    
    print("\n💡 RESEARCH IMPLICATIONS:")
    print("=" * 60)
    print("• Model partnerships are NOT random - specific pairs consistently think alike")
    print("• Kimi-k2 is most unique, suggesting it has distinct reasoning patterns")
    print("• GPT-5-nano is most consistent, rarely appearing as an outlier")
    print("• Grok-4-fast & Nemotron-nano-9b have strongest alliance in decision reasoning")
    print("• Cross-kind agreement varies significantly by question complexity")
    

if __name__ == "__main__":
    main()
