# Enhanced Analysis Guide: Advanced Clustering & TensorFlow Projector Integration

This guide explains the enhanced clustering and visualization system for ethical dilemma embeddings, inspired by the movie quotes topic naming approach and integrated with TensorFlow Projector for professional visualization.

## 🎯 What's New

### Enhanced Clustering
- **BisectingKMeans**: Advanced clustering on full 1024D embeddings (not 2D PCA)
- **Quality Metrics**: Silhouette scores and cluster composition analysis
- **Smart Sampling**: Representative examples from each cluster

### Automatic Topic Naming
- **LLM-Powered**: Uses GPT-4 or similar to generate meaningful cluster names
- **Ethical Focus**: Names reflect ethical frameworks and reasoning patterns
- **Descriptions**: Detailed explanations of each cluster's characteristics

### TensorFlow Projector Integration
- **Professional Visualization**: Export for https://projector.tensorflow.org/
- **Rich Metadata**: Model types, cluster labels, text previews
- **Multiple Views**: Color by model, kind, cluster, or custom attributes

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure your embeddings are ready
uv run scripts/embed_cache.py \
  --merged data/merged/merged_dilemmas_responses.json \
  --db data/embeddings.sqlite3 \
  --bge-model BAAI/bge-large-en-v1.5 \
  --placeholder-empty "[none]"
```

### Run Complete Analysis
```bash
# With topic naming (requires API key)
uv run scripts/run_enhanced_analysis.py \
  --api-key YOUR_OPENAI_API_KEY \
  --n-clusters 25

# Without topic naming
uv run scripts/run_enhanced_analysis.py \
  --skip-topic-naming \
  --n-clusters 25
```

### Visualize Results
1. Go to https://projector.tensorflow.org/
2. Upload `docs/tensorflow_projector/embeddings.tsv`
3. Upload `docs/tensorflow_projector/metadata.tsv`
4. Explore!

## 📋 Step-by-Step Workflow

### Step 1: Enhanced Clustering
```bash
uv run scripts/enhanced_clustering.py \
  --db data/embeddings.sqlite3 \
  --merged data/merged/merged_dilemmas_responses.json \
  --out data/analysis/clusters.json \
  --n-clusters 25
```

**Output**: `data/analysis/clusters.json` with:
- Cluster labels for all 2,877 embeddings
- Silhouette scores (quality metrics)
- Representative examples per cluster
- Cluster composition analysis

### Step 2: Topic Naming (Optional)
```bash
uv run scripts/cluster_topic_naming.py \
  --clusters data/analysis/clusters.json \
  --merged data/merged/merged_dilemmas_responses.json \
  --out data/analysis/named_clusters.json \
  --api-key YOUR_API_KEY \
  --model gpt-4o-mini
```

**Output**: `data/analysis/named_clusters.json` with:
- Meaningful topic names for each cluster
- Detailed descriptions of ethical reasoning patterns
- Enhanced metadata for analysis

### Step 3: TensorFlow Projector Export
```bash
uv run scripts/export_tensorflow_projector.py \
  --db data/embeddings.sqlite3 \
  --merged data/merged/merged_dilemmas_responses.json \
  --clusters data/analysis/named_clusters.json \
  --out-dir docs/tensorflow_projector/
```

**Output**: `docs/tensorflow_projector/` with:
- `embeddings.tsv`: 1024D vectors for visualization
- `metadata.tsv`: Rich metadata for each point
- `projector_config.pbtxt`: TensorBoard configuration
- `README.md`: Detailed usage instructions

## 🎨 Visualization Strategies

### Color Coding Options

1. **By Model**: See how different AI systems cluster
   - `gpt5-decisions` (reference)
   - `deepseek-chat-v3.1`
   - `gemma-3-27b`
   - `gpt-5-nano`
   - `grok-4-fast`
   - `kimi-k2`
   - `nemotron-nano-9b`

2. **By Kind**: Compare reasoning types
   - `body`: Complete decision + reasoning
   - `in_favor`: Arguments supporting the decision
   - `against`: Arguments opposing the decision

3. **By Cluster**: Explore ethical patterns
   - Utilitarian reasoning
   - Deontological principles
   - Virtue ethics approaches
   - Risk-based decisions
   - Stakeholder analysis

4. **By Point Type**: Visualization categories
   - `gpt5_reference`: Original GPT-5 decisions
   - `decision_body`: Full AI responses
   - `reasoning_in_favor`: Supporting arguments
   - `reasoning_against`: Opposing arguments

### Dimensionality Reduction

- **PCA**: Good for global structure, preserves distances
- **t-SNE**: Excellent for local clusters, reveals fine structure
- **UMAP**: Balance of global and local structure

## 🔍 Analysis Insights

### Expected Patterns

1. **Model Clustering**: Similar AI systems group together
2. **Ethical Frameworks**: Clear separation of reasoning styles
3. **Decision Types**: Body vs. reasoning embeddings cluster differently
4. **Consensus Areas**: High agreement across models
5. **Controversial Topics**: Scattered, diverse responses

### Research Questions

- Which AI models share similar ethical reasoning?
- What ethical frameworks emerge from clustering?
- Where do models disagree most strongly?
- How do different reasoning types (in_favor/against) relate?
- What unique ethical positions does each model take?

## 🛠️ Advanced Usage

### Custom Clustering
```bash
# Try different cluster counts
uv run scripts/enhanced_clustering.py --n-clusters 15
uv run scripts/enhanced_clustering.py --n-clusters 50

# Different random seeds
uv run scripts/enhanced_clustering.py --random-state 123
```

### Alternative LLM Providers
```bash
# Use different API endpoints
uv run scripts/cluster_topic_naming.py \
  --api-base "https://api.anthropic.com/v1" \
  --model "claude-3-sonnet" \
  --api-key YOUR_ANTHROPIC_KEY
```

### Subset Analysis
```bash
# Analyze specific models only
# (Modify the clustering script to filter by model)
```

## 📊 Output Files Reference

```
data/analysis/
├── clusters.json           # Core clustering results
├── named_clusters.json     # With LLM-generated topic names
└── analysis_summary.md     # Auto-generated summary

docs/tensorflow_projector/
├── embeddings.tsv         # 2,877 × 1024 embedding matrix
├── metadata.tsv           # Rich metadata for each point
├── projector_config.pbtxt # TensorBoard configuration
└── README.md              # Visualization instructions
```

## 🤝 Comparison with Current System

| Feature | Current System | Enhanced System |
|---------|---------------|-----------------|
| Clustering | K-means on 2D PCA | BisectingKMeans on 1024D |
| Visualization | Custom HTML5 Canvas | TensorFlow Projector |
| Topic Names | Manual/None | LLM-generated |
| Quality Metrics | Basic | Silhouette scores |
| Export Format | JSON | TSV (standard) |
| Interactivity | Custom controls | Professional interface |

## 🎯 Benefits

1. **Better Clustering**: Full-dimensional analysis reveals true semantic patterns
2. **Professional Visualization**: Industry-standard TensorFlow Projector interface
3. **Semantic Understanding**: Meaningful cluster names explain ethical reasoning
4. **Research-Ready**: Exportable data for academic analysis
5. **Scalable**: Works with larger datasets and different embedding models

## 🔮 Future Enhancements

- **Hierarchical Clustering**: Multi-level ethical taxonomy
- **Temporal Analysis**: How responses evolve over time
- **Cross-Model Consensus**: Measure agreement patterns
- **Custom Embeddings**: Domain-specific ethical embeddings
- **Interactive Filtering**: Real-time subset analysis

---

*This enhanced system provides a comprehensive platform for analyzing ethical reasoning patterns across AI systems, combining advanced clustering with professional visualization tools.*
