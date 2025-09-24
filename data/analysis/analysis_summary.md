# Enhanced Analysis Summary

## Overview
- **Analysis Date**: Generated automatically
- **Embeddings**: data\embeddings.sqlite3
- **Source Data**: data\merged\merged_dilemmas_responses.json  
- **Clusters**: 25
- **Topic Naming**: ❌ Skipped

## Files Generated

### Clustering Analysis
- `clusters.json`: Core clustering results with BisectingKMeans


### TensorFlow Projector
- `docs\tensorflow_projector/embeddings.tsv`: 1024D embeddings for visualization
- `docs\tensorflow_projector/metadata.tsv`: Point metadata and cluster labels
- `docs\tensorflow_projector/projector_config.pbtxt`: TensorBoard configuration
- `docs\tensorflow_projector/README.md`: Detailed usage instructions

## Next Steps

### 1. Explore with TensorFlow Projector
```bash
# Option A: Online (Recommended)
# 1. Go to https://projector.tensorflow.org/
# 2. Upload embeddings.tsv and metadata.tsv from docs\tensorflow_projector/

# Option B: Local TensorBoard
pip install tensorboard
tensorboard --logdir=docs
```

### 2. Analyze Results
- Color by **model** to see AI system differences
- Color by **kind** to compare decision types (body/in_favor/against)
- Color by **cluster** to explore ethical reasoning patterns

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
