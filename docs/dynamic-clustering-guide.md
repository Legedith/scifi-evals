# Dynamic Clustering Guide 🎯

## Overview

You now have **three powerful options** for dynamic clustering exploration with interactive sliders to adjust k values and identify outliers:

1. **🚀 FastAPI Web App** - Professional real-time clustering API with interactive visualization
2. **🎨 Streamlit App** - User-friendly interface for rapid exploration  
3. **🧠 Enhanced TensorFlow Projector** - Multiple clustering presets for projector.tensorflow.org

## 🚀 Option 1: FastAPI Web App (Recommended)

### Features
- ⚡ Real-time clustering with adjustable k slider (2-100)
- 🎯 Dynamic outlier detection with threshold control
- 📊 Live quality metrics (silhouette score, inertia)
- 🎨 Interactive D3.js visualization
- 🔍 Hover tooltips with detailed information
- 📱 Responsive design

### Usage
```bash
# Start the server
uv run scripts/dynamic_clustering_api.py --port 8080

# Open in browser
# Interactive interface: http://localhost:8080/viz
# API documentation: http://localhost:8080/docs
```

### Interface Controls
- **Cluster Slider**: Adjust k from 2 to 100 clusters
- **Outlier Threshold**: Control outlier sensitivity (-0.5 to 0.2)
- **Live Metrics**: Silhouette score, outlier count, inertia
- **Color Coding**: Clusters (colored) vs Outliers (red X marks)
- **Hover Details**: Model, kind, text preview, quality metrics

## 🎨 Option 2: Streamlit App (Easiest)

### Features
- 🎛️ Easy-to-use sliders and controls
- 📊 Plotly interactive charts
- 📈 Multiple visualization tabs
- 🚨 Detailed outlier analysis
- 💾 Export functionality
- 📋 Cluster statistics tables

### Usage
```bash
# Start Streamlit app  
uv run scripts/interactive_clustering_app.py

# Streamlit will automatically open your browser
# Usually at: http://localhost:8501
```

### Interface Features
- **Cluster Visualization Tab**: Interactive scatter plots with color options
- **Outlier Analysis Tab**: Detailed outlier breakdown and statistics
- **Cluster Statistics Tab**: Tables and charts of cluster compositions
- **Export Results**: Save clustering results to JSON

## 🧠 Option 3: Enhanced TensorFlow Projector

### Features
- 🎯 Pre-computed clustering presets (k=5,10,15,25,50)
- 🌐 Works with projector.tensorflow.org
- 🎨 Multiple metadata fields for visualization
- 📊 Quality metrics for each preset
- 🏆 Automatic "best" clustering recommendation

### Setup
```bash
# Generate enhanced projector files
uv run scripts/enhanced_tensorflow_projector.py --generate-presets

# Output: docs/tensorflow_projector_enhanced/
#   - embeddings.tsv
#   - metadata.tsv  
#   - index.html (preset selector)
#   - README.md
```

### Usage
1. Open `docs/tensorflow_projector_enhanced/index.html`
2. Select your preferred clustering preset
3. Go to https://projector.tensorflow.org/
4. Upload `embeddings.tsv` and `metadata.tsv`
5. Color by `cluster_k25` (or your chosen k value)

## 🎯 Finding the Natural Number of Clusters

### Strategy 1: Silhouette Score Analysis
```bash
# Use the FastAPI app
uv run scripts/dynamic_clustering_api.py --port 8080
# Visit http://localhost:8080/viz
# Adjust k slider and watch silhouette score
# Higher scores = better clustering
```

**What to look for:**
- 📈 **Peak silhouette scores** indicate natural cluster counts
- 📊 **Stable plateaus** suggest robust clustering
- 📉 **Sharp drops** indicate over-clustering

### Strategy 2: Outlier Analysis
- 🎯 **Few outliers** = clusters fit the data well
- 🚨 **Many outliers** = clusters may be too rigid
- ⚖️ **Balance** outlier count with cluster interpretability

### Strategy 3: Elbow Method
- 📉 Watch **inertia** (sum of squared distances)
- 🔍 Look for the **"elbow"** where improvement slows
- ⚖️ Balance complexity vs. quality

## 🔍 Outlier Identification Strategy

### What are Outliers?
- **Low silhouette scores**: Points that don't fit well in any cluster
- **Unique ethical positions**: Novel or rare reasoning patterns
- **Model-specific quirks**: Unusual responses from particular AI systems

### Outlier Threshold Guidelines
- **-0.1**: Moderate outlier detection (recommended start)
- **-0.2**: More aggressive outlier detection
- **0.0**: Only negative silhouette scores (very conservative)

### Using Outliers for Insights
1. **Identify unique ethical positions** that don't cluster well
2. **Find controversial topics** where models disagree
3. **Discover model-specific reasoning patterns**
4. **Highlight edge cases** for further analysis

## 📊 Recommended Workflow

### 1. Start with Streamlit (Easiest)
```bash
uv run scripts/interactive_clustering_app.py
```
- 🎛️ Use sliders to explore k values (try 5, 10, 15, 25, 50)
- 📊 Check silhouette scores in the metrics
- 🚨 Examine outliers in the outlier analysis tab
- 💾 Export promising configurations

### 2. Refine with FastAPI (Real-time)
```bash
uv run scripts/dynamic_clustering_api.py --port 8080
```
- ⚡ Fine-tune k values around promising ranges
- 🎯 Adjust outlier thresholds for your analysis needs
- 🔍 Use hover tooltips to understand specific points

### 3. Publish with TensorFlow Projector (Professional)
```bash
uv run scripts/enhanced_tensorflow_projector.py --generate-presets --k-values 10 15 25
```
- 🧠 Generate professional visualizations
- 🌐 Share with collaborators via projector.tensorflow.org
- 📊 Use multiple presets for different analysis levels

## 🎨 Visualization Tips

### Color Strategies
1. **By Cluster**: See overall groupings
2. **By Model**: Compare AI system behaviors  
3. **By Kind**: Distinguish decision types (body/in_favor/against)
4. **By Outlier Status**: Highlight unusual points
5. **By Quality**: See clustering confidence

### Exploration Techniques
- 🔍 **Start broad** (k=5-10) to see major patterns
- 🎯 **Go granular** (k=25-50) for detailed analysis
- 🚨 **Focus on outliers** to find unique insights
- ⚖️ **Compare models** to see different ethical approaches

## 🎯 Expected Discoveries

### Natural Cluster Counts
- **~5-8 clusters**: Major ethical frameworks (utilitarian, deontological, etc.)
- **~15-25 clusters**: Specific reasoning patterns and approaches
- **~40+ clusters**: Fine-grained decision nuances

### Outlier Patterns
- **GPT-5 decisions** may be outliers (reference standard)
- **Controversial dilemmas** create more outliers
- **Model-specific reasoning** styles create consistent outliers

### Ethical Insights
- **Utilitarian clusters**: Outcome-focused reasoning
- **Deontological clusters**: Rule-based, principled approaches  
- **Virtue ethics clusters**: Character and virtue-focused
- **Pragmatic clusters**: Practical, situation-specific reasoning

## 🛠️ Troubleshooting

### Performance Issues
- 📊 Large k values (>50) may be slow
- 💾 Consider exporting and analyzing subsets
- ⚡ FastAPI app caches data for better performance

### Interpretation Challenges
- 🔍 Use hover tooltips to understand clusters
- 📖 Read outlier text content for insights
- 🎨 Try different color schemes
- 📊 Compare multiple k values side-by-side

### Technical Issues
```bash
# Missing dependencies
pip install fastapi uvicorn streamlit plotly

# Database not found
uv run scripts/embed_cache.py  # Regenerate embeddings

# Port conflicts
# Change port: --port 8081
```

## 🎉 Success Metrics

### Good Clustering
- ✅ **High silhouette scores** (>0.3)
- ✅ **Reasonable outlier count** (<10% of points)
- ✅ **Interpretable clusters** with clear themes
- ✅ **Stable across k values** (similar patterns)

### Insightful Analysis
- 🧠 **Clear ethical frameworks** emerge
- 🤝 **Model differences** are visible
- 🎯 **Outliers reveal unique positions**
- 📊 **Patterns match domain knowledge**

---

**Happy clustering! 🎯 The combination of these three tools gives you unprecedented flexibility in exploring ethical reasoning patterns across AI systems.**
