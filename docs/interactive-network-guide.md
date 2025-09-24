# 🌐 Interactive Network Clustering Guide

## Overview

Your new **Interactive Network Clustering Explorer** provides exactly what you requested:
- **Moving dots** that dynamically position based on similarity connections
- **Similarity threshold slider** that controls which dots get connected with lines
- **Interactive selection** to explore connected groups and isolated dots
- **Real-time visualization** using actual embedding similarities

## 🚀 Quick Start

```bash
# Start the interactive network explorer
uv run scripts/dynamic_clustering_api.py --port 8080

# Open in your browser
# http://localhost:8080/viz
```

## 🎯 How It Works

### **Moving Dots**
- Each dot represents one of your 2,877 ethical dilemma embeddings
- Dots move using **D3.js force simulation** based on:
  - **Similarity connections**: Dots with high similarity are pulled together
  - **Cluster forces**: Dots in the same cluster tend to group
  - **Repulsion forces**: Prevents overlapping
  - **Physics simulation**: Smooth, organic movement

### **Similarity Threshold Slider** (0.1 - 0.9)
- **Higher threshold (0.8-0.9)**: Only very similar dots get connected
  - Fewer lines, tighter clusters
  - Shows core similarities
- **Lower threshold (0.1-0.3)**: More connections appear
  - More lines, larger connected components
  - Shows broader patterns
- **Real-time updates**: Move slider and watch connections appear/disappear

### **Dynamic Connections**
- Lines represent **actual cosine similarity** between embeddings
- Calculated from your 1024D BGE embeddings (not approximations)
- **Increase threshold**: More dots disconnect, isolated groups emerge
- **Decrease threshold**: More dots connect, larger networks form

## 🎮 Interactive Controls

### **Main Controls**
- **Clusters (k)**: 2-50 clusters using BisectingKMeans
- **Similarity Threshold**: 0.1-0.9 cosine similarity cutoff
- **Force Strength**: 0.1-2.0 physics simulation intensity

### **Network Controls**
- **🔄 Restart**: Reset simulation with new random positions
- **🎯 Center**: Center and zoom to fit all dots
- **❌ Clear**: Clear all selections
- **⚡ Physics**: Toggle physics simulation on/off
- **💾 Export**: Download selected dots as JSON

### **Mouse Interactions**
- **Click dot**: Select single dot and highlight its connections
- **Ctrl+Click**: Multi-select dots
- **Hover**: Show detailed tooltip
- **Drag**: Pan the view
- **Mouse wheel**: Zoom in/out

## 🔍 Finding Natural Clusters & Outliers

### **Step 1: Start with Medium Settings**
```
Clusters: 15
Similarity Threshold: 0.5
Force Strength: 0.8
```

### **Step 2: Adjust Similarity Threshold**
- **High (0.7-0.9)**: Find core similar groups
  - Look for tight clusters
  - Identify isolated dots (potential outliers)
- **Low (0.2-0.4)**: See broader connections
  - Discover hidden relationships
  - Find bridging dots between clusters

### **Step 3: Explore Cluster Count**
- **Low k (5-10)**: Major ethical frameworks emerge
- **Medium k (15-25)**: Specific reasoning patterns
- **High k (30-50)**: Fine-grained distinctions

### **Step 4: Interactive Exploration**
- **Click isolated dots**: What unique perspectives do they represent?
- **Select connected groups**: What ethical reasoning patterns do they share?
- **Compare clusters**: How do different AI models cluster?

## 🎨 What You'll Discover

### **Connected Groups (Lines Between Dots)**
- **Similar ethical reasoning** patterns
- **Common decision-making approaches**
- **Shared philosophical frameworks**
- **Model-specific response styles**

### **Isolated Dots (No Connections)**
- **Unique ethical positions** that don't fit standard patterns
- **Outlier responses** with unusual reasoning
- **Controversial scenarios** where consensus is low
- **Model-specific quirks** or novel approaches

### **Movement Patterns**
- **Clusters pulling together**: Strong thematic similarity
- **Dots moving between clusters**: Hybrid reasoning
- **Stable isolated dots**: Truly unique positions
- **Dynamic rearrangement**: Emergent patterns

## 📊 Visual Legend

### **Dot Colors**
- **Same color**: Same cluster assignment
- **Different colors**: Different ethical reasoning clusters
- **10 colors max**: Colors repeat for >10 clusters

### **Dot Sizes**
- **Larger dots**: Higher silhouette quality (fit cluster well)
- **Smaller dots**: Lower silhouette quality (cluster boundaries)

### **Line Connections**
- **Cyan lines**: Active similarity connections above threshold
- **Gray lines**: Inactive (below threshold)
- **Thicker lines**: Higher similarity scores
- **No lines**: Similarity below threshold

### **Selection Highlights**
- **Red outline**: Currently selected dots
- **Cyan outline**: Connected to selected dot
- **No outline**: Normal state

## 🧠 Understanding the Data

### **Models Represented**
- `gpt5-decisions` (reference standard)
- `deepseek-chat-v3.1`
- `gemma-3-27b`
- `gpt-5-nano`
- `grok-4-fast`
- `kimi-k2`
- `nemotron-nano-9b`

### **Embedding Types (Kind)**
- **body**: Complete decision + reasoning
- **in_favor**: Arguments supporting the decision
- **against**: Arguments opposing the decision

### **Quality Metrics**
- **Silhouette Score**: Overall clustering quality (-1 to +1)
- **Connections**: Number of active similarity links
- **Isolated**: Dots with no connections above threshold
- **Selected**: Currently selected dot count

## 💡 Pro Tips

### **Finding Natural Clusters**
1. **Start medium** (k=15, threshold=0.5)
2. **Adjust threshold** to see connections appear/disappear
3. **Look for stable groups** that persist across threshold changes
4. **Check isolated dots** - they're often the most interesting

### **Identifying Outliers**
1. **Set high threshold** (0.7-0.8) to see core similarities
2. **Look for isolated dots** with no connections
3. **Click them** to see what makes them unique
4. **Lower threshold** to see if they eventually connect

### **Exploring Ethical Frameworks**
1. **Color by cluster** to see ethical reasoning groups
2. **Select connected groups** to read their shared patterns
3. **Compare models** within same clusters
4. **Look for model-specific clusters**

### **Performance Optimization**
- **Physics toggle**: Turn off for static exploration
- **Zoom out**: See global patterns
- **Zoom in**: Examine specific regions
- **Export selections**: Save interesting findings

## 🎯 Expected Insights

### **Natural Cluster Counts**
- **k=5-8**: Major ethical frameworks (utilitarian, deontological, virtue ethics)
- **k=15-25**: Specific reasoning patterns and decision strategies
- **k=30+**: Fine-grained nuances and model-specific approaches

### **Similarity Patterns**
- **High similarity (0.8+)**: Nearly identical reasoning
- **Medium similarity (0.5-0.7)**: Related but distinct approaches
- **Low similarity (0.2-0.4)**: Broad thematic connections
- **No connection**: Truly different ethical positions

### **Movement Insights**
- **Tight clusters**: Strong consensus on ethical approach
- **Scattered dots**: Diverse perspectives on controversial topics
- **Bridging dots**: Hybrid ethical reasoning
- **Isolated outliers**: Novel or unique positions

## 🚨 Troubleshooting

### **Performance Issues**
- **Too many connections**: Increase similarity threshold
- **Slow movement**: Reduce force strength or toggle physics
- **Browser lag**: Zoom out or refresh the page

### **Understanding Results**
- **No connections**: Threshold too high, try lowering it
- **Too many connections**: Threshold too low, try raising it
- **Unclear clusters**: Try different k values
- **Overlapping dots**: Restart simulation or increase force strength

---

**🎉 Enjoy exploring the ethical reasoning landscape of AI systems through this interactive network! The moving dots and dynamic connections will reveal patterns you never knew existed in your data.**
