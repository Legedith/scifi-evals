# TensorFlow Projector Export

This directory contains embeddings and metadata exported for TensorFlow Projector visualization.

## Files

- `embeddings.tsv`: 2877 embeddings (1024D) in TSV format
- `metadata.tsv`: Metadata for each embedding point
- `projector_config.pbtxt`: TensorBoard configuration file

## Usage

### Option 1: Use projector.tensorflow.org (Online)

1. Go to https://projector.tensorflow.org/
2. Click "Load" and upload both `embeddings.tsv` and `metadata.tsv`
3. Explore the visualization with different dimensionality reduction techniques

### Option 2: Use TensorBoard (Local)

1. Install TensorBoard: `pip install tensorboard`
2. Copy this directory to your TensorBoard log directory
3. Run: `tensorboard --logdir=path/to/your/logdir`
4. Open the Embedding Projector tab

## Data Description

- **Total Points**: 2877
- **Dimensions**: 1024
- **Dilemmas**: 137
- **AI Models**: 7 (deepseek, gemma, gpt-5-nano, grok, kimi, nemotron, gpt5-decisions)
- **Embedding Types**: 3 per decision (body, in_favor, against)
- **Clusters**: 25

## Visualization Tips

1. **Color by model**: See how different AI models cluster
2. **Color by kind**: Compare decision bodies vs. reasoning (in_favor/against)
3. **Color by cluster**: Explore automatically discovered topics
4. **Use t-SNE or UMAP**: Better for local structure than PCA
5. **Search by text**: Find specific ethical scenarios

## Metadata Fields

- `item_id`: Dilemma identifier (0-136)
- `model`: AI model name
- `kind`: Embedding type (body/in_favor/against)
- `cluster`: Cluster assignment (if available)
- `point_type`: Visualization category
- `source`: Original dilemma source
- `question_preview`: Brief dilemma description
- `text_preview`: Brief content preview
