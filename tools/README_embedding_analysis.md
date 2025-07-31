# TinyLlama Embedding Outlier Analysis with NNCF Weight Compression

This directory contains scripts to compress the TinyLlama model using `nncf.compress_weights` and visualize outliers in token embedding weights before and after compression.

## Files

- `compress_and_visualize_embedding_outliers.py`: Main script that performs model compression and outlier analysis
- `demo_embedding_analysis.py`: Demonstration script showing how to use the main tool
- `README_embedding_analysis.md`: This documentation file

## Features

### Weight Compression
- Uses NNCF to compress TinyLlama model weights
- Supports multiple compression modes (INT4_SYM, INT4_ASYM, INT8_SYM, INT8_ASYM, NF4)
- Configurable compression ratio and group size
- Uses calibration dataset for data-aware compression

### Embedding Outlier Analysis
- Extracts token embedding weights from both original and compressed models
- Implements multiple outlier detection methods:
  - **Z-Score**: Based on L2 norm deviations from mean
  - **Magnitude**: Simple L2 norm of embedding vectors
  - **Variance**: Variance across embedding dimensions
  - **Max Absolute**: Maximum absolute value across dimensions

### Visualization
- Comprehensive plots comparing original vs compressed outlier scores
- Distribution comparisons and scatter plots
- Top-k outlier token analysis
- Statistical summary with correlation analysis

## Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers datasets numpy matplotlib seaborn pandas nncf
```

## Usage

### Basic Usage

```bash
python compress_and_visualize_embedding_outliers.py
```

This will use default settings:
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Compression: INT4_SYM with ratio=0.8, group_size=128
- Outlier method: Z-score
- Output directory: embedding_outlier_analysis

### Advanced Usage

```bash
python compress_and_visualize_embedding_outliers.py \
    --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --compression-mode "int4_sym" \
    --ratio 0.8 \
    --group-size 128 \
    --outlier-method "zscore" \
    --top-k 20 \
    --dataset-size 128 \
    --output-dir "my_analysis" \
    --device "cpu"
```

### Command Line Options

- `--model-id`: Hugging Face model identifier
- `--compression-mode`: Compression mode (int4_sym, int4_asym, int8_sym, int8_asym, nf4)
- `--ratio`: Compression ratio (0.0 to 1.0)
- `--group-size`: Group size for quantization
- `--outlier-method`: Outlier detection method (zscore, magnitude, variance, max_abs)
- `--top-k`: Number of top outliers to analyze
- `--dataset-size`: Size of calibration dataset
- `--output-dir`: Directory to save results
- `--device`: Device to run on (cpu, cuda)

### Demo Script

For a quick demonstration:

```bash
python demo_embedding_analysis.py
```

This will present options for:
1. Basic analysis with default settings
2. Comparison across multiple outlier detection methods

## Output Files

The script generates several output files:

1. **Visualization**: `embedding_outliers_analysis_{method}.png`
   - Multi-panel plot showing score distributions, changes, and top outliers

2. **Detailed Data**: `outlier_analysis_results.csv`
   - Complete dataset with outlier scores for all tokens
   - Columns: token_idx, token, original_score, compressed_score, score_change, score_ratio

3. **Summary**: `top_outliers_summary.txt`
   - Text summary with top outlier tokens for both models
   - Analysis parameters and statistics

## Example Workflow

1. **Load and analyze original model**:
   ```python
   # Extract embeddings and calculate outlier scores
   original_weights, tokens = extract_embedding_weights(model, tokenizer)
   original_scores = calculate_outlier_metrics(original_weights, "zscore")
   ```

2. **Compress the model**:
   ```python
   # Apply NNCF compression
   compressed_model = nncf.compress_weights(
       model,
       dataset=calibration_dataset,
       mode=CompressWeightsMode.INT4_SYM,
       ratio=0.8,
       group_size=128
   )
   ```

3. **Analyze compressed model**:
   ```python
   # Extract compressed embeddings and compare
   compressed_weights, _ = extract_embedding_weights(compressed_model, tokenizer)
   compressed_scores = calculate_outlier_metrics(compressed_weights, "zscore")
   ```

4. **Visualize results**:
   ```python
   # Create comprehensive visualizations
   create_outlier_visualization(original_scores, compressed_scores, tokens)
   ```

## Understanding the Results

### Outlier Scores
- Higher scores indicate tokens whose embeddings are more different from the average
- Common outliers include special tokens, rare words, and tokens with unique semantic properties

### Compression Impact
- Compare how compression affects different types of tokens
- Analyze if outlier patterns are preserved after compression
- Check correlation between original and compressed outlier scores

### Use Cases
- **Model Analysis**: Understand which tokens are most affected by compression
- **Quality Assessment**: Evaluate compression impact on embedding space structure
- **Token Selection**: Identify problematic tokens for further analysis
- **Compression Tuning**: Guide compression parameter selection

## Technical Details

### Compression Parameters
- **Mode**: Determines quantization scheme (symmetric vs asymmetric, bit width)
- **Ratio**: Percentage of layers to compress (1.0 = all layers)
- **Group Size**: Number of weights sharing quantization parameters

### Outlier Methods
- **Z-Score**: `||(embedding_norm - mean_norm) / std_norm||`
- **Magnitude**: `||embedding||_2`
- **Variance**: `var(embedding_dimensions)`
- **Max Absolute**: `max(|embedding|)`

### Calibration Dataset
- Uses WikiText-2 dataset for data-aware compression
- Filters samples by minimum length
- Applies tokenization with padding/truncation

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `--dataset-size` or use smaller model
2. **CUDA Errors**: Use `--device cpu` for CPU-only execution
3. **Missing Dependencies**: Install required packages with pip
4. **Tokenizer Errors**: Some tokens may not decode properly (handled gracefully)

### Performance Tips

- Use smaller dataset size for faster execution during development
- Consider using GPU for larger models (set `--device cuda`)
- Adjust `--top-k` based on your analysis needs

## Example Output

```
=== Outlier Analysis Summary (ZSCORE) ===
Total tokens analyzed: 32000
Original model - Mean outlier score: 0.8234
Original model - Std outlier score: 0.5432
Compressed model - Mean outlier score: 0.8156
Compressed model - Std outlier score: 0.5398
Mean score change: -0.0078
Std score change: 0.0892
Correlation between original and compressed scores: 0.9456

Top outlier tokens (Original model):
 1. <unk>                  (idx:     0, score: 3.2134)
 2. <s>                    (idx:     1, score: 2.8765)
 3. </s>                   (idx:     2, score: 2.7432)
 ...
```

This analysis helps understand how NNCF weight compression affects the embedding space and identifies which tokens are most impacted by the compression process.
