# Model Output Comparison Tool

This tool compares outputs between original and compressed LLM models and identifies the top-k rows/tokens with the largest differences for specified layers.

## Files

- `compare_model_outputs.py` - Main comparison tool
- `test_model_comparison.py` - Simple test script demonstrating usage

## Features

- Compares any Hugging Face LLM model in original vs compressed form
- Supports multiple distance metrics (L2, cosine distance, MSE)
- Extracts outputs from specific layers (embedding, lm_head, etc.)
- Finds top-k rows/tokens with largest differences
- Shows corresponding tokens for easier interpretation
- Supports different compression configurations

## Usage

### Basic Usage

```bash
python tools/compare_model_outputs.py \
    --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --layers embed_tokens lm_head \
    --top-k 10 \
    --metric l2 \
    --num-samples 3 \
    --max-length 32
```

### Parameters

- `--model-id`: Hugging Face model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `--layers`: Layer name patterns to analyze (default: ["embed", "lm_head"])
- `--top-k`: Number of top different rows to show (default: 10)
- `--metric`: Distance metric ("l2", "cosine", "mse") (default: "l2")
- `--num-samples`: Number of input samples to test (default: 3)
- `--max-length`: Maximum sequence length (default: 32)
- `--compression-params`: JSON string with compression parameters
- `--device`: Device to run inference on (default: "cpu")

### Example with Custom Compression

```bash
python tools/compare_model_outputs.py \
    --model-id "microsoft/DialoGPT-small" \
    --layers transformer.wte transformer.ln_f \
    --top-k 5 \
    --metric cosine \
    --compression-params '{"mode": "int8_sym", "group_size": 64}' \
    --device cpu
```

### Running the Test

```bash
python tools/test_model_comparison.py
```

## Output Format

The tool outputs:

1. **Layer Summary**: For each layer, shows mean and max distances
2. **Top-K Analysis**: Lists the top-k positions with largest differences, including:
   - Batch and sequence position
   - Distance value
   - Corresponding token (if available)
   - Token ID

### Example Output

```bash
================================================================================
LAYER COMPARISON RESULTS
================================================================================

Layer: model.embed_tokens
Distance metric: l2
Mean distance: 0.051234
Max distance: 0.234567

Top-5 rows with largest differences:
  1. Batch 0, Position 7: distance = 0.234567
      Token: 'the' (ID: 279)
  2. Batch 1, Position 3: distance = 0.198432
      Token: 'and' (ID: 322)
  ...
```

## How It Works

1. **Load Models**: Loads original model and creates compressed version using NNCF
2. **Layer Hooks**: Registers forward hooks to capture outputs from specified layers
3. **Inference**: Runs both models on sample inputs from WikiText dataset
4. **Distance Calculation**: Computes distances between layer outputs using chosen metric
5. **Top-K Selection**: Identifies positions with largest differences
6. **Token Mapping**: Maps positions back to original tokens for interpretation

## Supported Distance Metrics

- **L2**: Euclidean distance `||orig - comp||_2`
- **Cosine**: Cosine distance `1 - cosine_similarity(orig, comp)`
- **MSE**: Mean squared error `mean((orig - comp)^2)`

## Layer Selection

The tool automatically finds layers matching the provided patterns. For example:

- `embed` matches embedding layers like `embed_tokens`, `word_embeddings`
- `lm_head` matches language modeling head layers
- `attention` matches attention layers

If no layers are found, the tool lists available layer names to help you choose.

## Requirements

- PyTorch
- Transformers
- NNCF
- Datasets
- NumPy

## Troubleshooting

1. **Memory Issues**: Reduce `--num-samples` or `--max-length`
2. **Layer Not Found**: Run once to see available layer names
3. **CUDA Issues**: Use `--device cpu` for CPU-only execution
4. **Model Download**: Ensure internet connection for first-time model download
