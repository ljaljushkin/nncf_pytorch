# Model Comparison Tool

A comprehensive tool for comparing the outputs of original vs compressed LLM models using NNCF weight compression. The tool supports both PyTorch and OpenVINO backends and can analyze layer-level differences to understand the impact of compression.

## Features

- **Dual Backend Support**: PyTorch (fully functional) and OpenVINO (experimental)
- **Layer-Level Analysis**: Extract and compare outputs from specific model layers
- **Multiple Distance Metrics**: L2 norm, cosine distance, and MSE
- **Top-K Analysis**: Find the positions/tokens with the largest compression differences
- **Compression Modes**: Support for various NNCF compression modes (int4_sym, int8_sym, etc.)
- **Flexible Input**: Use calibration datasets or custom text inputs

## Quick Start

### PyTorch Backend (Recommended)

```bash
# Basic comparison of gate projection layer
python compare_model_outputs.py \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --backend pytorch \
    --layers model.layers.0.mlp.gate_proj \
    --top-k 5 \
    --metric l2 \
    --compression-params '{"mode": "int4_sym", "group_size": 128}'

# Compare multiple layers with different metrics
python compare_model_outputs.py \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --backend pytorch \
    --layers model.layers.0.mlp.gate_proj model.layers.0.mlp.up_proj model.embed_tokens \
    --top-k 3 \
    --metric cosine \
    --num-samples 5 \
    --compression-params '{"mode": "int8_sym"}'
```

### OpenVINO Backend (Experimental)

⚠️ **Note**: OpenVINO backend is currently experimental and may have issues with layer extraction.

```bash
# OpenVINO comparison (when working)
python compare_model_outputs.py \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --backend openvino \
    --layers gate_proj up_proj \
    --top-k 3 \
    --metric l2 \
    --compression-params '{"mode": "int4_sym", "group_size": 128}'
```

## Parameters

### Required Arguments
- `--model-id`: HuggingFace model identifier (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `--backend`: Backend to use (`pytorch` or `openvino`)
- `--layers`: Layer names to analyze (space-separated)

### Compression Parameters
- `--compression-params`: JSON string with NNCF compression parameters
  - Available modes: `int4_sym`, `int4_asym`, `int8_sym`, `int8_asym`, `nf4`, `e2m1`
  - Example: `'{"mode": "int4_sym", "group_size": 128, "ratio": 0.8}'`

### Analysis Options
- `--top-k`: Number of top differences to report (default: 5)
- `--metric`: Distance metric (`l2`, `cosine`, `mse`) (default: `l2`)
- `--num-samples`: Number of input samples to process (default: 3)
- `--max-length`: Maximum input sequence length (default: 32)

### Device and Performance
- `--device`: Device to use (`cpu`, `cuda`) (default: `cpu`)

## Layer Name Discovery

Use the helper tool to find available layer names:

```bash
# Find all layer names in a model
python discover_layers.py TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Find layers matching a pattern
python discover_layers.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --pattern "mlp.*proj"
```

## Example Output

```
================================================================================
LAYER COMPARISON RESULTS
================================================================================

Layer: model.layers.0.mlp.gate_proj
Distance metric: l2
Mean distance: 2.491294
Max distance: 3.744726

Top-2 rows with largest differences:
  1. Batch 0, Position 1: distance = 3.744726
      Token: '' (ID: 29871)
  2. Batch 0, Position 2: distance = 2.616566
      Token: '=' (ID: 353)
------------------------------------------------------------

Analysis complete! Processed 1 samples across 1 layers.
```

## Backend Differences

### PyTorch Backend
- **Status**: ✅ Fully functional
- **Layer Names**: Use PyTorch module names (e.g., `model.layers.0.mlp.gate_proj`)
- **Extraction Method**: Forward hooks for real-time layer output capture
- **Performance**: Fast, direct access to intermediate outputs

### OpenVINO Backend
- **Status**: ⚠️ Experimental (issues with layer extraction)
- **Layer Names**: Use simplified names (e.g., `gate_proj`, `up_proj`)
- **Extraction Method**: Model graph modification to add output nodes
- **Performance**: Optimized for deployment scenarios

## Testing

Run the comprehensive test suite:

```bash
# Test both backends with various configurations
python test_model_comparison.py
```

## Known Issues

1. **OpenVINO Layer Extraction**: Current implementation has issues with OpenVINO model graph modification for intermediate layer outputs
2. **Memory Usage**: Large models may require significant memory for both original and compressed versions
3. **Layer Name Mapping**: Different backends use different layer naming conventions

## Implementation Notes

- **In-place Compression Fix**: The tool uses `copy.deepcopy()` to avoid modifying the original model during compression
- **Enum Conversion**: String compression parameters are automatically converted to NNCF enums
- **Error Handling**: Robust error handling for missing layers and invalid parameters

## Dependencies

```
torch
transformers
nncf
datasets
numpy
openvino (optional, for OpenVINO backend)
optimum[openvino] (optional, for OpenVINO conversion)
```
