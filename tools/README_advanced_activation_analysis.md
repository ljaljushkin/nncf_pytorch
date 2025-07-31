# Model Output Comparison Tool with Activation Analysis

This tool provides comprehensive functionality for comparing outputs between original and compressed LLM models, with special focus on OpenVINO models and activation analysis.

## New Features Added

### 1. Advanced Activation Collection (`collect_activations`)
Enhanced NNCF statistics-based activation collection for OpenVINO models:
- Uses proper NNCF graph conversion and statistics aggregation
- Collects both input and output activations for matched layers
- Supports regex pattern matching for layer selection

### 2. Weight Collection (`collect_weights`)
Specialized function for collecting weights from specific layers:
- Similar to activation collection but focuses on layer weights
- Useful for analyzing weight distribution changes after quantization

### 3. Histogram Visualization (`draw_activations_2`)
Advanced visualization function that creates detailed histogram plots:
- Compares floating point vs quantized model activations
- Shows distribution overlays with transparency
- Generates multi-subplot analysis with:
  - Mean and standard deviation error bars
  - Absolute difference plots
  - Relative difference analysis
- Supports multiple layer pattern matching

## Usage Examples

### Basic Model Comparison (Existing Functionality)
```bash
# Compare PyTorch models
python compare_model_outputs.py \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --backend pytorch \
    --layers embed lm_head \
    --top-k 10

# Compare OpenVINO models
python compare_model_outputs.py \
    --model-id path/to/openvino/model \
    --backend openvino \
    --layers "__module.model.layers.0.mlp.gate_proj/ov_ext::linear/MatMul" \
    --top-k 10
```

### Advanced Activation Visualization (New)
```bash
# Use the example script for histogram analysis
python example_draw_activations.py \
    --model-fp path/to/fp32/openvino/model \
    --model-int path/to/quantized/openvino/model \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --n-samples 2
```

### Programmatic Usage
```python
from compare_model_outputs import draw_activations_2, collect_activations, collect_weights
import re
from nncf import Dataset

# Create histogram comparison plots
draw_activations_2(
    model_name_fp="path/to/fp32/model",
    model_name_int="path/to/int8/model",
    tokenizer_name="microsoft/DialoGPT-medium",
    n_samples=1
)

# Collect activations for specific layers
model = load_openvino_model("path/to/model")
dataset = create_nncf_dataset()
pattern = re.compile(r'__module.model.layers.\d+.self_attn')
activations = collect_activations(model, dataset, pattern, subset_size=4)

# Collect weights
weights = collect_weights(model, dataset, pattern, subset_size=1)
```

## Layer Pattern Examples

### Common OpenVINO Layer Patterns
```python
# Self-attention layers
r'__module.model.layers.\d+.self_attn'

# MLP layers
r'__module.model.layers.\d+.mlp'

# Specific operations
r'__module.model.layers.\d+.self_attn.v_proj/ov_ext::linear/MatMul'
r'__module.model.layers.\d+.mlp.down_proj/ov_ext::linear/MatMul'

# Unsqueeze operations
r'__module.model.layers.\d+.self_attn/aten::unsqueeze/Unsqueeze_2'

# Embedding layers
r'__module.model.embed_tokens'
```

## Output Interpretation

### Histogram Plots
The `draw_activations_2` function generates several types of plots:

1. **Individual Layer Histograms**: Shows distribution overlap between FP32 and quantized versions
2. **Mean/Std Error Bar Plots**: Compares statistical moments across layers
3. **Absolute Difference Plots**: Shows magnitude of changes introduced by quantization
4. **Relative Difference Plots**: Shows proportional changes (most important for assessing impact)

### Key Metrics to Watch
- **Large relative differences**: Indicate layers most affected by quantization
- **Distribution shape changes**: May indicate clipping or range reduction
- **Statistical moment changes**: Mean/std shifts can affect model behavior

## Dependencies

### Required Packages
```bash
pip install torch transformers datasets
pip install openvino optimum[openvino]
pip install matplotlib numpy
pip install nncf
```

### Optional for Advanced Features
```bash
pip install jupyter  # For notebook environments
pip install seaborn  # For enhanced plotting (if desired)
```

## Technical Details

### NNCF Statistics Collection
The tool uses NNCF's experimental statistics collection framework:
- `TensorCollector` with `RawReducer` and `NoopAggregator`
- `OVStatisticsAggregator` for OpenVINO models
- `StatisticPointsContainer` for managing collection points

### Memory Considerations
- Activation collection stores full tensors in memory
- Use `subset_size` parameter to control memory usage
- Large models may require reducing `n_samples` parameter

### Supported Model Types
- **PyTorch**: HuggingFace transformers models
- **OpenVINO**: Converted models via optimum[openvino]
- **Quantization**: INT4, INT8, NF4, E2M1 modes supported

## Troubleshooting

### Common Issues
1. **"OpenVINO not available"**: Install openvino and optimum[openvino]
2. **"No operations found matching pattern"**: Check layer pattern regex
3. **Memory errors**: Reduce n_samples or subset_size
4. **Import errors**: Ensure all dependencies installed

### Debug Tips
- Use `--help` flag to see all available options
- Check model structure with simple patterns first
- Monitor memory usage with large models
- Verify regex patterns against actual layer names

## Performance Tips
- Start with small n_samples (1-2) for initial exploration
- Use specific layer patterns rather than broad matches
- Consider using subset_size < 10 for large models
- Close matplotlib plots to free memory between analyses
