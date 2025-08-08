# Flexible Benchmark Analysis

This script (`flexible_benchmark_analysis.py`) is a flexible version of the original `final_benchmark_analysis.py` that allows you to compare any set of benchmark files against a reference file using regex patterns.

## Features

- **Flexible reference selection**: Choose any benchmark file as your reference/baseline
- **Regex pattern matching**: Use regex patterns to select which files to compare
- **Multiple metrics**: Analyze forward_avg, backward_avg, forward_gb_avg, backward_gb_avg, or all metrics
- **Optional output files**: Save results to files with custom prefixes
- **Excel export**: Save results to formatted Excel files with proper number formatting and color coding
- **Clean formatted output**: Results in the format: `name tensor_type input_size ref_value comparison_value rel_performance`

## Usage

### Basic Usage

```bash
python flexible_benchmark_analysis.py --ref benchmark_cuda_507d96.csv --pattern "benchmark_triton_.*\.csv"
```

### Advanced Options

```bash
python flexible_benchmark_analysis.py \
    --ref benchmark_cuda_507d96.csv \
    --pattern "benchmark_triton_1[0-9]_.*\.csv" \
    --metric forward_avg \
    --output my_analysis \
    --excel
```

### Command Line Arguments

- `--ref`: **(Required)** Reference file to compare against
- `--pattern`: **(Required)** Regex pattern to match comparison files
- `--metric`: Metric to analyze (choices: forward_avg, backward_avg, forward_gb_avg, backward_gb_avg, all) [default: all]
- `--output`: Output file prefix (optional)
- `--excel`: Save results to Excel file with formatting (requires openpyxl)

## Examples

### Example 1: Compare all Triton implementations against CUDA baseline
```bash
python flexible_benchmark_analysis.py \
    --ref benchmark_cuda_507d96.csv \
    --pattern "benchmark_triton_.*\.csv" \
    --output triton_vs_cuda
```

### Example 2: Compare only Triton versions 10-19 for forward pass only
```bash
python flexible_benchmark_analysis.py \
    --ref benchmark_cuda_507d96.csv \
    --pattern "benchmark_triton_1[0-9]_.*\.csv" \
    --metric forward_avg \
    --output triton_10s_forward
```

### Example 3: Compare compile results against a Triton baseline
```bash
python flexible_benchmark_analysis.py \
    --ref benchmark_triton_13_02b9d9.csv \
    --pattern "benchmark_compile_.*\.csv" \
    --output compile_vs_triton
```

### Example 4: Compare specific versions
```bash
python flexible_benchmark_analysis.py \
    --ref benchmark_cuda_507d96.csv \
    --pattern "benchmark_(triton_12|triton_13|compile_02b9d9).*\.csv"
```

### Example 5: Export to Excel with formatting
```bash
python flexible_benchmark_analysis.py \
    --ref benchmark_cuda_507d96.csv \
    --pattern "benchmark_triton_1[0-3]_.*\.csv" \
    --output analysis_results \
    --excel
```

## Output Format

The script outputs results in a clean tabular format:

```
name tensor_type input_size ref_value comparison_value rel_performance
cuda_507d96 float16 (2048, 1) 0.05 - -
cuda_507d96 float16 (2048, 62) 0.12 - -
triton_10 float16 (2048, 1) 0.05 0.04 20%
triton_10 float16 (2048, 62) 0.12 0.10 17%
```

## Regular Expression Patterns

Here are some useful regex patterns:

- `benchmark_triton_.*\.csv` - All Triton benchmarks
- `benchmark_cuda_.*\.csv` - All CUDA benchmarks
- `benchmark_compile_.*\.csv` - All compile benchmarks
- `benchmark_triton_1[0-9]_.*\.csv` - Triton versions 10-19
- `benchmark_(triton_12|triton_13).*\.csv` - Specific Triton versions
- `benchmark_.*_per_group.*\.csv` - All per-group benchmarks

## Output Files

When using the `--output` parameter, the script creates:

- `{prefix}_{metric}_results.txt` - Individual metric results
- `{prefix}_all_metrics_results.txt` - All metrics combined (when analyzing all metrics)

When using the `--excel` parameter, the script creates:

- `{prefix}.xlsx` (or `benchmark_analysis.xlsx` if no prefix) - Formatted Excel file with:
  - Separate worksheets for each metric
  - Proper number formatting (2 decimal places for times)
  - Percentage formatting for relative performance
  - Color coding (green for better performance, red for worse)
  - Auto-adjusted column widths and row heights
  - Professional styling with headers and titles

## Excel Features

The Excel export includes advanced formatting:

- **Number formatting**: Time values show 2 decimal places
- **Percentage formatting**: Relative performance shown as percentages
- **Color coding**:
  - Green background for positive performance improvements
  - Red background for performance regressions
  - Professional blue headers
- **Auto-sizing**: Columns and rows automatically adjust to content
- **Multiple worksheets**: Each metric gets its own worksheet
- **Professional styling**: Clean, readable format suitable for reports

### Requirements for Excel Export

To use the Excel export feature, install the required dependency:

```bash
pip install openpyxl
```

## Running Examples

Use the provided `example_usage.py` script to see the tool in action:

```bash
python example_usage.py
```

This will run several example analyses and show different usage patterns.

## Differences from Original Script

- **Flexible reference**: Choose any file as baseline (not hardcoded to cuda_507d96)
- **Regex matching**: Use patterns to select comparison files (not hardcoded to triton files)
- **Single metric option**: Analyze specific metrics instead of always analyzing all
- **Better output control**: Optional file output with custom prefixes
- **Excel export**: Formatted Excel files with professional styling and color coding
- **More descriptive output**: Shows both reference and comparison values
