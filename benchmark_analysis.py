#!/usr/bin/env python3  # noqa: CPY001
"""
Benchmark analysis script to compare CUDA vs Triton performance.
Calculates relative performance as (cuda - triton) / cuda.
"""

import glob
from pathlib import Path

import pandas as pd


def extract_benchmark_name(filename):
    """Extract benchmark name from filename."""
    base = Path(filename).stem
    if base.startswith("benchmark_"):
        return base[10:]  # Remove 'benchmark_' prefix
    return base


def load_benchmark_data():
    """Load all benchmark CSV files and combine them."""
    benchmark_files = glob.glob("benchmark_*.csv")

    all_data = []

    for file in benchmark_files:
        name = extract_benchmark_name(file)
        print(f"Processing {file} -> {name}")

        try:
            df = pd.read_csv(file)

            # Add the benchmark name
            df["name"] = name

            # Select relevant columns
            relevant_cols = ["name", "tensor_type", "input_size", "forward_avg"]
            if all(col in df.columns for col in relevant_cols):
                subset = df[relevant_cols].copy()
                all_data.append(subset)
            else:
                print(f"Warning: Missing columns in {file}")

        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        print("No valid benchmark data found!")
        return pd.DataFrame()

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    return combined


def calculate_relative_performance(df):
    """Calculate relative performance for triton vs cuda."""

    # Find the CUDA baseline (assuming it's cuda_507d96)
    cuda_baseline = df[df["name"] == "cuda_507d96"].copy()

    if cuda_baseline.empty:
        print("Warning: No CUDA baseline (cuda_507d96) found!")
        return df

    # Create a lookup dictionary for CUDA performance
    cuda_lookup = {}
    for _, row in cuda_baseline.iterrows():
        key = (row["tensor_type"], row["input_size"])
        cuda_lookup[key] = row["forward_avg"]

    # Calculate relative performance for all rows
    df_result = df.copy()
    df_result["cuda"] = None
    df_result["triton"] = None
    df_result["rel"] = None

    for idx, row in df_result.iterrows():
        key = (row["tensor_type"], row["input_size"])

        if row["name"].startswith("cuda_"):
            # This is a CUDA run
            df_result.loc[idx, "cuda"] = row["forward_avg"]
            df_result.loc[idx, "triton"] = None
            df_result.loc[idx, "rel"] = None
        elif row["name"].startswith("triton_"):
            # This is a Triton run
            df_result.loc[idx, "triton"] = row["forward_avg"]

            if key in cuda_lookup:
                cuda_time = cuda_lookup[key]
                triton_time = row["forward_avg"]

                # Calculate relative performance: (cuda - triton) / cuda
                rel_perf = (cuda_time - triton_time) / cuda_time

                df_result.loc[idx, "cuda"] = cuda_time
                df_result.loc[idx, "rel"] = f"{rel_perf:.0%}"
            else:
                print(f"Warning: No CUDA baseline found for {key}")
                df_result.loc[idx, "cuda"] = None
                df_result.loc[idx, "rel"] = None

    return df_result


def format_output(df):
    """Format the output as requested."""
    # Select and reorder columns
    output_cols = ["name", "tensor_type", "input_size", "cuda", "triton", "rel"]
    df_output = df[output_cols].copy()

    # Round numeric values, handling None values
    df_output["cuda"] = pd.to_numeric(df_output["cuda"], errors="coerce").round(2)
    df_output["triton"] = pd.to_numeric(df_output["triton"], errors="coerce").round(2)

    # Sort by name, tensor_type, and input_size
    df_output = df_output.sort_values(["name", "tensor_type", "input_size"])

    return df_output


def main():
    print("Loading benchmark data...")
    df = load_benchmark_data()

    if df.empty:
        print("No data to process!")
        return

    print(f"\nLoaded {len(df)} benchmark records")
    print(f"Unique benchmarks: {sorted(df['name'].unique())}")

    print("\nCalculating relative performance...")
    df_with_rel = calculate_relative_performance(df)

    print("\nFormatting output...")
    df_final = format_output(df_with_rel)

    # Display results
    print("\nResults:")
    print("=" * 80)

    # Set display options for better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 30)

    print(df_final.to_string(index=False))

    # Save to file
    output_file = "benchmark_analysis_results.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
