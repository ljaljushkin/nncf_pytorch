#!/usr/bin/env python3  # noqa: CPY001
"""
Clean benchmark analysis script to compare CUDA vs Triton performance.
Calculates relative performance as (cuda - triton) / cuda.
Produces output matching the requested format.
"""

import glob
from pathlib import Path

import pandas as pd


def extract_short_name(filename):
    """Extract short benchmark name from filename."""
    base = Path(filename).stem
    if base.startswith("benchmark_"):
        name = base[10:]  # Remove 'benchmark_' prefix

        # Extract shorter names for readability
        if name.startswith("cuda_"):
            return name  # Keep full cuda name
        elif name.startswith("triton_"):
            # Extract just the number or key identifier
            parts = name.split("_")
            if len(parts) >= 2:
                return f"triton_{parts[1]}"  # e.g., triton_10, triton_11
            return name
    return base


def load_and_process_data():
    """Load all benchmark CSV files and process them."""
    benchmark_files = glob.glob("benchmark_*.csv")

    all_data = []

    for file in benchmark_files:
        # name = extract_short_name(file)
        name = Path(file).stem.replace("benchmark_", "")

        try:
            df = pd.read_csv(file)

            # Select relevant data
            for _, row in df.iterrows():
                all_data.append(
                    {
                        "name": name,
                        "tensor_type": row["tensor_type"],
                        "input_size": row["input_size"],
                        "forward_avg": row["forward_avg"],
                    }
                )

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return pd.DataFrame(all_data)


def calculate_relative_performance(df):
    """Calculate relative performance for triton vs cuda."""

    # Find CUDA baseline
    cuda_data = df[df["name"] == "cuda_507d96"].copy()

    if cuda_data.empty:
        print("Error: No CUDA baseline found!")
        return pd.DataFrame()

    # Create lookup for CUDA times
    cuda_lookup = {}
    for _, row in cuda_data.iterrows():
        key = (row["tensor_type"], row["input_size"])
        cuda_lookup[key] = row["forward_avg"]

    # Create result rows
    results = []

    # Add CUDA rows first
    for _, row in cuda_data.iterrows():
        results.append(
            {
                "name": row["name"],
                "tensor_type": row["tensor_type"],
                "input_size": row["input_size"],
                "cuda": f"{row['forward_avg']:.2f}",
                "triton": "",
                "rel": "",
            }
        )

    # Add Triton rows with relative performance
    triton_data = df[df["name"].str.startswith("triton_")].copy()

    for _, row in triton_data.iterrows():
        key = (row["tensor_type"], row["input_size"])

        if key in cuda_lookup:
            cuda_time = cuda_lookup[key]
            triton_time = row["forward_avg"]
            rel_perf = (cuda_time - triton_time) / cuda_time

            results.append(
                {
                    "name": row["name"],
                    "tensor_type": row["tensor_type"],
                    "input_size": row["input_size"],
                    "cuda": "",
                    "triton": f"{triton_time:.2f}",
                    "rel": f"{rel_perf:.0%}",
                }
            )

    return pd.DataFrame(results)


def main():
    print("Loading and processing benchmark data...")
    df = load_and_process_data()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} records")

    print("\nCalculating relative performance...")
    results_df = calculate_relative_performance(df)

    print("\nFormatted Results:")
    print("=" * 80)

    final_df = results_df.sort_values(["name", "tensor_type", "input_size"])

    # Save to CSV
    output_file = "clean_benchmark_results.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
