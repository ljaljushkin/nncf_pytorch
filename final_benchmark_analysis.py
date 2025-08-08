#!/usr/bin/env python3  # noqa: CPY001
"""
Final benchmark analysis script - produces clean output for all metrics.
Shows results in the exact format requested: name tensor_type input_size cuda triton rel
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
        name = extract_short_name(file)

        try:
            df = pd.read_csv(file)

            # Select relevant data including all metrics
            for _, row in df.iterrows():
                all_data.append(
                    {
                        "name": name,
                        "tensor_type": row["tensor_type"],
                        "input_size": row["input_size"],
                        "forward_avg": row["forward_avg"],
                        "backward_avg": row["backward_avg"],
                        "forward_gb_avg": row["forward_gb_avg"],
                        "backward_gb_avg": row["backward_gb_avg"],
                    }
                )

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return pd.DataFrame(all_data)


def format_metric_results(df, metric_name):
    """Format results for a specific metric in the requested format."""

    # Find CUDA baseline
    cuda_data = df[df["name"] == "cuda_507d96"].copy()

    if cuda_data.empty:
        print("Error: No CUDA baseline found!")
        return []

    # Create lookup for CUDA times
    cuda_lookup = {}
    for _, row in cuda_data.iterrows():
        key = (row["tensor_type"], row["input_size"])
        cuda_lookup[key] = row[metric_name]

    lines = []
    lines.append("name tensor_type input_size cuda triton rel")

    # Add CUDA rows first
    for _, row in cuda_data.iterrows():
        cuda_val = f"{row[metric_name]:.2f}"
        lines.append(f"{row['name']} {row['tensor_type']} {row['input_size']} {cuda_val}")

    # Add Triton rows with relative performance
    triton_data = df[df["name"].str.startswith("triton_")].copy()
    triton_data = triton_data.sort_values(["name", "tensor_type", "input_size"])

    for _, row in triton_data.iterrows():
        key = (row["tensor_type"], row["input_size"])

        if key in cuda_lookup:
            cuda_time = cuda_lookup[key]
            triton_time = row[metric_name]
            rel_perf = (cuda_time - triton_time) / cuda_time

            triton_val = f"{triton_time:.2f}"
            rel_val = f"{rel_perf:.0%}"
            lines.append(f"{row['name']} {row['tensor_type']} {row['input_size']} {triton_val} {rel_val}")

    return lines


def main():
    print("Loading and processing benchmark data...")
    df = load_and_process_data()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} records")

    metrics = [
        ("forward_avg", "Forward Average Time"),
        ("backward_avg", "Backward Average Time"),
        ("forward_gb_avg", "Forward Memory Bandwidth Average"),
        ("backward_gb_avg", "Backward Memory Bandwidth Average"),
    ]

    all_results = {}

    for metric_key, metric_title in metrics:
        print(f"\n{'=' * 20} {metric_title.upper()} {'=' * 20}")
        lines = format_metric_results(df, metric_key)

        for line in lines:
            print(line)

        all_results[metric_key] = lines

        # Save individual metric results
        output_file = f"{metric_key}_results.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"Results saved to {output_file}")

    # Save all results to one file
    with open("all_metrics_results.txt", "w") as f:
        for metric_key, metric_title in metrics:
            f.write(f"\n{'=' * 20} {metric_title.upper()} {'=' * 20}\n")
            f.write("\n".join(all_results[metric_key]))
            f.write("\n\n")

    print("\nAll results saved to all_metrics_results.txt")


if __name__ == "__main__":
    main()
