#!/usr/bin/env python3  # noqa: CPY001
"""
Benchmark analysis script with relative performance columns for each metric.
Format: name tensor_type input_size forward_avg forward_avg_rel backward_avg
        backward_avg_rel forward_gb_avg forward_gb_avg_rel backward_gb_avg
        backward_gb_avg_rel
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


def calculate_relative_performance(df):
    """Calculate relative performance for all metrics."""

    # Find CUDA baseline
    cuda_data = df[df["name"] == "cuda_507d96"].copy()

    if cuda_data.empty:
        print("Error: No CUDA baseline found!")
        return df

    # Create lookup dictionaries for CUDA times for each metric
    metrics = ["forward_avg", "backward_avg", "forward_gb_avg", "backward_gb_avg"]
    cuda_lookups = {}

    for metric in metrics:
        cuda_lookups[metric] = {}
        for _, row in cuda_data.iterrows():
            key = (row["tensor_type"], row["input_size"])
            cuda_lookups[metric][key] = row[metric]

    # Add relative performance columns
    df_result = df.copy()

    for metric in metrics:
        rel_col = f"{metric}_rel"
        df_result[rel_col] = None

        for idx, row in df_result.iterrows():
            key = (row["tensor_type"], row["input_size"])

            if row["name"].startswith("cuda_"):
                # CUDA rows don't have relative performance
                df_result.loc[idx, rel_col] = ""
            elif row["name"].startswith("triton_") and key in cuda_lookups[metric]:
                # Calculate relative performance: (cuda - triton) / cuda
                cuda_time = cuda_lookups[metric][key]
                triton_time = row[metric]
                rel_perf = (cuda_time - triton_time) / cuda_time
                df_result.loc[idx, rel_col] = f"{rel_perf:.0%}"
            else:
                df_result.loc[idx, rel_col] = ""

    return df_result


def format_results_with_rel_columns(df):
    """Format results showing metrics with their relative performance columns."""

    lines = []
    header = (
        "name tensor_type input_size forward_avg forward_avg_rel backward_avg "
        "backward_avg_rel forward_gb_avg forward_gb_avg_rel backward_gb_avg "
        "backward_gb_avg_rel"
    )
    lines.append(header)

    # Sort by name, tensor_type, input_size
    df_sorted = df.sort_values(["name", "tensor_type", "input_size"])

    for _, row in df_sorted.iterrows():
        name = row["name"]
        tensor_type = row["tensor_type"]
        input_size = row["input_size"]
        forward_avg = f"{row['forward_avg']:.2f}"
        forward_avg_rel = row["forward_avg_rel"]
        backward_avg = f"{row['backward_avg']:.2f}"
        backward_avg_rel = row["backward_avg_rel"]
        forward_gb_avg = f"{row['forward_gb_avg']:.2f}"
        forward_gb_avg_rel = row["forward_gb_avg_rel"]
        backward_gb_avg = f"{row['backward_gb_avg']:.2f}"
        backward_gb_avg_rel = row["backward_gb_avg_rel"]

        line = (
            f"{name} {tensor_type} {input_size} {forward_avg} {forward_avg_rel} "
            f"{backward_avg} {backward_avg_rel} {forward_gb_avg} {forward_gb_avg_rel} "
            f"{backward_gb_avg} {backward_gb_avg_rel}"
        )
        lines.append(line)

    return lines


def create_csv_output(df):
    """Create a clean CSV output with all metrics and relative performance."""
    # Sort by name, tensor_type, input_size
    df_sorted = df.sort_values(["name", "tensor_type", "input_size"])

    # Select and round the columns
    columns = [
        "name",
        "tensor_type",
        "input_size",
        "forward_avg",
        "forward_avg_rel",
        "backward_avg",
        "backward_avg_rel",
        "forward_gb_avg",
        "forward_gb_avg_rel",
        "backward_gb_avg",
        "backward_gb_avg_rel",
    ]
    output_df = df_sorted[columns].copy()

    # Round numeric columns
    numeric_cols = ["forward_avg", "backward_avg", "forward_gb_avg", "backward_gb_avg"]
    for col in numeric_cols:
        output_df[col] = output_df[col].round(2)

    return output_df


def main():
    print("Loading and processing benchmark data...")
    df = load_and_process_data()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} records")

    print("\nCalculating relative performance for all metrics...")
    df_with_rel = calculate_relative_performance(df)

    print("\nFormatting results with relative performance columns...")
    lines = format_results_with_rel_columns(df_with_rel)

    # Display results
    print("\nResults:")
    print("=" * 150)
    for line in lines:
        print(line)

    # Save text output
    output_file = "metrics_columns_analysis_results.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nText results saved to {output_file}")

    # Save CSV output
    csv_df = create_csv_output(df_with_rel)
    csv_file = "metrics_columns_analysis_results.csv"
    csv_df.to_csv(csv_file, index=False)
    print(f"CSV results saved to {csv_file}")

    # Show summary
    print("\nSummary:")
    print(f"Total records: {len(df)}")
    print(f"Unique benchmarks: {sorted(df['name'].unique())}")
    print(f"Tensor types: {sorted(df['tensor_type'].unique())}")
    print(f"Input sizes: {sorted(df['input_size'].unique())}")


if __name__ == "__main__":
    main()
