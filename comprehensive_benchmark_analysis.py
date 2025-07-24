#!/usr/bin/env python3  # noqa: CPY001
"""
Comprehensive benchmark analysis script to compare CUDA vs Triton performance.
Calculates relative performance as (cuda - triton) / cuda for all metrics.
Handles: forward_avg, backward_avg, forward_mb_avg, backward_mb_avg
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
                        "forward_mb_avg": row["forward_mb_avg"],
                        "backward_mb_avg": row["backward_mb_avg"],
                    }
                )

        except Exception as e:
            print(f"Error reading {file}: {e}")

    return pd.DataFrame(all_data)


def calculate_relative_performance_for_metric(df, metric_name):
    """Calculate relative performance for a specific metric."""

    # Find CUDA baseline
    cuda_data = df[df["name"] == "cuda_507d96"].copy()

    if cuda_data.empty:
        print("Error: No CUDA baseline found!")
        return pd.DataFrame()

    # Create lookup for CUDA times
    cuda_lookup = {}
    for _, row in cuda_data.iterrows():
        key = (row["tensor_type"], row["input_size"])
        cuda_lookup[key] = row[metric_name]

    # Create result rows
    results = []

    # Add CUDA rows first
    for _, row in cuda_data.iterrows():
        results.append(
            {
                "name": row["name"],
                "tensor_type": row["tensor_type"],
                "input_size": row["input_size"],
                "metric": metric_name,
                "cuda": f"{row[metric_name]:.2f}",
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
            triton_time = row[metric_name]
            rel_perf = (cuda_time - triton_time) / cuda_time

            results.append(
                {
                    "name": row["name"],
                    "tensor_type": row["tensor_type"],
                    "input_size": row["input_size"],
                    "metric": metric_name,
                    "cuda": "",
                    "triton": f"{triton_time:.2f}",
                    "rel": f"{rel_perf:.0%}",
                }
            )

    return pd.DataFrame(results)


def process_all_metrics(df):
    """Process all metrics and combine results."""
    metrics = ["forward_avg", "backward_avg", "forward_mb_avg", "backward_mb_avg"]

    all_results = []

    for metric in metrics:
        print(f"Processing {metric}...")
        metric_results = calculate_relative_performance_for_metric(df, metric)
        all_results.append(metric_results)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results


def format_and_display_by_metric(df):
    """Format and display results grouped by metric."""

    metrics = ["forward_avg", "backward_avg", "forward_mb_avg", "backward_mb_avg"]

    for metric in metrics:
        print(f"\n{'=' * 20} {metric.upper()} {'=' * 20}")
        print("name tensor_type input_size cuda triton rel")

        metric_data = df[df["metric"] == metric].sort_values(["name", "tensor_type", "input_size"])

        for _, row in metric_data.iterrows():
            name = row["name"]
            tensor_type = row["tensor_type"]
            input_size = row["input_size"]
            cuda = row["cuda"]
            triton = row["triton"]
            rel = row["rel"]

            print(f"{name} {tensor_type} {input_size} {cuda} {triton} {rel}")


def create_summary_table(df):
    """Create a summary table showing all metrics side by side."""

    # Get unique combinations of name, tensor_type, input_size
    unique_combos = df[["name", "tensor_type", "input_size"]].drop_duplicates()

    summary_rows = []

    for _, combo in unique_combos.iterrows():
        name = combo["name"]
        tensor_type = combo["tensor_type"]
        input_size = combo["input_size"]

        # Get data for this combination
        combo_data = df[(df["name"] == name) & (df["tensor_type"] == tensor_type) & (df["input_size"] == input_size)]

        row = {
            "name": name,
            "tensor_type": tensor_type,
            "input_size": input_size,
        }

        # Add metrics
        for metric in ["forward_avg", "backward_avg", "forward_mb_avg", "backward_mb_avg"]:
            metric_data = combo_data[combo_data["metric"] == metric]
            if not metric_data.empty:
                metric_row = metric_data.iloc[0]
                if name.startswith("cuda_"):
                    row[f"{metric}_cuda"] = metric_row["cuda"]
                    row[f"{metric}_triton"] = ""
                    row[f"{metric}_rel"] = ""
                else:
                    row[f"{metric}_cuda"] = ""
                    row[f"{metric}_triton"] = metric_row["triton"]
                    row[f"{metric}_rel"] = metric_row["rel"]
            else:
                row[f"{metric}_cuda"] = ""
                row[f"{metric}_triton"] = ""
                row[f"{metric}_rel"] = ""

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    print("Loading and processing benchmark data...")
    df = load_and_process_data()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} records")

    print("\nCalculating relative performance for all metrics...")
    results_df = process_all_metrics(df)

    # Display results by metric
    format_and_display_by_metric(results_df)

    # Create and save summary table
    print(f"\n{'=' * 20} CREATING SUMMARY TABLE {'=' * 20}")
    summary_df = create_summary_table(results_df)

    # Save all results
    results_file = "comprehensive_benchmark_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to {results_file}")

    summary_file = "benchmark_summary_table.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to {summary_file}")

    # Display a sample of the summary
    print("\nSample of summary table:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
