#!/usr/bin/env python3  # noqa: CPY001
"""
Flexible benchmark analysis script - compares files matching a regex pattern against a reference file.
Usage: python flexible_benchmark_analysis.py --ref benchmark_cuda_507d96.csv --pattern "benchmark_triton_.*\.csv"
Shows results in the format: name tensor_type input_size ref_value comparison_value rel_performance
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

try:
    import openpyxl  # noqa: F401
    from openpyxl.styles import Alignment  # noqa: F401
    from openpyxl.styles import Font  # noqa: F401
    from openpyxl.styles import PatternFill  # noqa: F401
    from openpyxl.utils.dataframe import dataframe_to_rows  # noqa: F401

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


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
        elif name.startswith("compile_"):
            return name  # Keep full compile name
    return base


def find_matching_files(pattern):
    """Find files matching the regex pattern."""
    all_files = glob.glob("*.csv")
    matching_files = []

    compiled_pattern = re.compile(pattern)

    for file in all_files:
        if compiled_pattern.match(file):
            matching_files.append(file)

    return matching_files


def load_file_data(filename):
    """Load data from a single CSV file."""
    try:
        df = pd.read_csv(filename)
        name = extract_short_name(filename)

        data = []
        for _, row in df.iterrows():
            data.append(
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

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return pd.DataFrame()


def create_structured_data(ref_df, comparison_files, metric_name):
    """Create structured data for Excel export."""

    if ref_df.empty:
        return pd.DataFrame()

    # Create lookup for reference times
    ref_lookup = {}

    for _, row in ref_df.iterrows():
        key = (row["tensor_type"], row["input_size"])
        ref_lookup[key] = row[metric_name]

    data_rows = []

    # Add reference rows first
    for _, row in ref_df.iterrows():
        data_rows.append(
            {
                "name": row["name"],
                "tensor_type": row["tensor_type"],
                "input_size": row["input_size"],
                "ref_value": row[metric_name],
                "comparison_value": None,
                "rel_performance": None,
                "is_reference": True,
            }
        )

    # Process comparison files
    for comp_file in comparison_files:
        comp_df = load_file_data(comp_file)

        if comp_df.empty:
            continue

        comp_df = comp_df.sort_values(["tensor_type", "input_size"])

        for _, row in comp_df.iterrows():
            key = (row["tensor_type"], row["input_size"])

            if key in ref_lookup:
                ref_time = ref_lookup[key]
                comp_time = row[metric_name]
                rel_perf = (ref_time - comp_time) / ref_time

                data_rows.append(
                    {
                        "name": row["name"],
                        "tensor_type": row["tensor_type"],
                        "input_size": row["input_size"],
                        "ref_value": ref_time,
                        "comparison_value": comp_time,
                        "rel_performance": rel_perf,
                        "is_reference": False,
                    }
                )

    return pd.DataFrame(data_rows)


def aggregate_performance_data(all_structured_data, metrics):
    """Aggregate relative performance data across all metrics and implementations."""

    aggregated_results = []

    for metric_key, metric_title in metrics:
        if metric_key not in all_structured_data:
            continue

        df = all_structured_data[metric_key]
        if df.empty:
            continue

        # Include all rows with performance data (including reference with 0 values)
        comparison_df = df[df["rel_performance"].notna()].copy()

        if comparison_df.empty:
            continue

        # Group by implementation name and calculate statistics
        stats = (
            comparison_df.groupby("name")["rel_performance"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )

        stats["metric"] = metric_key
        stats["metric_title"] = metric_title
        stats.columns = [
            "name",
            "count",
            "mean_perf",
            "median_perf",
            "std_perf",
            "min_perf",
            "max_perf",
            "metric",
            "metric_title",
        ]

        aggregated_results.append(stats)

    if not aggregated_results:
        return pd.DataFrame()

    return pd.concat(aggregated_results, ignore_index=True)


def create_performance_plots(aggregated_data, all_structured_data, metrics, output_prefix="", baseline_name=""):
    """Create visualization plots for performance data."""

    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not available. Cannot create plots.")
        print("Install with: pip install matplotlib seaborn")
        return

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style for better-looking plots
        plt.style.use("default")
        sns.set_palette("husl")

        if aggregated_data.empty:
            print("No performance data available for plotting.")
            return

        # Create figure with subplots - only 2 plots now
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        baseline_text = f" (Baseline: {baseline_name})" if baseline_name else ""
        fig.suptitle(f"Benchmark Performance Analysis{baseline_text}", fontsize=16, fontweight="bold")

        # Plot 1: Mean Performance by Implementation (Bar Chart)
        ax1 = axes[0]
        pivot_mean = aggregated_data.pivot(index="name", columns="metric", values="mean_perf")
        pivot_mean.plot(kind="bar", ax=ax1, width=0.8)
        ax1.set_title("Mean Relative Performance by Implementation")
        ax1.set_xlabel("Implementation")
        ax1.set_ylabel("Mean Relative Performance")
        ax1.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.tick_params(axis="x", rotation=45)
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Performance Distribution (Box Plot)
        ax2 = axes[1]
        # Combine all performance data for box plot - include reference data
        all_perf_data = []
        for metric_key, _ in metrics:
            if metric_key in all_structured_data:
                df = all_structured_data[metric_key]
                comparison_df = df[df["rel_performance"].notna()].copy()
                if not comparison_df.empty:
                    comparison_df["metric"] = metric_key
                    all_perf_data.append(comparison_df[["name", "rel_performance", "metric"]])

        if all_perf_data:
            combined_perf = pd.concat(all_perf_data, ignore_index=True)
            sns.boxplot(data=combined_perf, x="name", y="rel_performance", hue="metric", ax=ax2)
            ax2.set_title("Performance Distribution by Implementation")
            ax2.set_xlabel("Implementation")
            ax2.set_ylabel("Relative Performance")
            ax2.tick_params(axis="x", rotation=45)
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax2.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        # Save plot
        plot_file = f"{output_prefix}_performance_analysis.png" if output_prefix else "performance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Performance plots saved to {plot_file}")

        # Create a second figure with detailed metric comparison
        if len(metrics) > 1:
            fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

            # Heatmap of performance across metrics and implementations - include reference data
            heatmap_data = aggregated_data.pivot(index="name", columns="metric", values="mean_perf")

            sns.heatmap(
                heatmap_data,
                annot=True,
                cmap="RdYlGn",
                center=0,
                fmt=".2%",
                ax=ax,
                cbar_kws={"label": "Relative Performance"},
            )
            heatmap_title = f"Performance Heatmap: Implementation vs Metric{baseline_text}"
            ax.set_title(heatmap_title)
            ax.set_xlabel("Metric")
            ax.set_ylabel("Implementation")

            plt.tight_layout()

            heatmap_file = f"{output_prefix}_performance_heatmap.png" if output_prefix else "performance_heatmap.png"
            plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
            print(f"Performance heatmap saved to {heatmap_file}")

        plt.show()

    except Exception as e:
        print(f"Error creating plots: {e}")


def save_to_excel(all_data, output_file, metrics):
    """Save results to Excel with proper formatting."""

    if not EXCEL_AVAILABLE:
        print("Warning: openpyxl not available. Cannot create Excel file.")
        print("Install with: pip install openpyxl")
        return

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment
        from openpyxl.styles import Font
        from openpyxl.styles import PatternFill

        # Create workbook with better compatibility
        wb = Workbook()

        # Set workbook properties for better Windows compatibility
        wb.properties.creator = "Flexible Benchmark Analysis"
        wb.properties.title = "Benchmark Analysis Results"

        # Remove default sheet
        if wb.worksheets:
            wb.remove(wb.active)

        for metric_key, metric_title in metrics:
            if metric_key not in all_data:
                continue

            df = all_data[metric_key]
            if df.empty:
                continue

            # Create worksheet with safe name
            safe_sheet_name = metric_key.replace("_", " ").title()[:31]  # Excel sheet name limit is 31 chars
            ws = wb.create_sheet(title=safe_sheet_name)

            # Prepare data for Excel
            excel_df = df.copy()
            excel_df = excel_df.drop("is_reference", axis=1)

            # Add title row
            ws.append([metric_title, "", "", "", "", ""])
            ws.merge_cells("A1:F1")

            # Style title
            title_cell = ws["A1"]
            title_cell.font = Font(bold=True, size=14)
            title_cell.alignment = Alignment(horizontal="center")
            title_cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            title_cell.font = Font(bold=True, size=14, color="FFFFFF")

            # Add empty row
            ws.append([])

            # Add headers
            headers = [
                "Name",
                "Tensor Type",
                "Input Size",
                "Reference Value",
                "Comparison Value",
                "Relative Performance",
            ]
            ws.append(headers)

            # Style headers
            header_row = ws.max_row
            for col_num, header in enumerate(headers, 1):
                cell = ws.cell(row=header_row, column=col_num)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
                cell.alignment = Alignment(horizontal="center")

            # Add data rows with proper formatting for Excel
            for _, row in excel_df.iterrows():
                # Clean and format the data for Excel compatibility
                name = str(row["name"]) if pd.notna(row["name"]) else ""
                tensor_type = str(row["tensor_type"]).replace("TensorType.", "") if pd.notna(row["tensor_type"]) else ""

                # Format input_size - remove brackets and quotes to make it Excel-friendly
                input_size = str(row["input_size"])
                if input_size.startswith("[") and input_size.endswith("]"):
                    input_size = input_size[1:-1]  # Remove brackets
                if input_size.startswith('"') and input_size.endswith('"'):
                    input_size = input_size[1:-1]  # Remove quotes

                ref_value = row["ref_value"] if pd.notna(row["ref_value"]) else None
                comp_value = row["comparison_value"] if pd.notna(row["comparison_value"]) else None
                rel_perf = row["rel_performance"] if pd.notna(row["rel_performance"]) else None

                ws.append([name, tensor_type, input_size, ref_value, comp_value, rel_perf])

            # Format columns
            for row_num in range(4, ws.max_row + 1):  # Start from data rows
                # Reference value (column D)
                ref_cell = ws.cell(row=row_num, column=4)
                if ref_cell.value is not None and isinstance(ref_cell.value, (int, float)):
                    ref_cell.number_format = "0.00"
                elif ref_cell.value is None:
                    ref_cell.value = "-"

                # Comparison value (column E)
                comp_cell = ws.cell(row=row_num, column=5)
                if comp_cell.value is not None and isinstance(comp_cell.value, (int, float)):
                    comp_cell.number_format = "0.00"
                elif comp_cell.value is None:
                    comp_cell.value = "-"

                # Relative performance (column F)
                perf_cell = ws.cell(row=row_num, column=6)
                if perf_cell.value is not None and isinstance(perf_cell.value, (int, float)):
                    perf_cell.number_format = "0%"

                    # Color coding for performance
                    if perf_cell.value > 0:  # Better performance
                        perf_cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                    elif perf_cell.value < 0:  # Worse performance
                        perf_cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                elif perf_cell.value is None:
                    perf_cell.value = "-"

            # Auto-adjust column widths - simpler approach
            column_widths = {}
            for row in ws.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        col_letter = cell.column_letter
                        cell_length = len(str(cell.value))
                        if col_letter not in column_widths or cell_length > column_widths[col_letter]:
                            column_widths[col_letter] = cell_length

            # Apply column widths
            for col_letter, width in column_widths.items():
                adjusted_width = min(width + 2, 50)
                ws.column_dimensions[col_letter].width = adjusted_width

            # Set minimum widths for common columns
            min_widths = {"A": 15, "B": 12, "C": 12, "D": 15, "E": 15, "F": 18}
            for col_letter, min_width in min_widths.items():
                if col_letter in ws.column_dimensions:
                    current_width = ws.column_dimensions[col_letter].width
                    ws.column_dimensions[col_letter].width = max(current_width, min_width)

        # Save workbook
        wb.save(output_file)
        print(f"Excel file saved to {output_file}")

    except Exception as e:
        print(f"Error creating Excel file: {e}")


def format_comparison_results(ref_df, comparison_files, metric_name):
    """Format comparison results for a specific metric."""

    if ref_df.empty:
        print("Error: Reference file is empty!")
        return []

    # Create lookup for reference times
    ref_lookup = {}

    for _, row in ref_df.iterrows():
        key = (row["tensor_type"], row["input_size"])
        ref_lookup[key] = row[metric_name]

    lines = []
    lines.append("name tensor_type input_size ref_value comparison_value rel_performance")

    # Add reference rows first
    for _, row in ref_df.iterrows():
        ref_val = f"{row[metric_name]:.2f}"
        lines.append(f"{row['name']} {row['tensor_type']} {row['input_size']} {ref_val} - -")

    # Process comparison files
    for comp_file in comparison_files:
        comp_df = load_file_data(comp_file)

        if comp_df.empty:
            continue

        comp_df = comp_df.sort_values(["tensor_type", "input_size"])

        for _, row in comp_df.iterrows():
            key = (row["tensor_type"], row["input_size"])

            if key in ref_lookup:
                ref_time = ref_lookup[key]
                comp_time = row[metric_name]
                rel_perf = (ref_time - comp_time) / ref_time

                comp_val = f"{comp_time:.2f}"
                rel_val = f"{rel_perf:.0%}"
                lines.append(
                    f"{row['name']} {row['tensor_type']} {row['input_size']} {ref_time:.2f} {comp_val} {rel_val}"
                )

    return lines


def main():
    parser = argparse.ArgumentParser(description="Flexible benchmark analysis")
    parser.add_argument("--ref", required=True, help="Reference file to compare against")
    parser.add_argument("--pattern", required=True, help="Regex pattern to match comparison files")
    parser.add_argument(
        "--metric",
        default="all",
        choices=["forward_avg", "backward_avg", "forward_mb_avg", "backward_mb_avg", "all"],
        help="Which metric to analyze (default: all)",
    )
    parser.add_argument("--output", help="Output file prefix (optional)")
    parser.add_argument("--excel", action="store_true", help="Save results to Excel file (requires openpyxl)")
    parser.add_argument(
        "--plot", action="store_true", help="Create performance visualization plots (requires matplotlib/seaborn)"
    )

    args = parser.parse_args()

    # Check if reference file exists
    if not Path(args.ref).exists():
        print(f"Error: Reference file '{args.ref}' not found!")
        return

    print(f"Loading reference file: {args.ref}")
    ref_df = load_file_data(args.ref)

    if ref_df.empty:
        print("Error: Could not load reference data!")
        return

    print(f"Finding files matching pattern: {args.pattern}")
    comparison_files = find_matching_files(args.pattern)

    if not comparison_files:
        print(f"No files found matching pattern: {args.pattern}")
        return

    print(f"Found {len(comparison_files)} comparison files:")
    for file in comparison_files:
        print(f"  - {file}")

    # Define metrics to analyze
    if args.metric == "all":
        metrics = [
            ("forward_avg", "Forward Average Time"),
            ("backward_avg", "Backward Average Time"),
            ("forward_mb_avg", "Forward Memory Bandwidth Average"),
            ("backward_mb_avg", "Backward Memory Bandwidth Average"),
        ]
    else:
        metric_titles = {
            "forward_avg": "Forward Average Time",
            "backward_avg": "Backward Average Time",
            "forward_mb_avg": "Forward Memory Bandwidth Average",
            "backward_mb_avg": "Backward Memory Bandwidth Average",
        }
        metrics = [(args.metric, metric_titles[args.metric])]

    all_results = {}
    all_structured_data = {}

    for metric_key, metric_title in metrics:
        print(f"\n{'=' * 20} {metric_title.upper()} {'=' * 20}")
        lines = format_comparison_results(ref_df, comparison_files, metric_key)
        structured_data = create_structured_data(ref_df, comparison_files, metric_key)

        for line in lines:
            print(line)

        all_results[metric_key] = lines
        all_structured_data[metric_key] = structured_data

        # Save individual metric results if output prefix specified
        if args.output:
            output_file = f"{args.output}_{metric_key}_results.txt"
            with open(output_file, "w") as f:
                f.write("\n".join(lines))
            print(f"Results saved to {output_file}")

    # Save all results to one file if output prefix specified
    if args.output:
        output_file = f"{args.output}_all_metrics_results.txt"
        with open(output_file, "w") as f:
            for metric_key, metric_title in metrics:
                f.write(f"\n{'=' * 20} {metric_title.upper()} {'=' * 20}\n")
                f.write("\n".join(all_results[metric_key]))
                f.write("\n\n")

        print(f"\nAll results saved to {output_file}")

    # Save to Excel if requested
    if args.excel:
        excel_file = f"{args.output if args.output else 'benchmark_analysis'}.xlsx"
        save_to_excel(all_structured_data, excel_file, metrics)

    # Create plots if requested
    if args.plot:
        print("\n" + "=" * 50)
        print("CREATING PERFORMANCE VISUALIZATIONS")
        print("=" * 50)

        # Aggregate performance data
        aggregated_data = aggregate_performance_data(all_structured_data, metrics)

        if not aggregated_data.empty:
            print(f"Aggregated performance data for {len(aggregated_data)} implementations")
            print("\nPerformance Summary:")
            summary = aggregated_data.groupby("name").agg({"mean_perf": "mean", "count": "sum"}).round(3)
            print(summary)

            # Create visualizations
            output_prefix = args.output if args.output else "benchmark"
            baseline_name = extract_short_name(args.ref)
            create_performance_plots(aggregated_data, all_structured_data, metrics, output_prefix, baseline_name)
        else:
            print("No performance data available for visualization.")
            print("Make sure you have comparison files that match the reference file structure.")


if __name__ == "__main__":
    main()
