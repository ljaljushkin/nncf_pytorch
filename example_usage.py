#!/usr/bin/env python3  # noqa: CPY001
"""
Example usage script for flexible_benchmark_analysis.py
"""

import subprocess
import sys


def run_analysis(ref_file, pattern, metric="all", output_prefix=None, excel=False, plot=False):
    """Run the flexible benchmark analysis with given parameters."""

    cmd = [
        sys.executable,
        "flexible_benchmark_analysis.py",
        "--ref",
        ref_file,
        "--pattern",
        pattern,
        "--metric",
        metric,
    ]

    if output_prefix:
        cmd.extend(["--output", output_prefix])

    if excel:
        cmd.append("--excel")

    if plot:
        cmd.append("--plot")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("SUCCESS:")
        print(result.stdout)
    else:
        print("ERROR:")
        print(result.stderr)

    return result.returncode == 0


def main():
    """Run example analyses."""

    # print("=== Flexible Benchmark Analysis Examples ===\n")

    # # Example 1: Compare all triton files against cuda reference
    # print("Example 1: Compare triton implementations against CUDA baseline")
    # run_analysis(
    #     ref_file="benchmark_cuda_507d96.csv", pattern=r"benchmark_triton_.*\.csv", output_prefix="triton_vs_cuda"
    # )

    # print("\n" + "=" * 60 + "\n")

    # # Example 2: Compare only forward_avg metric
    # print("Example 2: Compare only forward average times")
    # run_analysis(
    #     ref_file="benchmark_cuda_507d96.csv",
    #     pattern=r"benchmark_triton_1[0-9]_.*\.csv",
    #     metric="forward_avg",
    #     output_prefix="triton_10s_forward",
    # )

    # print("\n" + "=" * 60 + "\n")

    # # Example 3: Compare compile results against triton
    # print("Example 3: Compare compile results against triton baseline")
    # if Path("benchmark_triton_13_02b9d9.csv").exists():
    #     run_analysis(
    #         ref_file="benchmark_triton_13_02b9d9.csv",
    #         pattern=r"benchmark_compile_.*\.csv",
    #         output_prefix="compile_vs_triton",
    #     )
    # else:
    #     print("Triton reference file not found, skipping this example")

    # print("\n" + "=" * 60 + "\n")

    # # Example 4: Export to Excel with formatting
    # print("Example 4: Export results to Excel with formatting")
    # run_analysis(
    #     ref_file="benchmark_cuda_507d96.csv",
    #     pattern=r"benchmark_triton_1[0-3]_.*\.csv",
    #     output_prefix="excel_export_example",
    #     excel=True,
    # )

    print("\n" + "=" * 60 + "\n")

    # Example 5: Export to Excel with formatting and create plots
    print("Example 5: Export results to Excel with formatting and create performance plots")
    run_analysis(
        # ref_file="benchmark_reference_22.csv",
        # pattern=r"benchmark_cuda_24_all_optimized.csv|.*triton_22.*|.*compile_22.*",
        # ref_file="benchmark_reference_24_gs32.csv",
        # pattern=r"benchmark_cuda_24_all_optimized_gs32.csv|.*triton_24_gs32.*|.*compile_24_gs32.*",
        ref_file="benchmark_reference_24_gs64.csv",
        pattern=r"benchmark_cuda_24_all_optimized_gs64.csv|.*triton_24_gs64.*|.*compile_24_gs64.*",
        # ref_file="benchmark_cuda_22_all_not_optimized.csv",
        # pattern=r"benchmark_cuda_22_.*.csv",
        # ref_file="benchmark_cuda_22_100k_optimized.csv",
        # pattern=r"benchmark_cuda_24_all_optimized.csv",
        excel=True,
        plot=True,
        output_prefix="comprehensive_analysis_24_gs64",
    )


if __name__ == "__main__":
    main()
