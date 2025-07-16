#!/usr/bin/env python3
"""
Comprehensive benchmark for different Triton sum reduction kernels.

This benchmark compares:
- optimized_sum_reduction_kernel
- hierarchical_sum_like_kernel
- warp_level_sum_reduction_kernel
- warp_efficient_sum_like_kernel

With different configurations:
- Block sizes: 64, 128, 256, 512, 1024
- Input sizes: Various realistic quantization scenarios
- Weight vs activation modes
- Single-scale vs per-channel quantization
"""

import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton

from nncf.torch.quantization.triton.reference import calculate_contiguous_elements_per_scale
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta
from nncf.torch.quantization.triton.reference import optimize_block_size_for_contiguous_elements
from nncf.torch.quantization.triton.reference import optimized_sum_reduction_kernel
from nncf.torch.quantization.triton.reference import warp_efficient_sum_like_kernel
from nncf.torch.quantization.triton.reference import warp_level_sum_reduction_kernel


def generate_reference_tensor(
    input_size: List[int], scale_mode: str, is_weights: bool, dtype: torch.dtype
) -> torch.Tensor:
    """Generate reference tensor shape based on scale mode and tensor type"""
    if scale_mode == "single_scale":
        return torch.ones([1], dtype=dtype, device="cuda")
    elif scale_mode == "per_channel":
        if is_weights:
            # For weights, channel is dim 0: [C, 1, 1, 1]
            channel_count = input_size[0]
            ref_shape = [1 for _ in input_size]
            ref_shape[0] = channel_count
        else:
            # For activations, channel is dim 1: [1, C, 1, 1]
            channel_count = input_size[1]
            ref_shape = [1 for _ in input_size]
            ref_shape[1] = channel_count
        return torch.ones(ref_shape, dtype=dtype, device="cuda")
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")


def benchmark_kernel(
    kernel_func,
    input_tensor: torch.Tensor,
    ref_tensor: torch.Tensor,
    block_size: int,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    kernel_name: str = "kernel",
) -> Dict[str, Any]:
    """Benchmark a single kernel configuration"""

    # Prepare kernel inputs
    output = torch.zeros_like(ref_tensor)
    input_meta = get_4d_tensor_meta(input_tensor)
    output_meta = get_4d_tensor_meta(output)
    grid_size = triton.cdiv(input_tensor.numel(), block_size)

    # Warmup
    for _ in range(warmup_runs):
        try:
            kernel_func[(grid_size,)](
                input_tensor,
                input_meta,
                output,
                output_meta,
                BLOCK_SIZE=block_size,
            )
        except Exception as e:
            return {
                "kernel": kernel_name,
                "error": str(e),
                "time_ms": float("inf"),
                "throughput_gb_s": 0.0,
                "grid_size": grid_size,
                "block_size": block_size,
            }

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(benchmark_runs):
        kernel_func[(grid_size,)](
            input_tensor,
            input_meta,
            output,
            output_meta,
            BLOCK_SIZE=block_size,
        )

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    avg_time_s = (end_time - start_time) / benchmark_runs
    avg_time_ms = avg_time_s * 1000

    # Calculate throughput (GB/s)
    input_bytes = input_tensor.numel() * input_tensor.element_size()
    output_bytes = output.numel() * output.element_size()
    total_bytes = input_bytes + output_bytes
    throughput_gb_s = (total_bytes / 1e9) / avg_time_s

    return {
        "kernel": kernel_name,
        "time_ms": avg_time_ms,
        "throughput_gb_s": throughput_gb_s,
        "grid_size": grid_size,
        "block_size": block_size,
        "error": None,
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all configurations"""

    # Test configurations
    input_sizes = [
        [1, 16, 64, 64],  # Small 4D
        [4, 16, 16, 16],  # Medium 4D
        [8, 256, 32, 32],  # Large 4D
        [4, 64, 128, 128],  # Very large 4D
        [1024, 256],  # 2D medium
        [4096, 4096],  # 2D large
        [256],  # 1D small
        [1048576],  # 1D large
    ]

    scale_modes = ["single_scale", "per_channel"]
    is_weights_modes = [True, False]
    block_sizes = [64, 128, 256, 512, 1024]

    # Kernels to benchmark
    kernels = [
        (optimized_sum_reduction_kernel, "optimized_sum_reduction"),
        # (hierarchical_sum_like_kernel, "hierarchical_sum_like"),
        (warp_level_sum_reduction_kernel, "warp_level_sum_reduction"),
        (warp_efficient_sum_like_kernel, "warp_efficient_sum_like"),
    ]

    results = []
    total_configs = len(input_sizes) * len(scale_modes) * len(is_weights_modes) * len(block_sizes) * len(kernels)
    current_config = 0

    print(f"Running comprehensive benchmark with {total_configs} configurations...")
    print("=" * 80)

    for input_size in input_sizes:
        for scale_mode in scale_modes:
            for is_weights in is_weights_modes:
                # Skip invalid configurations
                if scale_mode == "per_channel":
                    if is_weights and len(input_size) > 0 and input_size[0] == 1:
                        continue  # Skip per-channel weights with single channel
                    if not is_weights and len(input_size) > 1 and input_size[1] == 1:
                        continue  # Skip per-channel activations with single channel

                # Generate test tensors
                input_tensor = torch.randn(input_size, device="cuda", dtype=torch.float16)
                ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, torch.float16)

                # Calculate contiguous elements per scale
                contiguous_elements = calculate_contiguous_elements_per_scale(input_tensor, ref_tensor)
                optimal_block_size = optimize_block_size_for_contiguous_elements(
                    contiguous_elements, input_tensor.numel()
                )

                config_info = {
                    "input_size": input_size,
                    "input_numel": input_tensor.numel(),
                    "ref_size": list(ref_tensor.shape),
                    "ref_numel": ref_tensor.numel(),
                    "scale_mode": scale_mode,
                    "is_weights": is_weights,
                    "contiguous_elements_per_scale": contiguous_elements,
                    "optimal_block_size": optimal_block_size,
                }

                print(
                    f"\nConfig: {input_size} -> {list(ref_tensor.shape)} ({scale_mode}, {'weights' if is_weights else 'activations'})"
                )
                print(f"  Contiguous elements per scale: {contiguous_elements}")
                print(f"  Optimal block size: {optimal_block_size}")

                for block_size in block_sizes:
                    for kernel_func, kernel_name in kernels:
                        current_config += 1

                        # Benchmark this configuration
                        result = benchmark_kernel(
                            kernel_func, input_tensor, ref_tensor, block_size, kernel_name=kernel_name
                        )

                        # Add configuration info
                        result.update(config_info)
                        result["is_optimal_block_size"] = block_size == optimal_block_size

                        results.append(result)

                        # Print progress
                        if result["error"] is None:
                            print(
                                f"  [{current_config:4d}/{total_configs}] {kernel_name:25s} bs={block_size:4d}: "
                                f"{result['time_ms']:7.3f}ms, {result['throughput_gb_s']:6.1f}GB/s"
                            )
                        else:
                            print(
                                f"  [{current_config:4d}/{total_configs}] {kernel_name:25s} bs={block_size:4d}: ERROR - {result['error']}"
                            )

    return results


def analyze_results(results: List[Dict[str, Any]]) -> None:
    """Analyze and visualize benchmark results"""

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Filter out error results
    df_success = df[df["error"].isna()].copy()

    print("\n" + "=" * 80)
    print("BENCHMARK ANALYSIS")
    print("=" * 80)

    # Overall statistics
    print(f"Total configurations tested: {len(df)}")
    print(f"Successful runs: {len(df_success)}")
    print(f"Failed runs: {len(df) - len(df_success)}")

    if len(df_success) == 0:
        print("No successful runs to analyze!")
        return

    # Best performers by kernel
    print("\nBest performance by kernel:")
    for kernel in df_success["kernel"].unique():
        kernel_df = df_success[df_success["kernel"] == kernel]
        best_row = kernel_df.loc[kernel_df["time_ms"].idxmin()]
        print(
            f"  {kernel:25s}: {best_row['time_ms']:7.3f}ms, {best_row['throughput_gb_s']:6.1f}GB/s "
            f"(bs={best_row['block_size']}, {best_row['input_size']} -> {best_row['ref_size']})"
        )

    # Analyze block size effectiveness
    print("\nBlock size analysis:")
    block_size_stats = (
        df_success.groupby("block_size")
        .agg(
            {
                "time_ms": ["mean", "median", "std"],
                "throughput_gb_s": ["mean", "median", "std"],
                "is_optimal_block_size": "sum",
            }
        )
        .round(3)
    )
    print(block_size_stats)

    # Analyze optimal block size effectiveness
    print("\nOptimal block size effectiveness:")
    optimal_stats = (
        df_success.groupby("is_optimal_block_size")
        .agg(
            {
                "time_ms": ["mean", "median"],
                "throughput_gb_s": ["mean", "median"],
            }
        )
        .round(3)
    )
    print(optimal_stats)

    # Analyze by quantization mode
    print("\nPerformance by quantization mode:")
    mode_stats = (
        df_success.groupby(["scale_mode", "is_weights"])
        .agg(
            {
                "time_ms": ["mean", "median"],
                "throughput_gb_s": ["mean", "median"],
            }
        )
        .round(3)
    )
    print(mode_stats)

    # Find best kernel for each configuration
    print("\nBest kernel by configuration:")
    config_cols = ["input_size", "scale_mode", "is_weights"]

    for config, group in df_success.groupby(config_cols):
        best_overall = group.loc[group["time_ms"].idxmin()]
        print(
            f"  {str(config):60s}: {best_overall['kernel']:25s} "
            f"(bs={best_overall['block_size']}, {best_overall['time_ms']:6.3f}ms)"
        )


def create_visualizations(results: List[Dict[str, Any]]) -> None:
    """Create visualizations for benchmark results"""

    df = pd.DataFrame(results)
    df_success = df[df["error"].isna()].copy()

    if len(df_success) == 0:
        print("No successful runs to visualize!")
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Performance by kernel and block size
    ax1 = axes[0, 0]
    for kernel in df_success["kernel"].unique():
        kernel_data = df_success[df_success["kernel"] == kernel]
        block_size_perf = kernel_data.groupby("block_size")["time_ms"].mean()
        ax1.plot(block_size_perf.index, block_size_perf.values, marker="o", label=kernel)

    ax1.set_xlabel("Block Size")
    ax1.set_ylabel("Average Time (ms)")
    ax1.set_title("Performance by Kernel and Block Size")
    ax1.legend()
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # 2. Throughput by kernel and block size
    ax2 = axes[0, 1]
    for kernel in df_success["kernel"].unique():
        kernel_data = df_success[df_success["kernel"] == kernel]
        block_size_throughput = kernel_data.groupby("block_size")["throughput_gb_s"].mean()
        ax2.plot(block_size_throughput.index, block_size_throughput.values, marker="o", label=kernel)

    ax2.set_xlabel("Block Size")
    ax2.set_ylabel("Average Throughput (GB/s)")
    ax2.set_title("Throughput by Kernel and Block Size")
    ax2.legend()
    ax2.set_xscale("log", base=2)
    ax2.grid(True, alpha=0.3)

    # 3. Performance by tensor size
    ax3 = axes[1, 0]
    for kernel in df_success["kernel"].unique():
        kernel_data = df_success[df_success["kernel"] == kernel]
        size_perf = kernel_data.groupby("input_numel")["time_ms"].mean()
        ax3.plot(size_perf.index, size_perf.values, marker="o", label=kernel)

    ax3.set_xlabel("Input Tensor Size (elements)")
    ax3.set_ylabel("Average Time (ms)")
    ax3.set_title("Performance by Tensor Size")
    ax3.legend()
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    # 4. Optimal block size effectiveness
    ax4 = axes[1, 1]
    optimal_data = df_success.groupby(["kernel", "is_optimal_block_size"])["time_ms"].mean().unstack()
    optimal_data.plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Kernel")
    ax4.set_ylabel("Average Time (ms)")
    ax4.set_title("Optimal vs Non-Optimal Block Size")
    ax4.legend(["Non-Optimal", "Optimal"])
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("triton_kernel_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nVisualization saved as 'triton_kernel_benchmark.png'")


def export_results(results: List[Dict[str, Any]], filename: str = "benchmark_results.csv") -> None:
    """Export results to CSV file"""

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nResults exported to {filename}")


def main():
    """Main benchmark function"""

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("Starting Triton Sum Reduction Kernel Benchmark")
    print("=" * 80)

    # Run benchmark
    results = run_comprehensive_benchmark()

    # Analyze results
    analyze_results(results)

    # Create visualizations
    try:
        create_visualizations(results)
    except Exception as e:
        print(f"Visualization failed: {e}")

    # Export results
    export_results(results)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
