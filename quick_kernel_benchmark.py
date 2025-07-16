#!/usr/bin/env python3
"""
Quick benchmark script to test different Triton sum reduction kernels.
"""

import time

import torch
import triton

from nncf.torch.quantization.triton.reference import calculate_contiguous_elements_per_scale
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta
from nncf.torch.quantization.triton.reference import optimize_block_size_for_contiguous_elements
from nncf.torch.quantization.triton.reference import optimized_sum_reduction_kernel
from nncf.torch.quantization.triton.reference import warp_level_sum_reduction_kernel


def generate_reference_tensor(input_size, scale_mode, is_weights, dtype):
    """Generate reference tensor shape based on scale mode and tensor type"""
    if scale_mode == "single_scale":
        return torch.ones([1], dtype=dtype, device="cuda")
    elif scale_mode == "per_channel":
        if is_weights:
            # For weights, channel is dim 0
            channel_count = input_size[0]
            ref_shape = [1 for _ in input_size]
            ref_shape[0] = channel_count
        else:
            # For activations, channel is dim 1
            channel_count = input_size[1]
            ref_shape = [1 for _ in input_size]
            ref_shape[1] = channel_count
        return torch.ones(ref_shape, dtype=dtype, device="cuda")
    else:
        msg = f"Unknown scale_mode: {scale_mode}"
        raise ValueError(msg)


def benchmark_kernel(kernel_func, input_tensor, ref_tensor, block_size, runs=50):
    """Benchmark a kernel function"""
    # Prepare kernel inputs
    output = torch.zeros_like(ref_tensor)
    input_meta = get_4d_tensor_meta(input_tensor)
    output_meta = get_4d_tensor_meta(output)
    grid_size = triton.cdiv(input_tensor.numel(), block_size)

    # Warmup
    for _ in range(10):
        try:
            kernel_func[(grid_size,)](input_tensor, input_meta, output, output_meta, BLOCK_SIZE=block_size)
        except Exception as e:
            return None, f"Error: {e}"

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(runs):
        kernel_func[(grid_size,)](input_tensor, input_meta, output, output_meta, BLOCK_SIZE=block_size)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / runs * 1000  # Convert to ms
    return avg_time, output


def run_quick_benchmark():
    """Run a quick benchmark comparison"""

    print("Quick Triton Kernel Benchmark")
    print("=" * 80)

    # Test configurations
    test_configs = [
        # (input_size, scale_mode, is_weights, description)
        ([1, 16, 64, 64], "single_scale", False, "Small 4D per-tensor"),
        ([4, 16, 16, 16], "per_channel", False, "Medium 4D per-channel activations"),
        # ([8, 256, 32, 32], "per_channel", True, "Large 4D per-channel weights"),
        # ([1024, 256], "per_channel", False, "2D per-channel activations"),
        # ([4096, 4096], "single_scale", False, "Large 2D per-tensor"),
    ]

    # Kernels to test
    kernels = [
        (optimized_sum_reduction_kernel, "optimized_sum_reduction"),
        # (hierarchical_sum_like_kernel, "hierarchical_sum_like"),
        (warp_level_sum_reduction_kernel, "warp_level_sum_reduction"),
        # (warp_efficient_sum_like_kernel, "warp_efficient_sum_like"),
    ]

    block_sizes = [64, 128, 256, 512, 1024]

    for input_size, scale_mode, is_weights, description in test_configs:
        print(f"\n{description}")
        print(f"Input size: {input_size}, Scale mode: {scale_mode}")

        # Generate test data
        input_tensor = torch.randn(input_size, device="cuda", dtype=torch.float16)
        ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, torch.float16)

        # Calculate optimal block size
        contiguous_elements = calculate_contiguous_elements_per_scale(input_tensor, ref_tensor)
        optimal_block_size = optimize_block_size_for_contiguous_elements(contiguous_elements, input_tensor.numel())

        print(f"Contiguous elements per scale: {contiguous_elements}")
        print(f"Optimal block size: {optimal_block_size}")
        print()

        # Test each kernel with different block sizes
        for kernel_func, kernel_name in kernels:
            print(f"{kernel_name:25s}:", end="")

            best_time = float("inf")
            best_block_size = None

            for block_size in block_sizes:
                avg_time, result = benchmark_kernel(kernel_func, input_tensor, ref_tensor, block_size)

                if avg_time is not None:
                    if avg_time < best_time:
                        best_time = avg_time
                        best_block_size = block_size

                    is_optimal = block_size == optimal_block_size
                    marker = " *" if is_optimal else "  "
                    print(f" {block_size}:{avg_time:6.3f}ms{marker}", end="")
                else:
                    print(f" {block_size}:ERROR", end="")

            print(f" | Best: {best_block_size} ({best_time:.3f}ms)")

        print("-" * 80)


if __name__ == "__main__":
    if torch.cuda.is_available():
        run_quick_benchmark()
    else:
        print("CUDA not available. Skipping benchmark.")
