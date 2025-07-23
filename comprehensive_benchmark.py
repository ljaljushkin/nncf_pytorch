#!/usr/bin/env python3  # noqa: CPY001

import os
import sys
import time

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def benchmark_different_shapes():
    """Benchmark different tensor shapes and quantization patterns"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print("=== Comprehensive Triton Kernel Benchmark ===\n")

    device = torch.device("cuda")
    torch.manual_seed(42)

    test_cases = [
        # Small tensors
        {
            "name": "Small weights per-channel [128, 2048] with params [128, 1]",
            "input_shape": [128, 2048],
            "input_low_shape": [128, 1],
            "input_range_shape": [128, 1],
            "dtype": torch.bfloat16,
        },
        {
            "name": "Small activations per-channel [128, 2048] with params [1, 2048]",
            "input_shape": [128, 2048],
            "input_low_shape": [1, 2048],
            "input_range_shape": [1, 2048],
            "dtype": torch.bfloat16,
        },
        # Large tensors - weights
        {
            "name": "Large weights per-channel [2048, 128256] with params [2048, 1]",
            "input_shape": [2048, 128256],
            "input_low_shape": [2048, 1],
            "input_range_shape": [2048, 1],
            "dtype": torch.bfloat16,
        },
        # Large tensors - activations
        {
            "name": "Large activations per-channel [2048, 128256] with params [1, 128256]",
            "input_shape": [2048, 128256],
            "input_low_shape": [1, 128256],
            "input_range_shape": [1, 128256],
            "dtype": torch.bfloat16,
        },
        # Single scale
        {
            "name": "Single scale [2048, 128256] with params [1]",
            "input_shape": [2048, 128256],
            "input_low_shape": [1],
            "input_range_shape": [1],
            "dtype": torch.bfloat16,
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i + 1}: {case['name']}")
        print(f"{'=' * 60}")

        # Create tensors
        input_ = torch.randn(case["input_shape"], device=device, dtype=case["dtype"])
        input_low = torch.randn(case["input_low_shape"], device=device, dtype=case["dtype"])
        input_range = torch.abs(torch.randn(case["input_range_shape"], device=device, dtype=case["dtype"])) + 0.1
        grad_output = torch.ones_like(input_)

        print(f"Input shape: {list(input_.shape)}")
        print(f"input_low shape: {list(input_low.shape)}")
        print(f"input_range shape: {list(input_range.shape)}")
        print(f"Total elements: {input_.numel():,}")

        if len(input_.shape) >= 2 and len(input_low.shape) >= 2:
            if input_low.shape[0] > 1:
                elements_per_scale = input_.numel() // input_.shape[0]
                print(f"Per-weight-channel: {input_.shape[0]} channels, {elements_per_scale:,} elements/channel")
            elif input_low.shape[1] > 1:
                elements_per_scale = input_.numel() // input_.shape[1]
                print(f"Per-activation-channel: {input_.shape[1]} channels, {elements_per_scale:,} elements/channel")

        # Warmup
        try:
            from nncf.torch.quantization.triton.reference import backward

            for _ in range(3):
                _ = backward(grad_output, input_, input_low, input_range, levels=16, level_low=0, level_high=15)
        except Exception as e:
            print(f"❌ FAILED during warmup: {e}")
            continue

        # Benchmark timing
        torch.cuda.synchronize()
        num_runs = 50 if input_.numel() < 1000000 else 10

        times = []
        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()

            grad_input, _, _ = backward(
                grad_output, input_, input_low, input_range, levels=16, level_low=0, level_high=15
            )

            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"\nTiming Results ({num_runs} runs):")
        print(f"  Average: {avg_time:.3f} ms")
        print(f"  Min:     {min_time:.3f} ms")
        print(f"  Max:     {max_time:.3f} ms")

        # Calculate throughput
        throughput_gb_s = (input_.numel() * 2) / (avg_time / 1000) / 1e9  # Assuming bfloat16 = 2 bytes
        print(f"  Throughput: {throughput_gb_s:.2f} GB/s")

        # Verify correctness
        if torch.isnan(grad_input).any() or torch.isinf(grad_input).any():
            print("❌ WARNING: Invalid values in grad_input")
        else:
            print("✅ Results are valid")


if __name__ == "__main__":
    benchmark_different_shapes()
