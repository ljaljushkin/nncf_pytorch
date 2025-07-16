#!/usr/bin/env python3
"""
Simple test to demonstrate the atomic contention issue with different block sizes
"""

import torch
import triton

from nncf.torch.quantization.triton.reference import get_4d_tensor_meta
from nncf.torch.quantization.triton.reference import optimized_sum_reduction_kernel


def test_atomic_contention():
    """Test atomic contention with different block sizes"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Create test data - single scale case (worst case for atomic contention)
    input_tensor = torch.randn(4, 16, 16, 16, device=device)  # 16384 elements
    ref_tensor = torch.ones([1], device=device)  # Single element - all atomic ops target this

    # Expected result using PyTorch
    expected = input_tensor.sum()

    print(f"Input shape: {input_tensor.shape}")
    print(f"Input elements: {input_tensor.numel()}")
    print(f"Reference shape: {ref_tensor.shape}")
    print(f"Expected result: {expected.item():.6f}")
    print()

    # Test different block sizes
    block_sizes = [64, 128, 256, 512, 1024]

    for block_size in block_sizes:
        # Calculate grid size
        grid_size = triton.cdiv(input_tensor.numel(), block_size)

        print(f"Block size: {block_size}, Grid size: {grid_size}")
        print(f"  Number of programs: {grid_size}")
        print(f"  Elements per program: {block_size}")
        print(f"  Atomic operations per program: {min(block_size, input_tensor.numel())}")
        print(f"  Total atomic operations: {input_tensor.numel()}")

        # Run the kernel multiple times to check consistency
        results = []
        for run in range(5):
            output = torch.zeros_like(ref_tensor)
            input_meta = get_4d_tensor_meta(input_tensor)
            output_meta = get_4d_tensor_meta(output)

            optimized_sum_reduction_kernel[(grid_size,)](
                input_tensor,
                input_meta,
                output,
                output_meta,
                BLOCK_SIZE=block_size,
            )

            results.append(output.item())

        # Check consistency
        min_result = min(results)
        max_result = max(results)
        avg_result = sum(results) / len(results)

        print(f"  Results: min={min_result:.6f}, max={max_result:.6f}, avg={avg_result:.6f}")
        print(f"  Expected: {expected.item():.6f}")
        print(f"  Difference from expected: {abs(avg_result - expected.item()):.6f}")
        print(f"  Ratio: {avg_result / expected.item():.6f}")
        print(f"  Consistency (max-min): {max_result - min_result:.6f}")

        if abs(avg_result - expected.item()) > 1e-3:
            print(f"  ❌ FAILED: Large difference from expected")
        elif max_result - min_result > 1e-6:
            print(f"  ⚠️  WARNING: Inconsistent results across runs")
        else:
            print(f"  ✅ PASSED")
        print()


if __name__ == "__main__":
    test_atomic_contention()
