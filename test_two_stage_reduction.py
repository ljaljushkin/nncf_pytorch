"""Test two-stage reduction approach to fix atomic contention."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nncf.torch.quantization.triton.reference import two_stage_sum_reduction


def test_two_stage_reduction():
    """Test that two-stage reduction produces consistent results."""
    print("Testing two-stage reduction for atomic contention fix...")

    # Set up test data
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test case: single-scale reduction (all elements sum to single output)
    input_tensor = torch.randn(4, 16, 16, 16, device=device, dtype=torch.float32)
    ref_tensor = torch.ones(1, device=device, dtype=torch.float32)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Input elements: {input_tensor.numel()}")
    print(f"Reference shape: {ref_tensor.shape}")

    # Compute expected result using PyTorch
    expected = input_tensor.sum()
    print(f"Expected result: {expected:.6f}")

    # Test different block sizes
    block_sizes = [64, 128, 256, 512, 1024]

    for block_size in block_sizes:
        print(f"\nBlock size: {block_size}")

        # Run multiple times to check consistency
        results = []
        for run in range(5):
            result = two_stage_sum_reduction(input_tensor, ref_tensor, block_size)
            results.append(result.item())

        # Analyze results
        min_val = min(results)
        max_val = max(results)
        avg_val = sum(results) / len(results)

        print(f"  Results: min={min_val:.6f}, max={max_val:.6f}, avg={avg_val:.6f}")
        print(f"  Expected: {expected:.6f}")
        print(f"  Difference from expected: {abs(avg_val - expected):.6f}")
        print(f"  Ratio: {avg_val / expected:.6f}")
        print(f"  Consistency (max-min): {max_val - min_val:.6f}")

        # Check if results are consistent
        if max_val - min_val < 1e-6:
            print("  ✅ Results are consistent!")
        else:
            print("  ⚠️  WARNING: Inconsistent results across runs")

        # Check if result is close to expected
        if abs(avg_val - expected) < 1e-4:
            print("  ✅ Results are accurate!")
        else:
            print("  ❌ Results are not accurate")


if __name__ == "__main__":
    test_two_stage_reduction()
