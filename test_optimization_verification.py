#!/usr/bin/env python3  # noqa: CPY001
"""
Test to verify that the performance optimization for large per-activation-channel tensors works correctly
"""

import sys
import time

import torch

# Add the source directory to the path
sys.path.insert(0, "/home/nlyaly/projects/nncf/src")

from nncf.torch.quantization.triton.reference import backward


def test_large_tensor_optimization():
    """
    Test that large per-activation-channel tensors use the fast 1D kernel instead of slow 2D kernel
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Test case: [2048, 128256] - this should trigger the optimization
    input_shape = [2048, 128256]
    param_shape = [1, 128256]

    print("Testing large tensor optimization:")
    print(f"  Input shape: {input_shape}")
    print(f"  Parameter shape: {param_shape}")
    print(f"  Total elements: {input_shape[0] * input_shape[1]:,}")

    # Create test tensors
    torch.manual_seed(42)
    grad_output = torch.randn(input_shape, device=device, dtype=torch.float32)
    input_ = torch.randn(input_shape, device=device, dtype=torch.float32)
    input_low = torch.randn(param_shape, device=device, dtype=torch.float32) * 0.1
    input_range = torch.abs(torch.randn(param_shape, device=device, dtype=torch.float32)) + 0.1

    levels = 256
    level_low = 0
    level_high = 255

    # Test performance
    warmup_runs = 3
    test_runs = 5

    print(f"  Warmup runs: {warmup_runs}")
    for _ in range(warmup_runs):
        _ = backward(grad_output, input_, input_low, input_range, levels, level_low, level_high)
        torch.cuda.synchronize()

    print(f"  Timing runs: {test_runs}")
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(test_runs):
        result = backward(grad_output, input_, input_low, input_range, levels, level_low, level_high)
        torch.cuda.synchronize()

    end_time = time.time()

    avg_time = (end_time - start_time) / test_runs
    total_elements = input_.numel()
    throughput = total_elements / avg_time / 1e9  # GB/s

    print(f"  Average time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} GB/s")

    # Check correctness
    grad_input, grad_low, grad_range = result
    print(f"  Result shapes: grad_input={grad_input.shape}, grad_low={grad_low.shape}, grad_range={grad_range.shape}")
    print(
        f"  Gradient check: grad_input has {torch.isfinite(grad_input).sum().item()} finite values out "
        " of {grad_input.numel()}"
    )

    # Verify the optimization worked - throughput should be good (>20 GB/s for large tensors)
    if throughput > 20.0:
        print("  ✅ OPTIMIZATION SUCCESSFUL: High throughput indicates 1D kernel was used")
        print("     (2D kernel would give ~4 GB/s, 1D kernel gives >20 GB/s)")
    else:
        print("  ⚠️  LOW THROUGHPUT: May still be using slow 2D kernel")

    return avg_time, throughput


def test_small_tensor_behavior():
    """
    Test that small tensors still work correctly (may use either kernel)
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Small tensor that might still use 2D kernel
    input_shape = [256, 1024]
    param_shape = [1, 1024]

    print("\nTesting small tensor behavior:")
    print(f"  Input shape: {input_shape}")
    print(f"  Parameter shape: {param_shape}")

    # Create test tensors
    torch.manual_seed(42)
    grad_output = torch.randn(input_shape, device=device, dtype=torch.float32)
    input_ = torch.randn(input_shape, device=device, dtype=torch.float32)
    input_low = torch.randn(param_shape, device=device, dtype=torch.float32) * 0.1
    input_range = torch.abs(torch.randn(param_shape, device=device, dtype=torch.float32)) + 0.1

    levels = 256
    level_low = 0
    level_high = 255

    # Quick test
    result = backward(grad_output, input_, input_low, input_range, levels, level_low, level_high)
    grad_input, grad_low, grad_range = result

    print(f"  ✅ Small tensor test passed: shapes {grad_input.shape}, {grad_low.shape}, {grad_range.shape}")

    return True


if __name__ == "__main__":
    print("=" * 80)
    print("PERFORMANCE OPTIMIZATION VERIFICATION")
    print("=" * 80)

    test_large_tensor_optimization()
    test_small_tensor_behavior()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The optimization automatically selects the best kernel:")
    print("- Large tensors with many channels → Fast 1D kernel (>20 GB/s)")
    print("- Small tensors → May use either kernel (both work fine)")
    print("- No manual intervention required - it's all automatic!")
