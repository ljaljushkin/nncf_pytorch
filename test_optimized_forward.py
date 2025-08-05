#!/usr/bin/env python3

import sys
import time

import torch

sys.path.insert(0, "/home/nlyaly/projects/nncf2/src")


def test_optimized_forward_kernel():
    """Test the optimized forward kernel performance"""

    print("=== Testing Optimized Forward Kernel ===")
    print()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()

    # Test cases: small and large tensors
    test_cases = [([128, 2048], "Small tensor"), ([2048, 128256], "Large tensor (original bottleneck)")]

    for shape, description in test_cases:
        print(f"=== {description}: {shape} ===")

        batch_size, seq_len = shape
        group_size = 128
        num_groups = seq_len // group_size
        total_elements = batch_size * seq_len

        print(f"Total elements: {total_elements:,}")
        print("Expected threshold: 100K elements (very aggressive)")
        print(f"Will use optimized kernel: {'Yes' if total_elements > 100000 else 'No'}")
        print()

        # Create test data
        input_tensor = torch.randn(batch_size, seq_len, device="cuda", requires_grad=True, dtype=torch.float16)

        # Per-group parameters
        input_low = torch.randn(batch_size, num_groups, device="cuda", dtype=torch.float16) * 0.1
        input_range = torch.rand(batch_size, num_groups, device="cuda", dtype=torch.float16) * 2.0 + 0.1

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                from nncf.torch.quantization.quantize_functions import asymmetric_quantize

                _ = asymmetric_quantize(input_tensor, 256, 0.0, 255.0, input_low, input_range, 1e-7)

        torch.cuda.synchronize()

        # Benchmark forward pass
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        output = asymmetric_quantize(input_tensor, 256, 0.0, 255.0, input_low, input_range, 1e-7)

        torch.cuda.synchronize()
        forward_time = time.perf_counter() - start_time

        print(f"Forward pass time: {forward_time * 1000:.2f}ms")
        print()

        # Compare with expected results
        if total_elements <= 300000:  # Small tensors
            expected_performance = "Should be similar to before (already good)"
        else:  # Large tensors
            original_time = 5.9  # ms from benchmark
            expected_improvement = (original_time - forward_time * 1000) / original_time * 100
            print(f"Original time: {original_time}ms")
            print(f"Expected improvement: {expected_improvement:.1f}%")
            if expected_improvement > 30:
                print("🎉 Significant improvement achieved!")
            elif expected_improvement > 10:
                print("✅ Good improvement achieved")
            else:
                print("⚠️  Improvement less than expected")

        print("-" * 50)
        print()


if __name__ == "__main__":
    test_optimized_forward_kernel()
