#!/usr/bin/env python3
"""
Test the working optimized kernel.
"""

import torch

from nncf.torch.quantization.triton.reference import triton_sum_like
from nncf.torch.quantization.triton.reference import triton_sum_like_optimized


def test_optimized_vs_simple():
    """Test optimized vs simple implementation"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test case
    input_tensor = torch.randn(2, 3, 4, 4, device=device)
    ref_tensor = torch.randn(1, 3, 1, 1, device=device)

    print("Input tensor shape:", input_tensor.shape)
    print("Reference tensor shape:", ref_tensor.shape)

    # Expected result using PyTorch
    expected = torch.zeros_like(ref_tensor)
    temp_tensor = input_tensor.clone()
    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            temp_tensor = temp_tensor.sum(dim, keepdim=True)
    expected = temp_tensor

    print("Expected result:", expected.flatten())

    # Test simple kernel
    simple_result = triton_sum_like(input_tensor, ref_tensor)
    print("Simple result:", simple_result.flatten())

    # Test optimized kernel
    optimized_result = triton_sum_like_optimized(input_tensor, ref_tensor)
    print("Optimized result:", optimized_result.flatten())

    # Check matches
    print("Simple matches expected:", torch.allclose(simple_result, expected))
    print("Optimized matches expected:", torch.allclose(optimized_result, expected))
    print("Simple vs Optimized match:", torch.allclose(simple_result, optimized_result))

    if not torch.allclose(optimized_result, expected):
        print("Optimized diff:", (optimized_result - expected).abs().max())
        print("Optimized ratio:", (optimized_result / expected).abs().max())


if __name__ == "__main__":
    test_optimized_vs_simple()
