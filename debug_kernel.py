#!/usr/bin/env python3
"""
Debug script to understand the difference between simple and optimized kernels.
"""

import torch

from nncf.torch.quantization.triton.reference import get_4d_tensor_meta
from nncf.torch.quantization.triton.reference import optimized_sum_reduction_kernel
from nncf.torch.quantization.triton.reference import triton_sum_like


def test_optimized_kernel():
    """Test the optimized kernel directly"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create small test case
    input_tensor = torch.randn(2, 3, 4, 4, device=device)
    ref_tensor = torch.randn(1, 3, 1, 1, device=device)

    print("Original input tensor shape:", input_tensor.shape)
    print("Original input tensor size:", input_tensor.numel())
    print("Reference tensor shape:", ref_tensor.shape)
    print("Reference tensor size:", ref_tensor.numel())

    # Expected result using PyTorch
    expected = torch.zeros_like(ref_tensor)
    temp_tensor = input_tensor.clone()
    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            temp_tensor = temp_tensor.sum(dim, keepdim=True)
    expected = temp_tensor

    print("Expected result:", expected.flatten())

    # Test optimized kernel
    output = torch.zeros_like(ref_tensor)
    input_meta = get_4d_tensor_meta(input_tensor)
    output_meta = get_4d_tensor_meta(output)

    # Launch kernel manually with explicit block size
    import triton

    original_input_size = input_tensor.numel()  # 2*3*4*4 = 96
    block_size = 256  # Fixed block size
    grid_size = triton.cdiv(original_input_size, block_size)  # Should be 1

    print("Input tensor shape:", input_tensor.shape)
    print("Input tensor size:", original_input_size)
    print("Block size:", block_size)
    print("Grid size:", grid_size)
    print("Input meta:", input_meta)
    print("Output meta:", output_meta)

    # Launch with fixed block size
    optimized_sum_reduction_kernel[(grid_size,)](
        input_tensor,
        input_meta,
        output,
        output_meta,
        BLOCK_SIZE=block_size,
    )

    print("Optimized result:", output.flatten())
    print("Simple result:", triton_sum_like(input_tensor, ref_tensor).flatten())

    # Check if they match
    print("Match:", torch.allclose(output, expected))
    print("Diff:", (output - expected).abs().max())
    print("Ratio:", (output / expected).abs().max())

    # Let's also test with different block sizes
    for test_block_size in [256, 512, 1024]:
        output_test = torch.zeros_like(ref_tensor)
        test_grid_size = triton.cdiv(original_input_size, test_block_size)
        print(f"Block size {test_block_size}: Grid size {test_grid_size}")
        optimized_sum_reduction_kernel[(test_grid_size,)](
            input_tensor,
            input_meta,
            output_test,
            output_meta,
            BLOCK_SIZE=test_block_size,
        )
        ratio = (output_test / expected).abs().max()
        print(f"  Result: {output_test.flatten()}, Ratio: {ratio}")


def test_simple_kernel():
    """Test the simple kernel directly"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create small test case
    input_tensor = torch.randn(2, 3, 4, 4, device=device)
    ref_tensor = torch.randn(1, 3, 1, 1, device=device)

    print("\n=== SIMPLE KERNEL TEST ===")
    print("Input tensor shape:", input_tensor.shape)
    print("Input tensor size:", input_tensor.numel())
    print("Reference tensor shape:", ref_tensor.shape)
    print("Reference tensor size:", ref_tensor.numel())

    # Expected result using PyTorch
    expected = torch.zeros_like(ref_tensor)
    temp_tensor = input_tensor.clone()
    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            temp_tensor = temp_tensor.sum(dim, keepdim=True)
    expected = temp_tensor

    print("Expected result:", expected.flatten())

    # Test simple kernel
    result = triton_sum_like(input_tensor, ref_tensor)
    print("Simple result:", result.flatten())

    print("Match:", torch.allclose(result, expected))
    print("Diff:", (result - expected).abs().max())


if __name__ == "__main__":
    test_optimized_kernel()
    test_simple_kernel()
