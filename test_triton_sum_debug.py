#!/usr/bin/env python3
"""
Debug test for Triton sum reduction to check numerical correctness.
"""

import torch

from nncf.torch.quantization.triton.reference import triton_sum_like


def test_triton_sum_simple():
    """Test with a simple case to debug the issue."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple test case
    input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device=device)
    ref_tensor = torch.tensor([[[[0.0]]]], device=device)

    print(f"Input tensor: {input_tensor}")
    print(f"Reference tensor shape: {ref_tensor.shape}")
    print(f"Input tensor shape: {input_tensor.shape}")

    # PyTorch reference
    pytorch_result = input_tensor.sum()
    print(f"PyTorch sum(): {pytorch_result}")

    # Triton result
    triton_result = triton_sum_like(input_tensor, ref_tensor)
    print(f"Triton result: {triton_result}")

    print(f"Are close: {torch.allclose(pytorch_result, triton_result)}")

    # Per-channel reduction test
    input_tensor_2d = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], device=device)
    ref_tensor_2d = torch.tensor([[[[0.0]], [[0.0]]]], device=device)

    print(f"\nInput tensor 2D: {input_tensor_2d}")
    print(f"Input tensor 2D shape: {input_tensor_2d.shape}")
    print(f"Reference tensor 2D shape: {ref_tensor_2d.shape}")

    # PyTorch reference - sum over dimensions 0, 2, 3 (keeping dimension 1)
    pytorch_result_2d = input_tensor_2d.sum(dim=[0, 2, 3], keepdim=True)
    print(f"PyTorch sum (per-channel): {pytorch_result_2d}")

    # Triton result
    triton_result_2d = triton_sum_like(input_tensor_2d, ref_tensor_2d)
    print(f"Triton result (per-channel): {triton_result_2d}")

    print(f"Are close: {torch.allclose(pytorch_result_2d, triton_result_2d)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_triton_sum_simple()
    else:
        print("CUDA not available. This test requires GPU support.")
