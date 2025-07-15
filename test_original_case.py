#!/usr/bin/env python3
"""
Debug test for the original failing case.
"""

import torch

from nncf.torch.quantization.triton.reference import triton_sum_like


def test_original_case():
    """Test the original failing case."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match the original example exactly
    torch.manual_seed(42)  # For reproducibility

    # Input tensor: [batch_size, channels, height, width] = [2, 3, 4, 4]
    input_tensor = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)

    # Per-channel quantization scales: [1, channels, 1, 1] = [1, 3, 1, 1]
    input_low = torch.randn(1, 3, 1, 1, device=device, requires_grad=True)
    input_range = torch.randn(1, 3, 1, 1, device=device, requires_grad=True)

    # Simulate per-element gradients (what the backward kernel produces)
    grad_low_full = torch.randn_like(input_tensor)  # Full tensor size
    grad_range_full = torch.randn_like(input_tensor)  # Full tensor size

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input low shape: {input_low.shape}")
    print(f"Grad low full shape: {grad_low_full.shape}")
    print(f"Grad range full shape: {grad_range_full.shape}")

    # Traditional PyTorch approach (what sum_like does):
    def pytorch_sum_like(tensor_to_sum, ref_tensor):
        """Reference implementation using PyTorch operations."""
        if ref_tensor.numel() == 1:
            return tensor_to_sum.sum()

        result = tensor_to_sum
        for dim, size in enumerate(ref_tensor.shape):
            if size == 1:
                result = result.sum(dim, keepdim=True)
        return result

    # Apply PyTorch sum reduction
    grad_low_reduced_pytorch = pytorch_sum_like(grad_low_full.clone(), input_low)
    grad_range_reduced_pytorch = pytorch_sum_like(grad_range_full.clone(), input_range)

    print(f"\nPyTorch result (grad_low): {grad_low_reduced_pytorch}")
    print(f"PyTorch result (grad_range): {grad_range_reduced_pytorch}")

    # Apply Triton sum reduction
    grad_low_reduced_triton = triton_sum_like(grad_low_full, input_low)
    grad_range_reduced_triton = triton_sum_like(grad_range_full, input_range)

    print(f"\nTriton result (grad_low): {grad_low_reduced_triton}")
    print(f"Triton result (grad_range): {grad_range_reduced_triton}")

    print(f"\nAre grad_low results close: {torch.allclose(grad_low_reduced_pytorch, grad_low_reduced_triton)}")
    print(f"Are grad_range results close: {torch.allclose(grad_range_reduced_pytorch, grad_range_reduced_triton)}")

    if not torch.allclose(grad_low_reduced_pytorch, grad_low_reduced_triton):
        print(f"Grad low difference: {(grad_low_reduced_pytorch - grad_low_reduced_triton).abs().max()}")

    if not torch.allclose(grad_range_reduced_pytorch, grad_range_reduced_triton):
        print(f"Grad range difference: {(grad_range_reduced_pytorch - grad_range_reduced_triton).abs().max()}")

    # Let's also check the manual sum approach to see if there's a fundamental issue
    print("\nManual verification:")
    manual_result = grad_low_full.sum(dim=(0, 2, 3), keepdim=True)
    print(f"Manual sum result: {manual_result}")
    print(f"Manual vs PyTorch: {torch.allclose(manual_result, grad_low_reduced_pytorch)}")
    print(f"Manual vs Triton: {torch.allclose(manual_result, grad_low_reduced_triton)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_original_case()
    else:
        print("CUDA not available. This test requires GPU support.")
