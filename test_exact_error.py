#!/usr/bin/env python3
"""
Test to reproduce the exact error case.
"""

import torch

from nncf.torch.quantization.triton.reference import triton_sum_like


def test_exact_error_case():
    """Test the exact error case that was reported."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the exact values from the error report
    grad_low_full = torch.tensor([[[[-1360.9210]], [[-19.7893]], [[2838.7156]]]], device=device)

    # This should produce the expected result
    expected_result = torch.tensor([[[[-1.4906]], [[-0.0217]], [[3.1092]]]], device=device)

    # Reference tensor shape to reduce to
    ref_tensor = torch.ones(1, 3, 1, 1, device=device)

    print(f"Input tensor: {grad_low_full}")
    print(f"Expected result: {expected_result}")
    print(f"Reference tensor shape: {ref_tensor.shape}")

    # Apply Triton sum reduction
    triton_result = triton_sum_like(grad_low_full, ref_tensor)
    print(f"Triton result: {triton_result}")

    # The issue is that the Triton kernel doesn't apply the proper scaling
    # Let's check if this is the issue
    print(f"Are results close: {torch.allclose(triton_result, expected_result)}")

    # Check if the issue is that the input values are much larger than expected
    # Maybe there's an issue with the test data generation
    print(f"Ratio of actual/expected: {triton_result / expected_result}")
    print(f"Scale factor: {grad_low_full.shape[0] * grad_low_full.shape[2] * grad_low_full.shape[3]}")

    # The issue might be that we're not applying the correct reduction
    # Let's check what the PyTorch reference would produce
    def pytorch_sum_like(tensor_to_sum, ref_tensor):
        """Reference implementation using PyTorch operations."""
        if ref_tensor.numel() == 1:
            return tensor_to_sum.sum()

        result = tensor_to_sum
        for dim, size in enumerate(ref_tensor.shape):
            if size == 1:
                result = result.sum(dim, keepdim=True)
        return result

    pytorch_result = pytorch_sum_like(grad_low_full, ref_tensor)
    print(f"PyTorch result: {pytorch_result}")
    print(f"PyTorch vs Triton: {torch.allclose(pytorch_result, triton_result)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_exact_error_case()
    else:
        print("CUDA not available. This test requires GPU support.")
