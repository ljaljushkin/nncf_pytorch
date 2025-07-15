#!/usr/bin/env python3
"""
Test to understand the scaling issue.
"""

import torch

from nncf.torch.quantization.triton.reference import triton_sum_like


def test_scaling_issue():
    """Test to understand the scaling issue."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The expected result values
    expected_per_channel = torch.tensor([[[[-1.4906]], [[-0.0217]], [[3.1092]]]], device=device)

    # If we multiply by the ratio (~913), we should get close to the reported values
    scaling_factor = 913
    scaled_values = expected_per_channel * scaling_factor

    print(f"Expected result: {expected_per_channel}")
    print(f"Scaled by {scaling_factor}: {scaled_values}")

    # This should be close to the reported "triton result"
    reported_triton = torch.tensor([[[[-1360.9210]], [[-19.7893]], [[2838.7156]]]], device=device)

    print(f"Reported triton result: {reported_triton}")
    print(f"Are scaled values close to reported: {torch.allclose(scaled_values, reported_triton, atol=1e-2)}")

    # Now let's create a tensor that, when reduced, gives us the expected result
    # If we want the sum to be expected_per_channel, and we have 32 elements per channel
    # then each element should be expected_per_channel / 32

    elements_per_channel = 32  # 2 * 4 * 4 for shape [2, 3, 4, 4]
    element_value = expected_per_channel / elements_per_channel

    print(f"Element value needed: {element_value}")

    # Create a tensor with this element value
    input_tensor = element_value.expand(2, 3, 4, 4)
    ref_tensor = torch.ones(1, 3, 1, 1, device=device)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor sample: {input_tensor[0, 0, 0, 0]}")

    # Apply PyTorch sum reduction
    def pytorch_sum_like(tensor_to_sum, ref_tensor):
        """Reference implementation using PyTorch operations."""
        if ref_tensor.numel() == 1:
            return tensor_to_sum.sum()

        result = tensor_to_sum
        for dim, size in enumerate(ref_tensor.shape):
            if size == 1:
                result = result.sum(dim, keepdim=True)
        return result

    pytorch_result = pytorch_sum_like(input_tensor, ref_tensor)
    print(f"PyTorch result: {pytorch_result}")
    print(f"PyTorch vs expected: {torch.allclose(pytorch_result, expected_per_channel)}")

    # Apply Triton sum reduction
    triton_result = triton_sum_like(input_tensor, ref_tensor)
    print(f"Triton result: {triton_result}")
    print(f"Triton vs expected: {torch.allclose(triton_result, expected_per_channel)}")
    print(f"Triton vs PyTorch: {torch.allclose(triton_result, pytorch_result)}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_scaling_issue()
    else:
        print("CUDA not available. This test requires GPU support.")
