#!/usr/bin/env python3
"""
Test script for the generalized sum_like_v2_two_stage function.
"""

import sys

import torch

sys.path.append("/home/nlyaly/projects/nncf")

from gemma_code import sum_like_v2_two_stage


def generate_reference_tensor(input_size, scale_mode, is_weights, dtype):
    """Generate reference tensor shape based on scale mode and tensor type"""
    if scale_mode == "single_scale":
        # Single scale: [1]
        return torch.ones([1], dtype=dtype)
    elif scale_mode == "per_channel_scale":
        if is_weights:
            # For weights, channel is dim 0: [C, 1, 1, 1]
            channel_count = input_size[0]
            ref_shape = [1 for _ in input_size]
            ref_shape[0] = channel_count
        else:
            # For activations, channel is dim 1: [1, C, 1, 1]
            channel_count = input_size[1]
            ref_shape = [1 for _ in input_size]
            ref_shape[1] = channel_count
        return torch.ones(ref_shape, dtype=dtype)
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")


def pytorch_sum_like(tensor_to_sum, ref_tensor):
    """PyTorch reference implementation of sum_like functionality"""
    # Ensure both tensors have the same number of dimensions
    while tensor_to_sum.dim() < ref_tensor.dim():
        tensor_to_sum = tensor_to_sum.unsqueeze(0)
    while ref_tensor.dim() < tensor_to_sum.dim():
        ref_tensor = ref_tensor.unsqueeze(0)

    # Sum over dimensions where ref_tensor has size 1
    result = tensor_to_sum
    for dim in range(result.dim()):
        if ref_tensor.size(dim) == 1 and result.size(dim) > 1:
            result = result.sum(dim, keepdim=True)

    return result


def test_sum_like_configurations():
    """Test different configurations"""
    device = torch.device("cuda")
    dtype = torch.float16

    # Test cases: (input_size, scale_mode, is_weights)
    test_cases = [
        # 4D tensors
        ([4, 16, 16, 16], "single_scale", False),
        ([4, 16, 16, 16], "per_channel_scale", True),  # weights
        ([4, 16, 16, 16], "per_channel_scale", False),  # activations
        # 2D tensors
        ([1024, 256], "single_scale", False),
        ([1024, 256], "per_channel_scale", True),  # weights
        ([1024, 256], "per_channel_scale", False),  # activations
    ]

    for input_size, scale_mode, is_weights in test_cases:
        print(f"\nTesting: {input_size}, {scale_mode}, {'weights' if is_weights else 'activations'}")

        # Generate test data
        torch.manual_seed(42)
        input_tensor = torch.randn(input_size, device=device, dtype=dtype)
        ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Reference shape: {ref_tensor.shape}")

        # Compute expected result using PyTorch
        expected = pytorch_sum_like(input_tensor, ref_tensor)

        # Compute result using our implementation
        try:
            result = sum_like_v2_two_stage(input_tensor, ref_tensor)

            print(f"Expected shape: {expected.shape}")
            print(f"Result shape: {result.shape}")

            # Check correctness
            rtol = 1e-1
            atol = 1e-2

            correct = torch.allclose(result, expected, rtol=rtol, atol=atol)
            print(f"Correct: {correct}")

            if not correct:
                print(f"Max diff: {(result - expected).abs().max()}")
                print(f"Mean diff: {(result - expected).abs().mean()}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_sum_like_configurations()
