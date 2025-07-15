#!/usr/bin/env python3
import torch

from nncf.torch.quantization.triton.reference import triton_sum_like

# Test the triton_sum_like function directly
device = torch.device("cuda")

print("=== Testing triton_sum_like function ===")

# Create test data
input_tensor = torch.randn(4, 16, 16, 16, device=device, dtype=torch.float16)
target_shape = torch.zeros(1, 16, 1, 1, device=device, dtype=torch.float16)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Target shape: {target_shape.shape}")

# Test triton_sum_like
result = triton_sum_like(input_tensor, target_shape)

print(f"Result shape: {result.shape}")
print(f"Result values: {result.flatten()[:5]}")

# Compare with PyTorch's sum
expected_result = input_tensor.sum(dim=(0, 2, 3), keepdim=True)
print(f"Expected result shape: {expected_result.shape}")
print(f"Expected result values: {expected_result.flatten()[:5]}")

# Check difference
diff = (result - expected_result).abs()
print(f"Max difference: {diff.max().item()}")
print(f"Mean difference: {diff.mean().item()}")

# Check if the results are close
print(f"Results are close: {torch.allclose(result, expected_result, rtol=1e-2)}")

# Test with a simple case
print("\n=== Testing simple case ===")
simple_input = torch.ones(2, 3, 4, 4, device=device, dtype=torch.float16)
simple_target = torch.zeros(1, 3, 1, 1, device=device, dtype=torch.float16)

simple_result = triton_sum_like(simple_input, simple_target)
simple_expected = simple_input.sum(dim=(0, 2, 3), keepdim=True)

print(f"Simple result: {simple_result.flatten()}")
print(f"Simple expected: {simple_expected.flatten()}")
print(f"Simple results match: {torch.allclose(simple_result, simple_expected)}")
