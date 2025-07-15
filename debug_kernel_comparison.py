#!/usr/bin/env python3
import torch

from nncf.torch.quantization.reference import ReferenceBackendType
from nncf.torch.quantization.reference import ReferenceQuantize
from nncf.torch.quantization.triton.reference import backward_kernel_separate
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta

# Test just the separate kernel vs reference gradients
device = torch.device("cuda")

# Small test case
input_size = [1, 2, 2, 2]
levels = 16
level_low = 0
level_high = 15

# Create reference implementation
RQ = ReferenceQuantize(ReferenceBackendType.TORCH)

# Generate test data
ref_input = torch.randn(input_size, device=device, dtype=torch.float16)
ref_input_low = torch.tensor([-8.0], device=device, dtype=torch.float16)
ref_input_range = torch.tensor([15.0], device=device, dtype=torch.float16)

print(f"Input: {ref_input.flatten()}")
print(f"Input_low: {ref_input_low.item()}")
print(f"Input_range: {ref_input_range.item()}")

# Test reference backward
mock_grad_output = torch.ones(input_size, device=device, dtype=torch.float16)
ref_grads = RQ.backward(mock_grad_output, ref_input, ref_input_low, ref_input_range, levels, level_low, level_high)

print("\nReference gradients:")
print(f"grad_input: {ref_grads[0].flatten()}")
print(f"grad_low: {ref_grads[1].flatten()}")
print(f"grad_range: {ref_grads[2].flatten()}")

# Test triton separate kernel
grad_input = torch.empty_like(ref_input)
grad_low_full = torch.empty_like(ref_input)
grad_range_full = torch.empty_like(ref_input)

# Get meta information
grad_output_meta = get_4d_tensor_meta(mock_grad_output)
input__meta = get_4d_tensor_meta(ref_input)
input_low_meta = get_4d_tensor_meta(ref_input_low)
input_range_meta = get_4d_tensor_meta(ref_input_range)

# Launch kernel
grid = lambda meta: (1,)  # Only one block for small test
backward_kernel_separate[grid](
    mock_grad_output,
    grad_output_meta,
    ref_input,
    input__meta,
    ref_input_low,
    input_low_meta,
    ref_input_range,
    input_range_meta,
    levels,
    level_low,
    level_high,
    grad_input,
    grad_low_full,
    grad_range_full,
)

print("\nTriton separate kernel gradients:")
print(f"grad_input: {grad_input.flatten()}")
print(f"grad_low_full: {grad_low_full.flatten()}")
print(f"grad_range_full: {grad_range_full.flatten()}")

# Check matches
print("\nMatches:")
print(f"grad_input: {torch.allclose(ref_grads[0], grad_input, rtol=1e-2)}")
print(f"grad_low_full: {torch.allclose(ref_grads[1], grad_low_full, rtol=1e-2)}")
print(f"grad_range_full: {torch.allclose(ref_grads[2], grad_range_full, rtol=1e-2)}")

# Test sum reduction
from nncf.torch.quantization.triton.reference import triton_sum_like

grad_low_reduced = triton_sum_like(grad_low_full, ref_input_low)
grad_range_reduced = triton_sum_like(grad_range_full, ref_input_range)

print("\nAfter sum reduction:")
print(f"grad_low_reduced: {grad_low_reduced.item()}")
print(f"grad_range_reduced: {grad_range_reduced.item()}")

# Reference sums
ref_grad_low_sum = ref_grads[1].sum().item()
ref_grad_range_sum = ref_grads[2].sum().item()

print("\nReference sums:")
print(f"grad_low sum: {ref_grad_low_sum}")
print(f"grad_range sum: {ref_grad_range_sum}")

print("\nFinal comparison:")
print(f"grad_low ratio: {grad_low_reduced.item() / ref_grad_low_sum if ref_grad_low_sum != 0 else 'N/A'}")
print(f"grad_range ratio: {grad_range_reduced.item() / ref_grad_range_sum if ref_grad_range_sum != 0 else 'N/A'}")
