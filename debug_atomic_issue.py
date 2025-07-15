#!/usr/bin/env python3
import torch
import triton

from nncf.torch.quantization.triton.reference import backward_kernel_separate_with_reduction
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta

# Test the atomic accumulation issue
device = torch.device("cuda")
torch.manual_seed(42)

# Simple test case
input_tensor = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)
input_low = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)
input_range = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)
grad_output = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)

# Create output tensors
grad_input = torch.zeros_like(input_tensor)
grad_low = torch.zeros_like(input_low)
grad_range = torch.zeros_like(input_range)

# Get meta information
grad_output_meta = get_4d_tensor_meta(grad_output)
input__meta = get_4d_tensor_meta(input_tensor)
input_low_meta = get_4d_tensor_meta(input_low)
input_range_meta = get_4d_tensor_meta(input_range)
grad_low_meta = get_4d_tensor_meta(grad_low)
grad_range_meta = get_4d_tensor_meta(grad_range)

print("Input shapes:")
print(f"input_tensor: {input_tensor.shape}")
print(f"input_low: {input_low.shape}")
print(f"input_range: {input_range.shape}")
print(f"grad_output: {grad_output.shape}")

print("\nOutput shapes:")
print(f"grad_input: {grad_input.shape}")
print(f"grad_low: {grad_low.shape}")
print(f"grad_range: {grad_range.shape}")

print("\nGrid calculation:")
total_elements = input_tensor.numel()
print(f"Total elements: {total_elements}")

# Check different block sizes
for block_size in [256, 512, 1024]:
    grid_size = triton.cdiv(total_elements, block_size)
    print(f"Block size {block_size}: Grid size {grid_size}")
    print(f"  Total threads: {grid_size * block_size}")
    print(f"  Threads per element: {(grid_size * block_size) / total_elements}")

# Parameters
levels = 255
level_low = 0
level_high = 255

# Test with minimal grid size to avoid over-accumulation
print("\n=== Testing with minimal grid ===")
# Reset output tensors
grad_input.zero_()
grad_low.zero_()
grad_range.zero_()

# Use a specific block size to debug
block_size = 256
grid_size = triton.cdiv(total_elements, block_size)
print(f"Using block size {block_size}, grid size {grid_size}")

# Run the kernel
grid = lambda meta: (grid_size,)
backward_kernel_separate_with_reduction[grid](
    grad_output,
    grad_output_meta,
    input_tensor,
    input__meta,
    input_low,
    input_low_meta,
    input_range,
    input_range_meta,
    levels,
    level_low,
    level_high,
    grad_input,
    grad_low,
    grad_low_meta,
    grad_range,
    grad_range_meta,
)

print("Results:")
print(f"grad_low: {grad_low.flatten()}")
print(f"grad_range: {grad_range.flatten()}")

# Calculate expected values manually
print("\n=== Manual calculation ===")
# For 2x2x2x2 tensor -> 1x2x1x1 reduction
# Each channel should accumulate from 8 elements (2x1x2x2)
elements_per_channel = 2 * 1 * 2 * 2  # 8 elements
print(f"Elements per channel: {elements_per_channel}")
print(f"Total elements: {total_elements}")
print(f"Expected accumulation factor: {elements_per_channel}")

# Check if we have over-accumulation
if grad_low.numel() > 0:
    avg_grad_low = grad_low.abs().mean().item()
    print(f"Average grad_low magnitude: {avg_grad_low}")
    if avg_grad_low > 100:  # Arbitrary threshold
        print("WARNING: Likely over-accumulation detected!")
