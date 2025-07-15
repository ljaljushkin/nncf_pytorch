#!/usr/bin/env python3
import torch
import triton

from nncf.torch.quantization.triton.reference import backward_kernel_separate_with_reduction
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta

# Create a simple test case
device = torch.device("cuda")
torch.manual_seed(42)

# Simple test case: 2x2x2x2 -> 1x2x1x1
input_tensor = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)
input_low = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)
input_range = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)
grad_output = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)

print("=== Input shapes ===")
print(f"input_tensor: {input_tensor.shape}")
print(f"input_low: {input_low.shape}")
print(f"input_range: {input_range.shape}")
print(f"grad_output: {grad_output.shape}")

# Create output tensors
grad_input = torch.zeros_like(input_tensor)
grad_low = torch.zeros_like(input_low)
grad_range = torch.zeros_like(input_range)

print("\n=== Output shapes ===")
print(f"grad_input: {grad_input.shape}")
print(f"grad_low: {grad_low.shape}")
print(f"grad_range: {grad_range.shape}")

# Parameters
levels = 255
level_low = 0
level_high = 255

# Get meta information
grad_output_meta = get_4d_tensor_meta(grad_output)
input__meta = get_4d_tensor_meta(input_tensor)
input_low_meta = get_4d_tensor_meta(input_low)
input_range_meta = get_4d_tensor_meta(input_range)
grad_low_meta = get_4d_tensor_meta(grad_low)
grad_range_meta = get_4d_tensor_meta(grad_range)

print("\n=== Meta information ===")
print(f"grad_output_meta: {grad_output_meta.cpu().numpy()}")
print(f"input__meta: {input__meta.cpu().numpy()}")
print(f"input_low_meta: {input_low_meta.cpu().numpy()}")
print(f"input_range_meta: {input_range_meta.cpu().numpy()}")
print(f"grad_low_meta: {grad_low_meta.cpu().numpy()}")
print(f"grad_range_meta: {grad_range_meta.cpu().numpy()}")

# Test the kernel
print("\n=== Testing integrated kernel ===")
try:
    grid = lambda meta: (triton.cdiv(input_tensor.numel(), meta["BLOCK_SIZE"]),)
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

    print("grad_low result:", grad_low.flatten())
    print("grad_range result:", grad_range.flatten())
    print("grad_input result:", grad_input.flatten())

except Exception as e:
    print("Error:", e)
    import traceback

    traceback.print_exc()

# Calculate what the expected values should be
print("\n=== Expected values calculation ===")
print(f"Total input elements: {input_tensor.numel()}")
print(f"Input tensor elements per channel: {input_tensor.shape[0] * input_tensor.shape[2] * input_tensor.shape[3]}")
print(f"Number of channels: {input_tensor.shape[1]}")
print(
    f"Elements that should accumulate to grad_low[0,0,0,0]: {input_tensor.shape[0] * input_tensor.shape[2] * input_tensor.shape[3]}"
)
print(
    f"Elements that should accumulate to grad_low[0,1,0,0]: {input_tensor.shape[0] * input_tensor.shape[2] * input_tensor.shape[3]}"
)
