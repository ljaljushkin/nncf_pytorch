#!/usr/bin/env python3
import torch

from nncf.torch.quantization.triton.reference import backward
from nncf.torch.quantization.triton.reference import backward_kernel_separate
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta
from nncf.torch.quantization.triton.reference import triton_sum_like

# Test comparison between integrated and separate approaches
device = torch.device("cuda")

# Create simple test data
input_size = [4, 16, 16, 16]
input_tensor = torch.randn(input_size, device=device, dtype=torch.float16)

# For per_channel_scale activations, input_low and input_range have shape [1, 16, 1, 1]
input_low = torch.randn(1, 16, 1, 1, device=device, dtype=torch.float16)
input_range = torch.abs(torch.randn(1, 16, 1, 1, device=device, dtype=torch.float16)) + 0.1

grad_output = torch.randn_like(input_tensor)
levels = 16
level_low = 0
level_high = 15

print("Input shapes:")
print(f"input_tensor: {input_tensor.shape}")
print(f"input_low: {input_low.shape}")
print(f"input_range: {input_range.shape}")

# Test 1: Our new integrated approach
print("\n=== Testing integrated approach ===")
grad_input_integrated, grad_low_integrated, grad_range_integrated = backward(
    grad_output, input_tensor, input_low, input_range, levels, level_low, level_high
)

print(f"grad_low_integrated: {grad_low_integrated.flatten()[:3]}")
print(f"grad_range_integrated: {grad_range_integrated.flatten()[:3]}")

# Test 2: Separate approach (old way)
print("\n=== Testing separate approach ===")
grad_input_separate = torch.empty_like(input_tensor)
grad_low_full = torch.empty_like(input_tensor)
grad_range_full = torch.empty_like(input_tensor)

# Get meta information
grad_output_meta = get_4d_tensor_meta(grad_output)
input__meta = get_4d_tensor_meta(input_tensor)
input_low_meta = get_4d_tensor_meta(input_low)
input_range_meta = get_4d_tensor_meta(input_range)

with torch.cuda.device(input_tensor.device):
    # Launch the separate kernel
    import triton

    grid = lambda meta: (triton.cdiv(input_tensor.numel(), meta["BLOCK_SIZE"]),)
    backward_kernel_separate[grid](
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
        grad_input_separate,
        grad_low_full,
        grad_range_full,
    )

# Apply separate sum reduction
grad_low_separate = triton_sum_like(grad_low_full, input_low)
grad_range_separate = triton_sum_like(grad_range_full, input_range)

print(f"grad_low_separate: {grad_low_separate.flatten()[:3]}")
print(f"grad_range_separate: {grad_range_separate.flatten()[:3]}")

# Compare
print("\n=== Comparison ===")
print(f"grad_low diff: {(grad_low_integrated - grad_low_separate).abs().max().item()}")
print(f"grad_range diff: {(grad_range_integrated - grad_range_separate).abs().max().item()}")
print(f"grad_input diff: {(grad_input_integrated - grad_input_separate).abs().max().item()}")

# Check if the issue is with accumulation
print("\n=== Accumulation check ===")
print(f"grad_low_integrated sum: {grad_low_integrated.sum().item()}")
print(f"grad_low_separate sum: {grad_low_separate.sum().item()}")
print(f"grad_low_full sum: {grad_low_full.sum().item()}")
print(f"Expected ratio: {grad_low_full.sum().item() / grad_low_separate.sum().item()}")

# Print individual channel values
print("\n=== Per-channel values ===")
print(f"grad_low_integrated[0, :, 0, 0]: {grad_low_integrated[0, :, 0, 0]}")
print(f"grad_low_separate[0, :, 0, 0]: {grad_low_separate[0, :, 0, 0]}")
