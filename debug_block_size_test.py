#!/usr/bin/env python3
import torch
import triton

from nncf.torch.quantization.triton.reference import backward_kernel_separate_with_reduction
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta

# Test the atomic accumulation issue
device = torch.device("cuda")
torch.manual_seed(42)

# Simple test case - same as what the test uses
input_tensor = torch.randn(4, 16, 16, 16, device=device, dtype=torch.float16)
input_low = torch.randn(1, 16, 1, 1, device=device, dtype=torch.float16)
input_range = torch.randn(1, 16, 1, 1, device=device, dtype=torch.float16)
grad_output = torch.randn(4, 16, 16, 16, device=device, dtype=torch.float16)

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

# Parameters
levels = 255
level_low = 0
level_high = 255
total_elements = input_tensor.numel()

print(f"\nTotal elements: {total_elements}")

# Test different block sizes
for block_size in [256, 512, 1024]:
    print(f"\n=== Testing with block size {block_size} ===")

    # Reset output tensors
    grad_input.zero_()
    grad_low.zero_()
    grad_range.zero_()

    grid_size = triton.cdiv(total_elements, block_size)
    print(f"Grid size: {grid_size}")
    print(f"Total threads: {grid_size * block_size}")
    print(f"Threads per element: {(grid_size * block_size) / total_elements}")

    try:
        # Run the kernel with explicit block size
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
            BLOCK_SIZE=block_size,
        )

        print("SUCCESS!")
        print(f"grad_low sample: {grad_low[0, 0, 0, 0]} (should be sum of {4 * 16 * 16} elements)")
        print(f"grad_range sample: {grad_range[0, 0, 0, 0]} (should be sum of {4 * 16 * 16} elements)")

        # Check for over-accumulation
        avg_grad_low = grad_low.abs().mean().item()
        avg_grad_range = grad_range.abs().mean().item()
        print(f"Average grad_low magnitude: {avg_grad_low}")
        print(f"Average grad_range magnitude: {avg_grad_range}")

        # Expected: each element should accumulate exactly 4*16*16 = 1024 times
        # If over-accumulation occurs, we'll see values that are too large

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
