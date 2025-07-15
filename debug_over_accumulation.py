#!/usr/bin/env python3
import torch
import triton

from nncf.torch.quantization.triton.reference import backward_kernel_separate_with_reduction
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta

# Test the atomic accumulation issue with smaller tensor
device = torch.device("cuda")
torch.manual_seed(42)

# Small test case that will cause over-accumulation
input_tensor = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)  # 16 elements
input_low = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)  # 2 elements
input_range = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)  # 2 elements
grad_output = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)  # 16 elements

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

# Parameters
levels = 255
level_low = 0
level_high = 255
total_elements = input_tensor.numel()

print(f"\nTotal elements: {total_elements}")

# Test different block sizes - this will show over-accumulation
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
        print(f"grad_low values: {grad_low.flatten()}")
        print(f"grad_range values: {grad_range.flatten()}")

        # Check for over-accumulation
        avg_grad_low = grad_low.abs().mean().item()
        avg_grad_range = grad_range.abs().mean().item()
        print(f"Average grad_low magnitude: {avg_grad_low}")
        print(f"Average grad_range magnitude: {avg_grad_range}")

        # For 2x2x2x2 -> 1x2x1x1, each output element should accumulate 8 input elements
        # But with over-threading, we might get more accumulation
        expected_accumulation = 8  # 2*1*2*2 = 8 elements per channel
        over_accumulation_factor = (grid_size * block_size) / total_elements
        print(f"Expected accumulation per element: {expected_accumulation}")
        print(f"Over-accumulation factor: {over_accumulation_factor}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
