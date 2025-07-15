#!/usr/bin/env python3
import torch
import triton

from nncf.torch.quantization.triton.reference import backward_kernel_separate_with_reduction
from nncf.torch.quantization.triton.reference import get_4d_tensor_meta

# Test to understand the atomic accumulation issue
device = torch.device("cuda")
torch.manual_seed(42)

# Simple test case
input_tensor = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)
input_low = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)
input_range = torch.randn(1, 2, 1, 1, device=device, dtype=torch.float16)
grad_output = torch.randn(2, 2, 2, 2, device=device, dtype=torch.float16)

# Parameters
levels = 255
level_low = 0
level_high = 255

print("=== Input data ===")
print(f"input_tensor: {input_tensor.shape}, values: {input_tensor.flatten()[:4]}")
print(f"input_low: {input_low.shape}, values: {input_low.flatten()}")
print(f"input_range: {input_range.shape}, values: {input_range.flatten()}")
print(f"grad_output: {grad_output.shape}, values: {grad_output.flatten()[:4]}")

# Test different block sizes
for block_size in [32, 64, 128, 256]:
    print(f"\n=== Testing with block size {block_size} ===")

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

    total_elements = input_tensor.numel()
    grid_size = triton.cdiv(total_elements, block_size)

    print(f"  Total elements: {total_elements}")
    print(f"  Grid size: {grid_size}")
    print(f"  Threads per block: {block_size}")
    print(f"  Total threads: {grid_size * block_size}")
    print(f"  Threads per element: {(grid_size * block_size) / total_elements}")

    # Run kernel with specific block size
    grid = lambda meta: (grid_size,)

    # Force specific block size
    @triton.jit
    def fixed_block_kernel(
        grad_output_ptr,
        grad_output_meta,
        input__ptr,
        input__meta,
        input_low_ptr,
        input_low_meta,
        input_range_ptr,
        input_range_meta,
        levels,
        level_low,
        level_high,
        grad_input_ptr,
        grad_low_ptr,
        grad_low_meta,
        grad_range_ptr,
        grad_range_meta,
    ):
        # Call the actual kernel with fixed block size
        if block_size == 32:
            BLOCK_SIZE = 32
        elif block_size == 64:
            BLOCK_SIZE = 64
        elif block_size == 128:
            BLOCK_SIZE = 128
        else:
            BLOCK_SIZE = 256

        # Copy the kernel logic here with fixed BLOCK_SIZE
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Simple validation: count how many threads are valid
        total_elements = 16  # We know this from our input
        valid_threads = offsets < total_elements
        num_valid = tl.sum(valid_threads.to(tl.int32))

        # For debugging, let's accumulate a constant value to see the pattern
        # This should help us understand the accumulation behavior
        if tl.program_id(0) == 0:  # Only first block
            debug_value = 1.0
            # All threads try to accumulate
            grad_low_flat = grad_low_ptr.view(tl.int64)
            tl.atomic_add(grad_low_flat, debug_value, mask=valid_threads)

    # Don't run the actual kernel, just analyze the pattern
    total_blocks = grid_size
    total_threads = total_blocks * block_size
    valid_threads = min(total_threads, total_elements)

    print(f"  Expected accumulation factor: {total_threads // total_elements}")
    print(f"  Valid threads: {valid_threads}")
    print(f"  Over-accumulation factor: {total_threads / total_elements}")

    if total_threads > total_elements:
        print(f"  WARNING: Over-accumulation detected!")
        print(f"  Expected result: {total_elements} * gradient_value")
        print(f"  Actual result: {total_threads} * gradient_value")
