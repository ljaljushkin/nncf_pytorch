#!/usr/bin/env python3
"""
Simple test to debug the optimized kernel without autotuner conflicts.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def debug_sum_reduction_kernel(
    input_ptr: torch.tensor,
    input_meta: torch.tensor,
    output_ptr: torch.tensor,
    output_meta: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Debug version of sum reduction kernel.
    """
    pid = tl.program_id(0)

    # Get shapes and strides
    input_s0 = tl.load(input_meta + 0)
    input_s1 = tl.load(input_meta + 1)
    input_s2 = tl.load(input_meta + 2)
    input_s3 = tl.load(input_meta + 3)

    output_s0 = tl.load(output_meta + 0)
    output_s1 = tl.load(output_meta + 1)
    output_s2 = tl.load(output_meta + 2)
    output_s3 = tl.load(output_meta + 3)

    input_st0 = tl.load(input_meta + 4)
    input_st1 = tl.load(input_meta + 5)
    input_st2 = tl.load(input_meta + 6)
    input_st3 = tl.load(input_meta + 7)

    output_st0 = tl.load(output_meta + 4)
    output_st1 = tl.load(output_meta + 5)
    output_st2 = tl.load(output_meta + 6)
    output_st3 = tl.load(output_meta + 7)

    input_elements = input_s0 * input_s1 * input_s2 * input_s3

    # Process input elements in blocks
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < input_elements

    # Convert linear input index to 4D coordinates
    tmp = input_offsets
    i3 = tmp % input_s3
    tmp //= input_s3
    i2 = tmp % input_s2
    tmp //= input_s2
    i1 = tmp % input_s1
    tmp //= input_s1
    i0 = tmp % input_s0

    # Calculate corresponding output coordinates (reduction mapping)
    o0 = tl.where(output_s0 == 1, 0, i0)
    o1 = tl.where(output_s1 == 1, 0, i1)
    o2 = tl.where(output_s2 == 1, 0, i2)
    o3 = tl.where(output_s3 == 1, 0, i3)

    # Load input values
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0).to(tl.float32)

    # Calculate output offsets for this block
    output_offsets = o0 * output_st0 + o1 * output_st1 + o2 * output_st2 + o3 * output_st3

    # Simple direct approach: each valid input element contributes to its output location
    # Use atomic operations to handle potential conflicts
    tl.atomic_add(output_ptr + output_offsets, input_vals, mask=input_mask)


def get_4d_tensor_meta(x: torch.tensor) -> torch.tensor:
    """
    Helper function for meta information creation.
    """
    shape = list(x.shape)
    stride = list(x.stride())
    size = len(shape)

    for i in range(4):
        if i >= size:
            shape += [1]
            stride += [0]
        elif shape[i] == 1:
            stride[i] = 0

    return torch.tensor(shape + stride, dtype=torch.int32).to(x.device)


def test_kernel():
    """Test the debug kernel directly"""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create small test case
    input_tensor = torch.randn(2, 3, 4, 4, device=device)
    ref_tensor = torch.randn(1, 3, 1, 1, device=device)

    print("Input tensor shape:", input_tensor.shape)
    print("Input tensor size:", input_tensor.numel())
    print("Reference tensor shape:", ref_tensor.shape)
    print("Reference tensor size:", ref_tensor.numel())

    # Expected result using PyTorch
    expected = torch.zeros_like(ref_tensor)
    temp_tensor = input_tensor.clone()
    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            temp_tensor = temp_tensor.sum(dim, keepdim=True)
    expected = temp_tensor

    print("Expected result:", expected.flatten())

    # Test debug kernel
    output = torch.zeros_like(ref_tensor)
    input_meta = get_4d_tensor_meta(input_tensor)
    output_meta = get_4d_tensor_meta(output)

    print("Input meta:", input_meta)
    print("Output meta:", output_meta)

    # Launch kernel manually with explicit block size
    block_size = 256
    grid_size = triton.cdiv(input_tensor.numel(), block_size)

    print("Block size:", block_size)
    print("Grid size:", grid_size)

    # Call the kernel directly
    debug_sum_reduction_kernel[(grid_size,)](
        input_tensor,
        input_meta,
        output,
        output_meta,
        BLOCK_SIZE=block_size,
    )

    print("Debug kernel result:", output.flatten())
    print("Match:", torch.allclose(output, expected))
    print("Diff:", (output - expected).abs().max())
    print("Ratio:", (output / expected).abs().max())


if __name__ == "__main__":
    test_kernel()
