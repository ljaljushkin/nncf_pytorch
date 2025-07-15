#!/usr/bin/env python3
import torch
import triton
import triton.language as tl


# Simple working version
@triton.jit
def simple_sum_reduction_kernel(
    input_ptr,
    output_ptr,
    input_meta,
    output_meta,
    BLOCK_SIZE: tl.constexpr,
):
    # Read metadata
    input_shape = input_meta[:4]
    input_stride = input_meta[4:]
    output_shape = output_meta[:4]
    output_stride = output_meta[4:]

    # Get program ID
    pid = tl.program_id(0)

    # Calculate total number of output elements
    output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]

    # Process output elements
    for output_idx in range(pid, output_size, tl.num_programs(0)):
        # Convert 1D index to 4D coordinates for output
        n = output_idx // (output_shape[1] * output_shape[2] * output_shape[3])
        remaining = output_idx % (output_shape[1] * output_shape[2] * output_shape[3])
        c = remaining // (output_shape[2] * output_shape[3])
        remaining = remaining % (output_shape[2] * output_shape[3])
        h = remaining // output_shape[3]
        w = remaining % output_shape[3]

        # Calculate output offset
        output_offset = n * output_stride[0] + c * output_stride[1] + h * output_stride[2] + w * output_stride[3]

        # Sum over input elements that map to this output element
        sum_val = 0.0

        # Iterate over all input elements
        for in_n in range(input_shape[0]):
            for in_c in range(input_shape[1]):
                for in_h in range(input_shape[2]):
                    for in_w in range(input_shape[3]):
                        # Check if this input element should contribute to this output element
                        out_n = in_n if output_shape[0] > 1 else 0
                        out_c = in_c if output_shape[1] > 1 else 0
                        out_h = in_h if output_shape[2] > 1 else 0
                        out_w = in_w if output_shape[3] > 1 else 0

                        # Check if this input element maps to our output element
                        if out_n == n and out_c == c and out_h == h and out_w == w:
                            input_offset = (
                                in_n * input_stride[0]
                                + in_c * input_stride[1]
                                + in_h * input_stride[2]
                                + in_w * input_stride[3]
                            )
                            val = tl.load(input_ptr + input_offset)
                            sum_val += val

        # Store result
        tl.store(output_ptr + output_offset, sum_val)


# Test it
device = torch.device("cuda")


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


def simple_triton_sum_like(tensor_to_sum, target_shape_tensor):
    """Simple triton sum_like implementation"""
    # Handle scalar case
    if tensor_to_sum.numel() == 1:
        return tensor_to_sum.expand_as(target_shape_tensor)

    # Create output tensor
    output = torch.zeros_like(target_shape_tensor)

    # Get metadata
    input_meta = get_4d_tensor_meta(tensor_to_sum)
    output_meta = get_4d_tensor_meta(output)

    # Calculate grid size
    total_output_elements = output.numel()
    grid_size = min(1024, total_output_elements)

    # Launch kernel
    simple_sum_reduction_kernel[(grid_size,)](
        tensor_to_sum,
        output,
        input_meta,
        output_meta,
        BLOCK_SIZE=256,
    )

    return output


# Test the simple version
print("=== Testing Simple Triton Sum Like ===")

# Test case 1: Simple case
input1 = torch.ones(2, 3, 4, 4, device=device, dtype=torch.float16)
target1 = torch.zeros(1, 3, 1, 1, device=device, dtype=torch.float16)
result1 = simple_triton_sum_like(input1, target1)
expected1 = input1.sum(dim=(0, 2, 3), keepdim=True)
print(f"Simple case - Result: {result1.flatten()}")
print(f"Simple case - Expected: {expected1.flatten()}")
print(f"Simple case - Match: {torch.allclose(result1, expected1)}")

# Test case 2: Complex case
input2 = torch.randn(4, 16, 16, 16, device=device, dtype=torch.float16)
target2 = torch.zeros(1, 16, 1, 1, device=device, dtype=torch.float16)
result2 = simple_triton_sum_like(input2, target2)
expected2 = input2.sum(dim=(0, 2, 3), keepdim=True)
print(f"\nComplex case - Result values: {result2.flatten()[:5]}")
print(f"Complex case - Expected values: {expected2.flatten()[:5]}")
print(f"Complex case - Max diff: {(result2 - expected2).abs().max().item()}")
print(f"Complex case - Match: {torch.allclose(result2, expected2, rtol=1e-2)}")
