"""
## Typical Solution for sum_like reduction in Triton for 4D tensors

For a reduction like: def sum_like(tensor_to_sum: Tensor[4, 16, 16, 16], ref_tensor: Tensor[1,16,1,1]) -> Tensor[1,16,1,1]

### Key approaches:

1. **Direct Atomic Approach**: Use atomic operations to accumulate results
   - Simple but suffers from atomic contention
   - Works well for small tensors

2. **Hierarchical Reduction**: Use larger blocks with internal reduction
   - Reduces atomic operations by doing more work per thread block
   - Better for medium-sized tensors

3. **Two-Stage Reduction**: Split into block-level and final reduction
   - Eliminates atomic contention entirely
   - Best for large tensors

### About tl.reduce:
- `tl.reduce` is excellent for reducing within a single thread block
- But for cross-block reduction (like our case), you still need atomic operations or multiple kernel launches
- The two-stage approach effectively uses `tl.reduce` internally (via `tl.sum`) then launches a second kernel

### Performance optimization for float16:
- Use fp32 accumulation internally, convert to fp16 for storage
- Larger block sizes (2048 vs 1024) reduce atomic contention
- Two-stage approach avoids atomic operations entirely

The two-stage approach is generally the best solution for large-scale reductions in Triton.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_like_4d_kernel_coalesced(
    input_ptr,
    output_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Memory-coalesced sum reduction kernel for 4D tensors.

    This kernel processes contiguous memory regions to maximize
    memory bandwidth utilization.
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate total elements to process for this channel
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    tmp = linear_idx
    i3 = tmp % input_s3
    tmp //= input_s3
    i2 = tmp % input_s2
    tmp //= input_s2
    i0 = tmp % input_s0

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)

    # Sum within the block and use atomic add for final result
    block_sum = tl.sum(data)
    tl.atomic_add(output_ptr + channel_idx, block_sum)


def sum_like_v2(tensor_to_sum, ref_tensor):
    """Optimized Triton implementation of sum_like for 4D tensors."""
    # Create output tensor with the same dtype as the input tensor
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Calculate grid dimensions for parallel processing
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Use 2D grid: (channels, blocks_per_channel)
    grid = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_coalesced[grid](
        tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(ref_tensor.shape)


# Optimized float16 kernel with hierarchical reduction
@triton.jit
def _sum_like_4d_kernel_fp16_optimized(
    input_ptr,
    output_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Float16-optimized sum reduction kernel using hierarchical reduction.

    This kernel reduces atomic contention by using larger blocks and
    hierarchical reduction to minimize atomic operations.
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate total elements to process for this channel
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    i3 = linear_idx % input_s3
    i2 = (linear_idx // input_s3) % input_s2
    i0 = linear_idx // (input_s3 * input_s2)

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data and convert to float32 for accumulation
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0).to(tl.float32)

    # Use hierarchical reduction within the block
    block_sum = tl.sum(data)

    # Convert back to float16 for atomic operation
    block_sum_fp16 = block_sum.to(tl.float16)

    # Use atomic add for final result
    tl.atomic_add(output_ptr + channel_idx, block_sum_fp16)


def sum_like_v2_fp16_optimized(tensor_to_sum, ref_tensor):
    """Float16-optimized Triton implementation with hierarchical reduction."""
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Calculate grid dimensions for parallel processing
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]
    print("elements_per_channel=", elements_per_channel)
    # Use larger block size for float16 to reduce atomic contention
    BLOCK_SIZE = 64 if tensor_to_sum.dtype == torch.float16 else 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)
    print("num_blocks=", num_blocks)

    # Use 2D grid: (channels, blocks_per_channel)
    grid = (tensor_to_sum.shape[1], num_blocks)
    print("DTYPE=", tensor_to_sum.dtype)
    if tensor_to_sum.dtype == torch.float16:
        _sum_like_4d_kernel_fp16_optimized[grid](
            tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        _sum_like_4d_kernel_coalesced[grid](
            tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
        )

    return output.reshape(ref_tensor.shape)


# Alternative approach: Two-stage reduction to minimize atomic operations
@triton.jit
def _sum_like_general_kernel_two_stage(
    input_ptr,
    temp_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    output_s0,
    output_s1,
    output_s2,
    output_s3,
    num_blocks_per_output,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generalized first stage: Reduce within blocks for any dimensional configuration.
    """
    # Get program IDs for 2D grid
    output_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # Calculate elements per output element
    elements_per_output = 1
    if output_s0 == 1 and input_s0 > 1:
        elements_per_output *= input_s0
    if output_s1 == 1 and input_s1 > 1:
        elements_per_output *= input_s1
    if output_s2 == 1 and input_s2 > 1:
        elements_per_output *= input_s2
    if output_s3 == 1 and input_s3 > 1:
        elements_per_output *= input_s3

    # Calculate total output elements
    output_elements = (
        (output_s0 if output_s0 > 1 else 1)
        * (output_s1 if output_s1 > 1 else 1)
        * (output_s2 if output_s2 > 1 else 1)
        * (output_s3 if output_s3 > 1 else 1)
    )

    if output_idx >= output_elements:
        return

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Convert output index to coordinates
    tmp = output_idx
    out_i3 = tmp % (output_s3 if output_s3 > 1 else 1)
    tmp //= output_s3 if output_s3 > 1 else 1
    out_i2 = tmp % (output_s2 if output_s2 > 1 else 1)
    tmp //= output_s2 if output_s2 > 1 else 1
    out_i1 = tmp % (output_s1 if output_s1 > 1 else 1)
    tmp //= output_s1 if output_s1 > 1 else 1
    out_i0 = tmp % (output_s0 if output_s0 > 1 else 1)

    # Accumulate over the elements that contribute to this output
    accumulated_sum = 0.0

    for i in range(BLOCK_SIZE):
        element_idx = start_offset + i
        if element_idx < elements_per_output:
            # Map element index to input coordinates
            tmp_elem = element_idx

            # Handle different reduction patterns
            if output_s0 == 1 and output_s1 > 1:  # Activations: reduce dims 0,2,3, keep dim 1
                i3 = tmp_elem % input_s3
                tmp_elem //= input_s3
                i2 = tmp_elem % input_s2
                tmp_elem //= input_s2
                i0 = tmp_elem % input_s0
                i1 = out_i1
            elif output_s1 == 1 and output_s0 > 1:  # Weights: reduce dims 1,2,3, keep dim 0
                i3 = tmp_elem % input_s3
                tmp_elem //= input_s3
                i2 = tmp_elem % input_s2
                tmp_elem //= input_s2
                i1 = tmp_elem % input_s1
                i0 = out_i0
            else:  # Single scale: reduce all dims
                i3 = tmp_elem % input_s3
                tmp_elem //= input_s3
                i2 = tmp_elem % input_s2
                tmp_elem //= input_s2
                i1 = tmp_elem % input_s1
                tmp_elem //= input_s1
                i0 = tmp_elem % input_s0

            # Calculate memory offset
            mem_offset = i0 * input_st0 + i1 * input_st1 + i2 * input_st2 + i3 * input_st3

            # Load and accumulate
            value = tl.load(input_ptr + mem_offset)
            accumulated_sum += value

    # Store intermediate result
    temp_offset = output_idx * num_blocks_per_output + block_idx
    tl.store(temp_ptr + temp_offset, accumulated_sum)


@triton.jit
def _sum_like_4d_kernel_second_stage(
    temp_ptr,
    output_ptr,
    num_output_elements,
    num_blocks_per_output,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second stage: Reduce intermediate results to final output.
    """
    output_idx = tl.program_id(0)

    if output_idx >= num_output_elements:
        return

    # Load intermediate results for this output element
    temp_offset = output_idx * num_blocks_per_output
    block_indices = tl.arange(0, BLOCK_SIZE)
    mask = block_indices < num_blocks_per_output

    intermediate_sums = tl.load(temp_ptr + temp_offset + block_indices, mask=mask, other=0.0)

    # Final reduction
    final_sum = tl.sum(intermediate_sums)

    # Store result
    tl.store(output_ptr + output_idx, final_sum)


def sum_like_v2_two_stage(tensor_to_sum, ref_tensor):
    """Two-stage reduction to minimize atomic operations. Works for 2D and 4D tensors with different reduction patterns."""

    # Ensure both tensors are 4D for consistent processing
    original_tensor_shape = tensor_to_sum.shape
    original_ref_shape = ref_tensor.shape

    # Convert to 4D if needed
    if len(original_tensor_shape) == 2:
        tensor_to_sum = tensor_to_sum.unsqueeze(2).unsqueeze(3)  # [N, C] -> [N, C, 1, 1]
    if len(original_ref_shape) == 1:
        ref_tensor = ref_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1] -> [1, 1, 1, 1]
    elif len(original_ref_shape) == 2:
        ref_tensor = ref_tensor.unsqueeze(2).unsqueeze(3)  # [N, C] -> [N, C, 1, 1]

    # Calculate total elements that contribute to each output element
    elements_per_output = 1
    for i in range(4):
        if ref_tensor.shape[i] == 1 and tensor_to_sum.shape[i] > 1:
            elements_per_output *= tensor_to_sum.shape[i]

    # Calculate total output elements
    output_elements = ref_tensor.numel()

    # Calculate grid dimensions
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_output, BLOCK_SIZE)

    # Create intermediate storage with proper dtype
    temp_size = output_elements * num_blocks
    temp_storage = torch.zeros(temp_size, device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # First stage: Block-level reduction
    grid1 = (output_elements, num_blocks)
    _sum_like_general_kernel_two_stage[grid1](
        tensor_to_sum,
        temp_storage,
        *tensor_to_sum.shape,
        *tensor_to_sum.stride(),
        *ref_tensor.shape,
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Second stage: Final reduction
    output = torch.zeros(ref_tensor.shape, device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Use a block size that can handle the number of intermediate results
    reduction_block_size = min(1024, triton.next_power_of_2(num_blocks))

    grid2 = (output_elements,)
    _sum_like_4d_kernel_second_stage[grid2](
        temp_storage, output.view(-1), output_elements, num_blocks, BLOCK_SIZE=reduction_block_size
    )

    # Restore original shape
    return output.view(original_ref_shape)


# Simplified two-stage reduction for 4D tensors only
@triton.jit
def _sum_like_4d_kernel_two_stage_simple(
    input_ptr,
    temp_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    num_blocks_per_channel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified first stage for 4D tensors: sum over dims 0,2,3, keep dim 1
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    i3 = linear_idx % input_s3
    i2 = (linear_idx // input_s3) % input_s2
    i0 = linear_idx // (input_s3 * input_s2)

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0).to(tl.float32)

    # Sum within the block
    block_sum = tl.sum(data).to(tl.float16)

    # Store intermediate result
    temp_offset = channel_idx * num_blocks_per_channel + block_idx
    tl.store(temp_ptr + temp_offset, block_sum)


@triton.jit
def _sum_like_4d_kernel_second_stage_simple(
    temp_ptr,
    output_ptr,
    input_s1,
    num_blocks_per_channel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second stage: Reduce intermediate results for each channel
    """
    channel_idx = tl.program_id(0)

    if channel_idx >= input_s1:
        return

    # Load intermediate results for this channel
    temp_offset = channel_idx * num_blocks_per_channel
    block_indices = tl.arange(0, BLOCK_SIZE)
    mask = block_indices < num_blocks_per_channel

    intermediate_sums = tl.load(temp_ptr + temp_offset + block_indices, mask=mask, other=0.0)

    # Final reduction
    final_sum = tl.sum(intermediate_sums)

    # Store result
    tl.store(output_ptr + channel_idx, final_sum)


def sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor):
    """Simplified two-stage reduction for 4D tensors only - much faster."""
    # Only works for 4D tensors with shape reduction pattern [N, C, H, W] -> [1, C, 1, 1]
    if len(tensor_to_sum.shape) != 4 or len(ref_tensor.shape) != 4:
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Check if it's the expected reduction pattern
    if not (
        ref_tensor.shape[0] == 1
        and ref_tensor.shape[1] == tensor_to_sum.shape[1]
        and ref_tensor.shape[2] == 1
        and ref_tensor.shape[3] == 1
    ):
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]

    # Calculate grid dimensions
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Create intermediate storage
    temp_size = tensor_to_sum.shape[1] * num_blocks
    temp_storage = torch.zeros(temp_size, device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # First stage: Block-level reduction
    grid1 = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_two_stage_simple[grid1](
        tensor_to_sum, temp_storage, *tensor_to_sum.shape, *tensor_to_sum.stride(), num_blocks, BLOCK_SIZE=BLOCK_SIZE
    )

    # Second stage: Final reduction
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Use a block size that can handle the number of intermediate results
    reduction_block_size = min(1024, triton.next_power_of_2(num_blocks))

    grid2 = (tensor_to_sum.shape[1],)
    _sum_like_4d_kernel_second_stage_simple[grid2](
        temp_storage, output, tensor_to_sum.shape[1], num_blocks, BLOCK_SIZE=reduction_block_size
    )

    return output.reshape(ref_tensor.shape)


# Single-stage implementation for comparison
@triton.jit
def _sum_like_4d_kernel_single_stage(
    input_ptr,
    output_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-stage reduction for 4D tensors: sum over dims 0,2,3, keep dim 1
    Each channel gets one or more blocks, results are atomically accumulated
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    i3 = linear_idx % input_s3
    i2 = (linear_idx // input_s3) % input_s2
    i0 = linear_idx // (input_s3 * input_s2)

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)

    # Sum within the block
    block_sum = tl.sum(data)

    # Direct atomic add to output (single stage)
    tl.atomic_add(output_ptr + channel_idx, block_sum)


def sum_like_v2_single_stage(tensor_to_sum, ref_tensor):
    """Single-stage reduction for 4D tensors - simpler but may have atomic contention."""
    # Only works for 4D tensors with shape reduction pattern [N, C, H, W] -> [1, C, 1, 1]
    if len(tensor_to_sum.shape) != 4 or len(ref_tensor.shape) != 4:
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Check if it's the expected reduction pattern
    if not (
        ref_tensor.shape[0] == 1
        and ref_tensor.shape[1] == tensor_to_sum.shape[1]
        and ref_tensor.shape[2] == 1
        and ref_tensor.shape[3] == 1
    ):
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]

    # Calculate grid dimensions
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Create output tensor (initialized to zero)
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Single stage: Direct reduction with atomic operations
    grid = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_single_stage[grid](
        tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(ref_tensor.shape)


# Single-stage kernel with intentional data races for performance measurement
@triton.jit
def _sum_like_4d_kernel_single_stage_race(
    input_ptr,
    output_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-stage kernel with intentional data races.
    WARNING: This produces incorrect results but measures raw performance.
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    i3 = linear_idx % input_s3
    i2 = (linear_idx // input_s3) % input_s2
    i0 = linear_idx // (input_s3 * input_s2)

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)

    # Sum within the block
    block_sum = tl.sum(data)

    # INTENTIONAL DATA RACE: Direct write without atomic operation
    # This will cause incorrect results but shows raw performance
    if block_idx == 0:
        # Only first block writes to avoid total chaos
        tl.store(output_ptr + channel_idx, block_sum)
    else:
        # Other blocks add their contribution (data race!)
        current_value = tl.load(output_ptr + channel_idx)
        tl.store(output_ptr + channel_idx, current_value + block_sum)


def sum_like_v2_single_stage_race(tensor_to_sum, ref_tensor):
    """
    Single-stage reduction with intentional data races for performance measurement.
    WARNING: Produces incorrect results but measures raw kernel overhead.
    """
    # Only works for 4D tensors with shape reduction pattern [N, C, H, W] -> [1, C, 1, 1]
    if len(tensor_to_sum.shape) != 4 or len(ref_tensor.shape) != 4:
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Check if it's the expected reduction pattern
    if not (
        ref_tensor.shape[0] == 1
        and ref_tensor.shape[1] == tensor_to_sum.shape[1]
        and ref_tensor.shape[2] == 1
        and ref_tensor.shape[3] == 1
    ):
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]

    # Calculate grid dimensions
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Create output tensor
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Single stage: Direct reduction with data races
    grid = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_single_stage_race[grid](
        tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(ref_tensor.shape)


# Single-stage kernel with proper atomic operations (for correctness comparison)
@triton.jit
def _sum_like_4d_kernel_single_stage_atomic(
    input_ptr,
    output_ptr,
    input_s0,
    input_s1,
    input_s2,
    input_s3,
    input_st0,
    input_st1,
    input_st2,
    input_st3,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-stage kernel with atomic operations for correctness.
    """
    # Get program IDs for 2D grid
    channel_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    if channel_idx >= input_s1:
        return

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = input_s0 * input_s2 * input_s3

    # Calculate the starting offset for this block
    start_offset = block_idx * BLOCK_SIZE

    # Generate linear indices for this block
    linear_idx = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = linear_idx < elements_per_channel

    # Convert to 4D coordinates for dimensions 0, 2, 3
    i3 = linear_idx % input_s3
    i2 = (linear_idx // input_s3) % input_s2
    i0 = linear_idx // (input_s3 * input_s2)

    # Calculate memory offsets
    mem_offsets = i0 * input_st0 + channel_idx * input_st1 + i2 * input_st2 + i3 * input_st3

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)

    # Sum within the block
    block_sum = tl.sum(data)

    # Atomic add for correctness
    tl.atomic_add(output_ptr + channel_idx, block_sum)


def sum_like_v2_single_stage_atomic(tensor_to_sum, ref_tensor):
    """
    Single-stage reduction with atomic operations for correctness.
    """
    # Only works for 4D tensors with shape reduction pattern [N, C, H, W] -> [1, C, 1, 1]
    if len(tensor_to_sum.shape) != 4 or len(ref_tensor.shape) != 4:
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Check if it's the expected reduction pattern
    if not (
        ref_tensor.shape[0] == 1
        and ref_tensor.shape[1] == tensor_to_sum.shape[1]
        and ref_tensor.shape[2] == 1
        and ref_tensor.shape[3] == 1
    ):
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)  # Fall back to general version

    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]

    # Calculate grid dimensions
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Create output tensor
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Single stage: Direct reduction with atomic operations
    grid = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_single_stage_atomic[grid](
        tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(ref_tensor.shape)


# --- End of re-included code ---


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_ELEMENTS"],
        x_vals=[64 * 128 * s * s for s in [16, 32, 64, 128, 256]],
        line_arg="provider",
        line_vals=[
            "pytorch",
            "custom_kernel_v2",
            "custom_kernel_v2_optimized",
            "custom_kernel_v2_two_stage_simple",
            "custom_kernel_v2_single_stage",
            "custom_kernel_v2_single_stage_race",
            "custom_kernel_v2_single_stage_atomic",
        ],
        line_names=[
            "PyTorch",
            "Custom Kernel (v2)",
            "Custom Kernel (v2 Optimized)",
            "Custom Kernel (v2 Two-Stage Simple)",
            "Custom Kernel (v2 Single Stage)",
            "Custom Kernel (v2 Single Stage Race)",
            "Custom Kernel (v2 Single Stage Atomic)",
        ],
        styles=[
            ("blue", "-"),
            ("red", "--"),
            ("green", "-."),
            ("purple", "-."),
            ("orange", ":"),
            ("brown", ":"),
            ("pink", ":"),
        ],
        ylabel="ms",
        plot_name="sum-like-4d-performance-comparison",
        args={"D1_size": 128},
    )
)
def benchmark(D1_size, N_ELEMENTS, provider):
    # Infer shapes from total elements
    D0_size = 64
    D2_size = int((N_ELEMENTS / (D0_size * D1_size)) ** 0.5)
    D3_size = int(N_ELEMENTS / (D0_size * D1_size * D2_size))

    shape = (D0_size, D1_size, D2_size, D3_size)
    ref_shape = (1, D1_size, 1, 1)

    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    ref = torch.empty(ref_shape, device="cuda", dtype=torch.float16)

    quantiles = [0.2, 0.5, 0.8]

    if provider == "pytorch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.sum(x, axis=(0, 2, 3), keepdim=True), quantiles=quantiles
        )
    elif provider == "custom_kernel_v2":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2(x, ref), quantiles=quantiles)
    elif provider == "custom_kernel_v2_optimized":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2_fp16_optimized(x, ref), quantiles=quantiles)
    elif provider == "custom_kernel_v2_two_stage":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2_two_stage(x, ref), quantiles=quantiles)
    elif provider == "custom_kernel_v2_two_stage_simple":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2_two_stage_simple(x, ref), quantiles=quantiles)
    elif provider == "custom_kernel_v2_single_stage":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2_single_stage(x, ref), quantiles=quantiles)
    elif provider == "custom_kernel_v2_single_stage_race":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: sum_like_v2_single_stage_race(x, ref), quantiles=quantiles)
    elif provider == "custom_kernel_v2_single_stage_atomic":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sum_like_v2_single_stage_atomic(x, ref), quantiles=quantiles
        )

    return ms, min_ms, max_ms


if __name__ == "__main__":
    # Test correctness first
    print("Testing correctness...")

    # Test case
    shape = [4, 16, 64, 64]
    tensor_to_sum = torch.randn(*shape, device="cuda", dtype=torch.float16)
    # tensor_to_sum = torch.arange(math.prod(shape), device="cuda", dtype=torch.float16).reshape(shape)
    print("Input tensor:")
    print(tensor_to_sum[0, 0, 0, :5])
    ref_tensor = torch.empty(1, 16, 1, 1, device="cuda", dtype=torch.float16)

    # PyTorch reference
    pytorch_result = torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)
    print("PyTorch result:")
    print(pytorch_result)

    # Test all implementations
    triton_result_two_stage = sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)
    triton_result_atomic = sum_like_v2_single_stage_atomic(tensor_to_sum, ref_tensor)
    triton_result_race = sum_like_v2_single_stage_race(tensor_to_sum, ref_tensor)

    print("Two-stage result:", triton_result_two_stage.flatten())
    print("Atomic result:", triton_result_atomic.flatten())
    print("Race result (may be incorrect):", triton_result_race.flatten())

    # Check correctness
    assert torch.allclose(pytorch_result, triton_result_two_stage, rtol=1, atol=1e-1), (
        f"Two-stage results don't match. Max diff: {(pytorch_result - triton_result_two_stage).abs().max()}"
    )

    assert torch.allclose(pytorch_result, triton_result_atomic, rtol=1, atol=1e-1), (
        f"Atomic results don't match. Max diff: {(pytorch_result - triton_result_atomic).abs().max()}"
    )

    print("✓ Correctness tests passed!")

    # Performance comparison
    print("\nPerformance comparison:")
    print("=" * 50)

    # Test with larger tensor for better performance measurement
    large_shape = [128, 64, 256, 256]
    large_tensor = torch.randn(large_shape, device="cuda", dtype=torch.float16)
    large_ref = torch.empty(1, 64, 1, 1, device="cuda", dtype=torch.float16)

    import time

    # # Warmup
    # for _ in range(10):
    #     _ = sum_like_v2_two_stage_simple(large_tensor, large_ref)
    #     _ = sum_like_v2_single_stage_atomic(large_tensor, large_ref)
    #     _ = sum_like_v2_single_stage_race(large_tensor, large_ref)

    # Benchmark
    def benchmark_func(func, tensor, ref, name):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            result = func(tensor, ref)
        torch.cuda.synchronize()
        end = time.time()
        avg_time = (end - start) / 100 * 1000  # Convert to ms
        print(f"{name}: {avg_time:.4f}ms")
        return result

    print(f"Tensor shape: {large_shape}")
    print(f"Elements per channel: {large_shape[0] * large_shape[2] * large_shape[3]}")
    print(f"Blocks per channel: {triton.cdiv(large_shape[0] * large_shape[2] * large_shape[3], 1024)}")

    two_stage_result = benchmark_func(sum_like_v2_two_stage_simple, large_tensor, large_ref, "Two-stage")
    atomic_result = benchmark_func(sum_like_v2_single_stage_atomic, large_tensor, large_ref, "Single-stage atomic")
    race_result = benchmark_func(sum_like_v2_single_stage_race, large_tensor, large_ref, "Single-stage race")
    optimized_result = benchmark_func(sum_like_v2_fp16_optimized, large_tensor, large_ref, "Optimized for fp16")
    pytorch_result = torch.sum(large_tensor, axis=(0, 2, 3), keepdim=True)

    # Check if results match (race version will likely be wrong)
    print(f"\nResults match two-stage:")
    print(f"Atomic vs Two-stage: {torch.allclose(two_stage_result, atomic_result, rtol=1e-2, atol=1e-2)}")
    print(f"Race vs Two-stage: {torch.allclose(two_stage_result, race_result, rtol=1e-2, atol=1e-2)}")
    print(f"PyTorch vs Two-stage: {torch.allclose(pytorch_result, two_stage_result, rtol=1e-2, atol=1e-2)}")
    print(f"PyTorch vs Optimized: {torch.allclose(pytorch_result, optimized_result, rtol=1e-2, atol=1e-2)}")

    # Show max differences
    print(f"\nMax differences:")
    print(f"Atomic vs Two-stage: {(two_stage_result - atomic_result).abs().max()}")
    print(f"Race vs Two-stage: {(two_stage_result - race_result).abs().max()}")
    print(f"PyTorch vs Two-stage: {(pytorch_result - two_stage_result).abs().max()}")
    print(f"PyTorch vs Optimized: {(pytorch_result - optimized_result).abs().max()}")

    # Small tensor test to see when single-stage is better
    print("\nSmall tensor test:")
    print("=" * 50)
    small_tensor = tensor_to_sum
    small_ref = ref_tensor

    print(f"Small tensor shape: {small_tensor.shape}")
    print(f"Elements per channel: {small_tensor.shape[0] * small_tensor.shape[2] * small_tensor.shape[3]}")
    print(
        f"Blocks per channel: {triton.cdiv(small_tensor.shape[0] * small_tensor.shape[2] * small_tensor.shape[3], 1024)}"
    )

    # benchmark_func(sum_like_v2_two_stage_simple, small_tensor, small_ref, "Two-stage")
    # benchmark_func(sum_like_v2_single_stage_atomic, small_tensor, small_ref, "Single-stage atomic")
    # benchmark_func(sum_like_v2_single_stage_race, small_tensor, small_ref, "Single-stage race")
    # for _ in range(10):
    #     sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)
    #     sum_like_v2_single_stage(tensor_to_sum, ref_tensor)
    #     sum_like_v2_single_stage_race(tensor_to_sum, ref_tensor)
    #     sum_like_v2_single_stage_atomic(tensor_to_sum, ref_tensor)

    # Benchmark
    # import time

    # # Two-stage
    # start = time.time()
    # for _ in range(1000):
    #     sum_like_v2_two_stage_simple(tensor_to_sum, ref_tensor)
    # torch.cuda.synchronize()
    # two_stage_time = time.time() - start

    # # Single-stage
    # start = time.time()
    # for _ in range(1000):
    #     sum_like_v2_single_stage(tensor_to_sum, ref_tensor)
    # torch.cuda.synchronize()
    # single_stage_time = time.time() - start

    # # Single-stage with race
    # start = time.time()
    # for _ in range(1000):
    #     sum_like_v2_single_stage_race(tensor_to_sum, ref_tensor)
    # torch.cuda.synchronize()
    # single_stage_race_time = time.time() - start

    # # Single-stage with atomic
    # start = time.time()
    # for _ in range(1000):
    #     sum_like_v2_single_stage_atomic(tensor_to_sum, ref_tensor)
    # torch.cuda.synchronize()
    # single_stage_atomic_time = time.time() - start

    # print(f"Two-stage time: {two_stage_time * 1000:.3f}ms")
    # print(f"Single-stage time: {single_stage_time * 1000:.3f}ms")
    # print(f"Single-stage with race time: {single_stage_race_time * 1000:.3f}ms")
    # print(f"Single-stage with atomic time: {single_stage_atomic_time * 1000:.3f}ms")
    # print(f"Single-stage is {two_stage_time / single_stage_time:.2f}x faster")
    # print(f"Single-stage with race is {two_stage_time / single_stage_race_time:.2f}x faster")
    # print(f"Single-stage with atomic is {two_stage_time / single_stage_atomic_time:.2f}x faster")

    # print("Running benchmark...")
    # # benchmark.run(show_plots=True, print_data=True, save_path="gemma.png")
