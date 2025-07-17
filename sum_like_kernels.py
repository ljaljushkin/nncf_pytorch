# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Solution for sum_like reduction in Triton for 4D tensors

def sum_like(tensor_to_sum: Tensor[4, 16, 16, 16], ref_tensor: Tensor[1,16,1,1]) -> Tensor[1,16,1,1]

### Approaches:

1. **Direct Atomic Approach**: Use atomic operations to accumulate results
   - Simple but suffers from atomic contention
   - Works well for small tensors

2. **Two-Stage Reduction**: Split into block-level and final reduction
   - Eliminates atomic contention entirely
   - Best for large tensors

### Performance optimization for float16:
- Use fp32 accumulation internally, convert to fp16 for storage
- Larger block sizes (2048 vs 1024) reduce atomic contention
- Two-stage approach avoids atomic operations entirely

The two-stage approach is generally the best solution for large-scale reductions in Triton.
"""

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
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)  # .to(tl.float32)

    # Sum within the block and use atomic add for final result
    block_sum = tl.sum(data)  # .to(tl.float16)

    # Use atomic add for final result
    tl.atomic_add(output_ptr + channel_idx, block_sum)


def sum_like_baseline(tensor_to_sum, ref_tensor):
    """Float16-optimized Triton implementation with hierarchical reduction."""
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Calculate grid dimensions for parallel processing
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]
    # Use larger block size for float16 to reduce atomic contention
    BLOCK_SIZE = 2048 if tensor_to_sum.dtype == torch.float16 else 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Use 2D grid: (channels, blocks_per_channel)
    grid = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_coalesced[grid](
        tensor_to_sum, output, *tensor_to_sum.shape, *tensor_to_sum.stride(), BLOCK_SIZE=BLOCK_SIZE
    )

    return output.reshape(ref_tensor.shape)


@triton.jit
def _sum_like_4d_kernel_two_stage(
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
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)  # .to(tl.float32)

    # Sum within the block
    block_sum = tl.sum(data)  # .to(tl.float16)

    # Store intermediate result
    temp_offset = channel_idx * num_blocks_per_channel + block_idx
    tl.store(temp_ptr + temp_offset, block_sum)


@triton.jit
def _sum_like_4d_kernel_second_stage(
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


def sum_like_two_stage(tensor_to_sum, ref_tensor):
    """Simplified two-stage reduction for 4D tensors only - much faster."""
    # Calculate elements per channel (dimensions 0, 2, 3)
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]

    # Calculate grid dimensions
    BLOCK_SIZE = 2048
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Create intermediate storage
    temp_size = tensor_to_sum.shape[1] * num_blocks
    temp_storage = torch.zeros(temp_size, device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # First stage: Block-level reduction
    grid1 = (tensor_to_sum.shape[1], num_blocks)
    _sum_like_4d_kernel_two_stage[grid1](
        tensor_to_sum, temp_storage, *tensor_to_sum.shape, *tensor_to_sum.stride(), num_blocks, BLOCK_SIZE=BLOCK_SIZE
    )

    # Second stage: Final reduction
    output = torch.zeros(tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=tensor_to_sum.dtype)

    # Use a block size that can handle the number of intermediate results
    reduction_block_size = min(1024, triton.next_power_of_2(num_blocks))

    grid2 = (tensor_to_sum.shape[1],)
    _sum_like_4d_kernel_second_stage[grid2](
        temp_storage, output, tensor_to_sum.shape[1], num_blocks, BLOCK_SIZE=reduction_block_size
    )

    return output.reshape(ref_tensor.shape)
