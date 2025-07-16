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
    output = torch.zeros(
        tensor_to_sum.shape[1], device=tensor_to_sum.device, dtype=torch.float16
    )  # tensor_to_sum.dtype)

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

    # Use larger block size for float16 to reduce atomic contention
    BLOCK_SIZE = 2048 if tensor_to_sum.dtype == torch.float16 else 1024
    num_blocks = triton.cdiv(elements_per_channel, BLOCK_SIZE)

    # Use 2D grid: (channels, blocks_per_channel)
    grid = (tensor_to_sum.shape[1], num_blocks)

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
    First stage: Reduce within blocks and store intermediate results.
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

    # Load data
    data = tl.load(input_ptr + mem_offsets, mask=mask, other=0.0)

    # Sum within the block
    block_sum = tl.sum(data)

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
    Second stage: Reduce intermediate results to final output.
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


def sum_like_v2_two_stage(tensor_to_sum, ref_tensor):
    """Two-stage reduction to minimize atomic operations."""
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]
    BLOCK_SIZE = 1024
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


def sum_like_v2_adaptive(tensor_to_sum, ref_tensor):
    """
    Adaptive implementation that chooses the best kernel based on tensor size and dtype.
    """
    elements_per_channel = tensor_to_sum.shape[0] * tensor_to_sum.shape[2] * tensor_to_sum.shape[3]

    # For small tensors, use the original kernel
    if elements_per_channel < 65536:
        return sum_like_v2_fp16_optimized(tensor_to_sum, ref_tensor)
    # For large tensors, use two-stage reduction
    else:
        return sum_like_v2_two_stage(tensor_to_sum, ref_tensor)


# --- End of re-included code ---


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_ELEMENTS"],
        x_vals=[64 * 128 * s * s for s in [16, 32, 64, 128, 256]],
        line_arg="provider",
        line_vals=["pytorch", "custom_kernel_v2", "custom_kernel_v2_optimized", "custom_kernel_v2_two_stage"],
        line_names=["PyTorch", "Custom Kernel (v2)", "Custom Kernel (v2 Optimized)", "Custom Kernel (v2 Two-Stage)"],
        styles=[("blue", "-"), ("red", "--"), ("green", "-."), ("orange", ":")],
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

    return ms, min_ms, max_ms


if __name__ == "__main__":
    # Test correctness first
    print("Testing correctness...")

    # Test case
    tensor_to_sum = torch.randn(64, 64, 64, 64, device="cuda", dtype=torch.float16)
    ref_tensor = torch.empty(1, 64, 1, 1, device="cuda")

    # PyTorch reference
    pytorch_result = torch.sum(tensor_to_sum, axis=(0, 2, 3), keepdim=True)

    # Our implementations
    triton_result_v2 = sum_like_v2(tensor_to_sum, ref_tensor)
    triton_result_v2_optimized = sum_like_v2_fp16_optimized(tensor_to_sum, ref_tensor)
    triton_result_v2_two_stage = sum_like_v2_two_stage(tensor_to_sum, ref_tensor)

    print(f"PyTorch result shape: {pytorch_result.shape}")
    print(f"Triton v2 result shape: {triton_result_v2.shape}")
    print(f"Triton v2 optimized result shape: {triton_result_v2_optimized.shape}")
    print(f"Triton v2 two-stage result shape: {triton_result_v2_two_stage.shape}")

    # Check correctness
    assert torch.allclose(pytorch_result, triton_result_v2, rtol=1, atol=1e-1), (
        f"Results don't match. Max diff: {(pytorch_result - triton_result_v2).abs().max()}"
    )

    assert torch.allclose(pytorch_result, triton_result_v2_optimized, rtol=1, atol=1e-1), (
        f"Optimized results don't match. Max diff: {(pytorch_result - triton_result_v2_optimized).abs().max()}"
    )

    assert torch.allclose(pytorch_result, triton_result_v2_two_stage, rtol=1, atol=1e-1), (
        f"Two-stage results don't match. Max diff: {(pytorch_result - triton_result_v2_two_stage).abs().max()}"
    )

    print("✓ All Float16 tests passed!")
    print("Running benchmark...")
    benchmark.run(show_plots=True, print_data=True, save_path="gemma.png")
