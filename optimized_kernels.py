#!/usr/bin/env python3  # noqa: CPY001
"""
Optimized per-activation-channel kernel implementations
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256, "CHANNELS_PER_BLOCK": 4}),
        triton.Config(kwargs={"BLOCK_SIZE": 256, "CHANNELS_PER_BLOCK": 8}),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "CHANNELS_PER_BLOCK": 4}),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "CHANNELS_PER_BLOCK": 8}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "CHANNELS_PER_BLOCK": 2}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "CHANNELS_PER_BLOCK": 4}),
    ],
    key=["BLOCK_SIZE", "CHANNELS_PER_BLOCK"],
)
@triton.jit
def backward_kernel_per_activation_channel_optimized(
    grad_output_ptr: torch.tensor,
    input__ptr: torch.tensor,
    input_low_ptr: torch.tensor,
    input_range_ptr: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    grad_input_ptr: torch.tensor,
    grad_low_ptr: torch.tensor,
    grad_range_ptr: torch.tensor,
    batch_size: int,
    channel_count: int,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_BLOCK: tl.constexpr,
) -> None:
    """
    Optimized kernel for per-activation-channel quantization using transposed access pattern.
    Processes multiple channels per block to improve memory coalescing.

    Key optimizations:
    1. Process multiple channels per thread block
    2. Vectorized loads where possible
    3. Better memory access pattern
    4. Reduced parameter loading overhead
    """
    # Get block and thread IDs
    batch_block_idx = tl.program_id(0)  # Which batch chunk
    channel_block_idx = tl.program_id(1)  # Which channel chunk

    # Calculate batch range for this block
    batch_start = batch_block_idx * BLOCK_SIZE
    batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE)
    batch_mask = batch_offsets < batch_size

    # Calculate channel range for this block
    channel_start = channel_block_idx * CHANNELS_PER_BLOCK
    channel_offsets = channel_start + tl.arange(0, CHANNELS_PER_BLOCK)
    channel_mask = channel_offsets < channel_count

    # Pre-load parameters for all channels in this block into registers
    input_low_vals = tl.load(input_low_ptr + channel_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    input_range_vals = tl.load(input_range_ptr + channel_offsets, mask=channel_mask, other=0.0).to(tl.float32)

    # Pre-calculate common values
    alpha = level_low / level_high
    scale_vals = (levels - 1) / input_range_vals
    reverted_range_vals = 1.0 / input_range_vals

    # Process each channel in this block
    for c_idx in range(CHANNELS_PER_BLOCK):
        if channel_start + c_idx >= channel_count:
            break

        channel_idx = channel_start + c_idx

        # Extract values for this specific channel
        input_low = tl.load(input_low_vals + c_idx)
        input_range = tl.load(input_range_vals + c_idx)
        scale = tl.load(scale_vals + c_idx)
        reverted_range = tl.load(reverted_range_vals + c_idx)
        range_low = input_low
        range_high = input_low + input_range

        # Calculate memory offsets for this channel
        # Memory layout: [batch, channel] -> offset = batch * channel_count + channel
        memory_offsets = batch_offsets * channel_count + channel_idx

        # Load input data for this channel and batch range
        grad_output = tl.load(grad_output_ptr + memory_offsets, mask=batch_mask, other=0.0).to(tl.float32)
        input_ = tl.load(input__ptr + memory_offsets, mask=batch_mask, other=0.0).to(tl.float32)

        # Forward quantization
        output = tl.clamp(input_, min=input_low, max=input_low + input_range)
        zero_point = libdevice.nearbyint(-input_low * scale)
        output -= input_low
        output *= scale
        output -= zero_point
        output = libdevice.nearbyint(output)
        output = output / scale

        # Calculate masks and gradients
        mask_lo = input_ < range_low
        mask_hi = input_ > range_high
        mask_in = ~(mask_lo | mask_hi)

        grad_input = tl.where(mask_in, grad_output, 0.0)
        grad_low = tl.where(mask_lo | mask_hi, grad_output, 0.0)
        grad_range = tl.where(
            mask_lo,
            alpha * grad_output,
            tl.where(mask_hi, grad_output, grad_output * (output - input_) * reverted_range),
        )

        # Store results
        tl.store(grad_input_ptr + memory_offsets, grad_input, mask=batch_mask)
        tl.store(grad_low_ptr + memory_offsets, grad_low, mask=batch_mask)
        tl.store(grad_range_ptr + memory_offsets, grad_range, mask=batch_mask)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE_BATCH": 256, "BLOCK_SIZE_CHANNEL": 32}),
        triton.Config(kwargs={"BLOCK_SIZE_BATCH": 512, "BLOCK_SIZE_CHANNEL": 16}),
        triton.Config(kwargs={"BLOCK_SIZE_BATCH": 1024, "BLOCK_SIZE_CHANNEL": 8}),
        triton.Config(kwargs={"BLOCK_SIZE_BATCH": 128, "BLOCK_SIZE_CHANNEL": 64}),
    ],
    key=["BLOCK_SIZE_BATCH", "BLOCK_SIZE_CHANNEL"],
)
@triton.jit
def backward_kernel_per_activation_channel_tiled(
    grad_output_ptr: torch.tensor,
    input__ptr: torch.tensor,
    input_low_ptr: torch.tensor,
    input_range_ptr: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    grad_input_ptr: torch.tensor,
    grad_low_ptr: torch.tensor,
    grad_range_ptr: torch.tensor,
    batch_size: int,
    channel_count: int,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CHANNEL: tl.constexpr,
) -> None:
    """
    Tiled approach: process rectangular tiles of [batch_chunk, channel_chunk]
    for much better memory locality and cache utilization.
    """
    # Get tile coordinates
    batch_tile_idx = tl.program_id(0)
    channel_tile_idx = tl.program_id(1)

    # Calculate batch and channel ranges for this tile
    batch_start = batch_tile_idx * BLOCK_SIZE_BATCH
    channel_start = channel_tile_idx * BLOCK_SIZE_CHANNEL

    batch_offsets = batch_start + tl.arange(0, BLOCK_SIZE_BATCH)
    channel_offsets = channel_start + tl.arange(0, BLOCK_SIZE_CHANNEL)

    batch_mask = batch_offsets < batch_size
    channel_mask = channel_offsets < channel_count

    # Load parameters for all channels in this tile
    input_low_tile = tl.load(input_low_ptr + channel_offsets, mask=channel_mask, other=0.0).to(tl.float32)
    input_range_tile = tl.load(input_range_ptr + channel_offsets, mask=channel_mask, other=0.0).to(tl.float32)

    # Pre-calculate values for all channels
    alpha = level_low / level_high
    scale_tile = (levels - 1) / input_range_tile
    reverted_range_tile = 1.0 / input_range_tile
    range_low_tile = input_low_tile
    range_high_tile = input_low_tile + input_range_tile

    # Process the tile: outer loop over channels, inner loop over batches for better locality
    for ch_idx in range(BLOCK_SIZE_CHANNEL):
        if channel_start + ch_idx >= channel_count:
            break

        channel_idx = channel_start + ch_idx

        # Extract per-channel values
        input_low = input_low_tile[ch_idx]
        scale = scale_tile[ch_idx]
        reverted_range = reverted_range_tile[ch_idx]
        range_low = range_low_tile[ch_idx]
        range_high = range_high_tile[ch_idx]

        # Calculate offsets for this channel across all batches in the tile
        memory_offsets = batch_offsets * channel_count + channel_idx

        # Load data for this channel
        grad_output = tl.load(grad_output_ptr + memory_offsets, mask=batch_mask, other=0.0).to(tl.float32)
        input_ = tl.load(input__ptr + memory_offsets, mask=batch_mask, other=0.0).to(tl.float32)

        # Forward quantization
        output = tl.clamp(input_, min=input_low, max=range_high)
        zero_point = libdevice.nearbyint(-input_low * scale)
        output -= input_low
        output *= scale
        output -= zero_point
        output = libdevice.nearbyint(output)
        output = output / scale

        # Gradient computation
        mask_lo = input_ < range_low
        mask_hi = input_ > range_high
        mask_in = ~(mask_lo | mask_hi)

        grad_input = tl.where(mask_in, grad_output, 0.0)
        grad_low = tl.where(mask_lo | mask_hi, grad_output, 0.0)
        grad_range = tl.where(
            mask_lo,
            alpha * grad_output,
            tl.where(mask_hi, grad_output, grad_output * (output - input_) * reverted_range),
        )

        # Store results
        tl.store(grad_input_ptr + memory_offsets, grad_input, mask=batch_mask)
        tl.store(grad_low_ptr + memory_offsets, grad_low, mask=batch_mask)
        tl.store(grad_range_ptr + memory_offsets, grad_range, mask=batch_mask)


def should_use_optimized_kernel(input_shape, param_shape):
    """
    Determine if we should use the optimized per-activation-channel kernel
    """
    if len(input_shape) != 2 or len(param_shape) != 2:
        return False

    batch_size, channel_count = input_shape
    param_batch, param_channels = param_shape

    # Check if it's per-activation-channel pattern
    if param_batch != 1 or param_channels != channel_count:
        return False

    # Use optimized kernel for large channel counts
    # For [2048, 128256]: batch_size=2048, channel_count=128256
    # Threshold: prefer optimized kernel when channels > 1024 and batch < channels
    return channel_count > 1024 and batch_size < channel_count * 0.1


if __name__ == "__main__":
    print("Optimized per-activation-channel kernels defined")
    print("Key optimizations:")
    print("1. backward_kernel_per_activation_channel_optimized: Multi-channel per block")
    print("2. backward_kernel_per_activation_channel_tiled: Rectangular tiling")
    print("3. should_use_optimized_kernel: Heuristic for kernel selection")
