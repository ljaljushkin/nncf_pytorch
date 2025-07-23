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

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from nncf.torch.utils import sum_like


def get_optimal_grid_for_per_channel(scale_count: int, elements_per_scale: int, block_size: int) -> tuple[int, int]:
    """
    Calculate optimal 2D grid size for per-channel quantization based on empirical performance.

    :param scale_count: Number of channels/scales
    :param elements_per_scale: Number of elements per channel
    :param block_size: Block size from autotune
    :return: Tuple of (grid_x, grid_y) optimized for performance
    """
    grid_x = scale_count

    # Performance-based heuristics:
    # 1. For large elements_per_scale, prefer single block per channel when possible
    # 2. Use multiple blocks only when necessary or when it provides clear benefit

    if elements_per_scale <= block_size:
        # Single block can handle all elements in the channel
        grid_y = 1
    elif elements_per_scale <= 4 * block_size:
        # Use minimal blocks for medium-sized channels
        grid_y = triton.cdiv(elements_per_scale, block_size)
    else:
        # For very large channels, balance between memory coalescing and parallelism
        # Prefer larger blocks with fewer launches
        if block_size >= 1024:
            # Large block size: prefer single block per channel for better coalescing
            grid_y = 1 if elements_per_scale <= 8 * block_size else triton.cdiv(elements_per_scale, block_size)
        else:
            # Smaller block size: use more blocks but limit to reasonable number
            max_blocks_per_channel = min(64, triton.cdiv(elements_per_scale, block_size))
            grid_y = max_blocks_per_channel

    return (grid_x, grid_y)


def get_2d_grid_size_for_per_channel(scale_count: int) -> tuple[int, int]:
    """
    Calculate 2D grid size for per-channel quantization, following CUDA's approach.

    :param scale_count: Number of channels/scales
    :return: Tuple of (grid_x, grid_y) where grid_x=scale_count and grid_y=blocks_per_scale
    """
    # Constants from CUDA implementation (approximate values for Triton)
    TARGET_SM_COUNT = 72  # Typical for modern GPUs
    TARGET_THREADS_PER_SM = 1024
    WARP_SIZE = 32
    MAX_WARPS_PER_BLOCK = 32  # 1024 threads / 32 threads per warp
    MAX_GRID_SIZE_Y = 65535

    # X corresponds to scale count, Y determined to hit thread-per-SM target
    grid_size_x = scale_count
    available_threads_per_scale = int((TARGET_SM_COUNT * TARGET_THREADS_PER_SM) / grid_size_x)

    # Align to warp boundaries
    available_warps_per_scale = (available_threads_per_scale + WARP_SIZE - 1) // WARP_SIZE
    blocks_per_scale = max(1, available_warps_per_scale // MAX_WARPS_PER_BLOCK)
    grid_size_y = min(blocks_per_scale, MAX_GRID_SIZE_Y)

    return (grid_size_x, grid_size_y)


def get_4d_tensor_meta(x: torch.tensor) -> torch.tensor:
    """
    Helper function for meta information creation.

    :param x: Torch tensor.
    :returns: Torch tensor as meta with 4D shape + tensor.
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


@triton.jit
def read_shape(meta: torch.tensor) -> tuple[tl.tensor]:
    """
    Helper kernel for the shapes loading from meta tensor.

    :param meta: Torch tensor with meta information (shape + stride).
    :returns: Tuple of 4D shapes.
    """
    s0 = tl.load(meta + 0)
    s1 = tl.load(meta + 1)
    s2 = tl.load(meta + 2)
    s3 = tl.load(meta + 3)
    return s0, s1, s2, s3


@triton.jit
def read_stride(meta: torch.tensor) -> tuple[tl.tensor]:
    """
    Helper kernel for the strides loading from meta tensor.

    :param meta: Torch tensor with meta information (shape + stride).
    :returns: Tuple of 4D strides.
    """
    st0 = tl.load(meta + 4)
    st1 = tl.load(meta + 5)
    st2 = tl.load(meta + 6)
    st3 = tl.load(meta + 7)
    return st0, st1, st2, st3


@triton.jit
def calculate_total_elements(meta: torch.tensor) -> tl.tensor:
    """
    Helper kernel for the total elements calculation based on meta tensor.

    :param meta: Torch tensor with meta information (shape + stride).
    :returns: Total number of elements for mask calculation.
    """
    s0, s1, s2, s3 = read_shape(meta)
    return s0 * s1 * s2 * s3


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def forward_kernel(
    input__ptr: torch.tensor,
    input__meta: torch.tensor,
    input_low_ptr: torch.tensor,
    input_low_meta: torch.tensor,
    input_range_ptr: torch.tensor,
    input_range_meta: torch.tensor,
    levels: int,
    output_ptr: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    "
    Forward kernel implementation based on reference formula - nncf/torch/quantization/reference.py

    :param input__ptr: Memory pointer to input_ torch.tensor.
    :param input_low_ptr: Memory pointer to input_low torch.tensor.
    :param input_range_ptr: Memory pointer to input_range torch.tensor.
    :param levels: Levels value as scalar.
    :param output_ptr: Memory pointer to output torch.tensor that would be filled with return value.
    :param last_dim: Scalar to calculate loading offset for input_low/range pointers.
    :param is_per_tensor: Bool value for offset correction in per-tensor case.
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input__s0, input__s1, input__s2, input__s3 = read_shape(input__meta)
    input__elements = input__s0 * input__s1 * input__s2 * input__s3

    tmp = offsets
    i3 = tmp % input__s3
    tmp //= input__s3
    i2 = tmp % input__s2
    tmp //= input__s2
    i1 = tmp % input__s1
    tmp //= input__s1
    i0 = tmp % input__s0

    input_low_st0, input_low_st1, input_low_st2, input_low_st3 = read_stride(input_low_meta)
    input_low_offset = i0 * input_low_st0 + i1 * input_low_st1 + i2 * input_low_st2 + i3 * input_low_st3
    input_low_elements = calculate_total_elements(input_low_meta)

    input_range_st0, input_range_st1, input_range_st2, input_range_st3 = read_stride(input_range_meta)
    input_range_offset = i0 * input_range_st0 + i1 * input_range_st1 + i2 * input_range_st2 + i3 * input_range_st3
    input_range_elements = calculate_total_elements(input_range_meta)

    input_ = tl.load(input__ptr + offsets, mask=offsets < input__elements).to(tl.float32)
    input_low = tl.load(input_low_ptr + input_low_offset, mask=input_low_offset < input_low_elements).to(tl.float32)
    input_range = tl.load(input_range_ptr + input_range_offset, mask=input_range_offset < input_range_elements).to(
        tl.float32
    )

    scale = (levels - 1) / input_range

    output = tl.clamp(input_, min=input_low, max=input_low + input_range)

    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    tl.store(output_ptr + offsets, output, mask=offsets < input__elements)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def backward_kernel(
    grad_output_ptr: torch.tensor,
    grad_output_meta: torch.tensor,
    input__ptr: torch.tensor,
    input__meta: torch.tensor,
    input_low_ptr: torch.tensor,
    input_low_meta: torch.tensor,
    input_range_ptr: torch.tensor,
    input_range_meta: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    grad_input_ptr: torch.tensor,
    grad_low_ptr: torch.tensor,
    grad_range_ptr: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    "
    Backward kernel implementation based on reference formula - nncf/torch/quantization/reference.py
    :param grad_output_ptr: Memory pointer to grad_output torch.tensor.
    :param input__ptr: Memory pointer to input_ torch.tensor.
    :param input_low_ptr: Memory pointer to input_low torch.tensor.
    :param input_range_ptr: Memory pointer to input_range torch.tensor.
    :param levels: Levels value as scalar.
    :param level_low: Level low value as scalar.
    :param level_high: Level high value as scalar.
    :param grad_input_ptr: Memory pointer to grad_input torch.tensor that would be filled with return value.
    :param grad_low_ptr: Memory pointer to grad_low torch.tensor that would be filled with return value.
    :param grad_range_ptr: Memory pointer to grad_range torch.tensor that would be filled with return value.
    :param last_dim: Scalar to calculate loading offset for input_low/range pointers.
    :param is_per_tensor: Bool value for offset correction in per-tensor case.
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input__s0, input__s1, input__s2, input__s3 = read_shape(input__meta)
    input__elements = input__s0 * input__s1 * input__s2 * input__s3

    tmp = offsets
    i3 = tmp % input__s3
    tmp //= input__s3
    i2 = tmp % input__s2
    tmp //= input__s2
    i1 = tmp % input__s1
    tmp //= input__s1
    i0 = tmp % input__s0

    input_low_st0, input_low_st1, input_low_st2, input_low_st3 = read_stride(input_low_meta)
    input_low_offset = i0 * input_low_st0 + i1 * input_low_st1 + i2 * input_low_st2 + i3 * input_low_st3
    input_low_elements = calculate_total_elements(input_low_meta)

    input_range_st0, input_range_st1, input_range_st2, input_range_st3 = read_stride(input_range_meta)
    input_range_offset = i0 * input_range_st0 + i1 * input_range_st1 + i2 * input_range_st2 + i3 * input_range_st3
    input_range_elements = calculate_total_elements(input_range_meta)

    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < input__elements).to(tl.float32)
    input_ = tl.load(input__ptr + offsets, mask=offsets < input__elements).to(tl.float32)
    input_low = tl.load(input_low_ptr + input_low_offset, mask=input_low_offset < input_low_elements).to(tl.float32)
    input_range = tl.load(input_range_ptr + input_range_offset, mask=input_range_offset < input_range_elements).to(
        tl.float32
    )

    mask_hi = input_ > (input_low + input_range)
    mask_hi = mask_hi.to(tl.float32)
    mask_lo = input_ < input_low
    mask_lo = mask_lo.to(tl.float32)

    mask_in = 1 - mask_hi - mask_lo

    scale = (levels - 1) / input_range
    output = tl.clamp(input_, min=input_low, max=input_low + input_range)
    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    # Signed range calculation
    input_range_above_zero = input_range > 0
    input_range_below_zero = input_range < 0
    range_sign = input_range_above_zero - input_range_below_zero
    # Reciprocal calculation
    reciprocal = 1 / (input_range * range_sign)
    err = (output - input_) * reciprocal
    grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)

    grad_input = grad_output * mask_in

    grad_low = grad_output * (mask_hi + mask_lo)

    tl.store(grad_input_ptr + offsets, grad_input, mask=offsets < input__elements)
    tl.store(grad_low_ptr + offsets, grad_low, mask=offsets < input__elements)
    tl.store(grad_range_ptr + offsets, grad_range, mask=offsets < input__elements)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def backward_kernel_per_channel_2d(
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
    elements_per_scale: int,  # Elements per channel (e.g., 128256)
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    2D grid per-channel backward kernel following CUDA's q_scale_per_weight_channel_cuda_backward_kernel pattern.
    Optimized to handle both single-block-per-channel and multi-block-per-channel cases efficiently.

    Grid organization:
    - program_id(0): scale/channel index (equivalent to blockIdx.x in CUDA)
    - program_id(1): block index within channel (equivalent to blockIdx.y in CUDA)
    """
    # Get 2D program IDs - equivalent to CUDA's blockIdx.x and blockIdx.y
    scale_idx = tl.program_id(0)  # Channel/scale index
    per_scale_block_idx = tl.program_id(1)  # Block within this channel

    # Calculate thread index within this channel (equivalent to CUDA's per_scale_tidx)
    thread_idx = tl.arange(0, BLOCK_SIZE)
    per_scale_tidx = per_scale_block_idx * BLOCK_SIZE + thread_idx

    # Calculate base offset for this channel's data
    base_offset = scale_idx * elements_per_scale
    offsets = base_offset + per_scale_tidx

    # Mask for valid elements within this channel
    mask = per_scale_tidx < elements_per_scale

    # Load input data with channel-aligned access (much better coalescing)
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask).to(tl.float32)
    input_ = tl.load(input__ptr + offsets, mask=mask).to(tl.float32)

    # Load per-channel parameters (one per scale/channel)
    # This is highly efficient as all threads in the block use the same parameter
    input_low = tl.load(input_low_ptr + scale_idx).to(tl.float32)
    input_range = tl.load(input_range_ptr + scale_idx).to(tl.float32)

    # Quantization forward pass (same as CUDA's fakeQuantize)
    scale = (levels - 1) / input_range
    output = tl.clamp(input_, min=input_low, max=input_low + input_range)
    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    # Gradient calculations (same as CUDA's calcGrad)
    alpha = level_low / level_high
    range_low = input_low
    range_high = input_low + input_range
    reverted_range = 1 / input_range

    # Calculate masks for different regions
    mask_lo = input_ < range_low
    mask_hi = input_ > range_high
    mask_in = ~(mask_lo | mask_hi)

    # Calculate gradients following CUDA logic
    grad_input = tl.where(mask_in, grad_output, 0.0)
    grad_low = tl.where(mask_lo | mask_hi, grad_output, 0.0)
    grad_range = tl.where(
        mask_lo, alpha * grad_output, tl.where(mask_hi, grad_output, grad_output * (output - input_) * reverted_range)
    )

    # Store results with channel-aligned access
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
    tl.store(grad_low_ptr + offsets, grad_low, mask=mask)
    tl.store(grad_range_ptr + offsets, grad_range, mask=mask)


def forward(input_: torch.tensor, input_low: torch.tensor, input_range: torch.tensor, levels: int) -> torch.tensor:
    """
    Wrapper for the forward kernel.
    It contains preparation steps like output memory allocation via tensor creation,
    additional values calculation and CUDA context management based on the input tensors.
    :param input_: input_ as torch.tensor.
    :param input_low: input_low as torch.tensor.
    :param input_range: input_range as torch.tensor.
    :param levels: Levels value.
    :return: Calculated output value as torch.tensor.
    """
    output = torch.empty_like(input_)

    input__meta = get_4d_tensor_meta(input_)
    input_low_meta = get_4d_tensor_meta(input_low)
    input_range_meta = get_4d_tensor_meta(input_range)

    with torch.cuda.device(input_.device):
        grid = lambda meta: (triton.cdiv(input_.numel(), meta["BLOCK_SIZE"]),)
        forward_kernel[grid](
            input_,
            input__meta,
            input_low,
            input_low_meta,
            input_range,
            input_range_meta,
            levels,
            output,
        )

    return output


def backward(
    grad_output: torch.tensor,
    input_: torch.tensor,
    input_low: torch.tensor,
    input_range: torch.tensor,
    levels: int,
    level_low: int,
    level_high: int,
    is_asymmetric: bool = False,
) -> tuple[torch.tensor]:
    """
    Wrapper for the backward kernel with optimized 2D grid for per-channel quantization.
    It contains preparation steps like output memory allocation via tensor creation,
    additional values calculation and CUDA context management based on the input tensors.
    :param grad_output: input_ as torch.tensor.
    :param input_: input_ as torch.tensor.
    :param input_low: input_low as torch.tensor.
    :param input_range: input_range as torch.tensor.
    :param levels: Levels value.
    :return: Calculated grad_input, grad_low and grad_range as tuple of torch.tensor values.
    """
    grad_input = torch.empty_like(input_)
    grad_low = torch.empty_like(input_)
    grad_range = torch.empty_like(input_)

    # Detect per-channel quantization patterns
    use_2d_grid = False
    elements_per_scale = 1
    scale_count = 1

    # Check for per-weight-channel pattern: input[N, C, ...] with params[N, 1, ...]
    if (
        len(input_.shape) >= 2
        and len(input_low.shape) >= 2
        and input_low.shape[0] == input_.shape[0]
        and input_low.shape[1] == 1
        and input_range.shape[0] == input_.shape[0]
        and input_range.shape[1] == 1
    ):
        use_2d_grid = True
        scale_count = input_.shape[0]  # Number of channels
        elements_per_scale = input_.numel() // scale_count

    # Check for per-activation-channel pattern: input[N, C, ...] with params[1, C, ...]
    elif (
        len(input_.shape) >= 2
        and len(input_low.shape) >= 2
        and input_low.shape[0] == 1
        and input_low.shape[1] == input_.shape[1]
        and input_range.shape[0] == 1
        and input_range.shape[1] == input_.shape[1]
    ):
        use_2d_grid = True
        scale_count = input_.shape[1]  # Number of channels
        elements_per_scale = input_.numel() // scale_count

    with torch.cuda.device(input_.device):
        if use_2d_grid:  # Use 2D grid for per-channel cases
            # Use performance-optimized grid calculation
            grid = lambda meta: get_optimal_grid_for_per_channel(scale_count, elements_per_scale, meta["BLOCK_SIZE"])

            backward_kernel_per_channel_2d[grid](
                grad_output,
                input_,
                input_low,
                input_range,
                levels,
                level_low,
                level_high,
                grad_input,
                grad_low,
                grad_range,
                elements_per_scale,
            )
        else:
            # Use original 1D grid kernel for single-scale or small tensors
            grad_output_meta = get_4d_tensor_meta(grad_output)
            input__meta = get_4d_tensor_meta(input_)
            input_low_meta = get_4d_tensor_meta(input_low)
            input_range_meta = get_4d_tensor_meta(input_range)

            grid = lambda meta: (triton.cdiv(input_.numel(), meta["BLOCK_SIZE"]),)
            backward_kernel[grid](
                grad_output,
                grad_output_meta,
                input_,
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
                grad_range,
            )

        # Temporary solution until possibility of sum like kernel in Triton would be confirmed.
        grad_low = sum_like(grad_low, input_low)
        grad_range = sum_like(grad_range, input_range)

    return grad_input, grad_low, grad_range
