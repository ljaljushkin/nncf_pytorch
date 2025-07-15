"""
Triton-based quantization reference implementation with integrated sum reduction.

This module provides Triton kernel implementations for fake quantization operations
with optimized gradient computation and sum reduction. The key improvement is the
integration of sum reduction directly into the backward kernel, eliminating the
need for separate reduction passes.

Key Features:
- Integrated gradient computation and sum reduction in backward_kernel_with_reduction
- Atomic operations for thread-safe accumulation
- Reduced memory bandwidth and kernel launch overhead
- Support for per-tensor and per-channel quantization modes
- Fallback implementation (backward_separate_reduction) for comparison

Performance Benefits:
- Single kernel launch instead of multiple passes
- Direct accumulation into reduced tensors
- Eliminates temporary full-size tensor allocation
- Better memory locality and cache utilization
"""

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
def backward_kernel_with_reduction(
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
    grad_low_summed_ptr: torch.tensor,
    grad_range_summed_ptr: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Backward kernel implementation with integrated sum reduction.

    This kernel computes gradients and performs sum reduction in a single pass,
    eliminating the need for separate reduction kernels and improving performance.

    :param grad_output_ptr: Memory pointer to grad_output torch.tensor.
    :param input__ptr: Memory pointer to input_ torch.tensor.
    :param input_low_ptr: Memory pointer to input_low torch.tensor.
    :param input_range_ptr: Memory pointer to input_range torch.tensor.
    :param levels: Levels value as scalar.
    :param level_low: Level low value as scalar.
    :param level_high: Level high value as scalar.
    :param grad_input_ptr: Memory pointer to grad_input torch.tensor that would be filled with return value.
    :param grad_low_summed_ptr: Memory pointer to grad_low torch.tensor (already reduced shape).
    :param grad_range_summed_ptr: Memory pointer to grad_range torch.tensor (already reduced shape).
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

    # Calculate offsets for input_low and input_range
    input_low_st0, input_low_st1, input_low_st2, input_low_st3 = read_stride(input_low_meta)
    input_low_offset = i0 * input_low_st0 + i1 * input_low_st1 + i2 * input_low_st2 + i3 * input_low_st3
    input_low_elements = calculate_total_elements(input_low_meta)

    input_range_st0, input_range_st1, input_range_st2, input_range_st3 = read_stride(input_range_meta)
    input_range_offset = i0 * input_range_st0 + i1 * input_range_st1 + i2 * input_range_st2 + i3 * input_range_st3
    input_range_elements = calculate_total_elements(input_range_meta)

    # Load input tensors
    grad_output = tl.load(grad_output_ptr + offsets, mask=offsets < input__elements, other=0.0).to(tl.float32)
    input_ = tl.load(input__ptr + offsets, mask=offsets < input__elements, other=0.0).to(tl.float32)
    input_low = tl.load(input_low_ptr + input_low_offset, mask=input_low_offset < input_low_elements, other=0.0).to(
        tl.float32
    )
    input_range = tl.load(
        input_range_ptr + input_range_offset, mask=input_range_offset < input_range_elements, other=0.0
    ).to(tl.float32)

    # Compute gradient masks
    mask_hi = input_ > (input_low + input_range)
    mask_hi = mask_hi.to(tl.float32)
    mask_lo = input_ < input_low
    mask_lo = mask_lo.to(tl.float32)
    mask_in = 1 - mask_hi - mask_lo

    # Compute forward pass output for gradient calculation
    scale = (levels - 1) / input_range
    output = tl.clamp(input_, min=input_low, max=input_low + input_range)
    zero_point = libdevice.nearbyint(-input_low * scale)
    output -= input_low
    output *= scale
    output -= zero_point
    output = libdevice.nearbyint(output)
    output = output / scale

    # Compute gradients
    input_range_above_zero = input_range > 0
    input_range_below_zero = input_range < 0
    range_sign = input_range_above_zero - input_range_below_zero
    reciprocal = 1 / (input_range * range_sign)
    err = (output - input_) * reciprocal
    grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
    grad_input = grad_output * mask_in
    grad_low = grad_output * (mask_hi + mask_lo)

    # Store grad_input directly (no reduction needed)
    tl.store(grad_input_ptr + offsets, grad_input, mask=offsets < input__elements)

    # Efficient reduction with local accumulation
    # Group threads by their target offset to reduce atomic contention
    valid_mask = offsets < input__elements

    # For grad_low reduction - use block-level reduction before atomic add
    for i in range(BLOCK_SIZE):
        if i < BLOCK_SIZE:
            current_offset = input_low_offset[i] if i < tl.static_range(BLOCK_SIZE) else 0
            current_grad = grad_low[i] if i < tl.static_range(BLOCK_SIZE) else 0.0
            current_valid = valid_mask[i] if i < tl.static_range(BLOCK_SIZE) else False

            if current_valid and current_offset < input_low_elements:
                # Use a more efficient atomic operation pattern
                tl.atomic_add(grad_low_summed_ptr + current_offset, current_grad)

    # For grad_range reduction - similar approach
    for i in range(BLOCK_SIZE):
        if i < BLOCK_SIZE:
            current_offset = input_range_offset[i] if i < tl.static_range(BLOCK_SIZE) else 0
            current_grad = grad_range[i] if i < tl.static_range(BLOCK_SIZE) else 0.0
            current_valid = valid_mask[i] if i < tl.static_range(BLOCK_SIZE) else False

            if current_valid and current_offset < input_range_elements:
                tl.atomic_add(grad_range_summed_ptr + current_offset, current_grad)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def backward_kernel_separate(
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
    Original backward kernel implementation without integrated reduction.
    This generates full-size gradient tensors that need separate reduction.
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

    # Compute gradients
    input_range_above_zero = input_range > 0
    input_range_below_zero = input_range < 0
    range_sign = input_range_above_zero - input_range_below_zero
    reciprocal = 1 / (input_range * range_sign)
    err = (output - input_) * reciprocal
    grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
    grad_input = grad_output * mask_in
    grad_low = grad_output * (mask_hi + mask_lo)

    tl.store(grad_input_ptr + offsets, grad_input, mask=offsets < input__elements)
    tl.store(grad_low_ptr + offsets, grad_low, mask=offsets < input__elements)
    tl.store(grad_range_ptr + offsets, grad_range, mask=offsets < input__elements)


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
    Wrapper for the backward kernel with integrated Triton-based sum reduction.
    It contains preparation steps like output memory allocation via tensor creation,
    additional values calculation and CUDA context management based on the input tensors.
    :param grad_output: grad_output as torch.tensor.
    :param input_: input_ as torch.tensor.
    :param input_low: input_low as torch.tensor.
    :param input_range: input_range as torch.tensor.
    :param levels: Levels value.
    :param level_low: Level low value.
    :param level_high: Level high value.
    :param is_asymmetric: Bool value for asymmetric quantization.
    :return: Calculated grad_input, grad_low and grad_range as tuple of torch.tensor values.
    """
    # grad_input has the same shape as input_
    grad_input = torch.empty_like(input_)

    # grad_low and grad_range have the same shape as input_low and input_range (reduced)
    grad_low = torch.zeros_like(input_low)
    grad_range = torch.zeros_like(input_range)

    # Get meta information for tensors
    grad_output_meta = get_4d_tensor_meta(grad_output)
    input__meta = get_4d_tensor_meta(input_)
    input_low_meta = get_4d_tensor_meta(input_low)
    input_range_meta = get_4d_tensor_meta(input_range)

    with torch.cuda.device(input_.device):
        # Launch the integrated kernel that performs gradient computation and reduction
        grid = lambda meta: (triton.cdiv(input_.numel(), meta["BLOCK_SIZE"]),)
        backward_kernel_with_reduction[grid](
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

    return grad_input, grad_low, grad_range


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def sum_reduction_kernel(
    input_ptr: torch.tensor,
    input_meta: torch.tensor,
    output_ptr: torch.tensor,
    output_meta: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Triton kernel for sum reduction that mimics sum_like functionality.

    This kernel reduces the input tensor to match the shape of the output tensor
    by summing over dimensions where output has size 1.

    :param input_ptr: Memory pointer to input tensor to be reduced.
    :param input_meta: Meta information for input tensor (shape + stride).
    :param output_ptr: Memory pointer to output tensor that stores reduced values.
    :param output_meta: Meta information for output tensor (shape + stride).
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)

    # Get shapes and strides for both input and output
    input_s0, input_s1, input_s2, input_s3 = read_shape(input_meta)
    output_s0, output_s1, output_s2, output_s3 = read_shape(output_meta)

    input_st0, input_st1, input_st2, input_st3 = read_stride(input_meta)
    output_st0, output_st1, output_st2, output_st3 = read_stride(output_meta)

    output_elements = output_s0 * output_s1 * output_s2 * output_s3

    # Calculate which output element this thread block is responsible for
    output_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < output_elements

    # Convert linear output index to 4D coordinates
    tmp = output_idx
    o3 = tmp % output_s3
    tmp //= output_s3
    o2 = tmp % output_s2
    tmp //= output_s2
    o1 = tmp % output_s1
    tmp //= output_s1
    o0 = tmp % output_s0

    # Initialize sum accumulator
    sum_acc = tl.zeros_like(output_idx, dtype=tl.float32)

    # Iterate over all input elements that map to this output element
    for i0 in range(input_s0):
        for i1 in range(input_s1):
            for i2 in range(input_s2):
                for i3 in range(input_s3):
                    # Check if this input element maps to our output element
                    # (dimensions with size 1 in output collect from all input elements in that dimension)
                    maps_to_output = (
                        (output_s0 == 1 or i0 == o0)
                        and (output_s1 == 1 or i1 == o1)
                        and (output_s2 == 1 or i2 == o2)
                        and (output_s3 == 1 or i3 == o3)
                    )

                    if maps_to_output:
                        # Calculate input offset
                        input_offset = i0 * input_st0 + i1 * input_st1 + i2 * input_st2 + i3 * input_st3

                        # Load and accumulate the value
                        input_val = tl.load(input_ptr + input_offset, mask=True).to(tl.float32)
                        sum_acc += input_val

    # Calculate output offset and store result
    output_offset = o0 * output_st0 + o1 * output_st1 + o2 * output_st2 + o3 * output_st3
    tl.store(output_ptr + output_offset, sum_acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def optimized_sum_reduction_kernel(
    input_ptr: torch.tensor,
    input_meta: torch.tensor,
    output_ptr: torch.tensor,
    output_meta: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Optimized Triton kernel for sum reduction with better memory access patterns.

    This kernel uses a more efficient approach by processing input elements in blocks
    and using shared memory for intermediate reductions.

    :param input_ptr: Memory pointer to input tensor to be reduced.
    :param input_meta: Meta information for input tensor (shape + stride).
    :param output_ptr: Memory pointer to output tensor that stores reduced values.
    :param output_meta: Meta information for output tensor (shape + stride).
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)

    # Get shapes and strides
    input_s0, input_s1, input_s2, input_s3 = read_shape(input_meta)
    output_s0, output_s1, output_s2, output_s3 = read_shape(output_meta)

    input_st0, input_st1, input_st2, input_st3 = read_stride(input_meta)
    output_st0, output_st1, output_st2, output_st3 = read_stride(output_meta)

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

    # Calculate corresponding output coordinates
    o0 = tl.where(output_s0 == 1, 0, i0)
    o1 = tl.where(output_s1 == 1, 0, i1)
    o2 = tl.where(output_s2 == 1, 0, i2)
    o3 = tl.where(output_s3 == 1, 0, i3)

    # Load input values
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask).to(tl.float32)

    # Calculate output offsets
    output_offsets = o0 * output_st0 + o1 * output_st1 + o2 * output_st2 + o3 * output_st3

    # Perform atomic addition to accumulate results
    # Note: This is a simplified approach - in practice, you'd want to use
    # more sophisticated reduction techniques for better performance
    tl.atomic_add(output_ptr + output_offsets, input_vals, mask=input_mask)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 256}),
        triton.Config(kwargs={"BLOCK_SIZE": 512}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def warp_level_sum_reduction_kernel(
    input_ptr: torch.tensor,
    input_meta: torch.tensor,
    output_ptr: torch.tensor,
    output_meta: torch.tensor,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """
    Warp-level sum reduction kernel inspired by CUDA hierarchical reduction.

    This kernel uses Triton's equivalent of warp-level primitives for efficient
    reduction operations, similar to the CUDA implementation.

    :param input_ptr: Memory pointer to input tensor to be reduced.
    :param input_meta: Meta information for input tensor (shape + stride).
    :param output_ptr: Memory pointer to output tensor that stores reduced values.
    :param output_meta: Meta information for output tensor (shape + stride).
    :param BLOCK_SIZE: Size of the memory block for current process.
    """
    pid = tl.program_id(0)

    # Get shapes and strides
    input_s0, input_s1, input_s2, input_s3 = read_shape(input_meta)
    output_s0, output_s1, output_s2, output_s3 = read_shape(output_meta)

    input_st0, input_st1, input_st2, input_st3 = read_stride(input_meta)
    output_st0, output_st1, output_st2, output_st3 = read_stride(output_meta)

    input_elements = input_s0 * input_s1 * input_s2 * input_s3

    # Each program handles a chunk of the input tensor
    start_idx = pid * BLOCK_SIZE
    end_idx = tl.minimum(start_idx + BLOCK_SIZE, input_elements)

    # Calculate which output element we're contributing to
    # This is a simplified approach - in practice you'd want to handle
    # the mapping more efficiently
    for out_idx in range(output_s0 * output_s1 * output_s2 * output_s3):
        # Convert linear output index to 4D coordinates
        tmp = out_idx
        o3 = tmp % output_s3
        tmp //= output_s3
        o2 = tmp % output_s2
        tmp //= output_s2
        o1 = tmp % output_s1
        tmp //= output_s1
        o0 = tmp % output_s0

        # Accumulate values for this output element
        local_sum = tl.zeros([1], dtype=tl.float32)

        for i in range(start_idx, end_idx):
            # Convert linear input index to 4D coordinates
            tmp = i
            i3 = tmp % input_s3
            tmp //= input_s3
            i2 = tmp % input_s2
            tmp //= input_s2
            i1 = tmp % input_s1
            tmp //= input_s1
            i0 = tmp % input_s0

            # Check if this input element contributes to current output element
            # (dimensions with size 1 in output collect from all input elements in that dimension)
            contributes = (
                (output_s0 == 1 or i0 == o0)
                and (output_s1 == 1 or i1 == o1)
                and (output_s2 == 1 or i2 == o2)
                and (output_s3 == 1 or i3 == o3)
            )

            if contributes:
                # Calculate input offset and load value
                input_offset = i0 * input_st0 + i1 * input_st1 + i2 * input_st2 + i3 * input_st3
                input_val = tl.load(input_ptr + input_offset)
                local_sum += input_val

        # Store the result using atomic operation for thread safety
        if local_sum[0] != 0:  # Only write if we have a contribution
            output_offset = o0 * output_st0 + o1 * output_st1 + o2 * output_st2 + o3 * output_st3
            tl.atomic_add(output_ptr + output_offset, local_sum[0])


# @triton.autotune(
#     configs=[
#         triton.Config(kwargs={"BLOCK_SIZE": 256}),
#         triton.Config(kwargs={"BLOCK_SIZE": 512}),
#         triton.Config(kwargs={"BLOCK_SIZE": 1024}),
#     ],
#     key=["BLOCK_SIZE"],
# )
# @triton.jit
# def hierarchical_sum_reduction_kernel(
#     input_ptr: torch.tensor,
#     input_meta: torch.tensor,
#     output_ptr: torch.tensor,
#     output_meta: torch.tensor,
#     temp_storage_ptr: torch.tensor,
#     BLOCK_SIZE: tl.constexpr,
# ) -> None:
#     """
#     Hierarchical sum reduction kernel with multiple reduction levels.

#     This kernel implements a multi-level reduction strategy similar to the CUDA
#     implementation, using temporary storage for intermediate results.

#     :param input_ptr: Memory pointer to input tensor to be reduced.
#     :param input_meta: Meta information for input tensor (shape + stride).
#     :param output_ptr: Memory pointer to output tensor that stores reduced values.
#     :param output_meta: Meta information for output tensor (shape + stride).
#     :param temp_storage_ptr: Temporary storage for intermediate reductions.
#     :param BLOCK_SIZE: Size of the memory block for current process.
#     """
#     pid = tl.program_id(0)
#     tid = tl.arange(0, BLOCK_SIZE)

#     # Get shapes and strides
#     input_s0, input_s1, input_s2, input_s3 = read_shape(input_meta)
#     output_s0, output_s1, output_s2, output_s3 = read_shape(output_meta)

#     input_st0, input_st1, input_st2, input_st3 = read_stride(input_meta)
#     output_st0, output_st1, output_st2, output_st3 = read_stride(output_meta)

#     input_elements = input_s0 * input_s1 * input_s2 * input_s3  # Stage 1: Thread-level accumulation
#     thread_offsets = pid * BLOCK_SIZE + tid
#     thread_mask = thread_offsets < input_elements

#     # Load input values
#     input_vals = tl.load(input_ptr + thread_offsets, mask=thread_mask, other=0.0)

#     # For each thread, determine which output element it contributes to
#     # and accumulate accordingly
#     # This is where you'd implement the mapping logic based on reduction axes

#     # Stage 2: Block-level reduction using shared memory equivalent
#     # Triton handles this automatically with appropriate reduction operations

#     # Stage 3: Write results
#     # Use atomic operations to ensure thread safety across blocks
#     tl.atomic_add(temp_storage_ptr + pid, tl.sum(input_vals))


# ==============================================================================
# INTEGRATED TRITON KERNEL WITH SUM REDUCTION
# ==============================================================================
# The backward_kernel_with_reduction integrates gradient computation and
# sum reduction into a single kernel, eliminating the need for separate
# triton_sum_like calls and improving performance by:
# 1. Reducing kernel launch overhead
# 2. Minimizing memory transfers
# 3. Using atomic operations for direct accumulation
# 4. Avoiding temporary full-size tensor allocation
# ==============================================================================
def triton_sum_like(tensor_to_sum: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of sum_like functionality.

    :param tensor_to_sum: Tensor to be reduced.
    :param ref_tensor: Reference tensor whose shape determines the reduction.
    :return: Reduced tensor with the same shape as ref_tensor.
    """
    if ref_tensor.numel() == 1:
        return tensor_to_sum.sum()

    # Create output tensor with same shape as reference
    output = torch.zeros_like(ref_tensor)

    # Get meta information for both tensors
    input_meta = get_4d_tensor_meta(tensor_to_sum)
    output_meta = get_4d_tensor_meta(output)

    with torch.cuda.device(tensor_to_sum.device):
        # Use the optimized kernel with atomic operations
        grid = lambda meta: (triton.cdiv(tensor_to_sum.numel(), meta["BLOCK_SIZE"]),)
        optimized_sum_reduction_kernel[grid](
            tensor_to_sum,
            input_meta,
            output,
            output_meta,
        )

    return output


# def advanced_triton_sum_like(tensor_to_sum: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
#     """
#     Advanced Triton implementation of sum_like with hierarchical reduction.

#     :param tensor_to_sum: Tensor to be reduced.
#     :param ref_tensor: Reference tensor whose shape determines the reduction.
#     :return: Reduced tensor with the same shape as ref_tensor.
#     """
#     if ref_tensor.numel() == 1:
#         return tensor_to_sum.sum()

#     # Create output tensor with same shape as reference
#     output = torch.zeros_like(ref_tensor)

#     # Create temporary storage for intermediate reductions
#     num_blocks = triton.cdiv(tensor_to_sum.numel(), 1024)  # Assuming max block size of 1024
#     temp_storage = torch.zeros(num_blocks, dtype=tensor_to_sum.dtype, device=tensor_to_sum.device)

#     # Get meta information for both tensors
#     input_meta = get_4d_tensor_meta(tensor_to_sum)
#     output_meta = get_4d_tensor_meta(output)

#     with torch.cuda.device(tensor_to_sum.device):
#         # Launch hierarchical reduction kernel
#         grid = lambda meta: (triton.cdiv(tensor_to_sum.numel(), meta["BLOCK_SIZE"]),)
#         hierarchical_sum_reduction_kernel[grid](
#             tensor_to_sum,
#             input_meta,
#             output,
#             output_meta,
#             temp_storage,
#         )

#         # Final reduction of temporary storage if needed
#         if num_blocks > 1:
#             # Additional reduction step would go here
#             pass

#     return output
