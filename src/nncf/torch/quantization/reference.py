# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import TypeVar

import numpy as np
import torch

import nncf
from nncf.torch.utils import CompilationWrapper

GeneralizedTensor = TypeVar("GeneralizedTensor", torch.Tensor, np.ndarray)


def fp32_accum_wrapper(func):
    def wrapper(tensor_to_sum, ret_tensor):
        half = tensor_to_sum.dtype == np.float16
        if half:
            tensor_to_sum = tensor_to_sum.astype(np.float32)
        retval = func(tensor_to_sum, ret_tensor)
        if half:
            retval = retval.astype(np.float16)
        return retval

    return wrapper


@fp32_accum_wrapper
def sum_like(tensor_to_sum, ref_tensor):
    """Warning: may modify tensor_to_sum"""
    if ref_tensor.size == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            if isinstance(tensor_to_sum, np.ndarray):
                tensor_to_sum = tensor_to_sum.sum(dim, keepdims=True)
            else:
                tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


class ReferenceBackendType(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


class ReferenceQuantize:
    def __init__(self, backend_type: ReferenceBackendType):
        if backend_type is ReferenceBackendType.NUMPY:
            self.backend = np
        elif backend_type is ReferenceBackendType.TORCH:
            self.backend = torch
        else:
            msg = "Unknown backend for ReferenceQuantize"
            raise nncf.UnsupportedBackendError(msg)

    def _astype(self, tensor: GeneralizedTensor, dtype) -> GeneralizedTensor:
        if self.backend is np:
            return tensor.astype(dtype)
        return tensor.type(dtype)

    def _sign(self, tensor: GeneralizedTensor) -> GeneralizedTensor:
        if self.backend is np:
            return np.sign(tensor)
        return torch.sign(tensor)

    def _reciprocal(self, tensor: GeneralizedTensor) -> GeneralizedTensor:
        if self.backend is np:
            return np.reciprocal(tensor)
        return torch.reciprocal(tensor)

    def forward(
        self, input_: GeneralizedTensor, input_low: GeneralizedTensor, input_range: GeneralizedTensor, levels: int
    ) -> GeneralizedTensor:
        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale).round()
        output -= input_low
        output *= scale
        output -= zero_point
        output = output.round()
        output = output / scale
        return output

    def backward(
        self,
        grad_output: GeneralizedTensor,
        input_: GeneralizedTensor,
        input_low: GeneralizedTensor,
        input_range: GeneralizedTensor,
        levels: int,
        level_low: int,
        level_high: int,
        is_asymmetric: bool = False,
    ) -> list[GeneralizedTensor]:
        # is_asymmetric is unused, present only to correspond to the CPU signature of calling "backward"
        mask_hi = input_ > (input_low + input_range)
        mask_hi = self._astype(mask_hi, input_.dtype)
        mask_lo = input_ < input_low
        mask_lo = self._astype(mask_lo, input_.dtype)

        mask_in = 1 - mask_hi - mask_lo
        range_sign = self._sign(input_range)
        output = self.forward(input_, input_low, input_range, levels)
        err = (output - input_) * self._reciprocal(input_range * range_sign)
        grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
        grad_range = sum_like(grad_range, input_range)

        grad_input = grad_output * mask_in

        grad_low = grad_output * (mask_hi + mask_lo)
        grad_low = sum_like(grad_low, input_low)
        return [grad_input, grad_low, grad_range]

    def tune_range(
        self, input_low: GeneralizedTensor, input_range: GeneralizedTensor, levels: int
    ) -> tuple[GeneralizedTensor, GeneralizedTensor]:
        input_high = input_range + input_low
        input_low[input_low > 0] = 0
        input_high[input_high < 0] = 0
        n = levels - 1
        scale = n / (input_high - input_low)
        scale = self._astype(scale, input_high.dtype)
        zp = self.backend.round(-input_low * scale)

        new_input_low = self.backend.where(zp < n, zp / (zp - n) * input_high, input_low)
        new_input_high = self.backend.where(zp > 0.0, (zp - n) / zp * input_low, input_high)

        range_1 = input_high - new_input_low
        range_2 = new_input_high - input_low

        mask = self._astype((range_1 > range_2), input_high.dtype)
        inv_mask = abs(1 - mask)

        new_input_low = mask * new_input_low + inv_mask * input_low
        new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

        return new_input_low, new_input_range


torch_executor = ReferenceQuantize(backend_type=ReferenceBackendType.TORCH)
torch_forward = CompilationWrapper(torch_executor.forward)
torch_backward = CompilationWrapper(torch_executor.backward)


class ReferenceQuantizedFunctions:
    Quantize_forward = torch_forward
    Quantize_backward = torch_backward


# =============================================================================
# Autograd-based Quantization with STE (Straight-Through Estimator)
# =============================================================================


class STERound(torch.autograd.Function):
    """
    Straight-Through Estimator for rounding.
    Forward: applies round()
    Backward: passes gradient through unchanged (as if round was identity)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class STEClamp(torch.autograd.Function):
    """
    Straight-Through Estimator for clamping with boundary gradients.
    Forward: clamps to [min_val, max_val]
    Backward:
        - For input: passes gradient only for values within range
        - For min_val: accumulates gradients where input < min_val
        - For max_val: accumulates gradients where input > max_val
    This allows learning the clamp boundaries.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, min_val, max_val)
        return x.clamp(min=min_val, max=max_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, min_val, max_val = ctx.saved_tensors

        # Masks for different regions
        mask_below = x < min_val  # Values clamped to min
        mask_above = x > max_val  # Values clamped to max
        mask_in = ~mask_below & ~mask_above  # Values within range

        # Gradient for input: only pass through for values within range
        grad_input = grad_output * mask_in.to(grad_output.dtype)

        # Gradient for min_val: sum of gradients where values were clamped to min
        # (for values below min, output = min_val, so d_output/d_min_val = 1)
        grad_min_raw = grad_output * mask_below.to(grad_output.dtype)

        # Gradient for max_val: sum of gradients where values were clamped to max
        grad_max_raw = grad_output * mask_above.to(grad_output.dtype)

        # Sum-reduce to match the shape of min_val and max_val
        # They are typically broadcastable shapes
        def sum_like(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Sum grad to match target shape."""
            # Sum over all dimensions that were broadcast
            while grad.dim() > target.dim():
                grad = grad.sum(0)
            for i in range(target.dim()):
                if target.shape[i] == 1 and grad.shape[i] > 1:
                    grad = grad.sum(i, keepdim=True)
            return grad

        grad_min = sum_like(grad_min_raw, min_val)
        grad_max = sum_like(grad_max_raw, max_val)

        return grad_input, grad_min, grad_max


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Apply round with STE."""
    return STERound.apply(x)


def ste_clamp(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """Apply clamp with STE."""
    return STEClamp.apply(x, min_val, max_val)


class ReferenceQuantizeAutograd:
    """
    Quantization using PyTorch autograd with STE for non-differentiable operations.

    This class only implements forward() - gradients are computed automatically by PyTorch.
    Non-differentiable operations (round, clamp) use Straight-Through Estimator (STE).

    Benefits over explicit backward:
    - Simpler, less error-prone
    - Automatic gradient computation
    - Natural handling of fp16/bf16 without manual precision management
    """

    @staticmethod
    def forward(
        input_: torch.Tensor,
        input_low: torch.Tensor,
        input_range: torch.Tensor,
        levels: int,
    ) -> torch.Tensor:
        """
        Quantize input tensor.

        Args:
            input_: Input tensor to quantize
            input_low: Lower bound of quantization range (learnable)
            input_range: Range of quantization (learnable, input_high = input_low + input_range)
            levels: Number of quantization levels (e.g., 4 for 2-bit)

        Returns:
            Quantized tensor
        """
        # Scale factor: maps range to [0, levels-1]
        scale = (levels - 1) / input_range

        # Clamp input to valid range with STE (gradient zero outside range)
        output = ste_clamp(input_, input_low, input_low + input_range)

        # Compute zero point (rounded)
        zero_point = ste_round(-input_low * scale)

        # Quantize: scale -> shift -> round -> unscale
        output = output - input_low
        output = output * scale
        output = output - zero_point
        output = ste_round(output)
        output = output / scale

        return output


# Wrapper for compatibility with existing code

autograd_quantize_forward = CompilationWrapper(ReferenceQuantizeAutograd.forward)

# class ReferenceQuantizedAutogradFunctions:
#     Quantize_forward = CompilationWrapper(ReferenceQuantizeAutograd.forward)
#     Quantize_backward = CompilationWrapper(ReferenceQuantizeAutograd.backward)
