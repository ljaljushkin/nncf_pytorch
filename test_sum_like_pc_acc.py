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

import pytest
import torch

from sum_like_kernels import sum_like_baseline
from sum_like_kernels import sum_like_two_stage


def idfn(val):
    """ID function for pytest parametrization"""
    if isinstance(val, list):
        return "[{}]".format("-".join([str(v) for v in val]))
    return None


def generate_reference_tensor(input_size, scale_mode, is_weights, dtype):
    """Generate reference tensor shape based on scale mode and tensor type"""
    if scale_mode == "single_scale":
        # Single scale: [1]
        return torch.ones([1], dtype=dtype)
    elif scale_mode == "per_channel_scale":
        if is_weights:
            # For weights, channel is dim 0: [C, 1, 1, 1]
            channel_count = input_size[0]
            if channel_count == 1:
                pytest.skip("Same case as for single scale mode")
            ref_shape = [1 for _ in input_size]
            ref_shape[0] = channel_count
        else:
            # For activations, channel is dim 1: [1, C, 1, 1]
            channel_count = input_size[1]
            if channel_count == 1:
                pytest.skip("Same case as for single scale mode")
            ref_shape = [1 for _ in input_size]
            ref_shape[1] = channel_count
        return torch.ones(ref_shape, dtype=dtype)
    else:
        msg = f"Unknown scale_mode: {scale_mode}"
        raise ValueError(msg)


def pytorch_sum_like(tensor_to_sum, ref_tensor):
    """PyTorch reference implementation of sum_like functionality"""
    # Ensure both tensors have the same number of dimensions
    while tensor_to_sum.dim() < ref_tensor.dim():
        tensor_to_sum = tensor_to_sum.unsqueeze(0)
    while ref_tensor.dim() < tensor_to_sum.dim():
        ref_tensor = ref_tensor.unsqueeze(0)

    # Sum over dimensions where ref_tensor has size 1
    result = tensor_to_sum
    for dim in range(result.dim()):
        if ref_tensor.size(dim) == 1 and result.size(dim) > 1:
            result = result.sum(dim, keepdim=True)

    return result


@pytest.mark.parametrize(
    "input_size",
    [
        [1, 16, 64, 64],
        [1, 48, 112, 112],
        [4, 16, 16, 16],
        [8, 256, 32, 32],
        [16, 192, 28, 28],
        [16, 96, 112, 112],
        [16, 576, 14, 14],
        [4, 64, 128, 128],
    ],
    ids=idfn,
)
@pytest.mark.parametrize("is_fp16", [True, False], ids=["fp16", "fp32"])
@pytest.mark.parametrize("kernel_impl", [sum_like_baseline, sum_like_two_stage], ids=["baseline", "two_stage"])
class TestTritonSumReduction:
    def test_triton_sum_like_correctness(self, input_size, is_fp16, kernel_impl):
        use_cuda = True
        scale_mode = "per_channel_scale"
        is_weights = False

        device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.float16 if is_fp16 else torch.float32

        # Generate test data
        torch.manual_seed(42)

        # input_tensor = torch.arange(math.prod(input_size), device=device, dtype=dtype).reshape(input_size)
        input_tensor = torch.randn(input_size, device=device, dtype=dtype)
        ref_tensor = generate_reference_tensor(input_size, scale_mode, is_weights, dtype).to(device)

        # Compute expected result using PyTorch
        expected = pytorch_sum_like(input_tensor, ref_tensor)
        expected_once = torch.sum(input_tensor, axis=(0, 2, 3), keepdim=True)

        # Compute result using triton_sum_like
        result = kernel_impl(input_tensor, ref_tensor)

        # Check results
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        rtol = 1 if is_fp16 else 1e-3
        atol = 1e-1 if is_fp16 else 1e-4

        assert torch.allclose(expected_once, expected, rtol=rtol / 10, atol=atol / 10), (
            f"torch_once vs torch don't match. Max diff: {(expected_once - expected).abs().max()}"
        )

        assert torch.allclose(result, expected, rtol=rtol, atol=atol), (
            f"kernel vs torch don't match. Max diff: {(result - expected).abs().max()}"
        )

        assert torch.allclose(result, expected_once, rtol=rtol, atol=atol), (
            f"kernel vs torch_once don't match. Max diff: {(result - expected_once).abs().max()}"
        )
