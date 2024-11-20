# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import torch
import torch.nn as nn

from nncf import CompressWeightsMode
from nncf import compress_weights
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_creation import wrap_model
from nncf.torch.model_transformer import PTModelTransformer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


OUT_DIM = 5
IN_DIM = 3


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(out_features=OUT_DIM, in_features=IN_DIM)
        # self.linear.weight.data.fill_(3)

    def forward(self, x):
        return self.linear(x)


def sum_like(tensor_to_sum, ref_tensor):
    """Warning: may modify tensor_to_sum"""
    if ref_tensor.size == 1:
        return tensor_to_sum.sum()

    for dim, size in enumerate(ref_tensor.shape):
        if size == 1:
            tensor_to_sum = tensor_to_sum.sum(dim, keepdim=True)
    return tensor_to_sum


class AdditiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, A, B):
        # Save the additive parameter for backward pass
        ctx.save_for_backward(A, B)
        # Perform the forward pass
        # print(W.shape)
        return W + B @ A

    @staticmethod
    def backward(ctx, grad_output):
        # print('grad shape', grad_output.shape)
        # Retrieve the saved tensor
        A, B = ctx.saved_tensors
        # Compute the gradient for the additive parameter
        # grad_A = grad_output.clone()
        grad_A = B.t() @ grad_output  # Gradient of the loss w.r.t. A
        grad_B = grad_output @ A.t()  # Gradient of the loss w.r.t. B
        # No gradient for W since it is frozen
        grad_W = None
        return grad_W, grad_A, grad_B


class AdditiveFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        return W

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class FQLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, A, B, input_low, input_range, level_low, level_high, levels, is_lora):
        # print('original weight:', W.data)
        input_ = W + B @ A
        # print("weights + adapters", input_)
        # input_ = W
        # print('original weight + adapters:', input_.data)

        # scale = ((input_high - input_low) / (levels - 1)).astype(TensorDataType.float32)
        # zero_point = - fns.round(input_low / scale)
        # zero_point = fns.clip(zero_point.astype(TensorDataType.int32), level_low, level_high)
        # compressed_weights = weight / scale
        # compressed_weights += zero_point.astype(weight.dtype)
        # compressed_weights = fns.round(compressed_weights)
        # compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(dtype)

        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        zero_point = -(input_low * scale).round()

        # print('IL: ', input_low)
        # print('IR: ', input_range)
        # compressed_weights = output * scale
        # compressed_weights = (compressed_weights + zero_point).clip(min=level_low, max=level_high)
        # compressed_weights = compressed_weights.round()
        # compressed_weights = compressed_weights.clip(min=level_low, max=level_high)
        # print('Q(weights + adapters): ', compressed_weights.data)

        output -= input_low
        output *= scale

        # print('ZP: ', zero_point.data)
        # print('Scale: ', scale.data)
        # print('A: ', A.data)
        # print('B: ', B.data)
        output -= zero_point
        output = output.round()
        output = output / scale
        # print('FQ(weights + adapters): ', output.data)

        # Save tensors for backward pass
        # if is_lora:
        #     ctx.save_for_backward(A, B)
        # else:
        ctx.save_for_backward(A, B, input_, output, input_low, input_range)

        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.is_lora = is_lora

        # print('FQ(original weight + adapters): ', output.data)
        # print("quant noise", torch.linalg.norm(output - input_, ord="fro").item())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # is_lora = ctx.is_lora

        # if is_lora:
        #     A, B = ctx.saved_tensors
        #     grad_A = B.t() @ grad_output  # Gradient of the loss w.r.t. A
        #     grad_B = grad_output @ A.t()  # Gradient of the loss w.r.t. B
        #     #      [W,   A,      B,      input_low, input_range, level_low, level_high, levels, is_lora
        #     return None, grad_A, grad_B, None,      None,        None,      None,       None,   None
        # else:
        A, B, input_, output, input_low, input_range = ctx.saved_tensors

        level_low = ctx.level_low
        level_high = ctx.level_high
        # group_shape = ctx.group_shape

        mask_hi = input_ > (input_low + input_range)
        mask_hi = mask_hi.type(input_.dtype)
        mask_lo = input_ < input_low
        mask_lo = mask_lo.type(input_.dtype)

        mask_in = 1 - mask_hi - mask_lo
        range_sign = torch.sign(input_range)
        err = (output - input_) * torch.reciprocal(input_range * range_sign)
        grad_range = grad_output * (err * mask_in + range_sign * (level_low / level_high) * mask_lo + mask_hi)
        grad_range = sum_like(grad_range, input_range)

        # NOTE: no gradient for weights
        # grad_input = grad_output * mask_in

        grad_low = grad_output * (mask_hi + mask_lo)
        grad_low = sum_like(grad_low, input_low)
        #      [W,   A,      B,      input_low, input_range, level_low, level_high, levels, is_lora
        # return None, grad_A, grad_B, grad_low, grad_range, None, None, None, None

        grad_A = B.t() @ grad_output  # Gradient of the loss w.r.t. A
        grad_B = grad_output @ A.t()  # Gradient of the loss w.r.t. B
        # grad_A = grad_B = None

        return None, grad_A, grad_B, grad_low, grad_range, None, None, None, None


class FQLora(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shape = [OUT_DIM, IN_DIM]
        out_features, in_features = weight_shape
        lora_rank = 2
        self._A = torch.nn.Parameter(
            torch.ones((lora_rank, in_features), dtype=torch.float32), requires_grad=True
        )  # [L, I]
        self._B = torch.nn.Parameter(
            torch.ones((out_features, lora_rank), dtype=torch.float32), requires_grad=True
        )  # [O, L]

        self._input_low = torch.nn.Parameter(
            torch.ones((1, in_features), dtype=torch.float32), requires_grad=True
        )  # [1, I]
        self._input_range = torch.nn.Parameter(
            torch.ones((1, in_features), dtype=torch.float32), requires_grad=True
        )  # [1, I]
        # reduction_axis = 1
        # scale_shape = list(weight_shape)
        # scale_shape[reduction_axis] = 1
        # print("A", self._A.data)
        # print("B", self._B.data)

    def forward(self, weight):
        # weight = weight.detach()
        # for name, param in self.named_parameters():
        #     print("CHECK: ", name, param.requires_grad)
        # print("CHECK: weight ", weight.requires_grad)
        # return AdditiveFunction.apply(weight, self._A, self._B)
        # return AdditiveFunction2.apply(weight + self._B @ self._A)
        return FQLoRAFunction.apply(weight, self._A, self._B, self._input_low, self._input_range, 0, 15, 16, False)
        # return FQLoRAFunction.apply(weight + self._B @ self._A, None, None,
        # self._input_low, self._input_range, 0, 15, 16, False)


set_seed(42)
model = MyModel()
input_ = torch.tensor([1.0, 2.0, 3.0])

model = wrap_model(model, example_input=input_, trace_parameters=True)
# print(model)


transformation_layout = TransformationLayout()
with torch.no_grad():
    lora_rank = 2
    w = model.linear.weight + torch.ones(OUT_DIM, lora_rank) @ torch.ones(lora_rank, IN_DIM)
    input_low = torch.amin(w, dim=0, keepdim=True)
    # print('input_low', input_low)
    input_high = torch.amax(w, dim=0, keepdim=True)
    input_range = input_high - input_low
    # print('input_range', input_range)
    quantizer = FQLora()
    quantizer._input_low = torch.nn.Parameter(input_low, requires_grad=True)  # [1, I]
    quantizer._input_range = torch.nn.Parameter(input_range, requires_grad=True)  # [1, I]

node_name = "MyModel/Linear[linear]/linear_0"
target_point = PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, node_name, input_port_id=1)
transformation_layout.register(
    PTSharedFnInsertionCommand(
        target_points=[target_point],
        fn=quantizer,
        op_unique_name="FQ_LORA_for_node_",
        compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
        priority=TransformationPriority.QUANTIZATION_PRIORITY,
    )
)
transformed_model = PTModelTransformer(model).transform(transformation_layout)
print(transformed_model)
model.nncf.get_graph().visualize_graph("fq_model.dot")


compress_weights(transformed_model, mode=CompressWeightsMode.NF4)
