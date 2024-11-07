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

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_creation import wrap_model
from nncf.torch.model_transformer import PTModelTransformer

OUT_DIM = 2
IN_DIM = 3


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(out_features=OUT_DIM, in_features=IN_DIM)
        print("original weights", self.linear.weight.data)
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
    # @staticmethod
    # def forward_old(ctx, W, group_shape, A, B, input_low, input_range, levels):
    #     original_shape = W.shape
    #     input_ = W + B @ A
    #     input_ = input_.reshape(group_shape)

    #     # Save tensors for backward pass
    #     ctx.save_for_backward(A, B)

    #     scale = (levels - 1) / input_range
    #     output = input_.clip(min=input_low, max=input_low + input_range)
    #     zero_point = (-input_low * scale).round()
    #     output -= input_low
    #     output *= scale
    #     output -= zero_point
    #     output = output.round()
    #     output = output / scale

    #     output = output.reshape(original_shape)
    #     return output

    # @staticmethod
    # def backward_old(ctx, grad_output):
    #     # Retrieve saved tensors
    #     A, B = ctx.saved_tensors

    #     # Compute the gradient for the additive parameter
    #     # grad_A = grad_output.clone()
    #     # grad_B = grad_output.clone()
    #     grad_A = B.t() @ grad_output  # Gradient of the loss w.r.t. A
    #     grad_B = grad_output @ A.t()  # Gradient of the loss w.r.t. B
    #     # No gradient for W since it is frozen
    #     return None, None, grad_A, grad_B, None, None, None

    @staticmethod
    def forward(ctx, W, A, B, input_low, input_range, level_low, level_high, levels, is_lora):
        # print('original weight:', W.data)
        input_ = W + B @ A
        # input_ = W
        # print('original weight + adapters:', input_.data)

        scale = (levels - 1) / input_range
        output = input_.clip(min=input_low, max=input_low + input_range)
        zero_point = (-input_low * scale).round()
        output -= input_low
        output *= scale
        # print('Q(original weight + adapters): ', output.data)
        # print('ZP: ', zero_point.data)
        # print('Scale: ', scale.data)
        output -= zero_point
        output = output.round()
        output = output / scale

        # Save tensors for backward pass
        # if is_lora:
        #     ctx.save_for_backward(A, B)
        # else:
        ctx.save_for_backward(A, B, input_, output, input_low, input_range)

        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.is_lora = is_lora

        # print('FQ(original weight + adapters): ', output.data)
        print("quant noise", torch.linalg.norm(output - input_, ord="fro").item())
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
        grad_A = B.t() @ grad_output  # Gradient of the loss w.r.t. A
        grad_B = grad_output @ A.t()  # Gradient of the loss w.r.t. B
        # grad_A = grad_B = None

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
        reduction_axis = 1
        scale_shape = list(weight_shape)
        scale_shape[reduction_axis] = 1

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


model = MyModel()
input_ = torch.tensor([1.0, 2.0, 3.0])

model = wrap_model(model, example_input=input_, trace_parameters=True)
# print(model)


transformation_layout = TransformationLayout()
w = model.linear.weight
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
# print(transformed_model)
model.nncf.get_graph().visualize_graph("fq_model.dot")


adapters_to_train = []
scales_to_train = []
for name, param in model.named_parameters():
    if "_A" in name or "_B" in name:
        param.requires_grad = True
        adapters_to_train.append(param)
    # if "input" in name:
    #     param.requires_grad = True
    #     scales_to_train.append(param)
    else:
        param.requires_grad = False

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# param_to_train = [
#     {"params": adapters_to_train, "lr": 1e-2},
#     # {"params": scales_to_train, "lr": 1e-5},
# ]
optimizer = torch.optim.Adam(adapters_to_train, lr=1e-2)

# Dummy input and target
input_ = torch.tensor([1.0, 2.0, 3.0])
# target = torch.tensor([3.1, 4.2, 5.3])
target = torch.tensor([10.0, 10.0])


# Training loop
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_)
    loss = nn.MSELoss()(output, target)
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

plt.plot(losses)
plt.title("Lora fine-tuning", fontsize=20)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
path = Path("loss.png").resolve()
plt.savefig(path)
print("Saving loss plot to:", path)

# Check the updated additive parameter
print("Weights: ", model.linear.weight)
print("Updated additive parameter A:", model.nncf.external_quantizers.FQ_LORA_for_node_._A)
print("Updated additive parameter B:", model.nncf.external_quantizers.FQ_LORA_for_node_._B)
print(output)
