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


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(out_features=1, in_features=3)
        self.linear.weight.data.fill_(3)

    def forward(self, x):
        return self.linear(x)


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


class FQLora(nn.Module):
    def __init__(self):
        super().__init__()
        out_features, in_features = 1, 3
        lora_rank = 2
        self._A = torch.nn.Parameter(
            torch.ones((lora_rank, in_features), dtype=torch.float32), requires_grad=True
        )  # [L, I]
        self._B = torch.nn.Parameter(
            torch.zeros((out_features, lora_rank), dtype=torch.float32), requires_grad=True
        )  # [O, L]
        print(self._A)
        print(self._B)

    def forward(self, weight):
        # weight = weight.detach()
        # for name, param in self.named_parameters():
        #     print("CHECK: ", name, param.requires_grad)
        # print("CHECK: weight ", weight.requires_grad)
        return AdditiveFunction.apply(weight, self._A, self._B)


model = MyModel()
input_ = torch.tensor([1.0, 2.0, 3.0])

model = wrap_model(model, example_input=input_, trace_parameters=True)
# print(model)
model.nncf.get_graph().visualize_graph("fq_model.dot")

transformation_layout = TransformationLayout()
quantizer = FQLora()
node_name = "MyModel/Linear[linear]/linear_0"
target_point = PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, node_name, input_port_id=0)
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


param_to_train = []
for name, param in model.named_parameters():
    if "_A" in name or "_B" in name:  # or "11.self_attn.v_proj.weight" in name:  # or 'input' in name:
        param.requires_grad = True
        param_to_train.append(param)
        print(name)
    else:
        param.requires_grad = False


optimizer = torch.optim.SGD(param_to_train, lr=0.01)

# Dummy input and target
input_ = torch.tensor([1.0, 2.0, 3.0])
# target = torch.tensor([3.1, 4.2, 5.3])
target = torch.tensor([10.0])


# Training loop
losses = []
for epoch in range(15):
    optimizer.zero_grad()
    output = model(input_)
    loss = nn.MSELoss()(output, target)
    losses.append(float(loss))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

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
