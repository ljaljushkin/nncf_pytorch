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

import torch
import torch.nn as nn


class AdditiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, A, B):
        # Save tensors for backward pass
        ctx.save_for_backward(A, B)
        # Perform the forward pass
        return W + A @ B

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        A, B = ctx.saved_tensors

        # Compute the gradient for the additive parameter
        # grad_A = grad_output.clone()
        # grad_B = grad_output.clone()
        grad_A = grad_output @ B.t()  # Gradient of the loss w.r.t. A
        grad_B = A.t() @ grad_output  # Gradient of the loss w.r.t. B
        # No gradient for W since it is frozen
        grad_W = None
        return grad_W, grad_A, grad_B


# Define a custom module using the custom autograd function
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.W = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False))
        self.A = nn.Parameter(torch.tensor([[0.1]]))
        self.B = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3]]))

    def forward(self, x):
        # Use the custom autograd function
        result = AdditiveFunction.apply(self.W, self.A, self.B) @ x
        return result


class AdditiveFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        return W

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False))
        self.A = nn.Parameter(torch.tensor([[0.1]]))
        self.B = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3]]))

    def forward(self, x):
        # Use the custom autograd function
        result = AdditiveFunction2.apply(self.W + self.A @ self.B) @ x
        return result


# Example usage
# model = MyModel()
model = MyModel2()
param_to_train = []
for name, param in model.named_parameters():
    if name in ["A", "B"]:  # or "11.self_attn.v_proj.weight" in name:  # or 'input' in name:
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
for epoch in range(10):
    optimizer.zero_grad()
    output = model(input_)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Check the updated additive parameter
print("Updated additive parameter A:", model.A)
print("Updated additive parameter B:", model.B)
print((model.W + model.A @ model.B) @ input_)
