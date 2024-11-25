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
from torch.autograd import Function


class STERound(Function):
    @staticmethod
    def forward(ctx, input_):
        return input_.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


input_data = torch.tensor([0.5, 1.5, 2.5, 3.5], requires_grad=True)
min_val = 1.0
max_val = 3.0

clipped_data = torch.round(input_data)  # , min_val, max_val)
clipped_data = STERound.apply(input_data)

output = clipped_data.sum()
output.backward()

print("Input:", input_data)
print("Clipped Data:", clipped_data)
print("Gradients:", input_data.grad)
