"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import List
from typing import Optional

import torch.nn as nn

from nncf.torch.layers import NNCF_PADDING_VALUE_ATTR_NAME


class BaseOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    @property
    def operand(self):
        return self.op

    def forward(self, *inputs, **kwargs):
        return self.op(*inputs, **kwargs)


class UpdateInputs(BaseOp):
    def __call__(self, _, inputs):
        return super().__call__(*inputs)


class UpdateParameter(BaseOp):
    def __init__(self, param_name, op):
        super().__init__(op)
        self._param_name = param_name

    def __call__(self, module, inputs):
        if not hasattr(module, self._param_name):
            raise TypeError('{} should have {} attribute'.format(type(module), self._param_name))

        value = getattr(module, self._param_name)
        result = super().__call__(value, *inputs)
        setattr(module, self._param_name, result)


class UpdateWeight(UpdateParameter):
    def __init__(self, op):
        super().__init__("weight", op)


class UpdatePaddingValue(UpdateParameter):
    def __init__(self, op):
        super().__init__(NNCF_PADDING_VALUE_ATTR_NAME, op)


class UpdatePadding(UpdateParameter):
    def __init__(self, op):
        super().__init__("padding", op)


class UpdateParameterList(BaseOp):
    def __init__(self, param_names: List[str], op, is_optional_list: Optional[List[bool]] = None):
        super().__init__(op)
        self._param_names = param_names
        if is_optional_list is None:
            is_optional_list = [False for _ in param_names]
        self._is_optional_list = is_optional_list

    def __call__(self, module, inputs):
        param_values = []
        for param_name, is_optional in zip(self._param_names, self._is_optional_list):
            if not hasattr(module, param_name):
                if is_optional:
                    param_values.append(None)
                    continue
                raise TypeError('{} should have {} attribute'.format(type(module), param_name))
            param_values.append(getattr(module, param_name))
        updated_kwargs = dict(zip(self._param_names, param_values))
        updated_values = super().__call__(*inputs, **updated_kwargs)

        for param_name, updated_value in zip(self._param_names, updated_values):
            setattr(module, param_name, updated_value)


class UpdateWeightAndOptionalBias(UpdateParameterList):
    def __init__(self, op):
        super().__init__(["weight", "bias"], op, [False, True])


class UpdateBatchNormParams(UpdateParameterList):
    def __init__(self, op):
        super().__init__(["weight", "bias", "running_mean", "running_var"], op)
