"""
 Copyright (c) 2021 Intel Corporation
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
from typing import NamedTuple

import torch
from torch import nn

from nncf.dynamic_graph.context import Scope
from nncf.layers import NNCFConv2d
from nncf.quantization.layers import BaseQuantizer
from nncf.quantization.layers import QuantizationMode
from nncf.quantization.layers import QuantizerConfig
from nncf.quantization.layers import SymmetricQuantizer


class AdjustPaddingArgs(NamedTuple):
    weight_bitwidth: int
    activation_quantizer: BaseQuantizer
    quantized_module: nn.Module
    module_scope: Scope


class AdjustPadding:
    """
    Calculates padding value to perform a workaround for U4 support on VPU.
    VPU supports only i4 for weights and activations with zero-point=0 and padding=0. This imposes some limitations on
    the quantization scheme we can apply. In case of unsigned input for a quantizer (e.g. output of ReLU) half of
    i4 range (8 values) is insufficient to preserve the accuracy. To overcome the problem it is proposed
    to transform u4 to i4 in the VPU plugin by shifting the input by half of the quantization range to left. Padding
    value should be shifted as well. And to make it zero after the shift (non-zero padding values are not
    supported), the model should be trained with padding value equal to the half of the quantization range.
    """

    def __init__(self, activation_quantizer: SymmetricQuantizer):
        self._activation_quantizer = activation_quantizer
        self._is_enabled = True

    @staticmethod
    def _is_applicable(args: AdjustPaddingArgs):
        weight_bitwidth, activation_quantizer, module, _ = args
        result = False
        if isinstance(module, NNCFConv2d):
            padding_values = set(module.padding)
            padding_enabled = len(padding_values) >= 1 and padding_values.pop()
            if padding_enabled:
                result = isinstance(activation_quantizer, SymmetricQuantizer) and \
                         not activation_quantizer.per_channel and \
                         activation_quantizer.num_bits == 4 and \
                         activation_quantizer.num_bits >= weight_bitwidth and \
                         not activation_quantizer.signed
        return result

    @staticmethod
    def create(args: AdjustPaddingArgs):
        if AdjustPadding._is_applicable(args):
            return AdjustPadding(args.activation_quantizer)
        return None

    def __call__(self, previous_padding_value) -> torch.Tensor:
        if self._is_enabled:
            scale = self._activation_quantizer.scale
            eps = self._activation_quantizer.eps
            safe_scale = abs(scale) + eps
            return safe_scale / 2
        return previous_padding_value

    @staticmethod
    def is_config_applicable(qconfig: QuantizerConfig):
        return not qconfig.is_weights and not qconfig.per_channel and qconfig.bits == 4 and \
               not qconfig.signedness_to_force and qconfig.mode == QuantizationMode.SYMMETRIC
