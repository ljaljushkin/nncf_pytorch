# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import torch
from torch import nn
import numpy as np

from nncf.torch.layers import NNCF_WRAPPED_USER_MODULES_DICT
from nncf.torch.layers import NNCFEmbedding
from nncf.torch.layers import NNCFLinear
from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high


class WeightsDecompressor(nn.Module):
    """Applies decompression of compressed weights in forward pass

    Attributes:
        zero_point: zero point in quantization scheme
        scale: scale in quantizatin scheme
    """

    def __init__(self, zero_point, scale):
        super().__init__()
        self.zero_point = zero_point
        self.scale = scale

    def forward(self, layer, op_arg):
        w = layer.weight.type(dtype=self.scale.dtype)
        layer.weight = (w - self.zero_point) * self.scale


class WeightsFQ(nn.Module):
    """Replaces weights with Torch's FakeQuantize operation on forward pass

    Attributes:
        zero_point: zero point in quantization scheme
        scale: scale in quantizatin scheme
        axis: channel for quantization
        level_high: maximal quant value in assymetric quantization
    """

    def __init__(self, zero_point, scale, axis=0, level_high=255):
        super().__init__()
        self.zero_point = zero_point
        self.scale = scale
        self.axis = axis
        self.level_high = level_high

    def forward(self, layer, op_arg):
        layer.weight = torch.fake_quantize_per_channel_affine(
            layer.weight, self.scale, self.zero_point, self.axis, 0, self.level_high
        )

def get_rnd_err(layer):
    if not type(layer) is NNCFLinear:
        return -1.0

    target_dim = layer.target_weight_dim_for_compression
    stat_dim = (target_dim + 1) % 2
    input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
    input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()

    level_high = 2**4-1

    scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

    scale = scale.unsqueeze(stat_dim)
    zero_point = zero_point.unsqueeze(stat_dim)

    compressed_weight = layer.weight.data / scale + zero_point
    compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

    w = compressed_weight.type(dtype=scale.dtype)
    w = (w - zero_point) * scale

    diff = (w - layer.weight.data)**2
    mean_err = torch.mean(diff)
    layer_err = torch.mean(diff, dim=1)
    top_k = torch.topk(layer_err, 10)[0]
    #val = float(mean_err)#top_k[0])
    val = float(top_k[0])
    return val

def get_rnd_errors(module, res):
    for name, layer in module.named_children():
        if not type(layer) is NNCFLinear:
            get_rnd_errors(layer, res)
            continue

        res.append(get_rnd_err(layer))

def _insert_pre_compression_operations(
    module: nn.Module, allowed_types: Dict, use_fake_quantize=False, level_high=255, stats=None
) -> Optional[nn.Module]:
    """
    Inserts weights compression with dequantization or quantization pre operation for Linear and Embedding layers.

    :param module: The module to insert the weights compression.
    :param allowed_types: list of allowed types for weights compression.
    :param use_fake_quantize: Disables real compression of weights in Linear and Embedding layers.
        If True inserts pytorch torch.fake_quantize_per_channel_affine(),
        else compress weights to int8 and inserts custom dequantization.
    :param level_high: highest  possible value of compressed weights (lower is 0 in assymetric quantization).
    :return: The module with inserted operations. The module is not trainable if use_fake_quantize is False.
    """
    for name, layer in module.named_children():
        if not type(layer) in allowed_types:
            _insert_pre_compression_operations(layer, allowed_types, use_fake_quantize, level_high, stats)
            continue
        # if 'embed_in' == name or 'embed_out' == name:
        # if 'embed_out' == name:
        #      print(f'name={name} in FP32')
        #      num_bits = 32
        #      num_weights = np.prod(layer.weight.shape)
        #      stats.append((name, num_weights, num_bits))
        #      continue
        local_level_high = level_high
        if name in ['embed_in', 'embed_out', 'query_key_value', 'dense_4h_to_h']:
            local_level_high = 255

        if len(stats) > 96:
            local_level_high = 15
        num_bits = 4 if local_level_high == 15 else 8
        num_weights = np.prod(layer.weight.shape)
        stats.append((name, num_weights, num_bits))

        print(f'name={name} level_high={local_level_high} num_bits={num_bits} num_weights={num_weights}')
        target_dim = layer.target_weight_dim_for_compression
        stat_dim = (target_dim + 1) % 2
        input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
        input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()
        scale, zero_point = get_scale_zp_from_input_low_input_high(0, local_level_high, input_low, input_high)

        if not use_fake_quantize:
            scale = scale.unsqueeze(stat_dim)
            zero_point = zero_point.unsqueeze(stat_dim)
            layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))

            compressed_weight = layer.weight.data / scale + zero_point
            compressed_weight = torch.clamp(torch.round(compressed_weight), 0, local_level_high)

            layer.weight.requires_grad = False
            layer.weight.data = compressed_weight.type(dtype=torch.uint8)
        else:
            zero_point = zero_point.type(dtype=torch.int32)
            layer.register_pre_forward_operation(WeightsFQ(zero_point, scale, target_dim))


def insert_pre_compression_operations(module: nn.Module, use_fake_quantize=False, bits=4) -> Optional[nn.Module]:
    """
    Inserts weights compression with dequantization or quantization pre operation for Linear and Embedding layers.

    :param module: The module to insert the weights compression.
    :param use_fake_quantize: Disables real compression of weights in Linear and Embedding layers.
        If True inserts torch.fake_quantize_per_channel_affine(),
        else compress weights to int8 and inserts custom dequantization.
    :param bits: number of bits for compression/quantization. Note: compressed weights type is
        uint8 with one element per 8 bit.
    :return: The module with inserted operations. The module is not trainable if use_fake_quantize is False.
    """
    user_types = list(NNCF_WRAPPED_USER_MODULES_DICT.values())
    allowed_types = [NNCFEmbedding, NNCFLinear]
    level_high = 2**bits - 1

    assert level_high < 256

    for user_type in user_types:
        allowed_types.append(user_type)

    stats = []
    _insert_pre_compression_operations(module, allowed_types, use_fake_quantize, level_high, stats)
    print(*stats, sep='\n')
    total_num_weights = sum(num_weights for _, num_weights, _ in stats)
    compressed_bits = sum(num_weights * num_bits for _, num_weights, num_bits in stats)
    c = {4: 0, 8:0, 32:0}
    print(len(stats))
    for _, num_weights, num_bits in stats:
        c[num_bits] += num_weights
    for num_bits, num_weights in c.items():
        print(f'% weights in {num_bits} = {num_weights / total_num_weights * 100:.3f}')
    print(f'weight compression={total_num_weights * 32 / compressed_bits :.3f} (fp32 - 1x, int8 - 4x, int4 - 8x, 50% int4/int8 ~ 6x)')

