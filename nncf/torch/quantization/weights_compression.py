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
import numpy as np

import torch
from torch import nn

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


# class WeightsFQ(nn.Module):
    # """Replaces weights with Torch's FakeQuantize operation on forward pass

    # Attributes:
    #     zero_point: zero point in quantization scheme
    #     scale: scale in quantizatin scheme
    #     axis: channel for quantization
    #     level_high: maximal quant value in assymetric quantization
    # """

    # def __init__(self, zero_point, scale, axis=0, level_high=255):
    #     super().__init__()
    #     self.zero_point = zero_point
    #     self.scale = scale
    #     self.axis = axis
    #     self.level_high = level_high

    # def forward(self, layer, op_arg):
    #     layer.weight = torch.fake_quantize_per_channel_affine(
    #         layer.weight, self.scale, self.zero_point, self.axis, 0, self.level_high
    #     )
    # pass


# def _insert_pre_compression_operations(
#     module: nn.Module, allowed_types: Dict, use_fake_quantize=False, level_high=255
# ) -> Optional[nn.Module]:
    # """
    # Inserts weights compression with dequantization or quantization pre operation for Linear and Embedding layers.

    # :param module: The module to insert the weights compression.
    # :param allowed_types: list of allowed types for weights compression.
    # :param use_fake_quantize: Disables real compression of weights in Linear and Embedding layers.
    #     If True inserts pytorch torch.fake_quantize_per_channel_affine(),
    #     else compress weights to int8 and inserts custom dequantization.
    # :param level_high: highest  possible value of compressed weights (lower is 0 in assymetric quantization).
    # :return: The module with inserted operations. The module is not trainable if use_fake_quantize is False.
    # """
    # for _, layer in module.named_children():
    #     if not type(layer) in allowed_types:
    #         _insert_pre_compression_operations(layer, allowed_types, use_fake_quantize, level_high)
    #         continue
    #     target_dim = layer.target_weight_dim_for_compression
    #     stat_dim = (target_dim + 1) % 2
    #     input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
    #     input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()
    #     scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

    #     if not use_fake_quantize:
    #         scale = scale.unsqueeze(stat_dim)
    #         zero_point = zero_point.unsqueeze(stat_dim)
    #         layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))

    #         compressed_weight = layer.weight.data / scale + zero_point
    #         compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

    #         layer.weight.requires_grad = False
    #         layer.weight.data = compressed_weight.type(dtype=torch.uint8)
    #     else:
    #         zero_point = zero_point.type(dtype=torch.int32)
    #         layer.register_pre_forward_operation(WeightsFQ(zero_point, scale, target_dim))

def QuantizeNF4(x):
  if x > 0.03979014977812767:
    if x > 0.3893125355243683: # 1
      if x > 0.6427869200706482: # 11
        if x > 0.8614784181118011: # 111
          return 0b1111
        else:
          return 0b1110
      else:
        if x > 0.5016634166240692: # 110
          return 0b1101
        else:
          return 0b1100
    else:
      if x > 0.2035212516784668: # 10
        if x > 0.2920137718319893: # 101
          return 0b1011
        else:
          return 0b1010
      else:
        if x > 0.1202552504837513: # 100
          return 0b1001
        else:
          return 0b1000
  else:
    if x > -0.33967943489551544: # 0
      if x > -0.13791173323988914: # 01
        if x > -0.045525018125772476: # 011
          return 0b0111
        else:
          return 0b0110
      else:
        if x > -0.23460740596055984: # 010
          return 0b0101
        else:
          return 0b0100
    else:
      if x > -0.6106329262256622: # 00
        if x > -0.4599952697753906: # 001
          return 0b0011
        else:
          return 0b0010
      else:
        if x > -0.8480964004993439: # 000
          return 0b0001
        else:
          return 0b0000


def DequantizeNF4(x):
    lookup = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
    return lookup[x]


def quant_deguant(x):
    return DequantizeNF4(QuantizeNF4(x))

def get_scale_zp_from_input_low_input_high_nf4(level_low, level_high, input_low, input_high):
    nf4_convert = np.vectorize(quant_deguant)

    y_scale = (input_high - input_low) / (level_high - level_low)
    y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    y_zero_point = y_zero_point.numpy()
    y_zero_point = nf4_convert(y_zero_point)
    y_zero_point = torch.from_numpy(y_zero_point)

    y_scale = torch.squeeze(y_scale)
    y_zero_point = torch.squeeze(y_zero_point)
    return y_scale, y_zero_point

def ref_ASYM(level_low, level_high, input_low, input_high):
    y_scale = (input_high - input_low) / (level_high - level_low)
    y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    type_ = torch.int8 if level_low < 0 else torch.uint8
    level_low *= torch.ones_like(y_zero_point).to(type_)
    level_high *= torch.ones_like(y_zero_point).to(type_)
    level_low = level_low.to(y_zero_point.device)
    level_high = level_high.to(y_zero_point.device)
    y_zero_point = torch.min(torch.max(level_low, torch.round(y_zero_point).to(type_)), level_high)

    y_scale = torch.squeeze(y_scale)
    y_zero_point = torch.squeeze(y_zero_point)
    return y_scale, y_zero_point

def _fake_fp_to_nf4(
    module: nn.Module, allowed_types: Dict, iter
) -> Optional[nn.Module]:
    """
    Replace weights with nf4 for Linear and Embedding layers.
    """
    nf4_convert = np.vectorize(quant_deguant)
    for lname, layer in module.named_children():
        print('\t'*iter, lname)
        if not type(layer) in allowed_types:
            _fake_fp_to_nf4(layer, allowed_types, iter+1)
            continue

        if 'emb' in lname:
            target_dim = layer.target_weight_dim_for_compression
            stat_dim = (target_dim + 1) % 2
            input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
            input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()
            scale, zero_point = get_scale_zp_from_input_low_input_high(0, 255, input_low, input_high)

            scale = scale.unsqueeze(stat_dim)
            zero_point = zero_point.unsqueeze(stat_dim)
            layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))

            compressed_weight = layer.weight.data / scale + zero_point
            compressed_weight = torch.clamp(torch.round(compressed_weight), 0, 255)

            layer.weight.requires_grad = False
            layer.weight.data = compressed_weight.type(dtype=torch.uint8)
            continue


        layer.weight.requires_grad = False
        target_dim = layer.target_weight_dim_for_compression
        stat_dim = (target_dim + 1) % 2
        # scale = torch.max(torch.abs(layer.weight), dim=stat_dim)[0].detach()
        # scale = scale.unsqueeze(1)

        # tmp = (layer.weight / scale).detach()
        # tmp = tmp.numpy()
        # tmp = nf4_convert(tmp)
        # print("Type before: ", layer.weight.type())
        # layer.weight.data = (torch.from_numpy(tmp) * scale).type(torch.FloatTensor)
        # print("Type after: ", layer.weight.type())

        input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
        input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()
        input_low  = input_low.unsqueeze(1)
        input_high = input_high.unsqueeze(1)

        # delta = 0.5 * (input_low + input_high)
        #scale = (input_high - zp).abs()

        # zp = (zp / scale).numpy()
        # zp = nf4_convert(zp)
        delta = zp = 0.0
        w = layer.weight
        # print(w[:5,:5], w.shape)
        num_columns = w.shape[stat_dim]
        group_size = 256
        scale = []
        assert num_columns % group_size == 0
        for i1 in range(0, num_columns, group_size):
            i2 = i1 + group_size
            current_columns = w[:, i1:i2]  # [c_out, c_in // group_size]
            input_low = torch.min(current_columns, dim=stat_dim)[0].detach()  # [c_out]
            input_low = input_low.unsqueeze(dim=stat_dim)  # [c_out, 1]
            input_high = torch.max(current_columns, dim=stat_dim)[0].detach()  # [c_out]
            input_high = input_high.unsqueeze(dim=stat_dim)  # [c_out, 1]
            scale.append(torch.max((input_high - zp).abs(), (input_low - zp).abs()))
        scale = torch.cat(scale, dim=stat_dim)  # [c_out, c_in // group_size]
        scale = torch.repeat_interleave(scale, group_size, dim=stat_dim)  # [c_out, c_in]
        # print(scale[:5, :5], scale.shape)

        # scale = torch.max((input_high - zp).abs(), (input_low - zp).abs())

        # scale = input_high - delta
        # scale, zp = get_scale_zp_from_input_low_input_high_nf4(0.0, 2.0, input_low, input_high)
        # scale = scale.unsqueeze(stat_dim)
        # zp = zp.unsqueeze(stat_dim)
        tmp = ((layer.weight.data - delta) / scale).detach()
        # print(tmp[:5, :5], tmp.shape)
        tmp = tmp.numpy()
        tmp = nf4_convert(tmp)
        # TODO: nf4 quantize zp=0
        #print("Type before: ", layer.weight.type())
        nf4_data = (torch.from_numpy(tmp) * scale + delta).type(torch.FloatTensor)
        diff = torch.mean((nf4_data - layer.weight.data)**2)
        print('\t'*iter, "diff: ", diff)
        layer.weight.data = nf4_data
        #print("Type after: ", layer.weight.type())


def insert_pre_compression_operations(module: nn.Module, use_fake_quantize=False, bits=8) -> Optional[nn.Module]:
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

    # _insert_pre_compression_operations(module, allowed_types, use_fake_quantize, level_high)
    _fake_fp_to_nf4(module, allowed_types, 0)



