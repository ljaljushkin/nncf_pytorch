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

from dataclasses import dataclass
from typing import Dict, List, Optional

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

class WeightsDecompressorPowerQuant(nn.Module):
    """Applies decompression of compressed weights in forward pass

    Attributes:
        zero_point: zero point in quantization scheme
        scale: scale in quantizatin scheme
    """

    def __init__(self, zero_point, scale, alpha):
        super().__init__()
        self.zero_point = zero_point
        self.scale = scale
        if alpha >= 0.9:
            alpha = int(alpha)
        self.alpha = alpha
        # self.unpacker = Unpacker()

    def forward(self, layer, op_arg):
        # w = self.unpacker(layer.weight)
        w = layer.weight
        w = w.type(dtype=self.scale.dtype)
        w = (w - self.zero_point) * self.scale
        s = torch.sign(w)
        if self.alpha == 2:
            layer.weight = torch.square(w) * s
        else:
            layer.weight = torch.pow(w, self.alpha) * s
        # layer.weight = torch.exp(w) * s

    def dequantize(self, input):
        w = input.type(dtype=self.scale.dtype)
        w = (w - self.zero_point) * self.scale
        s = torch.sign(w)
        return torch.pow(w, self.alpha) * s
        # return torch.exp(w) * s

def get_total_num_weights(model, allowed_types, res=None):
    for _, module in model.named_children():
        if type(module) not in allowed_types:
            get_total_num_weights(module, allowed_types, res)
            continue
        res.append(module.weight.data.numel())

# total_num_weights = sum(a for a in all_num_weights)

def _insert_pre_compression_operations_power_quant(
    module: nn.Module, allowed_types: Dict, use_fake_quantize=False, level_high=15, th=1000.0, iter=0, precisions=None, num_weights_per_precision=None
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
    for lname, layer in module.named_children():
        # if 'emb' in lname:
        #     print(f'Skip {lname}')
        #     continue
        # if type(layer) is NNCFEmbedding:
        #     _insert_pre_compression_operations(layer, allowed_types, use_fake_quantize, 255)
        #     continue
        print("\t"*iter, lname)
        alphas = [0.125/2, 0.125, 0.25, 0.5, 1.0]#[0.25, 0.5, 1.0, 2.0]
        if not type(layer) in allowed_types:
            _insert_pre_compression_operations_power_quant(layer, allowed_types, use_fake_quantize, level_high, th, iter+1, precisions, num_weights_per_precision)
            continue

        err = get_relative_error(layer) #get_power_quant_error(layer)
        if 'emb' in lname or 'lm_head' in lname or err > th:
            print("\t"*iter, "INT8")
            precisions.append(8)
            # print("Skip embeddings ", lname)
            # continue
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
            num_weights_per_precision[8] += layer.weight.data.numel()
            continue

        # best_alpha = -1
        # min_err = layer.weight.numel()
        w_sign = torch.sign(layer.weight)

        # for alpha in alphas:
        #     w_pow = torch.pow(torch.abs(layer.weight), alpha) * w_sign
        #     target_dim = layer.target_weight_dim_for_compression
        #     stat_dim = (target_dim + 1) % 2
        #     input_low = torch.min(w_pow, dim=stat_dim)[0].detach()
        #     input_high = torch.max(w_pow, dim=stat_dim)[0].detach()
        #     scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

        #     scale = scale.unsqueeze(stat_dim)
        #     zero_point = zero_point.unsqueeze(stat_dim)
        #     op = WeightsDecompressorPowerQuant(zero_point, scale, 1/alpha)

        #     compressed_weight = w_pow.data / scale + zero_point
        #     compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

        #     decompressed_weight = op.dequantize(compressed_weight)
        #     diff = (decompressed_weight - layer.weight.data)**2
        #     cur_err = torch.mean(diff)

        #     layer_err = torch.mean(diff, dim=1)
        #     top_k = torch.topk(layer_err, 10)[0]
        #     #val = float(mean_err)#top_k[0])
        #     cur_err = float(top_k[0])

        #     if cur_err < min_err:
        #         min_err = cur_err
        #         best_alpha = alpha
        # print(f"layer {lname}, alpha {best_alpha}, min_err {min_err}")
        print("\t"*iter, "POWER QUANT INT4")
        precisions.append(4)
        num_weights_per_precision[4] += layer.weight.data.numel()

        best_alpha = 0.5
        w_pow = torch.pow(torch.abs(layer.weight), best_alpha) * w_sign
        target_dim = layer.target_weight_dim_for_compression
        stat_dim = (target_dim + 1) % 2
        group_mode = False

        if group_mode:
          group_size = 256
          w = w_pow
          # print(w[:5,:5], w.shape)
          num_columns = w.shape[stat_dim]
          scale = []
          zero_point = []
          assert num_columns % group_size == 0
          for i1 in range(0, num_columns, group_size):
              i2 = i1 + group_size
              current_columns = w[:, i1:i2]  # [c_out, c_in // group_size]
              input_low = torch.min(current_columns, dim=stat_dim)[0].detach()  # [c_out]
              input_high = torch.max(current_columns, dim=stat_dim)[0].detach()  # [c_out]

              scale_g, zero_point_g = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

              scale_g = scale_g.unsqueeze(dim=stat_dim)  # [c_out, 1]
              zero_point_g = zero_point_g.unsqueeze(dim=stat_dim)  # [c_out, 1]

              scale.append(scale_g)
              zero_point.append(zero_point_g)

          scale = torch.cat(scale, dim=stat_dim)  # [c_out, c_in // group_size]
          scale = torch.repeat_interleave(scale, group_size, dim=stat_dim)  # [c_out, c_in]

          zero_point = torch.cat(zero_point, dim=stat_dim)  # [c_out, c_in // group_size]
          zero_point = torch.repeat_interleave(zero_point, group_size, dim=stat_dim)  # [c_out, c_in]
        else:
          input_low = torch.min(w_pow, dim=stat_dim)[0].detach()
          input_high = torch.max(w_pow, dim=stat_dim)[0].detach()
          scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)
          scale = scale.unsqueeze(stat_dim)
          zero_point = zero_point.unsqueeze(stat_dim)

        compressed_weight = w_pow.data / scale + zero_point
        compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

        layer.weight.requires_grad = False
        compressed_weight = compressed_weight.type(dtype=torch.uint8)

        # original_weights = layer.weight.data.clone()
        # w = compressed_weight
        # w = w.type(dtype=scale.dtype)
        # w = (w - zero_point) * scale
        # s = torch.sign(w)
        # decompressed = torch.square(w) * s
        # diff = torch.mean((original_weights - decompressed)**2)
        # print(diff)
        # layer.weight.data = Packer.pack(compressed_weight)
        layer.weight.data = compressed_weight

        layer.register_pre_forward_operation(WeightsDecompressorPowerQuant(zero_point, scale, 1/best_alpha))


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
        # delta = []
        assert num_columns % group_size == 0
        for i1 in range(0, num_columns, group_size):
            i2 = i1 + group_size
            current_columns = w[:, i1:i2]  # [c_out, c_in // group_size]
            input_low = torch.min(current_columns, dim=stat_dim)[0].detach()  # [c_out]
            input_low = input_low.unsqueeze(dim=stat_dim)  # [c_out, 1]
            input_high = torch.max(current_columns, dim=stat_dim)[0].detach()  # [c_out]
            input_high = input_high.unsqueeze(dim=stat_dim)  # [c_out, 1]
            scale.append(torch.max((input_high - zp).abs(), (input_low - zp).abs()))
            # delta.append(0.5 * (input_low + input_high))
        scale = torch.cat(scale, dim=stat_dim)  # [c_out, c_in // group_size]
        scale = torch.repeat_interleave(scale, group_size, dim=stat_dim)  # [c_out, c_in]
        # delta = torch.cat(delta, dim=stat_dim)  # [c_out, c_in // group_size]
        # delta = torch.repeat_interleave(delta, group_size, dim=stat_dim)  # [c_out, c_in]
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


def get_int8_err(layer):
    if not type(layer) is NNCFLinear:
        return -1.0

    target_dim = layer.target_weight_dim_for_compression
    stat_dim = (target_dim + 1) % 2
    input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
    input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()

    level_high = 2**8-1

    scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

    scale = scale.unsqueeze(stat_dim)
    zero_point = zero_point.unsqueeze(stat_dim)

    compressed_weight = layer.weight.data / scale + zero_point
    compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

    w = compressed_weight.type(dtype=scale.dtype)
    w = (w - zero_point) * scale

    diff = (w - layer.weight.data)**2
    #mean_err = torch.mean(diff)
    layer_err = torch.mean(diff, dim=1)
    top_k = torch.topk(layer_err, 10)[0]
    #val = float(mean_err)#top_k[0])
    val = float(top_k[0])
    return val


def get_power_quant_error(layer):
    if not type(layer) is NNCFLinear:
        return -1.0
    alpha = 0.5

    level_high = 2**4 - 1
    w_sign = torch.sign(layer.weight)
    w_pow = torch.pow(torch.abs(layer.weight), alpha) * w_sign
    # w_pow = torch.log(torch.abs(layer.weight)) * w_sign
    target_dim = layer.target_weight_dim_for_compression
    stat_dim = (target_dim + 1) % 2
    input_low = torch.min(w_pow, dim=stat_dim)[0].detach()
    input_high = torch.max(w_pow, dim=stat_dim)[0].detach()
    scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

    scale = scale.unsqueeze(stat_dim)
    zero_point = zero_point.unsqueeze(stat_dim)
    op = WeightsDecompressorPowerQuant(zero_point, scale, 1/alpha)

    compressed_weight = w_pow.data / scale + zero_point
    compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

    decompressed_weight = op.dequantize(compressed_weight)
    diff = (decompressed_weight - layer.weight.data)**2

    # mean_err = torch.mean(diff)
    # return float(mean_err)
    layer_err = torch.mean(diff, dim=1)
    top_k = torch.topk(layer_err, 10)[0]
    #val = float(mean_err)#top_k[0])
    val = float(top_k[0])
    return val

def get_relative_error(layer):
    return get_power_quant_error(layer) / (get_int8_err(layer) + 0.0000000001)
    #return get_int8_err(layer) / (get_power_quant_error(layer) + 0.0000000001)


def get_power_quant_errors(module, res):
    for name, layer in module.named_children():
        if not type(layer) is NNCFLinear:
            get_power_quant_errors(layer, res)
            continue

        #res.append(get_power_quant_error(layer))
        res.append(get_relative_error(layer))

def get_node_name(module, module_name, prefix):
    return "{prefix}/{cls}[{name}]".format(prefix=prefix, cls=module.__class__.__name__, name=module_name)

@dataclass
class LayerData:
    name: str
    error: float
    num_weights: int
    module: nn.Module # TODO: access module via name
    is_skipped: bool
    precision: int = None

    def __str__(self):
        return f'\n\tname={self.name}\n\terr={self.error}\n\tnum_weights={self.num_weights}\n'

def _insert_pre_compression_operations_simple(
    data_list: List[LayerData]=None
) -> Optional[nn.Module]:
    """
    Inserts weights compression with dequantization or quantization pre operation for Linear and Embedding layers.

    :param model: The module to insert the weights compression.
    :param layer_id_vs_precision_map: TBD
    :return: The module with inserted operations.
    """
    best_alpha = 0.5
    group_mode = True
    group_size = 64
    is_power_quant_fn = lambda x: x == 4
    is_zp = True

    for data in data_list:
        layer = data.module
        bits = data.precision
        if is_zp:
            level_high = 2**bits - 1
            level_low = 0
        else:
            level_high = 2**(bits-1) - 1
            level_low = -2**(bits-1) + 1

        target_dim = layer.target_weight_dim_for_compression
        stat_dim = (target_dim + 1) % 2
        print(f'{data.precision} bits for {data.name}')
        is_power_quant = is_power_quant_fn(bits)
        w = layer.weight
        original_weights = w.data.clone()
        if is_power_quant:
            w_sign = torch.sign(layer.weight)
            w = torch.pow(torch.abs(layer.weight), best_alpha) * w_sign
            assert not torch.any(torch.isnan(w)) or not torch.any(torch.isinf(w))
        if group_mode:
            # print(w[:5,:5], w.shape)
            num_columns = w.shape[stat_dim]
            scale = []
            zero_point = []
            if not is_zp:
                zero_point = 0
            assert num_columns % group_size == 0
            for i1 in range(0, num_columns, group_size):
                i2 = i1 + group_size
                current_columns = w[:, i1:i2]  # [c_out, c_in // group_size]
                input_low = torch.min(current_columns, dim=stat_dim)[0].detach()  # [c_out]
                input_high = torch.max(current_columns, dim=stat_dim)[0].detach()  # [c_out]

                if not is_zp:
                    scale_g = torch.max(input_high.abs(), input_low.abs()) / level_high
                    scale_g = scale_g.unsqueeze(dim=stat_dim)  # [c_out, 1]
                    scale.append(scale_g)
                else:
                    scale_g, zero_point_g = get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high)
                    scale_g = scale_g.unsqueeze(dim=stat_dim)  # [c_out, 1]
                    zero_point_g = zero_point_g.unsqueeze(dim=stat_dim)  # [c_out, 1]
                    scale.append(scale_g)
                    zero_point.append(zero_point_g)

            scale = torch.cat(scale, dim=stat_dim)  # [c_out, c_in // group_size]
            scale = torch.repeat_interleave(scale, group_size, dim=stat_dim)  # [c_out, c_in]

            if is_zp:
                zero_point = torch.cat(zero_point, dim=stat_dim)  # [c_out, c_in // group_size]
                zero_point = torch.repeat_interleave(zero_point, group_size, dim=stat_dim)  # [c_out, c_in]
        else:
            # TODO: always with ZP!
            input_low = torch.min(w, dim=stat_dim)[0].detach()
            input_high = torch.max(w, dim=stat_dim)[0].detach()
            scale, zero_point = get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high)
            scale = scale.unsqueeze(stat_dim)
            zero_point = zero_point.unsqueeze(stat_dim)

        compressed_weight = w.data / scale + zero_point
        compressed_weight = torch.clamp(torch.round(compressed_weight), level_low, level_high)

        w = compressed_weight
        w = w.type(dtype=scale.dtype)
        decompressed = (w - zero_point) * scale
        if is_power_quant:
            s = torch.sign(decompressed)
            decompressed = torch.square(decompressed) * s
            assert not torch.any(torch.isnan(decompressed)) or not torch.any(torch.isinf(decompressed))
        diff = torch.mean((original_weights - decompressed)**2)
        print(diff)
        layer.weight.requires_grad = False
        compressed_weight = compressed_weight.type(dtype=torch.uint8)
        layer.weight.data = compressed_weight
        if is_power_quant:
            pre_forward_operation = WeightsDecompressorPowerQuant(zero_point, scale, 1/best_alpha)
        else:
           pre_forward_operation = WeightsDecompressor(zero_point, scale)
        layer.register_pre_forward_operation(pre_forward_operation)


def get_all_layer_data(model, allowed_types, prefix=None, res: List[LayerData]=None, is_skipped_fn=None):
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if type(module) not in allowed_types:
            get_all_layer_data(module, allowed_types, prefix=full_node_name, res=res, is_skipped_fn=is_skipped_fn)
            continue

        num_weights = module.weight.data.numel()
        is_skipped = is_skipped_fn(module)
        error = 0 # if is_skipped else get_relative_error(module)
        data = LayerData(full_node_name, error, num_weights, module, is_skipped)
        res.append(data)

def insert_pre_compression_operations(module: nn.Module) -> Optional[nn.Module]:
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
    # user_types = list(NNCF_WRAPPED_USER_MODULES_DICT.values())
    # for user_type in user_types:
    #     allowed_types.append(user_type)
    allowed_types = [NNCFEmbedding, NNCFLinear]

    # NOTE: dolly-v2-3b
    target_ratio_in_4_bit = 0.638
    # NOTE: llama-3b
    target_ratio_in_4_bit = 0.713
    # NOTE: llama-13b
    target_ratio_in_4_bit = 0.762
    # NOTE: bloom-7b1
    target_ratio_in_4_bit = 0.556
    # NOTE: opt-6.7b
    target_ratio_in_4_bit = 0.62
    # NOTE: pajama-7b
    target_ratio_in_4_bit = 0.744

    target_ratio_in_4_bit = 0.5
    # ratio_updated = 0.25

    all_data_list = []

    # NOTE: GPTNeoXForCausalLM
    is_skipped_neox = lambda name: 'embed_in' == name or 'embed_out' == name
    # NOTE: LlamaForCausalLM
    is_skipped_llama = lambda name: 'embed_tokens' == name or 'lm_head' == name
    get_all_layer_data(module, allowed_types=allowed_types, prefix=None, res=all_data_list, is_skipped_fn=is_skipped_llama)
    total_num_weights = sum(d.num_weights for d in all_data_list)
    # print(f'num all layers={len(all_data_list)}, num all weights={total_num_weights}')

    data_list = list(filter(lambda x: not x.is_skipped, all_data_list))
    errors = [data.error for data in data_list]
    # NOTE: force first 25% in int8
    # max_error = max(errors)
    # num_updated = int(target_ratio_in_4_bit * ratio_updated * len(errors))
    # for i in range(num_updated):
    #    errors[i] = max_error * 2
    # NOTE: visualize
    layers=list(range(len(errors)))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(layers, errors)

    indexes_of_layers_in_ascending_order_of_errors = [
        i[0] for i in sorted(enumerate(errors), reverse=False, key=lambda x: x[1])
    ]
    # print(f'\nindexes_of_layers_in_ascending_order_of_errors={indexes_of_layers_in_ascending_order_of_errors}')
    # sorted_errors = [data_list[i].error for i in indexes_of_layers_in_ascending_order_of_errors]
    # print(f'\nsorted errors: {sorted_errors}')

    current_num_weights = 0
    for i, index in enumerate(indexes_of_layers_in_ascending_order_of_errors):
        data = data_list[index]
        current_ratio = (current_num_weights + data.num_weights) / total_num_weights
        if current_ratio >= target_ratio_in_4_bit:
            for j in indexes_of_layers_in_ascending_order_of_errors[i:]:
                data_list[j].precision = 8
            boundary_error = errors[indexes_of_layers_in_ascending_order_of_errors[i-1]]
            plt.axhline(y = boundary_error, color = 'r', linestyle = '-')
            plt.savefig('errors_with_boundary.png')
            plt.close(fig)
            break
        data.precision = 4
        current_num_weights += data.num_weights

    for data in all_data_list:
        if data.is_skipped:
            data.precision = 8
    # bit_config = [data.precision for data in all_data_list]
    bit_config = [4] * len(all_data_list)
    bit_config[0] = bit_config[-1] = 8
    # NOTE: llama-3b (71.27% in 4bit - 63.44)
    # bit_config = [8, 4, 4, 8, 4, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 4, 4, 8, 4, 8, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8, 8, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 8, 8, 8, 8, 4, 8, 4, 4, 8, 8, 4, 4, 4, 4, 8, 8, 4, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 8, 4, 4, 8, 4, 4, 4, 8, 8, 4, 8, 4, 4, 4, 8, 8, 4, 8, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
    for bits, data in zip(bit_config, all_data_list):
        data.precision = bits
    print(bit_config)

    num_weights_per_precision = {}
    for data in all_data_list:
        precision = data.precision
        num_weights_per_precision[precision] = num_weights_per_precision.get(precision, 0) + data.num_weights
    for num_bits, num_weights in num_weights_per_precision.items():
        print(f'% weights in {num_bits} bit = {num_weights / total_num_weights * 100:.3f}')
    occupied_bits = sum(num_weights * num_bits for num_bits, num_weights in num_weights_per_precision.items())
    print(f'weight compression={total_num_weights * 32 / occupied_bits :.3f} (fp32 - 1x, int8 - 4x, int4 - 8x, 50% int4/int8 ~ 6x)')
    assert total_num_weights == sum(nw for nw in num_weights_per_precision.values())

    _insert_pre_compression_operations_simple(all_data_list)

    # level_high = 2**bits - 1
    # assert level_high < 256
    # for user_type in user_types:
    #     allowed_types.append(user_type)
    # errors = sorted(errors)
    # th = errors[int(0.7 * len(errors))]
    # _insert_pre_compression_operations(module, allowed_types, use_fake_quantize, level_high)
    # _fake_fp_to_nf4(module, allowed_types, 0)
    # all_num_weights = []
    # get_total_num_weights(module, allowed_types, all_num_weights)
    # total_num_weights = sum(a for a in all_num_weights)

    # precisions = []
    # num_weights_per_precision = {4: 0, 8: 0, 32: 0}
    # _insert_pre_compression_operations_power_quant(module, allowed_types, use_fake_quantize, level_high, th, 0, precisions, num_weights_per_precision)
    # print(precisions)
    # for num_bits, num_weights in num_weights_per_precision.items():
    #     print(f'% weights in {num_bits} bit = {num_weights / total_num_weights * 100:.3f}')
    # occupied_bits = sum(num_weights * num_bits for num_bits, num_weights in num_weights_per_precision.items())
    # print(f'weight compression={total_num_weights * 32 / occupied_bits :.3f} (fp32 - 1x, int8 - 4x, int4 - 8x, 50% int4/int8 ~ 6x)')





