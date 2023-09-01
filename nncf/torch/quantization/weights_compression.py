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
        # w = torch.pow(w, self.alpha)
        # s_w = w * self.sign
        if self.alpha == 2:
            layer.weight = torch.square(w) * s
        else:
            layer.weight = torch.pow(w, self.alpha) * s


    def dequantize(self, input):
        w = input.type(dtype=self.scale.dtype)
        w = (w - self.zero_point) * self.scale
        s = torch.sign(w)
        return torch.pow(w, self.alpha) * s


def metric_top_k(orig, quantized, k = 10):
    diff = (orig - quantized)**2
    layer_err = torch.mean(diff, dim=1)
    top_k = torch.topk(layer_err, k)[0]
    val = float(torch.mean(top_k))
    return val

def metric_mean(orig, quantized):
    diff = (orig - quantized)**2
    mean_err = torch.mean(diff)
    return float(mean_err)


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

    return metric_top_k(layer.weight.data, w)


def get_power_quant_error(layer):
    if not type(layer) is NNCFLinear:
        return -1.0
    alpha = 0.5

    level_high = 2**4 - 1
    w_sign = torch.sign(layer.weight)
    w_pow = torch.pow(torch.abs(layer.weight), alpha) * w_sign
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
    
    return metric_top_k(layer.weight.data, decompressed_weight)


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


def get_relative_error(layer):
    return get_power_quant_error(layer)# / (get_int8_err(layer) + 0.0000000001)
    #return get_int8_err(layer) / (get_power_quant_error(layer) + 0.0000000001)


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
    group_mode = False

    for data in data_list:
        layer = data.module
        bits = data.precision
        level_high = 2**bits - 1
        print(f'{data.precision} bits for {data.name}')

        if data.precision == 4:
            w_sign = torch.sign(layer.weight)
            w_pow = torch.pow(torch.abs(layer.weight), best_alpha) * w_sign
        else:
            w_pow = layer.weight

        target_dim = layer.target_weight_dim_for_compression
        stat_dim = (target_dim + 1) % 2

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

        if data.precision == 8:
            layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))
        else:
            layer.register_pre_forward_operation(WeightsDecompressorPowerQuant(zero_point, scale, 1/best_alpha))


def get_all_layer_data(model, allowed_types, prefix=None, res: List[LayerData]=None):
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        if type(module) not in allowed_types:
            get_all_layer_data(module, allowed_types, prefix=full_node_name, res=res)
            continue
        # TODO: GPTNeoXForCausalLM
        # is_skipped = 'embed_in' == name or 'embed_out' == name
        # TODO: LlamaForCausalLM
        is_skipped = 'embed_' in name or 'lm_head' in name
        num_weights = module.weight.data.numel()
        error = 0 if is_skipped else get_power_quant_error(module)
        data = LayerData(full_node_name, error, num_weights, module, is_skipped, 8)
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
    # dolly-v2-3b
    target_ratio_in_4_bit = 0.80
    # # llama-3b
    # target_ratio_in_4_bit = 0.713
    # # llama-13b
    # target_ratio_in_4_bit = 0.762
    # # bloom-7b1
    # target_ratio_in_4_bit = 0.556
    # # opt-6.7b
    # target_ratio_in_4_bit = 0.62
    # # pajama-7b
    # target_ratio_in_4_bit = 0.744


    all_data_list = []
    get_all_layer_data(module, allowed_types=allowed_types, prefix=None, res=all_data_list)
    total_num_weights = sum(d.num_weights for d in all_data_list)
    # print(f'num all layers={len(all_data_list)}, num all weights={total_num_weights}')

    data_list = list(filter(lambda x: not x.is_skipped, all_data_list))
    #total_num_weights = sum(d.num_weights for d in data_list)
    errors = sorted([data.error for data in data_list])
    num_updated = int(target_ratio_in_4_bit * len(errors))
    th = errors[num_updated]
    
    prev = -1
    theoretical = 0
    practical = 0
    for i in range(len(data_list)):
        cur_err = data_list[i].error
        if cur_err <= th:
            theoretical += 1
            if prev > -1 and cur_err / (prev + 0.00000001) > 1.5:# and False:
                prev = cur_err
                data_list[i].precision = 8
                continue
            practical += 1
            prev = cur_err
            data_list[i].precision = 4
        else:
            data_list[i].precision = 8

    sz = len(data_list)
    print(f"All {sz}. Candidates: {theoretical}. Winners: {practical}")

    for data in all_data_list:
        if data.is_skipped:
            data.precision = 8
    bit_config = [data.precision for data in all_data_list]
    print(bit_config)

    num_weights_per_precision = {}
    for data in all_data_list:
        # if data.is_skipped:
        #     continue
        precision = data.precision
        num_weights_per_precision[precision] = num_weights_per_precision.get(precision, 0) + data.num_weights
    for num_bits, num_weights in num_weights_per_precision.items():
        print(f'% weights in {num_bits} bit = {num_weights / total_num_weights * 100:.3f}')
    occupied_bits = sum(num_weights * num_bits for num_bits, num_weights in num_weights_per_precision.items())
    print(f'weight compression={total_num_weights * 32 / occupied_bits :.3f} (fp32 - 1x, int8 - 4x, int4 - 8x, 50% int4/int8 ~ 6x)')
    assert total_num_weights == sum(nw for nw in num_weights_per_precision.values())

    _insert_pre_compression_operations_simple(all_data_list)
