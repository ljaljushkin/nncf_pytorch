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

from collections import OrderedDict
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


def get_node_name(module, module_name, prefix):
    return "{prefix}/{cls}[{name}]".format(prefix=prefix, cls=module.__class__.__name__, name=module_name)

@dataclass
class LayerData:
    name: str
    error: float
    num_weights: int
    module_id: int

    def __str__(self):
        return f'\n\tname={self.name}\n\terr={self.error}\n\tnum_weights={self.num_weights}\n'


def get_all_layer_data(model, allowed_types, prefix=None, res: List[LayerData]=None):
    if prefix is None:
        prefix = model.__class__.__name__
    for name, module in model.named_children():
        full_node_name = get_node_name(module, name, prefix)
        is_skipped = 'embed_in' == name or 'embed_out' == name or type(module) not in allowed_types
        if is_skipped:
            get_all_layer_data(module, allowed_types, prefix=full_node_name, res=res)
            continue
        num_weights = module.weight.data.numel()
        error = get_rnd_err(module)
        data = LayerData(full_node_name, error, num_weights, id(module))
        res.append(data)



def _insert_pre_compression_operations(
    module: nn.Module, use_fake_quantize=False, layer_id_vs_precision_map=None
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
    for _, layer in module.named_children():
        layer_id = id(layer)
        if layer_id not in layer_id_vs_precision_map:
            _insert_pre_compression_operations(layer, use_fake_quantize, layer_id_vs_precision_map)
            continue
        bits, name = layer_id_vs_precision_map[layer_id]
        level_high = 2**bits - 1
        # if not type(layer) in allowed_types:
        #     _insert_pre_compression_operations(layer, allowed_types, use_fake_quantize, level_high, layer_id_vs_precision_map)
        #     continue
        # if 'embed_in' == name or 'embed_out' == name:
        # # if 'embed_out' == name:
        #      print(f'name={name} in FP32')
        #      num_bits = 32
        #      num_weights = np.prod(layer.weight.shape)
        #      stats.append((name, num_weights, num_bits))
        #      continue
        # local_level_high = level_high
        # if name in ['embed_in', 'embed_out', 'query_key_value', 'dense_4h_to_h']:
        # if name in ['query_key_value', 'dense_4h_to_h']:
        #     local_level_high = 255

        # if get_rnd_err(layer) > th and type(layer) is NNCFLinear:
        #     local_level_high = 255

        # if len(stats) > 96:
        #     local_level_high = 15
        # num_bits = 4 if local_level_high == 15 else 8
        # num_weights = np.prod(layer.weight.shape)
        # stats.append((name, num_weights, num_bits))

        print(f'{bits} bits for {name} level_high={level_high}')
        target_dim = layer.target_weight_dim_for_compression
        stat_dim = (target_dim + 1) % 2
        input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
        input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()
        scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

        if not use_fake_quantize:
            scale = scale.unsqueeze(stat_dim)
            zero_point = zero_point.unsqueeze(stat_dim)
            layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))

            compressed_weight = layer.weight.data / scale + zero_point
            compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

            layer.weight.requires_grad = False
            layer.weight.data = compressed_weight.type(dtype=torch.uint8)
        else:
            zero_point = zero_point.type(dtype=torch.int32)
            layer.register_pre_forward_operation(WeightsFQ(zero_point, scale, target_dim))

# TODO: get all modules, then iterate
def get_total_num_weights(model, allowed_types, res=None):
    for _, module in model.named_children():
        if type(module) not in allowed_types:
            get_total_num_weights(module, allowed_types, res)
            continue
        res.append(module.weight.data.numel())

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
    # level_high = 2**bits - 1
    # assert level_high < 256

    for user_type in user_types:
        allowed_types.append(user_type)

    all_num_weights = []
    get_total_num_weights(module, allowed_types, all_num_weights)
    num_all_weights = len(all_num_weights)
    # print(num_all_weights)
    total_num_weights = sum(a for a in all_num_weights)
    # print(total_num_weights)

    data_list = []
    get_all_layer_data(module, allowed_types=allowed_types, prefix=None, res=data_list)
    # print(f'data={data_list}')
    n = len(data_list)
    errors = [data.error for data in data_list]
    for data in data_list:
        print(data.name)
    max_error = max(errors)
    layers=list(range(n))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(layers, errors)
    plt.savefig('errors.png')
    plt.close(fig)
    # print(f'\nerrors={errors}')


    errors = [e/max_error for e in errors]
    fig = plt.figure()
    plt.plot(layers, errors)
    plt.savefig('norm_errors.png')
    plt.close(fig)
    # print(f'\nnorm_errors={errors}')

    assert (num_all_weights - 2) == n, f'{num_all_weights}_{n}'

    ratio_updated = 0.25
    alpha = 0.5
    num_updated = n * ratio_updated
    # TODO: no linear rule??
    position_weight = [ (1 - i / num_updated) * alpha if i < num_updated else 0 for i in range(n)]
    print(f'\nposition_weight={position_weight}')
    # TODO: normalize position weights to be between min and max error
    # TODO: find mean std!
    updated_errors = [e + p for e,p in zip(errors, position_weight)]
    fig = plt.figure()
    plt.plot(layers, updated_errors)
    plt.plot(layers, position_weight)
    # updated_errors = errors
    # print(f'\nupdated_errors={updated_errors}')
    plt.savefig('updated_errors.png')
    plt.close(fig)
    indexes_of_layers_in_ascending_order_of_errors = [
        i[0] for i in sorted(enumerate(updated_errors), reverse=False, key=lambda x: x[1])
    ]
    # print(f'\nindexes_of_layers_in_ascending_order_of_errors={indexes_of_layers_in_ascending_order_of_errors}')
    sorted_errors = [data_list[i].error for i in indexes_of_layers_in_ascending_order_of_errors]
    print(f'\nsorted errors: {sorted_errors}')
    target_ratio_in_4_bit = 0.43
    current_num_weights = 0

    last_idx = None
    for i, index in enumerate(indexes_of_layers_in_ascending_order_of_errors):
        data = data_list[index]
        current_ratio = (current_num_weights + data.num_weights) / total_num_weights
        if current_ratio >= target_ratio_in_4_bit:
            last_idx = i-1
            total_num_weights_in_8_bit = sum(data_list[j].num_weights for j in indexes_of_layers_in_ascending_order_of_errors[i:])
            total_num_weights_in_4_bit = sum(data_list[j].num_weights for j in indexes_of_layers_in_ascending_order_of_errors[:i])
            assert total_num_weights_in_4_bit == current_num_weights, f'{total_num_weights_in_4_bit} vs {current_num_weights - data.num_weights}'
            assert total_num_weights - total_num_weights_in_8_bit - total_num_weights_in_4_bit > 0, f'{total_num_weights - total_num_weights_in_8_bit - total_num_weights_in_4_bit}'
            num_weights_per_precision = {
                4: current_num_weights,
                8: total_num_weights_in_8_bit,
                32: total_num_weights - total_num_weights_in_8_bit - current_num_weights
            }
            for num_bits, num_weights in num_weights_per_precision.items():
                print(f'% weights in {num_bits} bit = {num_weights / total_num_weights * 100:.3f}')
            occupied_bits = sum(num_weights * num_bits for num_bits, num_weights in num_weights_per_precision.items())
            print(f'weight compression={total_num_weights * 32 / occupied_bits :.3f} (fp32 - 1x, int8 - 4x, int4 - 8x, 50% int4/int8 ~ 6x)')
            break
        current_num_weights += data.num_weights

    layer_id_vs_precision_map = {}
    for i, index in enumerate(indexes_of_layers_in_ascending_order_of_errors):
        data = data_list[index]
        precision = 4 if i <= last_idx else 8
        layer_id_vs_precision_map[data.module_id] = (precision, data.name)
        # print(f'{precision} bit for {data.name}')

    bit_config = []
    for data in data_list:
        precision, _ = layer_id_vs_precision_map[data.module_id]
        bit_config.append(precision)
        print(f'{precision} bit for {data.name}')
    print(bit_config)

    _insert_pre_compression_operations(module, use_fake_quantize, layer_id_vs_precision_map)

