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
from typing import Any, List, Tuple, Type, TypeVar, Union

import numpy as np
import openvino.runtime as ov
from openvino.runtime import opset9 as opset

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVEmbeddingMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_node_metatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op
from nncf.openvino.graph.node_utils import get_const_value
from nncf.openvino.graph.node_utils import get_matmul_channel_axes
from nncf.parameters import CompressWeightsMode
from nncf.quantization.fake_quantize import calculate_scale_zero_point

# def ref_group_mode(self):
#     if group_mode:
#         group_size = 256
#         w = w_pow
#         # print(w[:5,:5], w.shape)
#         num_columns = w.shape[stat_dim]
#         scale = []
#         zero_point = []
#         assert num_columns % group_size == 0
#         for i1 in range(0, num_columns, group_size):
#             i2 = i1 + group_size
#             current_columns = w[:, i1:i2]  # [c_out, c_in // group_size]
#             input_low = torch.min(current_columns, dim=stat_dim)[0].detach()  # [c_out]
#             input_high = torch.max(current_columns, dim=stat_dim)[0].detach()  # [c_out]

#             scale_g, zero_point_g = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

#             scale_g = scale_g.unsqueeze(dim=stat_dim)  # [c_out, 1]
#             zero_point_g = zero_point_g.unsqueeze(dim=stat_dim)  # [c_out, 1]

#             scale.append(scale_g)
#             zero_point.append(zero_point_g)

#         scale = torch.cat(scale, dim=stat_dim)  # [c_out, c_in // group_size]
#         scale = torch.repeat_interleave(scale, group_size, dim=stat_dim)  # [c_out, c_in]

#         zero_point = torch.cat(zero_point, dim=stat_dim)  # [c_out, c_in // group_size]
#         zero_point = torch.repeat_interleave(zero_point, group_size, dim=stat_dim)  # [c_out, c_in]


# def get_int8_err(layer):
#     if not type(layer) is NNCFLinear:
#         return -1.0

#     target_dim = layer.target_weight_dim_for_compression
#     stat_dim = (target_dim + 1) % 2
#     input_low = torch.min(layer.weight, dim=stat_dim)[0].detach()
#     input_high = torch.max(layer.weight, dim=stat_dim)[0].detach()

#     level_high = 2**8 - 1

#     scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

#     scale = scale.unsqueeze(stat_dim)
#     zero_point = zero_point.unsqueeze(stat_dim)

#     compressed_weight = layer.weight.data / scale + zero_point
#     compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

#     w = compressed_weight.type(dtype=scale.dtype)
#     w = (w - zero_point) * scale

#     diff = (w - layer.weight.data) ** 2
#     # mean_err = torch.mean(diff)
#     layer_err = torch.mean(diff, dim=1)
#     top_k = torch.topk(layer_err, 10)[0]
#     # val = float(mean_err)#top_k[0])
#     val = float(top_k[0])
#     return val


TWeightType = TypeVar("TWeightType")


@dataclass
class WeightCompressionConfig:
    num_bits: int = 8
    is_power_quant: bool = False
    group_size: int = -1


@dataclass
# TODO: rename
class WeightNodeParams:
    axes: Union[int, Tuple[int]]
    num_weights: int
    fq_name: str
    weight_node: ov.Node
    original_weight_dtype: TWeightType
    compression_config = WeightCompressionConfig()


# TODO: combine with power quant and int8 errors and with actual weight compression
def get_int8_err(wp: WeightNodeParams):
    weight = get_const_value(wp.weight_node)
    num_bits = 8

    level_low = 0
    level_high = 2**num_bits - 1

    min_values = np.min(weight, axis=wp.axes, keepdims=True)
    max_values = np.max(weight, axis=wp.axes, keepdims=True)

    scale, zero_point = calculate_scale_zero_point(min_values, max_values, level_low, level_high, narrow_range=False)

    compressed_weights = np.round(weight / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    compressed_weights[:10, :10]

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    # TODO: optimize max mean
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=1)
    # print(layer_err.shape)
    val = np.max(layer_err)
    return val


# def get_power_quant_error(layer):
#     if not type(layer) is NNCFLinear:
#         return -1.0
#     alpha = 0.5

#     level_high = 2**4 - 1
#     w_sign = torch.sign(layer.weight)
#     w_pow = torch.pow(torch.abs(layer.weight), alpha) * w_sign
#     # w_pow = torch.log(torch.abs(layer.weight)) * w_sign
#     target_dim = layer.target_weight_dim_for_compression
#     stat_dim = (target_dim + 1) % 2
#     input_low = torch.min(w_pow, dim=stat_dim)[0].detach()
#     input_high = torch.max(w_pow, dim=stat_dim)[0].detach()
#     scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

#     scale = scale.unsqueeze(stat_dim)
#     zero_point = zero_point.unsqueeze(stat_dim)
#     op = WeightsDecompressorPowerQuant(zero_point, scale, 1 / alpha)

#     compressed_weight = w_pow.data / scale + zero_point
#     compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

#     decompressed_weight = op.dequantize(compressed_weight)
#     diff = (decompressed_weight - layer.weight.data) ** 2

#     # mean_err = torch.mean(diff)
#     # return float(mean_err)
#     layer_err = torch.mean(diff, dim=1)
#     top_k = torch.topk(layer_err, 10)[0]
#     # val = float(mean_err)#top_k[0])
#     val = float(top_k[0])
#     return val


def get_power_quant_error(wp: WeightNodeParams):
    weight = get_const_value(wp.weight_node)
    alpha = 0.5
    num_bits = 4

    level_low = 0
    level_high = 2**num_bits - 1
    w_sign = np.sign(weight)

    w_pow = np.power(np.abs(weight), alpha) * w_sign
    min_values = np.min(w_pow, axis=wp.axes, keepdims=True)
    max_values = np.max(w_pow, axis=wp.axes, keepdims=True)

    scale, zero_point = calculate_scale_zero_point(min_values, max_values, level_low, level_high, narrow_range=False)

    compressed_weights = np.round(w_pow / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    # print(compressed_weights[:5, :5])

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (decompressed_weight - zero_point) * scale
    decompressed_weight = np.power(decompressed_weight, 1 / alpha) * w_sign

    # TODO: optimize
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=1)
    val = np.max(layer_err)
    print(val)
    return val


def get_relative_error(weight_node):
    return get_power_quant_error(weight_node) / (get_int8_err(weight_node) + 0.0000000001)


def insert_pre_compression_operations(
    model: ov.Model,
    mode: CompressWeightsMode = CompressWeightsMode.COMPRESSED_NF4,
    ratio: float = 0.5,
    group_size: int = -1,
) -> None:
    """
    Compress weights of Linear and Embedding layers to uint8.
    The result of compression is the same as asymmetric weight quantization.

    :param model: The model to be transformed.
    :param bits: Number of bits for quantization.
    """
    allowed_metatypes_to_const_port = {OVEmbeddingMetatype: [0], OVMatMulMetatype: [0, 1]}

    all_weight_params = []  # type: List[WeightNodeParams]
    for node in model.get_ordered_ops():
        metatype = get_node_metatype(node)
        if metatype not in allowed_metatypes_to_const_port:
            continue

        for const_port_id in allowed_metatypes_to_const_port[metatype]:
            weight_node = get_operation_const_op(node, const_port_id)
            if weight_node is None:
                continue

            weight_output = weight_node.output(0)
            weight_name = weight_node.get_friendly_name()
            target_inputs = weight_output.get_target_inputs()

            original_weight_dtype = weight_output.get_element_type().to_dtype()
            if original_weight_dtype not in [np.float32, np.float16, np.float64]:
                continue
            axes = _get_reduction_axes(metatype, node, const_port_id)
            fq_name = f"{node.get_friendly_name()}/fq_weights_{const_port_id}"
            weight = get_const_value(weight_node)
            num_weights = weight.size
            weight_params = WeightNodeParams(axes, num_weights, fq_name, weight_node, original_weight_dtype)
            all_weight_params.append(weight_params)

    if mode == CompressWeightsMode.COMPRESSED_NF4:
        total_num_weights = sum(wp.num_weights for wp in all_weight_params)
        # NOTE: first and last layer is always in 8 bit.
        num_internal_weights = total_num_weights - all_weight_params[0].num_weights - all_weight_params[-1].num_weights
        errors = []
        for weight_param in all_weight_params[1:-1]:
            print(weight_param.weight_node.get_friendly_name())
            error = get_relative_error(weight_param)
            errors.append(error)
        print(f"\errors: {errors}")

        layers = list(range(len(errors)))
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(layers, errors)

        # NOTE: index is defined in the array of all weight params by taking into account that errors were not
        # calculated for first and last layers.
        indexes_of_layers_in_ascending_order_of_errors = [
            i[0] + 1 for i in sorted(enumerate(errors), reverse=False, key=lambda x: x[1])
        ]
        print(f"\nindexes_of_layers_in_ascending_order_of_errors={indexes_of_layers_in_ascending_order_of_errors}")
        sorted_errors = [errors[i - 1] for i in indexes_of_layers_in_ascending_order_of_errors]
        print(f"\nsorted errors: {sorted_errors}")

        num_weights_in_4bit = 0
        for i, index in enumerate(indexes_of_layers_in_ascending_order_of_errors):
            weight_param = all_weight_params[index]
            current_ratio = (num_weights_in_4bit + weight_param.num_weights) / total_num_weights
            if current_ratio >= ratio:
                # for j in indexes_of_layers_in_ascending_order_of_errors[i:]:
                #     weight_param.compression_config = config_8bit
                boundary_error = errors[indexes_of_layers_in_ascending_order_of_errors[i - 1] - 1]
                plt.axhline(y=boundary_error, color="r", linestyle="-")
                plt.savefig("errors_with_boundary.png")
                plt.close(fig)
                break
            config_4bit = WeightCompressionConfig(num_bits=4, is_power_quant=True, group_size=group_size)
            weight_param.compression_config = config_4bit
            num_weights_in_4bit += weight_param.num_weights

        bit_config = [data.compression_config.num_bits for data in all_weight_params]
        print(bit_config)
        for data in all_weight_params:
            print(f"{data.num_weights / total_num_weights * 100:.3f}")

        print(f"{num_weights_in_4bit / total_num_weights * 100:.0f}% all weights in 4 bit")
        print(f"{num_weights_in_4bit / num_internal_weights * 100:.0f}% internal weights in 4 bit")

    for wp in all_weight_params:
        weight_node = wp.weight_node
        original_weight_dtype = wp.original_weight_dtype

        weight_output = weight_node.output(0)
        weight_name = weight_node.get_friendly_name()
        target_inputs = weight_output.get_target_inputs()

        weight = get_const_value(weight_node)

        min_values = np.min(weight, axis=wp.axes, keepdims=True)
        max_values = np.max(weight, axis=wp.axes, keepdims=True)
        config = wp.compression_config
        if config.is_power_quant:
            # TODO: take sqrt from min and max values
            pass

        level_low = 0
        level_high = 2**config.num_bits - 1

        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )

        compressed_weights = np.round(weight / scale + zero_point)
        compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)

        compressed_const = opset.constant(compressed_weights, dtype=np.uint8, name=weight_name)
        convert = opset.convert(compressed_const, original_weight_dtype)
        sub = opset.subtract(convert, zero_point.astype(original_weight_dtype))

        mul = opset.multiply(sub, scale.astype(original_weight_dtype), name=wp.fq_name)

        for target_input in target_inputs:
            target_input.replace_source_output(mul.output(0))


def _get_reduction_axes(metatype: Type[OperatorMetatype], node: ov.Node, weight_port_id: int) -> Union[int, Tuple[int]]:
    """
    Determines reduction axes by given metatype and node information.

    :param metatype: The metatype of the operator.
    :param node: The OpenVINO node.
    :param weight_port_id: The weight port ID.

    :return: The reduction axes as an integer or a tuple of integers.
    """
    if metatype is OVMatMulMetatype:
        transpose = node.get_attributes()[f"transpose_{'a' if weight_port_id == 0 else 'b'}"]
        ndims = node.input(weight_port_id).get_partial_shape().rank.get_max_length()
        channel_axes = get_matmul_channel_axes(weight_port_id, ndims, transpose)
        axes = tuple(i for i in range(ndims) if i not in channel_axes)
    elif metatype is OVEmbeddingMetatype:
        axes = (metatype.const_channel_axis[0] + 1) % 2
    else:
        RuntimeError("Unsupported metatype to find reduction axes.")
    return axes
